#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author :       	lance
@Email :  lance.wang@vastaitech.com
@Time  : 	2021/11/11 13:56:52
"""

"""
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
"""
import glob
import json
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .tools import (check, create_temp_files, draw_plot_func,
                    draw_text_in_image, error, file_lines_to_list,
                    is_float_between_0_and_1, log_average_miss_rate)


class Map:
    def __init__(
        self,
        gt_path,
        dt_path,
        no_animation=False,
        no_plot=False,
        quiet=True,
        overlap=0.5,
        ignore_classes=[],
        set_class_iou={},
        IMG_PATH="",
        output_path="./",
    ):
        """[map]

        Args:
            no_animation (bool, optional): [是否将检测结果打印在图片上]. Defaults to True.
            no_plot (bool, optional): [是否显示指标图]. Defaults to True.
            quiet (bool, optional): [只打印最终map]. Defaults to True.
            overlap (float, optional): [iou]. Defaults to 0.5.
            ignore_classes (list, optional): [不计入map统计的类别，形式为[person,book,...]]. Defaults to [].
            set_class_iou (dict, optional): [为指定类别设计独立的iou,形式为{"person":0.3,"book":0.7}]. Defaults to {}.
        """
        # 初始化超参数
        self.no_animation = no_animation
        self.no_plot = no_plot
        self.quiet = quiet
        self.overlap = overlap
        self.ignore_classes = ignore_classes
        self.set_class_iou = set_class_iou

        self.GT_PATH = gt_path
        self.DR_PATH = dt_path

        # 1.检查检测结果和真实标注文件是否对齐
        check(GT_PATH=self.GT_PATH, DR_PATH=self.DR_PATH)

        # 2.检查画图以及是否有图像存在，则相应得画图和画框参数动态改变
        self.IMG_PATH = IMG_PATH
        if os.path.exists(self.IMG_PATH):
            if len(os.listdir(self.IMG_PATH)) == 0:
                self.no_animation = True
        else:
            self.no_animation = True

        self.show_animation = False
        if not self.no_animation:
            try:
                import cv2

                self.show_animation = True
            except ImportError:
                print('"opencv-python" not found, please install to visualize the results.')
                self.no_animation = True

        # try to import Matplotlib if the user didn't choose the option --no-plot
        self.draw_plot = False
        if not self.no_plot:
            try:
                import matplotlib.pyplot as plt

                self.draw_plot = True
            except ImportError:
                print('"matplotlib" not found, please install it to get the resulting plots.')
                self.no_plot = True

        # 3.生成临时文件用来保存所有类的结果
        self.output_path = output_path
        self.temp_files_path, self.output_files_path = create_temp_files(
            self.show_animation, self.draw_plot, self.output_path
        )

        # 4.获取gt_files相关信息并按类别保存在temp_files,此时会判断是否有忽略的class
        self.ground_truth_files_list = glob.glob(self.GT_PATH + "/*.txt")
        self.ground_truth_files_list.sort()
        # dictionary with counter per class
        self.gt_counter_per_class = {}
        self.counter_images_per_class = {}
        self.gt_files = []
        self.gt_classes, self.n_classes = self.make_gt_files()

        # 5.检查指定class的iou阈值
        if self.set_class_iou != {}:
            self.check_spec_iou()

        # 6.获取dr_files相关信息并按类别保存在temp_files
        self.dr_files_list = glob.glob(self.DR_PATH + "/*.txt")
        self.dr_files_list.sort()
        self.make_dr_files()

        # 7.读取本地temp_files按类别计算ap,删除temp_files
        self.mAP, self.ap_dictionary, self.lamr_dictionary, self.count_true_positives = self.calculate_ap()

        # 8.画图 & 输出结果图
        self.output_with_ap(self.mAP, self.ap_dictionary, self.lamr_dictionary, self.count_true_positives)

    def voc_ap(self, rec, prec):
        """
        --- Official matlab code VOC2012---
        mrec=[0 ; rec ; 1];
        mpre=[0 ; prec ; 0];
        for i=numel(mpre)-1:-1:1
                mpre(i)=max(mpre(i),mpre(i+1));
        end
        i=find(mrec(2:end)~=mrec(1:end-1))+1;
        ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        rec.insert(0, 0.0)  # insert 0.0 at begining of list
        rec.append(1.0)  # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0)  # insert 0.0 at begining of list
        prec.append(0.0)  # insert 0.0 at end of list
        mpre = prec[:]
        """
        This part makes the precision monotonically decreasing
            (goes from the end to the beginning)
            matlab: for i=numel(mpre)-1:-1:1
                        mpre(i)=max(mpre(i),mpre(i+1));
        """
        # matlab indexes start in 1 but python in 0, so I have to do:
        #     range(start=(len(mpre) - 2), end=0, step=-1)
        # also the python function range excludes the end, resulting in:
        #     range(start=(len(mpre) - 2), end=-1, step=-1)
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        """
        This part creates a list of indexes where the recall changes
            matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
        """
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                i_list.append(i)  # if it was matlab would be i + 1
        """
        The Average Precision (AP) is the area under the curve
            (numerical integration)
            matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        ap = 0.0
        for i in i_list:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
        return ap, mrec, mpre

    def make_gt_files(self):
        for txt_file in self.ground_truth_files_list:
            # print(txt_file)
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            # check if there is a correspondent detection-results file
            temp_path = os.path.join(self.DR_PATH, (file_id + ".txt"))
            if not os.path.exists(temp_path):
                error_msg = "Error. File not found: {}\n".format(temp_path)
                error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                error(error_msg)
            lines_list = file_lines_to_list(txt_file)
            # create ground-truth dictionary
            bounding_boxes = []
            is_difficult = False
            already_seen_classes = []
            for line in lines_list:
                try:
                    if "difficult" in line:
                        class_name, left, top, right, bottom, _difficult = line.split()
                        is_difficult = True
                    else:
                        class_name, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                    error_msg += " Received: " + line
                    error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                    error_msg += 'by running the script "remove_space.py" or "rename_class.py" in the "extra/" folder.'
                    error(error_msg)
                # check if class is in the ignore list, if yes skip
                if class_name in self.ignore_classes:
                    continue
                bbox = left + " " + top + " " + right + " " + bottom
                if is_difficult:
                    bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                    is_difficult = False
                else:
                    bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                    # count that object
                    if class_name in self.gt_counter_per_class:
                        self.gt_counter_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        self.gt_counter_per_class[class_name] = 1

                    if class_name not in already_seen_classes:
                        if class_name in self.counter_images_per_class:
                            self.counter_images_per_class[class_name] += 1
                        else:
                            # if class didn't exist yet
                            self.counter_images_per_class[class_name] = 1
                        already_seen_classes.append(class_name)

            # dump bounding_boxes into a ".json" file
            new_temp_file = self.temp_files_path + "/" + file_id + "_ground_truth.json"
            self.gt_files.append(new_temp_file)
            with open(new_temp_file, "w") as outfile:
                json.dump(bounding_boxes, outfile)

        gt_classes = list(self.gt_counter_per_class.keys())
        # let's sort the classes alphabetically
        gt_classes = sorted(gt_classes)
        n_classes = len(gt_classes)
        # print(gt_classes)
        # print(gt_counter_per_class)
        return gt_classes, n_classes

    def check_spec_iou(self):
        # specific_iou_classes = ['class_1', 'class_2']
        # iou_list = ['IoU_1', 'IoU_2']

        for tmp_class in self.set_class_iou.keys():
            if tmp_class not in self.gt_classes:
                # 没有在列表中的将直接删除键值对
                del self.set_class_iou[tmp_class]
                error('Error, unknown class "' + tmp_class)
        for num in self.set_class_iou.values():
            if not is_float_between_0_and_1(num):
                error("Error, IoU must be between 0.0 and 1.0")

    def make_dr_files(self):
        for class_index, class_name in enumerate(self.gt_classes):
            bounding_boxes = []
            for txt_file in self.dr_files_list:
                # print(txt_file)
                # the first time it checks if all the corresponding ground-truth files exist
                file_id = txt_file.split(".txt", 1)[0]
                file_id = os.path.basename(os.path.normpath(file_id))
                temp_path = os.path.join(self.GT_PATH, (file_id + ".txt"))
                if class_index == 0:
                    if not os.path.exists(temp_path):
                        error_msg = "Error. File not found: {}\n".format(temp_path)
                        error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                        error(error_msg)
                lines = file_lines_to_list(txt_file)
                for line in lines:
                    try:
                        tmp_class_name, confidence, left, top, right, bottom = line.split()
                    except ValueError:
                        # TODO 空样本的处理
                        error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                        error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                        error_msg += " Received: " + line
                        error(error_msg)
                    if tmp_class_name == class_name:
                        # print("match")
                        bbox = left + " " + top + " " + right + " " + bottom
                        bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
                        # print(bounding_boxes)
            # sort detection-results by decreasing confidence
            bounding_boxes.sort(key=lambda x: float(x["confidence"]), reverse=True)
            with open(self.temp_files_path + "/" + class_name + "_dr.json", "w") as outfile:
                json.dump(bounding_boxes, outfile)

    def calculate_ap(self):
        sum_AP = 0.0
        ap_dictionary = {}
        lamr_dictionary = {}
        # open file to store the output
        with open(self.output_files_path + "/output.txt", "w") as output_file:
            output_file.write("# AP and precision/recall per class\n")
            count_true_positives = {}
            for class_index, class_name in enumerate(self.gt_classes):
                count_true_positives[class_name] = 0
                """
                Load detection-results of that class
                """
                dr_file = self.temp_files_path + "/" + class_name + "_dr.json"
                dr_data = json.load(open(dr_file))

                """
                Assign detection-results to ground-truth objects
                """
                nd = len(dr_data)
                tp = [0] * nd  # creates an array of zeros of size nd
                fp = [0] * nd
                for idx, detection in enumerate(dr_data):
                    file_id = detection["file_id"]
                    if self.show_animation:
                        # find ground truth image
                        ground_truth_img = glob.glob1(self.IMG_PATH, file_id + ".*")
                        # tifCounter = len(glob.glob1(myPath,"*.tif"))
                        if len(ground_truth_img) == 0:
                            error("Error. Image not found with id: " + file_id)
                        elif len(ground_truth_img) > 1:
                            error("Error. Multiple image with id: " + file_id)
                        else:  # found image
                            # print(IMG_PATH + "/" + ground_truth_img[0])
                            # Load image
                            # print("img:", os.path.join(IMG_PATH, ground_truth_img[0]))
                            # img = cv2.imread(os.path.join(IMG_PATH, ground_truth_img[0]))
                            img = cv2.imdecode(
                                np.fromfile(os.path.join(self.IMG_PATH, ground_truth_img[0]), dtype=np.uint8), -1
                            )
                            # load image with draws of multiple detections
                            img_cumulative_path = self.output_files_path + "/images/" + ground_truth_img[0]
                            if os.path.isfile(img_cumulative_path):
                                img_cumulative = cv2.imread(img_cumulative_path)
                            else:
                                img_cumulative = img.copy()
                            # Add bottom border to image
                            bottom_border = 60
                            BLACK = [0, 0, 0]
                            img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
                    # assign detection-results to ground truth object if any
                    # open ground-truth with that file_id
                    gt_file = self.temp_files_path + "/" + file_id + "_ground_truth.json"
                    ground_truth_data = json.load(open(gt_file))
                    ovmax = -1
                    gt_match = -1
                    # load detected object bounding-box
                    bb = [float(x) for x in detection["bbox"].split()]
                    for obj in ground_truth_data:
                        # look for a class_name match
                        if obj["class_name"] == class_name:
                            bbgt = [float(x) for x in obj["bbox"].split()]
                            bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                            iw = bi[2] - bi[0] + 1
                            ih = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:
                                # compute overlap (IoU) = area of intersection / area of union
                                ua = (
                                    (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
                                    + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1)
                                    - iw * ih
                                )
                                ov = iw * ih / ua
                                if ov > ovmax:
                                    ovmax = ov
                                    gt_match = obj

                    # assign detection as true positive/don't care/false positive
                    if self.show_animation:
                        status = "NO MATCH FOUND!"  # status is only used in the animation
                    # set minimum overlap
                    min_overlap = self.overlap
                    if self.set_class_iou != {}:
                        if class_name in self.specific_iou_classes.keys():
                            min_overlap = float(self.specific_iou_classes[class_name])
                    if ovmax >= min_overlap:
                        if "difficult" not in gt_match:
                            if not bool(gt_match["used"]):
                                # true positive
                                tp[idx] = 1
                                gt_match["used"] = True
                                count_true_positives[class_name] += 1
                                # update the ".json" file
                                with open(gt_file, "w") as f:
                                    f.write(json.dumps(ground_truth_data))
                                if self.show_animation:
                                    status = "MATCH!"
                            else:
                                # false positive (multiple detection)
                                fp[idx] = 1
                                if self.show_animation:
                                    status = "REPEATED MATCH!"
                    else:
                        # false positive
                        fp[idx] = 1
                        if ovmax > 0:
                            status = "INSUFFICIENT OVERLAP"

                    """
                    Draw image to show animation
                    """
                    if self.show_animation:
                        height, widht = img.shape[:2]
                        # colors (OpenCV works with BGR)
                        white = (255, 255, 255)
                        light_blue = (255, 200, 100)
                        green = (0, 255, 0)
                        light_red = (30, 30, 255)
                        # 1st line
                        margin = 10
                        v_pos = int(height - margin - (bottom_border / 2.0))
                        text = "Image: " + ground_truth_img[0] + " "
                        img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                        text = "Class [" + str(class_index) + "/" + str(self.n_classes) + "]: " + class_name + " "
                        img, line_width = draw_text_in_image(
                            img, text, (margin + line_width, v_pos), light_blue, line_width
                        )
                        if ovmax != -1:
                            color = light_red
                            if status == "INSUFFICIENT OVERLAP":
                                text = "IoU: {0:.2f}% ".format(ovmax * 100) + "< {0:.2f}% ".format(min_overlap * 100)
                            else:
                                text = "IoU: {0:.2f}% ".format(ovmax * 100) + ">= {0:.2f}% ".format(min_overlap * 100)
                                color = green
                            img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                        # 2nd line
                        v_pos += int(bottom_border / 2.0)
                        rank_pos = str(idx + 1)  # rank position (idx starts at 0)
                        text = (
                            "Detection #rank: "
                            + rank_pos
                            + " confidence: {0:.2f}% ".format(float(detection["confidence"]) * 100)
                        )
                        img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                        color = light_red
                        if status == "MATCH!":
                            color = green
                        text = "Result: " + status + " "
                        img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        if ovmax > 0:  # if there is intersections between the bounding-boxes
                            bbgt = [int(round(float(x))) for x in gt_match["bbox"].split()]
                            cv2.rectangle(img, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                            cv2.rectangle(img_cumulative, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                            cv2.putText(
                                img_cumulative,
                                class_name,
                                (bbgt[0], bbgt[1] - 5),
                                font,
                                0.6,
                                light_blue,
                                1,
                                cv2.LINE_AA,
                            )
                        bb = [int(i) for i in bb]
                        cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                        cv2.rectangle(img_cumulative, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                        cv2.putText(img_cumulative, class_name, (bb[0], bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
                        # show image
                        # cv2.imshow("Animation", img)
                        # cv2.waitKey(20)  # show for 20 ms
                        # save image to output
                        output_img_path = (
                            self.output_files_path
                            + "/images/detections_one_by_one/"
                            + class_name
                            + "_detection"
                            + str(idx)
                            + ".jpg"
                        )
                        cv2.imwrite(output_img_path, img)
                        # save the image with all the objects drawn to it
                        cv2.imwrite(img_cumulative_path, img_cumulative)

                # print(tp)
                # compute precision/recall
                cumsum = 0
                for idx, val in enumerate(fp):
                    fp[idx] += cumsum
                    cumsum += val
                cumsum = 0
                for idx, val in enumerate(tp):
                    tp[idx] += cumsum
                    cumsum += val
                # print(tp)
                rec = tp[:]
                for idx, val in enumerate(tp):
                    rec[idx] = float(tp[idx]) / self.gt_counter_per_class[class_name]
                # print(rec)
                prec = tp[:]
                for idx, val in enumerate(tp):
                    prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
                # print(prec)

                ap, mrec, mprec = self.voc_ap(rec[:], prec[:])
                sum_AP += ap
                text = (
                    "{0:.2f}%".format(ap * 100) + " = " + class_name + " AP "
                )  # class_name + " AP = {0:.2f}%".format(ap*100)
                """
                Write to output.txt
                """
                rounded_prec = ["%.2f" % elem for elem in prec]
                rounded_rec = ["%.2f" % elem for elem in rec]
                output_file.write(
                    text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n"
                )
                if not self.quiet:
                    print(text)
                ap_dictionary[class_name] = ap

                n_images = self.counter_images_per_class[class_name]
                lamr, mr, fppi = log_average_miss_rate(np.array(prec), np.array(rec), n_images)
                lamr_dictionary[class_name] = lamr

                """
                Draw plot
                """
                if not self.no_plot:
                    plt.plot(rec, prec, "-o")
                    # add a new penultimate point to the list (mrec[-2], 0.0)
                    # since the last line segment (and respective area) do not affect the AP value
                    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                    area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                    plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor="r")
                    # set window title
                    fig = plt.gcf()  # gcf - get current figure
                    fig.canvas.set_window_title("AP " + class_name)
                    # set plot title
                    plt.title("class: " + text)
                    # plt.suptitle('This is a somewhat long figure title', fontsize=16)
                    # set axis titles
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    # optional - set axes
                    axes = plt.gca()  # gca - get current axes
                    axes.set_xlim([0.0, 1.0])
                    axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
                    # Alternative option -> wait for button to be pressed
                    # while not plt.waitforbuttonpress(): pass # wait for key display
                    # Alternative option -> normal display
                    # plt.show()
                    # save the plot
                    fig.savefig(self.output_files_path + "/classes/" + class_name + ".png")
                    plt.cla()  # clear axes for next plot

            if self.show_animation:
                cv2.destroyAllWindows()

            output_file.write("\n# mAP of all classes\n")
            mAP = sum_AP / self.n_classes
            text = "mAP = {0:.2f}%".format(mAP * 100)
            output_file.write(text + "\n")
            # print(text)

        """
        Draw false negatives
        """
        if self.show_animation:
            pink = (203, 192, 255)
            for tmp_file in self.gt_files:
                ground_truth_data = json.load(open(tmp_file))
                # print(ground_truth_data)
                # get name of corresponding image
                start = self.temp_files_path + "/"
                img_id = tmp_file[tmp_file.find(start) + len(start) : tmp_file.rfind("_ground_truth.json")]
                img_cumulative_path = self.output_files_path + "/images/" + img_id + ".jpg"
                img = cv2.imread(img_cumulative_path)
                if img is None:
                    img_path = self.IMG_PATH + "/" + img_id + ".jpg"
                    img = cv2.imread(img_path)
                # draw false negatives
                for obj in ground_truth_data:
                    if not obj["used"]:
                        bbgt = [int(round(float(x))) for x in obj["bbox"].split()]
                        cv2.rectangle(img, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), pink, 2)
                cv2.imwrite(img_cumulative_path, img)

        # remove the temp_files directory
        shutil.rmtree(self.temp_files_path)

        return mAP, ap_dictionary, lamr_dictionary, count_true_positives

    def output_with_ap(self, mAP, ap_dictionary, lamr_dictionary, count_true_positives):
        # Count total of detection-results
        # iterate through all the files
        det_counter_per_class = {}
        for txt_file in self.dr_files_list:
            # get lines to list
            lines_list = file_lines_to_list(txt_file)
            for line in lines_list:
                class_name = line.split()[0]
                # check if class is in the ignore list, if yes skip
                if class_name in self.ignore_classes:
                    continue
                # count that object
                if class_name in det_counter_per_class:
                    det_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    det_counter_per_class[class_name] = 1
        # print(det_counter_per_class)
        dr_classes = list(det_counter_per_class.keys())

        """
        Plot the total number of occurences of each class in the ground-truth
        """
        if self.draw_plot:
            window_title = "ground-truth-info"
            plot_title = "ground-truth\n"
            plot_title += (
                "(" + str(len(self.ground_truth_files_list)) + " files and " + str(self.n_classes) + " classes)"
            )
            x_label = "Number of objects per class"
            output_path = self.output_files_path + "/ground-truth-info.png"
            to_show = False
            plot_color = "forestgreen"
            draw_plot_func(
                self.gt_counter_per_class,
                self.n_classes,
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                "",
            )

        """
        Write number of ground-truth objects per class to results.txt
        """
        with open(self.output_files_path + "/output.txt", "a") as output_file:
            output_file.write("\n# Number of ground-truth objects per class\n")
            for class_name in sorted(self.gt_counter_per_class):
                output_file.write(class_name + ": " + str(self.gt_counter_per_class[class_name]) + "\n")

        """
        Finish counting true positives
        """
        for class_name in dr_classes:
            # if class exists in detection-result but not in ground-truth then there are no true positives in that class
            if class_name not in self.gt_classes:
                count_true_positives[class_name] = 0
        # print(count_true_positives)

        """
        Plot the total number of occurences of each class in the "detection-results" folder
        """
        if self.draw_plot:
            window_title = "detection-results-info"
            # Plot title
            plot_title = "detection-results\n"
            plot_title += "(" + str(len(self.dr_files_list)) + " files and "
            count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
            plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
            # end Plot title
            x_label = "Number of objects per class"
            output_path = self.output_files_path + "/detection-results-info.png"
            to_show = False
            plot_color = "forestgreen"
            true_p_bar = count_true_positives
            draw_plot_func(
                det_counter_per_class,
                len(det_counter_per_class),
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                true_p_bar,
            )

        """
        Write number of detected objects per class to output.txt
        """
        with open(self.output_files_path + "/output.txt", "a") as output_file:
            output_file.write("\n# Number of detected objects per class\n")
            for class_name in sorted(dr_classes):
                n_det = det_counter_per_class[class_name]
                text = class_name + ": " + str(n_det)
                text += " (tp:" + str(count_true_positives[class_name]) + ""
                text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
                output_file.write(text)

        """
        Draw log-average miss rate plot (Show lamr of all classes in decreasing order)
        """
        if self.draw_plot:
            window_title = "lamr"
            plot_title = "log-average miss rate"
            x_label = "log-average miss rate"
            output_path = self.output_files_path + "/lamr.png"
            to_show = False
            plot_color = "royalblue"
            draw_plot_func(
                lamr_dictionary,
                self.n_classes,
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                "",
            )

        """
        Draw mAP plot (Show AP's of all classes in decreasing order)
        """
        if self.draw_plot:
            window_title = "mAP"
            plot_title = "mAP = {0:.2f}%".format(mAP * 100)
            x_label = "Average Precision"
            output_path = self.output_files_path + "/mAP.png"
            to_show = True
            plot_color = "royalblue"
            draw_plot_func(
                ap_dictionary, self.n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, "",
            )


if __name__ == "__main__":
    gt_path = "/home/lance/workspace/tools/sample_test/voc/gt_info"
    dt_path = "/home/lance/workspace/tools/sample_test/voc/dt_info"
    map = Map(gt_path, dt_path, no_animation=False, no_plot=False)
    print("================================================================================")
    print("mAP:", map.mAP)
    print("map.ap_dictionary:", map.ap_dictionary)
    print("map.lamr_dictionary:", map.lamr_dictionary)
    print("map.count_true_positives:", map.count_true_positives)
