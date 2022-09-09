#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author :          lance
@Email :  	  wangyl306@163.com
@Time  : 	2022/06/20 17:22:01
"""

import os
import sys
import xml.etree.ElementTree as ET


def convert(xml_list, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if len(xml_list) == 0:
        print("Error: no .xml files found in ground-truth")
        sys.exit()
    for tmp_file in xml_list:
        with open(
            os.path.join(
                save_dir, os.path.split(tmp_file)[-1].replace(".xml", ".txt")
            ),
            "a",
        ) as new_f:
            root = ET.parse(tmp_file).getroot()
            for obj in root.findall("object"):
                obj_name = obj.find("name").text
                bndbox = obj.find("bndbox")
                left = bndbox.find("xmin").text
                top = bndbox.find("ymin").text
                right = bndbox.find("xmax").text
                bottom = bndbox.find("ymax").text
                new_f.write(
                    "%s %s %s %s %s\n" % (obj_name, left, top, right, bottom)
                )

    print("Conversion completed!")


def generate_gt_txt(voc_path, image_set, save_dir):
    image_ids = (
        open(os.path.join(voc_path, "ImageSets/Main/%s.txt" % (image_set)))
        .read()
        .strip()
        .split()
    )
    xml_path_list = []
    for image_id in image_ids:
        # 判断是否包含后缀
        if os.path.splitext(image_id)[-1] in [".jpg", ".png", ".bmp", ".JPG"]:
            xml_file = os.path.join(
                voc_path, "Annotations", os.path.splitext(image_id)[0] + ".xml"
            )
        else:
            xml_file = os.path.join(voc_path, "Annotations", image_id + ".xml")

        xml_path_list.append(xml_file)
    convert(xml_path_list, save_dir)


if __name__ == "__main__":
    voc_path = "/home/lance/workspace/tools/sample_test/voc"
    image_set = "test"
    save_dir = "/home/lance/workspace/tools/sample_test/voc/gt_info"
    generate_gt_txt(voc_path, image_set, save_dir)
