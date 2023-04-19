#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from ast import Pass
import imp
import os
import sys
import time
import ctypes
import argparse
import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

from image_batcher import ImageBatcher
from visualize import visualize_detections

import torch
import torchvision


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


class TensorRTInfer:
    """
    Implements inference for the EfficientDet TensorRT engine.
    """

    def __init__(self, engine_path, nms_threshold=0.45, conf_thres=0.25, fclasses=None,max_det=1000, agnostic_nms=False):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        self.nms_threshold = nms_threshold
        self.conf_thres = conf_thres
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.fclasses = fclasses

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            shape = self.context.get_binding_shape(i)
            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                'index': i,
                'name': name,
                'dtype': dtype,
                'shape': list(shape),
                'allocation': allocation,
                'host_allocation': host_allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            print("{} '{}' with shape {} and dtype {}".format(
                "Input" if is_input else "Output",
                binding['name'], binding['shape'], binding['dtype']))

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, batch):
        """
        Execute inference on a batch of images.
        :param batch: A numpy array holding the image batch.
        :return A list of outputs as numpy arrays.
        """
        # Copy I/O and Execute
        cuda.memcpy_htod(self.inputs[0]['allocation'], batch)
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            cuda.memcpy_dtoh(self.outputs[o]['host_allocation'], self.outputs[o]['allocation'])
        return [o['host_allocation'] for o in self.outputs]

    def process(self, batch, scales=None, nms_threshold=None):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """
        # Run inference
        outputs = self.infer(batch)

        # Process the results
        nums = outputs[0]
        boxes = outputs[1]
        scores = outputs[2]
        classes = outputs[3]
        detections = []
        normalized = (np.max(boxes) < 2.0)
        for i in range(self.batch_size):
            detections.append([])
            for n in range(int(nums[i])):
                scale = self.inputs[0]['shape'][2] if normalized else 1.0
                if scales and i < len(scales):
                    scale /= scales[i]
                if nms_threshold and scores[i][n] < nms_threshold:
                    continue
                detections[i].append({
                    'ymin': boxes[i][n][0] * scale,
                    'xmin': boxes[i][n][1] * scale,
                    'ymax': boxes[i][n][2] * scale,
                    'xmax': boxes[i][n][3] * scale,
                    'score': scores[i][n],
                    'class': int(classes[i][n]),
                })
        return detections
    
    
    def clip_coords(self, boxes, shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                            labels=(), max_det=300):
        """Runs Non-Maximum Suppression (NMS) on inference results

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded

        return output



    def process_yolo(self, batch, scales=None, nms_threshold=None):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """
        # Run inference
        # print("process in")
        outputs = self.infer(batch)
        # print("len(outputs):", len(outputs))
        # print("outputs[3]", outputs[3].shape)
        # print(outputs[3][0,:10,:])

        pred = torch.tensor(outputs[3])
        # print("type(pred):",type(pred))
        pred = self.non_max_suppression(pred, self.conf_thres, self.nms_threshold, self.fclasses, self.agnostic_nms) #, max_det=self.max_det
        # print("pred[0].shape:",pred[0].shape)
        # print("pred:{}, type:{},len():{}".format(pred,type(pred),len(pred)))

        detections = []
        # Process predictions
        for i, det in enumerate(pred):  # per image
            det = det.numpy()
            # Rescale boxes from img_size to im0 size
            # det[:, :4] = self.scale_coords(img.shape[2:], det[:, :4], im0.shape).round() 
            # print("i:",i,"- scales:",scales)
            # print(det)
            detections.append([])
            for *xyxy, conf, cls in reversed(det):
                detections[i].append({
                'ymin': xyxy[0] ,
                'xmin': xyxy[1] ,
                'ymax': xyxy[2] ,
                'xmax': xyxy[3] ,
                'score': conf,
                'class': int(cls),
            })

        return detections

        # Process the results
        
        # normalized = (np.max(boxes) < 2.0)
        # for i in range(self.batch_size):
        #     
        #     for n in range(int(nums[i])):
        #         scale = self.inputs[0]['shape'][2] if normalized else 1.0
        #         if scales and i < len(scales):
        #             scale /= scales[i]
        #         if nms_threshold and scores[i][n] < nms_threshold:
        #             continue
        #         detections[i].append({
        #             'ymin': boxes[i][n][0] * scale,
        #             'xmin': boxes[i][n][1] * scale,
        #             'ymax': boxes[i][n][2] * scale,
        #             'xmax': boxes[i][n][3] * scale,
        #             'score': scores[i][n],
        #             'class': int(classes[i][n]),
        #         })
        


def main(args):
    if args.output:
        output_dir = os.path.realpath(args.output)
        os.makedirs(output_dir, exist_ok=True)

    labels = []
    if args.labels:
        with open(args.labels) as f:
            for i, label in enumerate(f):
                labels.append(label.strip())

    trt_infer = TensorRTInfer(args.engine, args.nms_threshold, args.conf_thres, args.fclasses,args.max_det, args.agnostic_nms)
    if args.input:
        print("Inferring data in {}".format(args.input))
        batcher = ImageBatcher(args.input, *trt_infer.input_spec(), preprocessor=args.preprocessor)
        for batch, images, scales in batcher.get_batch():
            print("Processing Image {} / {}".format(batcher.image_index, batcher.num_images), end="\r")
            if args.preprocessor=="EfficientDet":
                detections = trt_infer.process(batch, scales, args.nms_threshold)
                if args.output:
                    for i in range(len(images)):
                        basename = os.path.splitext(os.path.basename(images[i]))[0]
                        # Image Visualizations
                        output_path = os.path.join(output_dir, "{}.png".format(basename))
                        visualize_detections(images[i], output_path, detections[i], labels)
                        # Text Results
                        output_results = ""
                        for d in detections[i]:
                            line = [d['xmin'], d['ymin'], d['xmax'], d['ymax'], d['score'], d['class']]
                            output_results += "\t".join([str(f) for f in line]) + "\n"
                        with open(os.path.join(output_dir, "{}.txt".format(basename)), "w") as f:
                            f.write(output_results)
            elif args.preprocessor=="Yolo":
                detections = trt_infer.process_yolo(batch, scales, args.nms_threshold)
                if args.output:
                    for i in range(len(images)):
                        basename = os.path.splitext(os.path.basename(images[i]))[0]
                        # Image Visualizations
                        output_path = os.path.join(output_dir, "{}.png".format(basename))
                        visualize_detections(images[i], output_path, detections[i], labels)
                        # Text Results
                        output_results = ""
                        for d in detections[i]:
                            line = [d['xmin'], d['ymin'], d['xmax'], d['ymax'], d['score'], d['class']]
                            output_results += "\t".join([str(f) for f in line]) + "\n"
                        with open(os.path.join(output_dir, "{}.txt".format(basename)), "w") as f:
                            f.write(output_results)

    else:
        print("No input provided, running in benchmark mode")
        spec = trt_infer.input_spec()
        batch = 255 * np.random.rand(*spec[0]).astype(spec[1])

        iterations = 200
        times = []
        for i in range(20):  # GPU warmup iterations
            trt_infer.infer(batch)
        for i in range(iterations):
            start = time.time()
            trt_infer.infer(batch)
            times.append(time.time() - start)
            print("Iteration {} / {}".format(i + 1, iterations), end="\r")
        print("Benchmark results include time for H2D and D2H memory copies")
        print("Average Latency: {:.3f} ms".format(
            1000 * np.average(times)))
        print("Average Throughput: {:.1f} ips".format(
            trt_infer.batch_size / np.average(times)))

    print()
    print("Finished Processing")

# /////////////////////////////////////////////////////////////////////////////////////////////
import torch.utils.data as data
import glob
import cv2
# from yolov5.utils.augmentations import letterbox
from collections import OrderedDict, namedtuple
import logging

def letterbox2(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class dataset_of_gs():
    def __init__(self, img_dir, number_zoom, img_size) -> None:
        super().__init__()
        print(f"image dir:{img_dir}")
        self.imageslist = glob.glob(img_dir + "/*.jpg") 
        self.number_zoom = number_zoom
        self.numbers = len(self.imageslist)
        self.count = 0 
        self.img_size = img_size
    
    def _processor(self, path):
        self.count += 1
        im0 = cv2.imread(path)  # BGR
        assert im0 is not None, f'Image Not Found {path}'
        s = f'image {self.count}/{self.numbers} {path}: '
        print(s)
        im = letterbox2(im0, (self.img_size, self.img_size), stride=32, auto=False)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # im = np.ascontiguousarray(im)  # contiguous
        # im = torch.from_numpy(im).to("cuda:0")
        # im = im.half() #if model.fp16 else im.float()  # uint8 to fp16/32
        # im = np.asarray(im,dtype=np.float)
        # print(im.shape)
        # if len(im.shape) == 3:
        #     im = im[None]  # expand for batch dim
        im = np.ascontiguousarray(im)  # contiguous
        im = im / 255.0  # 0 - 255 to 0.0 - 1.0
        return im


    def __len__(self):
        return  self.numbers * self.number_zoom

    def __getitem__(self, index) :
        index = index % self.numbers
        return self._processor(self.imageslist[index])


class DetectMultiBackend2():
    def __init__(self, weights='yolov5s.engine', device=torch.device('cpu'), img_size=320, 
                    fp16=False, fuse=True):
        LOGGER = logging.getLogger("yolov5")
        LOGGER.info(f'Loading {weights} for TensorRT inference...')
        
        logger = trt.Logger(trt.Logger.INFO)
        self.device = device
        if self.device.type == 'cpu':
            self.device = torch.device('cuda:0')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))

        with open(weights, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        # self.cfx= cuda.Device(0).make_context()
        self.context = model.create_execution_context()
        self.bindings = OrderedDict()
        self.fp16 = False  # default updated below
        self.dynamic = False
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            if model.binding_is_input(index):
                # if -1 in tuple(model.get_binding_shape(index)):  # dynamic
                #     self.dynamic = True
                #     self.context.set_binding_shape(index, tuple(model.get_profile_shape(0, index)[2]))
                if dtype == np.float16:
                    self.fp16 = True
            shape = tuple(self.context.get_binding_shape(index))
            print(f"bindings index:{index}, name:{name},dtype:{dtype},shape:{shape},is_input:{model.binding_is_input(index)}")
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.batch_size = self.bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        # batch_size = bindings['input'].shape[0]
        print("init done,self.dynamic:{}",self.dynamic)

    def infer(self, im,  val=False):
        
        # b, ch, h, w = im.shape  # batch, channel, height, width
        # if self.fp16 and im.dtype != torch.float16:
        #     im = im.half()  # to FP16
        #     print("im.dtype:{im.dtype}")
        # print(f"b:{b}, ch:{ch}, h:{h}, w:{w}, im.dtype:{im.dtype},im.device:{im.device}")
        # if self.dynamic and im.shape != self.bindings['images'].shape:
        #     i_in, i_out = (self.model.get_binding_index(x) for x in ('images', 'output'))
        #     self.context.set_binding_shape(i_in, im.shape)  # reshape if dynamic
        #     self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
        #     self.bindings['output'].data.resize_(tuple(self.context.get_binding_shape(i_out)))
        s = self.bindings['images'].shape
        # s = self.bindings['input'].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs['images'] = int(im.data_ptr())
        # self.cfx.push()
        self.context.execute_v2(list(self.binding_addrs.values()))
        # self.cfx.pop()
        y = self.bindings['output'].data
        
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return (y, []) if val else y
            



def main_yolo_reposity(args):
    device = torch.device("cuda:0")
    model = DetectMultiBackend2(args.engine, device=device, img_size=args.img_size, fp16=True)

    gsdata = dataset_of_gs(args.input, number_zoom=1, img_size=416)
    # gs_data_loader = data.DataLoader(gsdata, batch_size=1, num_workers=1, pin_memory=True )
    # for img in gs_data_loader:
    for idx in range(len(gsdata)):
        img = gsdata[idx]
        print(f"1 - img.shape:{img.shape}, type:{type(img)}, img.device:{img.device}")
        model.infer(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", default="/workspace/TensorRT/samples/python/yolo_detection/output/engine/yolov5s_1_416_int8.engine", #required=True,
                        help="The serialized TensorRT engine")
    parser.add_argument("-i", "--input", default="/workspace/project/dataset/benchmarch_data/detection/test",#None,#
                        help="Path to the image or directory to process")
    parser.add_argument("--img_size", type=int, default=416,help="image input size for inference")
    parser.add_argument("-o", "--output", default="./output", #None,#
                        help="Directory where to save the visualization results")
    parser.add_argument("-l", "--labels", default="./labels_coco.txt",
                        help="File to use for reading the class labels from, default: ./labels_coco.txt")
    parser.add_argument("-t", "--nms_threshold", type=float, default=0.45,
                        help="Override the score threshold for the NMS operation, if higher than the built-in threshold")
    
    parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--max_det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS  不按类别进行nms')
    parser.add_argument('--fclasses', nargs='+', type=int, help='只取特定类别的目标,filter by class: --classes 0, or --classes 0 2 3')


    parser.add_argument("-p", "--preprocessor", default="Yolo", choices=["EfficientDet", "Yolo"],
                        help="Select the image preprocessor to use, either 'Yolo', 'EfficientDet', default: Yolo")                        
    args = parser.parse_args()
    # main(args)
    main_yolo_reposity(args)
