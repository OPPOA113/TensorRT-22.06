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

import os
import sys
import random

import numpy as np
from PIL import Image
import cv2

# from yolov5.utils.general import imwrite

class ImageBatcher:
    """
    Creates batches of pre-processed images.
    """

    def __init__(self, input, shape, dtype, max_num_images=None, exact_batches=False, 
                preprocessor="EfficientDet", shuffle_files=False,open_type="IPL"):
        """
        :param input: The input directory to read images from.
        :param shape: The tensor shape of the batch to prepare, either in NCHW or NHWC format.
        :param dtype: The (numpy) datatype to cast the batched data to.
        :param max_num_images: The maximum number of images to read from the directory.
        :param exact_batches: This defines how to handle a number of images that is not an exact multiple of the batch
        size. If false, it will pad the final batch with zeros to reach the batch size. If true, it will *remove* the
        last few images in excess of a batch size multiple, to guarantee batches are exact (useful for calibration).
        :param preprocessor: Set the preprocessor to use, depending on which network is being used.
        :param shuffle_files: Shuffle the list of files before batching.
        :open_type: yolo系列的读取方式,在preprocessor=="Yolo"时才生效; 可选["IPL","CV"]
        """
        # Find images in the given input path
        input = os.path.realpath(input)
        self.images = []

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        def is_image(path):
            return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions

        if os.path.isdir(input):
            self.images = [os.path.join(input, f) for f in os.listdir(input) if is_image(os.path.join(input, f))]
            self.images.sort()
            if shuffle_files:
                random.seed(47)
                random.shuffle(self.images)
        elif os.path.isfile(input):
            if is_image(input):
                self.images.append(input)
        self.num_images = len(self.images)
        if self.num_images < 1:
            print("No valid {} images found in {}".format("/".join(extensions), input))
            sys.exit(1)

        # Handle Tensor Shape
        self.dtype = dtype
        self.shape = shape
        assert len(self.shape) == 4
        self.batch_size = shape[0]
        assert self.batch_size > 0
        self.format = None
        self.width = -1
        self.height = -1
        if self.shape[1] == 3:
            self.format = "NCHW"
            self.height = self.shape[2]
            self.width = self.shape[3]
        elif self.shape[3] == 3:
            self.format = "NHWC"
            self.height = self.shape[1]
            self.width = self.shape[2]
        assert all([self.format, self.width > 0, self.height > 0])

        # Adapt the number of images as needed
        if max_num_images and 0 < max_num_images < len(self.images):
            self.num_images = max_num_images
        if exact_batches:
            self.num_images = self.batch_size * (self.num_images // self.batch_size)
        if self.num_images < 1:
            print("Not enough images to create batches")
            sys.exit(1)
        self.images = self.images[0:self.num_images]

        # Subdivide the list of images into batches
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self.batches.append(self.images[start:end])

        # Indices
        self.image_index = 0
        self.batch_index = 0

        self.preprocessor = preprocessor
        self.open_type = open_type

    def letterbox(slef, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
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

        if auto:  # minimum rectangle  保持宽高比，取最小的能整除stride的长宽
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch 非letterbox的resize, 没有padding
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
        
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
            # cv2.imwrite("resize.jpg",im)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        # cv2.imwrite("copyMakeBorder.jpg",im)
        return im, ratio, (dw, dh)        

    def preprocess_retina(self, input_image_path):
            """
            description: Read an image from image path, resize and pad it to target size,
                        normalize to [-1,1],transform to NCHW format.
            param:
                input_image_path: str, image path
            return:
                image:  the processed image
                image_raw: the original image
                h: original height
                w: original width
            """
            image_raw = cv2.imread(input_image_path)
            h, w, c = image_raw.shape

            # Calculate widht and height and paddings
            INPUT_W = self.width
            INPUT_H = self.height
            r_w = INPUT_W / w
            r_h = INPUT_H / h
            scale = 1.0 / min(r_w, r_h)
            if r_h > r_w:
                tw = INPUT_W
                th = int(r_w * h)
                tx1 = tx2 = 0
                ty1 = int((INPUT_H - th) / 2)
                ty2 = INPUT_H - th - ty1
            else:
                tw = int(r_h * w)
                th = INPUT_H
                tx1 = int((INPUT_W - tw) / 2)
                tx2 = INPUT_W - tw - tx1
                ty1 = ty2 = 0

            # Resize the image with long side while maintaining ratio
            image = cv2.resize(image_raw, (tw, th))
            # Pad the short side with (128,128,128)
            image = cv2.copyMakeBorder(
                image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
            )
            image = image.astype(np.float32)

            # HWC to CHW format:
            image -= (104, 117, 123)
            # image = np.transpose(image, [2, 0, 1])
            # CHW to NCHW format
            # image = np.expand_dims(image, axis=0)
            # Convert the image to row-major order, also known as "C order":
            # image = np.ascontiguousarray(image)
            return image, scale

    def preprocess_image(self, image_path):
        """
        The image preprocessor loads an image from disk and prepares it as needed for batching. This includes padding,
        resizing, normalization, data type casting, and transposing.
        This Image Batcher implements one algorithm for now:
        * EfficientDet: Resizes and pads the image to fit the input size.
        :param image_path: The path to the image on disk to load.
        :return: Two values: A numpy array holding the image sample, ready to be contacatenated into the rest of the
        batch, and the resize scale used, if any.
        """

        def resize_pad(image, pad_color=(0, 0, 0)):
            """
            A subroutine to implement padding and resizing. This will resize the image to fit fully within the input
            size, and pads the remaining bottom-right portions with the value provided.
            :param image: The PIL image object
            :pad_color: The RGB values to use for the padded area. Default: Black/Zeros.
            :return: Two values: The PIL image object already padded and cropped, and the resize scale used.
            """
            width, height = image.size
            width_scale = width / self.width
            height_scale = height / self.height
            scale = 1.0 / max(width_scale, height_scale)
            image = image.resize((round(width * scale), round(height * scale)), resample=Image.BILINEAR)
            pad = Image.new("RGB", (self.width, self.height))
            pad.paste(pad_color, [0, 0, self.width, self.height])
            pad.paste(image)
            return pad, scale

        scale = None
        image = Image.open(image_path)
        image = image.convert(mode='RGB')
        if self.preprocessor == "EfficientDet":
            # For EfficientNet V2: Resize & Pad with ImageNet mean values and keep as [0,255] Normalization
            image, scale = resize_pad(image, (124, 116, 104))
            image = np.asarray(image, dtype=self.dtype)
            # [0-1] Normalization, Mean subtraction and Std Dev scaling are part of the EfficientDet graph, so
            # no need to do it during preprocessing here
        elif self.preprocessor == "Yolo":
            if self.open_type == "IPL":
                # For Yolo v3 or v5: Resize & keep as [0,255] Normalization
                image, scale = resize_pad(image, (127, 127, 127))  # 保持宽高比，但粘贴在左上角
                image = np.asarray(image, dtype=self.dtype)
                image = image / 255.0
            elif self.open_type == "CV":
                image0 = cv2.imread(image_path)  # BGR                  """ self.stride """
                image, scale = self.letterbox(image0, (self.width,self.height), stride=32, auto=False, scaleup=False)[:2]
                image = np.asarray(image, dtype=self.dtype)
            else:
                print("open_type method {} not supported".format(self.open_type))
                sys.exit(1)
        elif self.preprocessor == "Retina":
            image,scale = self.preprocess_retina(image_path)
        else:
            print("Preprocessing method {} not supported".format(self.preprocessor))
            sys.exit(1)

        if self.format == "NCHW":
            image = np.transpose(image, (2, 0, 1))
        
        if self.preprocessor == "Yolo" and self.open_type == "CV":
            image = image[::-1]  # BGR to RGB
            image = np.ascontiguousarray(image)  # contiguous
            image = image / 255.0
            scale = scale[0]

        return image, scale

    def get_batch(self):
        """
        Retrieve the batches. This is a generator object, so you can use it within a loop as:
        for batch, images in batcher.get_batch():
           ...
        Or outside of a batch with the next() function.
        :return: A generator yielding three items per iteration: a numpy array holding a batch of images, the list of
        paths to the images loaded within this batch, and the list of resize scales for each image in the batch.
        """
        for i, batch_images in enumerate(self.batches):
            batch_data = np.zeros(self.shape, dtype=self.dtype)
            batch_scales = [None] * len(batch_images)
            for i, image in enumerate(batch_images):
                self.image_index += 1
                batch_data[i], batch_scales[i] = self.preprocess_image(image)
            self.batch_index += 1
            yield batch_data, batch_images, batch_scales
