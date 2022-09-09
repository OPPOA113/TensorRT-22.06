#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author :          lance
@Email :  	  wangyl306@163.com
@Time  : 	2022/07/07 11:29:49

支持resnet系列, torchvision的pth模型转为onnx

"""

import argparse
import os

import thop
import torch
from thop import clever_format

parse = argparse.ArgumentParser(description="MAKE MODELS CLS FOR tensorrt")
parse.add_argument("--model_library", type=str, default="torchvision", choices=["timm", "torchvision"])
parse.add_argument("--model_name", type=str, default="resnet50")
parse.add_argument("--save_dir", type=str, default=r"./torchvision_onnx/")
parse.add_argument("--size", type=int, default=224)
parse.add_argument("--batchsize", type=int, default=1)
parse.add_argument(
    "--pretrained_weights",
    type=str,
    default=r"./torchvision/resnet50-19c8e357.pth",
    help="timm or torchvision or custom onnx weights path",
)
parse.add_argument(
    "--convert_mode",
    type=str,
    default="onnx",
    choices=["onnx", "torchscript"],
)
args = parse.parse_args()
print(args)


class ModelHUb:
    def __init__(self, opt):
        self.model_name = opt.model_name
        self.pretrained_weights = opt.pretrained_weights
        self.convert_mode = opt.convert_mode
        self.num_class = 1000
        self.img = torch.randn(opt.batchsize, 3, opt.size, opt.size)
        self.save_file = os.path.join(opt.save_dir, self.model_name +"_"+ str(opt.batchsize)+"_"+str(opt.size)+  "." + self.convert_mode)
        if opt.model_library == "timm":
            self.model = self._get_model_timm()
        else:
            self.model = self._get_model_torchvision()

        count_op(self.model, self.img)

    def get_model(self):
        if self.convert_mode == "onnx":
            torch.onnx.export(self.model, self.img, self.save_file, input_names=["input"], opset_version=10)
        else:
            self.model(self.img)  # dry runs
            scripted_model = torch.jit.trace(self.model, self.img, strict=False)
            torch.jit.save(scripted_model, self.save_file)
        print("[INFO] convert model save:", self.save_file)

    def _get_model_torchvision(self):
        """通过torchvision加载预训练模型"""
        import torchvision

        if self.pretrained_weights:
            model = torchvision.models.__dict__[self.model_name](pretrained=False, num_classes=self.num_class)
            checkpoint = torch.load(self.pretrained_weights)
            model.load_state_dict(checkpoint)
        else:
            model = torchvision.models.__dict__[self.model_name](pretrained=True)
        model.eval()
        return model

    def _get_model_timm(self):
        """通过timm加载预训练模型"""
        import timm

        if self.pretrained_weights:
            model = timm.create_model(
                model_name=self.model_name,
                num_classes=self.num_class,
                pretrained=False,
                checkpoint_path=self.pretrained_weights,
            )
        else:
            model = timm.create_model(
                model_name=self.model_name,
                num_classes=self.num_class,
                pretrained=True,
            )
        model.eval()
        return model


def count_op(model, input):
    flops, params = thop.profile(model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)

if __name__ == "__main__":
    import timm


    maker = ModelHUb(args)
    maker.get_model()
