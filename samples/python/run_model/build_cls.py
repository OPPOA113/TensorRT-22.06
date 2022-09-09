#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author :          lance
@Email :  	  wangyl306@163.com
@Time  : 	2022/05/23 17:45:59
"""

import argparse

from vastdeploy import cls

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (0.0167 * 255)] * 3)
DEFAULT_INPUT = {"Resize": [224, 224], "CenterCrop": [256, 256]}
OTHER_INPUT = {"Resize": [299, 299], "CenterCrop": [342, 342]}

'''
input_hints = int(math.floor(img_size[0] / crop_pct))
'''




parse = argparse.ArgumentParser(description="Deploy CLS ON VACC")
parse.add_argument("--calib_path", type=str, default="../../data/eval/ILSVRC2012_img_calib", help="calib  path")
parse.add_argument("--quant_mode", type=str, default="percentile", choices=["percentile", "kl_divergence", "max"])
parse.add_argument("--model_name", type=str, default="inception_v4", help="model name")
parse.add_argument("--with_odsp_softmax", action="store_true", help="with softmax")
parse.add_argument(
    "--pretrained_weights",
    type=str,
    default="../../weights/inception_v4.torchscript",
    help="timm or torchvision or custom onnx weights path",
)
parse.add_argument(
    "--weights_dir",
    type=str,
    default="../../weights",
    help="weights_dir",
)
parse.add_argument("--prec", type=str, default="int8", choices=["int8", "fp16"])
parse.add_argument("--input_name", type=str, default="input")
parse.add_argument("--mean", nargs=3, type=float, default=[0.5, 0.5, 0.5])
parse.add_argument("--std", nargs=3, type=float, default=[0.5, 0.5, 0.5])
parse.add_argument("--input_size", nargs=2, type=int, default=[299,299], help="input hints resize")
parse.add_argument("--input_hints", nargs=2, type=int, default=[342,342], help="input hints crop, optional []")
parse.add_argument("--run_stream", action="store_true", help="run with SDK need True")
parse.add_argument(
    "--skip_conv_layers",
    nargs="+",
    type=int,
    default=[],
    help="skip_conv_layers for quantize",
)
parse.add_argument("--codesource", type=str, default="timm", help="algo from where")
args = parse.parse_args()
print(args)
build_func = cls.BuildVACC(
    model_name=args.model_name,
    with_odsp_softmax=args.with_odsp_softmax,
    pretrained_weights=args.pretrained_weights,
    save_weights_dir=args.weights_dir,
    quant_config={"quant_mode": args.quant_mode, "calib_path": args.calib_path},
    quant=args.prec,
    input_name=args.input_name,
    input_hints={"Resize": args.input_hints, "CenterCrop": args.input_size, "MEAN": args.mean, "STD": args.std},
    run_stream=args.run_stream,
    skip_conv_layers=args.skip_conv_layers,
    codesource=args.codesource,
)
build_func.build_vacc()
