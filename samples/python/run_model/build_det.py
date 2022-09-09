#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author :          lance
@Email :  	  wangyl306@163.com
@Time  : 	2022/05/24 11:34:08
"""

import argparse
import os

from vastdeploy import det

parse = argparse.ArgumentParser(description="Deploy YOLO ON VACC")
parse.add_argument(
    "--calib_path",
    type=str,
    default="../../data/eval/det_coco_calib",
    help="calib  path",
)
parse.add_argument("--quant_mode", type=str, default="percentile", choices=["percentile", "kl_divergence", "max"])
parse.add_argument("--prec", type=str, default="int8", choices=["int8", "fp16"])
parse.add_argument(
    "--img_shape",
    nargs=2,
    type=int,
    default=[640, 640],
    help="img shape-[h,W]",
)
parse.add_argument("--model_name", type=str, default="yolov5s", help="model name")
parse.add_argument(
    "--checkpoints",
    type=str,
    default="../../weights/yolov5s_640.torchscript.pt",
    help="pretrained weights",
)
parse.add_argument(
    "--weights_dir",
    type=str,
    default="../../weights",
    help="weights_dir",
)
parse.add_argument("--num_class", type=int, default=80, help="num_class")
parse.add_argument("--nms_thresh", type=float, default=0.45)
parse.add_argument("--confidence_thresh", type=float, default=0.25)
parse.add_argument("--run_stream", action="store_true", help="run with SDK need True")
parse.add_argument("--pipeline", action="store_true", help="pipeline")
parse.add_argument(
    "--anchor",
    nargs="+",
    type=float,
    default=[10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326
    ],
    help="anchor",
)
parse.add_argument("--strides", nargs="+", type=int, default=[8, 16, 32])
parse.add_argument(
    "--skip_conv_layers",
    nargs="+",
    type=int,
    default=[],
    help="skip_conv_layers for quantize",
)
parse.add_argument("--input_name", type=str, default="input")
parse.add_argument("--with_nms", action="store_true")
parse.add_argument("--algo_mode", type=str, default="yolov5", choices=["yolov5", "yolox", "mmdet_yolov3", "retinaface"])
parse.add_argument("--codesource", type=str, default="u", help="algo from where")
args = parse.parse_args()
print(args)
if not os.path.exists(args.weights_dir):
    os.makedirs(args.weights_dir)

builder = det.BuildVACC(
    img_shape=args.img_shape,
    model_name=args.model_name,
    checkpoints=args.checkpoints,
    weights_dir=args.weights_dir,
    prec=args.prec,
    with_nms=args.with_nms,
    input_name=args.input_name,
    skip_conv_layers=args.skip_conv_layers,
    run_stream=args.run_stream,
    pipeline=args.pipeline,
    quant_conf={"quant_mode": args.quant_mode, "calib_path": args.calib_path},
    nms_conf={
        "num_class": args.num_class,
        "nms_thresh": args.nms_thresh,
        "confidence_thresh": args.confidence_thresh,
        "anchor": tuple(args.anchor),
        "strides": tuple(args.strides),
    },
    algo_mode=args.algo_mode,
    codesource=args.codesource,
)

builder.build()
