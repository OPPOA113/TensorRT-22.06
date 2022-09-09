#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author :          lance
@Email :  	  wangyl306@163.com
@Time  : 	2022/05/24 11:55:44
"""

import argparse

from vastdeploy import det

parse = argparse.ArgumentParser(description="RUN YOLO ON VACC")
parse.add_argument("--imgs_path", type=str, default="../../data/eval/coco_val2017", help="img path or dir path")
parse.add_argument("--img_shape", nargs=2, type=int, default=[640, 640], help="img shape-[w,h]")
parse.add_argument(
    "--model_name", type=str, default="yolov5s-u-torchscript-int8-percentile-640-640-runmodel", help="model name"
)
parse.add_argument("--save_dir", type=str, default="../../output", help="save dir for img")
parse.add_argument(
    "--weights_dir",
    type=str,
    default="../../weights/yolov5s-u-torchscript-int8-percentile-640-640-runmodel",
    help="weights_dir",
)
parse.add_argument(
    "--classes_list",
    nargs="+",
    type=str,
    default=[],
    help="classes_list,default coco classes",
)
parse.add_argument("--hw_config_path", type=str, default="../hw_config.json", help="hw_config_path")
parse.add_argument("--input_name", type=str, default="input")
parse.add_argument("--device", type=int, default=0, help="die id")
parse.add_argument("--delay", action="store_true", help="model run cost time")
parse.add_argument("--save_img", action="store_true", help="save result img")
parse.add_argument("--algo_mode", type=str, default="yolov5", choices=["yolov5", "mmdet_yolov3", "retinaface"])
parse.add_argument("--run_mode", type=str, default="pipeline", choices=["pipeline", "forward"])
args = parse.parse_args()
print(args)

runner = det.RunVACC(
    imgs_path=args.imgs_path,
    img_shape=args.img_shape,
    weights_dir=args.weights_dir,
    model_name=args.model_name,
    device=args.device,
    save_dir=args.save_dir,
    hw_config_path=args.hw_config_path,
    classes_list=args.classes_list,
    input_name=args.input_name,
    save_img=args.save_img,
    delay=args.delay,
    algo_mode=args.algo_mode,
    run_mode=args.run_mode,
)
actual_batch, run_model_eta_with_sec = runner.run()
print("Running Info:", "batch:", actual_batch, "ETA:", run_model_eta_with_sec, "s")
