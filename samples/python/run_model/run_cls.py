#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author :          lance
@Email :  	  wangyl306@163.com
@Time  : 	2022/05/24 10:19:19
"""

import argparse

from vastdeploy import cls

parse = argparse.ArgumentParser(description="Deploy CLS ON VACC")
parse.add_argument("task", type=str, default="run", choices=["run", "topk"])
parse.add_argument(
    "--file_path",
    type=str,
    default="../../data/eval/ILSVRC2012_img_val",
    help="img or dir  path",
)
parse.add_argument(
    "--model_name",
    type=str,
    default="inception_v4-timm-torchscript-int8-percentile-299-299-runmodel",
    help="model name",
)
parse.add_argument(
    "--weights_dir",
    type=str,
    default="../../weights/inception_v4-timm-torchscript-int8-percentile-299-299-runmodel",
    help="weights_dir",
)
parse.add_argument("--save_dir", type=str, default="../../output", help="save_dir")
parse.add_argument("--num_class", type=int, default=1000, help="num_class")
parse.add_argument(
    "--label_dict", type=str, default="../../data/eval/imagenet1000_clsid_to_human.txt", help="label_dict"
)
parse.add_argument("--hw_config_path", type=str, default="../hw_config.json", help="hw_config_path")
parse.add_argument("--save_img", action="store_true", help="save pred images")
parse.add_argument("--delay", action="store_true", help="time test")
parse.add_argument("--topk", type=int, default=5, help="top1 or top5")
parse.add_argument("--device", type=int, default=0, help="device id")
parse.add_argument("--input_name", type=str, default="input")
parse.add_argument("--mean", nargs=3, type=float, default=[0.485, 0.456, 0.406])
parse.add_argument("--std", nargs=3, type=float, default=[0.229, 0.224, 0.225])
parse.add_argument("--input_size", nargs=2, type=int, default=[224,224], help="input hints resize")
parse.add_argument("--input_hints", nargs=2, type=int, default=[256,256], help="input hints crop, optional []")
args = parse.parse_args()
print(args)
run_func = cls.RunVACC(
    model_name=args.model_name,
    weights_dir=args.weights_dir,
    save_dir=args.save_dir,
    label_dict=args.label_dict,
    save_img=args.save_img,
    num_class=args.num_class,
    device=args.device,
    topk=args.topk,
    delay=args.delay,
    file_path=args.file_path,
    hw_config_path=args.hw_config_path,
    input_name=args.input_name,
     input_hints={"Resize": args.input_hints, "CenterCrop": args.input_size, "MEAN": args.mean, "STD": args.std},
)
# 测试
if args.task == "run":
    actual_batch, run_model_eta_with_sec = run_func.run_vacc()
    print("Running Info:", "batch:", actual_batch, "ETA:", run_model_eta_with_sec, "s")
else:
    # benchmark
    run_func.run_vacc_benchmark()
