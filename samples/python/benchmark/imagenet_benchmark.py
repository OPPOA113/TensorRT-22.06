#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author :       	lance
@Email :  lance.wang@vastaitech.com
@Time  : 	2021/12/10 10:08:59
"""
# NOTE
# 主要针对ImageNet Val的benchmark测试脚本
# 1.批量测试 & 保存txt
# xxx/imagement/n15075141/ILSVRC2012_val_00049174.JPEG: Relay top-0 id: 999, prob: 0.23510742, class name: toilet tissue, toilet paper, bathroom tissue
# xxx/imagement/n15075141/ILSVRC2012_val_00049174.JPEG: Relay top-1 id: 700, prob: 0.18310547, class name: paper towel
# xxx/imagement/n15075141/ILSVRC2012_val_00049174.JPEG: Relay top-2 id: 772, prob: 0.02722168, class name: safety pin
# xxx/imagement/n15075141/ILSVRC2012_val_00049174.JPEG: Relay top-3 id: 549, prob: 0.02722168, class name: envelope
# xxx/imagement/n15075141/ILSVRC2012_val_00049174.JPEG: Relay top-4 id: 674, prob: 0.01930237, class name: mousetrap
# 2.结果对比生成top1 & top5

# [timm result] https://github.com/rwightman/pytorch-image-models/blob/v0.4.12/results/results-imagenet.csv
# [torchvision result] https://pytorch.org/vision/0.9/models.html


import argparse
import csv
import datetime
import os
import re

from .imagenet_info import imagenet_info


def parse_opt():
    parse = argparse.ArgumentParser(description="test classificiation models on vacc")
    parse.add_argument("--model", type=str, default="resnet18", help="model name")
    parse.add_argument(
        "--model_library",
        type=str,
        default="torchvision_0.9.0",
        help="pretrained weights from where: timm_0.4.12 or torchvision_0.9.0",
    )
    parse.add_argument("--input_size", type=int, default="224", help="input shape")
    parse.add_argument("--quant", type=str, default="int8_max", help="quant config")
    parse.add_argument("--txt_dir", type=str, default="../output", help="save dir for img")
    opt = parse.parse_args()
    return opt


def get_timm_bench(model_name, bench_csv_file="./pretrained_benchmark/timm_imagenet_0.4.12.csv"):
    bench = ["model", "top1", "top5", "input_size"]
    with open(bench_csv_file) as file_val:
        lines = file_val.readlines()
        for i in range(1, len(lines)):  # first line is table head, skip
            line = lines[i].rstrip("\n")
            columns = line.split(",")
            if columns[0].lower() == model_name.lower():
                bench[0] = columns[0].lower()
                bench[1] = float(columns[1])
                bench[2] = float(columns[3])
                bench[3] = int(columns[6])
                break
    return bench


def get_tv_bench(model_name, bench_csv_file="./pretrained_benchmark/tv_imagenet_0.9.0.csv"):
    bench = ["model", "top1", "top5", "input_size"]
    with open(bench_csv_file) as file_val:
        lines = file_val.readlines()
        for i in range(1, len(lines)):  # first line is table head, skip
            line = lines[i].rstrip("\n")
            columns = line.split(",")
            if columns[0].lower() == model_name.lower():
                bench[0] = columns[0].lower()
                bench[1] = float(columns[1])
                bench[2] = float(columns[2])
                break
    print("[torchvison]: ", bench[1], bench[2])
    return bench


def is_match_class(line, label_info):
    # get gt & dt
    line = line.rstrip("\n")
    if len(line) == 0:
        return False
    line_info_path = line.split("/")
    line_info_relay = line.split(":")
    gt_label_dir = line_info_path[-2]
    dt_label_name = line_info_relay[4].strip()
    gt_label_name = label_info[gt_label_dir]
    return gt_label_name == dt_label_name


def get_vacc_result(txt_dir, model_name, topk=5, label_info=imagenet_info):
    """获取单个模型txt文件的top1&top5"""
    total_count = 0
    top1_count = 0
    top5_count = 0
    with open(os.path.join(txt_dir, model_name + ".txt"), "r") as fout:
        lines = fout.readlines()
        for i in range(0, len(lines), topk):
            total_count += 1
            five_lines = lines[i : i + topk]
            matches = [is_match_class(line, label_info) for line in five_lines]
            if matches[0]:
                top1_count += 1
                top5_count += 1
            elif True in matches:
                top5_count += 1
    top1_rate = top1_count / total_count * 100
    top5_rate = top5_count / total_count * 100
    print("[VACC]: ", "top1_rate:", top1_rate, "top5_rate:", top5_rate)
    return top1_rate, top5_rate


def save_csv(opt):
    model_name, model_library, input_size, quant, txt_dir = (
        opt.model,
        opt.model_library,
        opt.input_size,
        opt.quant,
        opt.txt_dir,
    )
    # 初始化csv
    if not os.path.exists("./vacc_cls_benchmark.csv"):
        with open("./vacc_cls_benchmark.csv", "w") as f:
            csv_write = csv.writer(f)
            csv_head = [
                "model",
                "top1_vacc",
                "top5_vacc",
                "top1_benchmark",
                "top5_benchmark",
                "top1_gap",
                "top5_gap",
                "quant_method",
                "input_size",
                "FPS_forward bs1",  # forward：仅推理
                "FPS_forward bs4",
                "FPS_forward bs8",
                "FPS_pipline bs1",  # pipline：AI处理全流程；数据预处理（decode、resize） + forward + 后处理（分类无后处理）
                "FPS_pipline bs4",
                "FPS_pipline bs8",
                "model_library",
                "UpdateTime",
            ]
            csv_write.writerow(csv_head)
    # 获取结果
    dayTime = datetime.datetime.now().strftime("%Y-%m-%d")
    if re.search(r"timm", model_library.lower(), re.M | re.I):
        print("[INFO] model library -> timm")
        bench = get_timm_bench(model_name)
        if bench[3] != input_size:
            bench[3] = str(input_size) + "_" + str(bench[3])
        top1_vacc, top5_vacc = get_vacc_result(txt_dir, model_name)
    elif re.search(r"torchvision", model_library.lower(), re.M | re.I):
        print("[INFO] model library -> torchvision")
        bench = get_tv_bench(model_name)
        bench[3] = str(input_size)
        top1_vacc, top5_vacc = get_vacc_result(txt_dir, model_name)
    # 追加结果
    with open("./vacc_cls_benchmark.csv", "a") as fadd:
        csv_add = csv.writer(fadd)
        csv_res = [
            model_name,
            top1_vacc,
            top5_vacc,
            bench[1],
            bench[2],
            top1_vacc - bench[1],
            top5_vacc - bench[2],
            quant,
            bench[3],
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            model_library,
            dayTime,
        ]
        csv_add.writerow(csv_res)


if __name__ == "__main__":
    opt = parse_opt()
    save_csv(opt)

