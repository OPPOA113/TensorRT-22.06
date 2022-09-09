#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author :       	lance
@Email :  lance.wang@vastaitech.com
@Time  : 	2021/11/10 11:06:55
"""

import os

# NOTE 解决真实样本和检测样本数量不对齐情况
# 1.真实样本数 > 检测结果样本数  ---> 生成空白的检测结果txt  ---> 未检测到的结果其类别的 FN + 1
# 2.真实样本数 < 检测结果样本数  ---> 移除多余的检测结果txt
import sys


def backup(src_folder, backup_files, backup_folder):
    # non-intersection files (txt format) will be moved to a backup folder
    if not backup_files:
        print("No backup required for", src_folder)
        return
    os.chdir(src_folder)
    # create the backup dir if it doesn't exist already
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    for file in backup_files:
        os.rename(file, backup_folder + "/" + file)


def makeup(gt_files, dr_files, target_folder):
    makeup_files = gt_files.difference(dr_files)
    for file in makeup_files:
        f = open(os.path.join(target_folder, file), "w")
        f.close()


def check(GT_PATH="", DR_PATH=""):
    """[解决真实样本和检测样本数量不对齐情况]

    Args:
        GT_PATH ([str]): [ground truth]
        DR_PATH ([str]): [detected result]
    """

    backup_folder = "backup_no_matches_found"  # must end without slash

    gt_files = os.listdir(GT_PATH)

    if len(gt_files) == 0:
        print("Error: no .txt files found in", GT_PATH)
        sys.exit()

    dr_files = os.listdir(DR_PATH)
    if len(dr_files) == 0:
        print("Error: no .txt files found in", DR_PATH)
        sys.exit()

    gt_files = [os.path.split(file)[-1] for file in gt_files]
    dr_files = [os.path.split(file)[-1] for file in dr_files]
    gt_files = set(gt_files)
    dr_files = set(dr_files)
    print("total ground-truth files:", len(gt_files))
    print("total detection-results files:", len(dr_files))

    gt_backup = gt_files - dr_files
    dr_backup = dr_files - gt_files

    if gt_backup:
        print("total ground-truth backup files:", len(gt_backup))
        # 在detection-results中生成对应的空txt
        makeup(gt_files, dr_files, DR_PATH)

    if dr_backup:
        # 移除多余的检测样本
        print("total detection-results backup files:", len(dr_backup))
        backup(DR_PATH, dr_backup, backup_folder)

    intersection = gt_files & dr_files
    print("total intersected files:", len(intersection))
    print("Intersection completed!")


if __name__ == "__main__":
    GT_PATH = "../input/ground-truth"
    DR_PATH = "../input/detection-results"
    check(GT_PATH=GT_PATH, DR_PATH=DR_PATH)
