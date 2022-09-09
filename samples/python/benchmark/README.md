# Benchmark

> lance.wang

## 1. classificiation

|       model        |      dataset       | input | batch | run model | run stream |  top1  | top5   | int8 quant | top1&top5_source | Hardware |  framework  | release  |
| :----------------: | :----------------: | :---: | :---: | :-------: | :--------: | :----: | ------ | :--------: | :--------------: | :------: | :---------: | :------: |
| gluon_resnet50_v1b | ILSVRC2012_img_val |  224  |   1   |  0.71ms   |     -      | 77.42  | 93.590 |   auto     | 77.576 & 93.722  |   VA1    | ali_release | 20211217 |
| gluon_resnet50_v1b | ILSVRC2012_img_val |  224  |   1   |  0.71ms   |     -      | 77.264 | 93.548 |    max     | 77.576 & 93.722  |   VA1    | ali_release | 20211210 |
| gluon_resnet50_v1b | ILSVRC2012_img_val |  224  |   4   |  1.72ms   |     -      | 77.264 | 93.548 |    max     | 77.576 & 93.722  |   VA1    | ali_release | 20211210 |
| gluon_resnet50_v1b | ILSVRC2012_img_val |  224  |   8   |  2.95ms   |     -      | 77.264 | 93.548 |    max     | 77.576 & 93.722  |   VA1    | ali_release | 20211210 |

### 指标说明

+ run model : forward
+ run stream : data_process + forward

### 模型源码

+ [gluon_resnet50_v1b](https://github.com/rwightman/pytorch-image-models)

## 2. object detection

| model   |   dataset    | input | batch | run model | run stream | map@0.5 | int8 quant | map@0.5_source | Hardware |  framework               | release  |
| :-----: | :----------: | :---: | :---: | :-------: | :--------: | :-----: | :--------: | :------------: | :------: | :----------------------: | :------: |
| yolov3  | COCO_val2017 |  416  |   1   |  3.66ms   |     -      |  0.601  |    auto    |     0.605      |   VA1    | ali_release              | 20211207 |
| yolov3  | COCO_val2017 |  416  |   2   |  6.07ms   |     -      |  0.601  |    auto    |     0.605      |   VA1    | ali_release              | 20211207 |
| yolov5s | COCO_val2017 |  640  |   1   |  3.04ms   |     -      |  0.543  |    auto    |     0.560      |   VA1    | fast_develop_all_commits | 20211220 |

### 指标说明

+ run model : forward + nms
+ run stream : data_process + forward + nms

### 模型源码

+ [yolov3](https://github.com/ultralytics/yolov3/tree/v9.5.0)
  + speed：` conf 0.25;iou 0.45`
  + ap：`conf 0.001;iou 0.65`

+ [yolov5](https://github.com/ultralytics/yolov5/tree/v6.0)
  + speed：` conf 0.25;iou 0.45`
  + ap：`conf 0.001;iou 0.65`