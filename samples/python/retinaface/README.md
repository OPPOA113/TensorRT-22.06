# 编译环境
* 176 vtxdemo 容器

* tensorrt 执行推理步骤：
```   
    scp -r vast@10.23.4.176:/home/vast/install_package/nvidia-pkg/TensorRT-22.06/samples/python/retinaface .
    cd retinaface/build
    export LD_LIBRARY_PATH=`pwd`/../opencv3410/jasper/:`pwd`/../opencv3410/lib:${LD_LIBRARY_PATH}
    
    ## 1. 序列化文件
    for use_type in USE_FP32 USE_FP16 USE_INT8;do
        for size in 320 416 512 608 640;do
            for model in retina_r50 retina_mnet;do
                sed -i "s/INPUT_W=/INPUT_W=$size;\/\//g" ../decode.h && sed -i "s/INPUT_H=/INPUT_H=$size;\/\//g" ../decode.h
                make -j$(nproc)
                ./$model -s -$use_type
            done 
        done
    done
    
    ../engine 文件夹下生产序列化engine文件

    ## 2. 推理
python test_widerface.py \
    --engine_model ./engine/retina_r50_1_640x640_int8.engine \
    --input_size_w 640 \
    --input_size_h 640 \
    --dataset_folder /workspace/project/dataset/benchmarch_data/widerface/val/images/ \
    --save_folder /workspace/TensorRT/samples/python/retinaface/output/retina_r50_1_640x640_int8

    ## 3.测试map
    cd widerface_evaluate
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple Cython
    python3 setup.py build_ext --inplace
    python3 evaluation.py -p ../output/retina_r50_1_640x640_in8 -g ./ground_truth
    
    result:
==================== Results ====================
Easy   Val AP: 0.9416307124371939
Medium Val AP: 0.9063059701023011
Hard   Val AP: 0.6590038805853572
=================================================

* sudo apt-get install -y libopencv-dev

deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
```
    

# RetinaFace

 The pytorch implementation is [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface), I forked it into 
[wang-xinyu/Pytorch_Retinaface](https://github.com/wang-xinyu/Pytorch_Retinaface) and add genwts.py

This branch is using TensorRT 7 API, branch [trt4->retinaface](https://github.com/wang-xinyu/tensorrtx/tree/trt4/retinaface) is using TensorRT 4.

## Config

- Input shape `INPUT_H`, `INPUT_W` defined in `decode.h`
- INT8/FP16/FP32 can be selected by the macro `USE_FP16` or `USE_INT8` or `USE_FP32` in `retina_r50.cpp`
- GPU id can be selected by the macro `DEVICE` in `retina_r50.cpp`
- Batchsize can be selected by the macro `BATCHSIZE` in `retina_r50.cpp`

## Run

The following described how to run `retina_r50`. While `retina_mnet` is nearly the same, just generate `retinaface.wts` with `mobilenet0.25_Final.pth` and run `retina_mnet`.

1. generate retinaface.wts from pytorch implementation https://github.com/wang-xinyu/Pytorch_Retinaface

```
git clone https://github.com/wang-xinyu/Pytorch_Retinaface.git
// download its weights 'Resnet50_Final.pth', put it in Pytorch_Retinaface/weights
cd Pytorch_Retinaface
python detect.py --save_model
python genwts.py
// a file 'retinaface.wts' will be generated.
```

2. put retinaface.wts into tensorrtx/retinaface, build and run

```
git clone https://github.com/wang-xinyu/tensorrtx.git
cd tensorrtx/retinaface
// put retinaface.wts here
mkdir build
cd build
cmake ..
make
sudo ./retina_r50 -s  // build and serialize model to file i.e. 'retina_r50.engine'
wget https://github.com/Tencent/FaceDetection-DSFD/raw/master/data/worlds-largest-selfie.jpg
sudo ./retina_r50 -d  // deserialize model file and run inference.
```

3. check the images generated, as follows. 0_result.jpg

4. we also provide a python wrapper

```
// install python-tensorrt, pycuda, etc.
// ensure the retina_r50.engine and libdecodeplugin.so have been built
python retinaface_trt.py
```

# INT8 Quantization

1. Prepare calibration images, you can randomly select 1000s images from your train set. For widerface, you can also download my calibration images `widerface_calib` from [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh

2. unzip it in retinaface/build

3. set the macro `USE_INT8` in retina_r50.cpp and make

4. serialize the model and test

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78901890-9077fb80-7aab-11ea-94f1-237f51fcc347.jpg">
</p>

## More Information

Check the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

