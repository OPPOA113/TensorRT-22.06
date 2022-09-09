nvidia TensorRT 8.2  benchmark测试
```
onnx                    1.10.2
onnx-graphsurgeon       0.3.20
onnxruntime             1.8.1
torch                   1.10.2+cu113
torchvision             0.11.3+cu113
```

# 1.环境安装
```shell
# 1. download
scp -r vast@10.23.4.176:/home/vast/install_package/nvidia-pkg/benchmark_a2_3080 .
  passwd: vast
 
 # 2.load image 
docker load -i ./benchmark_a2_3080/tensorrt-ubuntu18.04-cuda11.4.tar

# 3.tar -x
cd benchmark_a2_3080
tar -xvf TensorRT-22.06.tar
cd TensorRT-22.06
docker run --name trtdev --gpus all -v ${PWD}:/workspace/TensorRT --privileged=true -p 11422:22 -it tensorrt-ubuntu18.04-cuda11.4:latest /bin/bash

## 4.容器内编译tensorrt
cd /workspace/TensorRT
 mkdir -p build && cd build
 cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out
 make -j$(nproc)
 
 ## 5. 添加环境变量
 vim ~/.bashrc
 export PATH=/workspace/TensorRT/build/out:${PATH}
 source ~/.bashrc

## 6.验证环境
trtexec --help
```


# 2.分类网络benchmark测试
```shell
## 1. 转换onnx模型
cd /workspace/TensorRT/samples/python
scp -r vast@10.23.4.176:/home/vast/projects/VastDeploy_v0.4.0/example/run_model .
passwd:vast
pip install thop==0.1.0.post2207010342 timm
python cls_mode_hub.py \
--batchsize 64 \
  --model_name resnet50
  --model_library torchvision
  --pretrained_weights /path/resnet50-19c8e357.pth \
  --save_dir /path/onnx_models/
#注：
--batchsize 生成onnx模型的batch,即benchmark测试的batchsize：[1 2 4 8 16 32 64 128 512];如生成onnx失败，则表示已经达到最大batchsize
--model_name 模型名字[resnet18\resnet34\resnet50]
--model_library 模型库名称[timm\torchvision]
--pretrained_weights pth文件：从nextcloud下载，见下面注释
-save_dir onnx文件保存文件夹路径


## 2. 生成序列化文件
cd /workspace/TensorRT/samples/python/efficientnet
## fp16
CUDA_VISIBLE_DEVICES=0 python build_engine.py \
    --onnx /path/to/model_name.onnx \
    --engine /path/to/model_name_size_bs_fp16.engine \
    --precision fp16
#注：
--onnx /path/to/model_name.onnx 第1步生成的onnx模型
--engine /path/to/model_name_size_bs_fp16.engine  序列化文件，名字带input_size, batchsize信息 

## int8
CUDA_VISIBLE_DEVICES=0 python build_engine.py \
    --onnx /path/to/model_size.onnx \
    --engine /path/to/model_size_int8.engine \
    --precision int8 \
    --calib_input /path/to/calib_data \
    --calib_cache /path/to/cls/model.cache \
    --calib_preprocessor V2
#注： 
CUDA_VISIBLE_DEVICES=0  选择第0个GPU卡
--onnx /path/to/model_size.onnx第1步生成的onnx模型
--engine /path/to/model_size_bs_int8.engine序列化文件，名字带input_size, batchsize信息
--calib_input 校准数据文件夹，算法小组提供
--calib_cache  校准文件，同个算法取同一个名字即可；不同inputsize\batchsize用同一个名字，如resnet18\resnet50\yolov3\Retinaface


## 3. benchmark测试
trtexec \
--device=N \
    --loadEngine=/path/to/model.engine \
    --useCudaGraph --noDataTransfers \
    --duration=N 
    --batch=N# 与第1步生成onnx模型的batchsize数相同！！！！！

注： 
--device=N 选择第几个NV卡 0 ~ N-1
  --loadEngine=/path/to/model.engine第2步序列化生成的文件
--duration=N  推理最少时间（秒）
  --streams=<N> 并行运行多个流的推理
  --batch=Nbatchsize设置， 与第1步生成onnx模型的batchsize数相同！！！！！
如：./trtexec --loadEngine=./benchmark_test/retinaface/resnet50final_320x320_1_fp16.engine --useCudaGraph --noDataTransfers --duration=30
```
> Note 
  注：pth模型文件下载地址：取各算法类型下，torchvision文件夹下的pth文件；如：
  /算法权重/calssification/resnet/torchvision/*.pth
  测试脚本输出结果记录：
  Throughput: 768 qps			# 吞吐量
  GPU Compute Time: mean=1.29695	# latency
  ```
  1.qps: 768.031
  2.latency: 1.29695 ms
  3.shape ：320x320        从模型文件名获取
  4.batchsize: 1                从模型文件名获取
  5.显存-功耗-利用率：  在第三步的推理的N秒内，新起终端，使用nvidia-smi工具查看
  ```

# 3. 分类网络topk测试
```shell
## 获取脚本：
cd /workspace/TensorRT/samples/python
scp -r vast@10.23.4.176:/home/vast/install_package/nvidia-pkg/TensorRT-22.06/samples/python/classifer .
#passwd:vast
cd classifer

## 每类模型，fp16\int8,topk精度测试只验证batchsize=1的情形即可：

# 1. 参考上面第2步方式，用本文件夹下脚本生成的bs=1的engine模型，测试topk精度。
 

# 2. 统计topk   
python eval_gt.py \
    --engine path \
    --input path \
    --annotations file \
#注：  
--engine 序列化的engine模型
--annotations 图片和标签，与算法人员同步格式
--input图片文件夹路径，与算法人员同步格式

## 测试输出结果：
Processing 50000 / 50000 : Top-1 74.5% , Top-5: 92.0%     
Top-1 Accuracy: 74.452%
Top-5 Accuracy: 92.016%
```

# 4. 检测网络beachMark测试
```shell
检测网络模型的测试，在efficientdet目录下，操作与classification基本相同。
## 切换路径，注意与classification的不同
cd /workspace/TensorRT/samples/python/efficientdet

## 1. 序列化文件
## fp16
CUDA_VISIBLE_DEVICES=0 python3 build_engine.py \
--batch_size N \
    --onnx /path/to/model.onnx \
    --engine /path/to/model_fp16.engine \
    --precision fp16
    
## int8    
CUDA_VISIBLE_DEVICES=0 python3 build_engine.py \
--batch_size N \
    --onnx /path/to/model.onnx \
    --engine /path/to/model_int8.engine \
    --precision int8 \
    --calib_input /path/to/calibration/images \
    --calib_cache /path/to/calibration.cache
# 注： 
CUDA_VISIBLE_DEVICES=0  选择第0个GPU卡
--batch_size N 与onnx名字的batchsize大小一致！！！
--onnx /path/to/model_size.onnx
# yolov3\yolov5s，人脸Retinaface onnx获取：
scp -r vast@10.23.4.176:/home/vast/projects/yolov5-6.0/yolov5_onnx/yolov3 .
scp -r vast@10.23.4.176:/home/vast/projects/yolov5-6.0/yolov5_onnx/yolov5s .
scp -r vast@10.23.4.176:/home/vast/projects/Pytorch_Retinaface/weights_onnx .
#passwd: vast
--engine /path/to/model_size_bs_int8.engine序列化文件，名字带input_size, batchsize信息
--calib_input 校准数据文件夹，算法小组提供
--calib_cache  校准文件，同个算法取同一个名字即可；不同inputsize\batchsize用同一个名字，如resnet18\resnet50\yolov3\Retinaface

    
## 2. benchmark测试
trtexec \
--device=N \
    --loadEngine=/path/to/model_fp16_in8.engine \
    --useCudaGraph --noDataTransfers \
    --duration=N 
    --batch=N# 设置与onnx模型的batchsize数相同！！！！！

## 测试结果参数获取方式，与classification相同。
```

# 5. yolo系检测网络mAp精度测试
```shell
cd /workspace/TensorRT/samples/python
scp -r vast@10.23.4.176:/home/vast/install_package/nvidia-pkg/TensorRT-22.06/samples/python/yolo_detection .
# passwd:vast
cd yolo_detection

## 安装依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python PyYAML tqdm pandas matplotlib seaborn pycocotools
sudo apt-get install libgl1
# passwd: nvidia

## 每类模型，fp16\int8,topk精度测试只验证batchsize=1的情形即可：

# 1. 参考上面第2步方式，分别用本文件夹下脚本生成的bs=1的fp16\int8 engine模型，测试map精度。
 
 # 2. 模型推理结果
 python eval_yolo.py \
  --engine /path/to/yolov5s_1_320_int8.engine \
    --inputsize 320 \
    --input /path/to/coco/val2017 \
    --include map \
    --txtpath ./output/yolov5s_1_320_int8 \
    --conf_t 0.01 \
    --iou_t 0.65
 
 #注：
 --engine  前一步生成的fp16\int8模型engine;
 --inputsize 320 测试的输入图片size
 --input 图片路径，测试图片路径，与算法确认
 --txtpath 保存的检测结果文件夹，可以按engine名命名
 其余变量不变
 
 # 3. 计算map
python coco_benchmark.py \
    --gt ./instances_val2017.json \
    --txt /workspace/TensorRT/samples/python/yolo_detection/output/yolov5s_1_320_int8
# 注：
   --gt  标签文件，不用改
   --txt 上一步的 txtpath文件夹路径
   
  注：第3步计算map的输出；记录红框的结果
  记录最后一行的’bbox_mAP’: 的值
```

# 6.retinaface Map精度测试：
```shell
# 1. 获取脚本
cd /workspace/TensorRT/samples/python
scp -r vast@10.23.4.176:/home/vast/install_package/nvidia-pkg/TensorRT-22.06/samples/python/retinaface .
#passwd:vast
mkdir build && cd build
cmake ..
make -j$(nproc)
export LD_LIBRARY_PATH=`pwd`/../opencv3410/jasper/:`pwd`/../opencv3410/lib:${LD_LIBRARY_PATH}
# 校准图片软链接
ln -s /path/widerface/val/calib ./calib 

# 先安装依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple Cython
cd ../widerface_evaluate
python3 setup.py build_ext --inplace
cd ..

# 2. 在build目录下，序列化engine：
for use_type in USE_FP32 USE_FP16 USE_INT8;do
        for size in 320 416 512 608 640;do
            for model in retina_r50 retina_mnet;do
                sed -i "s/INPUT_W=/INPUT_W=$size;\/\//g" ../decode.h && sed -i "s/INPUT_H=/INPUT_H=$size;\/\//g" ../decode.h
                make -j$(nproc)
                ./$model -s -$use_type
            done 
        done
    done
## 注：ls ../engine  # 在固定的路径文件夹下生成各个case的序列化engine文件

# 3. 测试widerface
cd ..
python test_widerface.py \
    --engine_model ./engine/retina_r50_1_640x640_fp32.engine \
    --input_size_w 640 \
    --input_size_h 640 \
    --dataset_folder /path/widerface/val/images/ \
    --save_folder /path/output/retina_r50_1_640x640_fp32
#注： 
  --engine_model  上一步上次的engine文件 
    --input_size_w  engine模型的宽    
    --input_size_h  engine模型的高
    --dataset_folder /path/widerface/val/images/ 图片路径,路径要以“images/”结尾
    --save_folder 图片推理结果保存文件夹,建议按engine名命名
    
# 4. 计算map
cd widerface_evaluate
python3 evaluation.py -p ../output/retina_r50_1_640x640_fp32 -g ./ground_truth
# 注： 
-p ../output/retina_r50_1_640x640_fp32 上一步推理结果文件夹
  -g 参数按示例，不用变
  
执行结果：
==================== Results ====================
Easy   Val AP: 0.9416307124371939
Medium Val AP: 0.9063059701023011
Hard   Val AP: 0.6590038805853572
=================================================
```

# 7.阿丘 FCN 性能测试
```shell
# 从Lance Wang处拿的阿丘FCN模型
# 获取onnx模型
scp -r vast@10.23.4.176:/home/vast/projects/custom_model/pytorch-fcn/deploy_onnx .
  # passwd: vast
  
# 性能测试，按 “第2步 分类网络benchmark测试”的步骤进行操作
``` 



