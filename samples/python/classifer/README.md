# classifer in TensorRT

modelzoo分类模型在nv卡上benchmark测试

## dependency
    
    tensorrt 22.06(Tensorrt-8.2.5)

## docker env

```shell
# 1. clone repo
git clone -b master https://github.com/nvidia/TensorRT TensorRT
cd TensorRT
git submodule update --init --recursive
git checkout release/8.2

# 2. build docker
./docker/build.sh --file docker/ubuntu-18.04.Dockerfile --tag tensorrt-ubuntu18.04-cuda11.4
# TD
# 此处是否修改ubuntu-18.04.Dockerfile文件，将编译好的可执行文件、so、lib和头文件 copy到镜像内???

# 3. run docker
# 注意目录切换、run时目录挂载
cd TensorRT
docker run --name trt-benchmark --gpus '"device=0"' -v ${PWD}:/workspace/TensorRT -v /home/ermengz/nv_benchmark/resource:/workspace/resource --privileged=true -p 11222:22 -it tensorrt-ubuntu18.04-cuda11.4:latest /bin/bash
# containner user：trtuser
# containner pw:   nvidia

# 4. step into docker
docker exec -it trt-benchmark /bin/bash

# 5. complier TensorRT:
# 如果编译出错，check是否是protobuf版本问题
cd TensorRT
mkdir -p build && cd build
cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out
make -j$(nproc)

# 6. env path
# 设置环境变量
vim ~/.bashrc
export PATH=/workspace/TensorRT/build/out:/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/workspace/TensorRT/build/out:${LD_LIBRARY_PATH}
source ~/.bashrc
```

## 目录文件说明
  - cls_model_hub.py pth-->onnx  需要包thop==0.1.0.post2207010342\timm
  - build_engine.py onnx -> trt.engine
  - infer.py trt.engine模型推理
  - eval_gt.py 计算top k

  * create_onnx.py 是将tf模型转成onnx
  * compare_tf.py 是tf模型推理和trt模型推理的结果比较

## how to run

1. cls_mode_hub.py脚本生成 onnx模型；

2. build_engine.py文件，将onnx模型转换到trt序列化模型：fp16 int8精度

3. eval_gt.py脚本，测试fps或者模型的topk（bs=1时）

## 脚本参数说明  
```
STEP1: pth-->onnx

python cls_mode_hub.py 
    --batchsize 1 \
    --model_library torchvision \
    --pretrained_weights ./resnet50/resnet50-19c8e357.pth \
    --save_dir ./resnet50/ \
    --model_name resnet50 

-----------------------------------
STEP2: onnx-->engine

python build_engine.py \
    --onnx /path/to/model_size.onnx \
    --engine /path/to/model_size_int8.engine \
    --precision int8 \
    --calib_input /path/to/calib_data \
    --calib_cache /path/to/cls/model.cache \
    --calib_preprocessor resnet

e.g. --int8
python build_engine.py --onnx ./resnet50/resnet50_1_224.onnx --engine ./resnet50/resnet50_1_224_int8.engine --precision int8 --calib_input /workspace/project/dataset/benchmarch_data/recognise/ILSVRC2012_img_calib --calib_cache ./resnet50/resnet50.cache --calib_preprocessor RESNET
e.g. --fp16
python build_engine.py 
    --onnx ./resnet50/resnet50_1_224.onnx \
    --engine ./resnet50/resnet50_1_224_fp16.engine \
    --precision fp16

-----------------------------------
STEP3: trtexec --> qps

trtexec \
	--device=N \
    --loadEngine=/path/to/model.engine \
    --useCudaGraph --noDataTransfers \
    --duration=N \
    --batch=N

-----------------------------------
STEP4: cal -->  top k or fps
python eval_gt.py \
    --engine path \
    --input path \
    --annotations file \
    --benchmark \
	--topk \
	--preprocessor alexnet
注：--benchmark 表示测试fps
--topk 如果engine的bs=1,则增加输入topk精度 
--preprocessor 表示测试的模型名字，参考modelzoo上的名字填写   
eg. 
python eval_gt.py --engine ./resnet50/resnet50_1_224_fp16.trt --input /workspace/project/dataset/benchmarch_data/recognise/images_resnet50_ILSVRC2012_val --annotations /workspace/project/dataset/benchmarch_data/recognise/ILSVRC2012_val.txt

Processing 50000 / 50000 : Top-1 74.5% , Top-5: 92.0%     
Top-1 Accuracy: 74.452%
Top-5 Accuracy: 92.016%
-----------------------------------
```

## 完成测试sample

```shell
#1. pth-->onnx
python cls_mode_hub.py --model_library timm --batchsize 1 --size 256 --pretrained_weights ../classifer_resnet50/cspresnet50_ra-d3e8d487.pth --save_dir ../classifer_resnet50/ --model_name cspresnet50

#2. onnx-->engine-fp16
python build_engine.py --onnx ../classifer_resnet50/cspresnet50-timm-1_3_256_256.onnx --engine ../classifer_resnet50/cspresnet50-1_3_256_256_fp16.engine --precision fp16
#2. onnx-->engine-int8
python build_engine.py \
    --onnx ../classifer_resnet50/cspresnet50-timm-1_3_256_256.onnx \
    --engine ../classifer_resnet50/cspresnet50-1_3_256_256-int8.engine \
    --precision int8 \
    --calib_input /workspace/project/dataset/benchmarch_data/recognise/ILSVRC2012_img_calib \
    --calib_cache ../classifer_resnet50/cspresnet50.cache \
    --calib_preprocessor alexnet
#3. engine-fp16\int8 performace qps c++
trtexec \
	--device=0 \
    --loadEngine=../classifer_resnet50/cspresnet50-1_3_256_256_fp16.engine \
    --useCudaGraph --noDataTransfers \
    --duration=30 \
    --batch=1
#4. engine-fp16\int8 performace qps\topk fps python
python eval_gt.py \
    --engine ../classifer_resnet50/cspresnet50-1_3_256_256_fp16.engine \
	--input /workspace/project/dataset/benchmarch_data/recognise/images_resnet50_ILSVRC2012_val \
	--annotations /workspace/project/dataset/benchmarch_data/recognise/ILSVRC2012_val.txt \
	--benchmark \
	--topk \
	--preprocessor alexnet
```

## result sample 

| model | batchsize| python | trtexec | type | top1 | top5 |
| :---: | :----: |:----: | :---: | :---: | :-------: | :-------: |
|resnet50|1|1480.1|1635.0|fp16|--|--|
|alxnet|1|2041.1|2110.68|fp16|52.6%|76.2%|
|alxnet|1|3761.9|4027.55|int8|52.6%|76.2%|
|cspresnet50|1|1271.9|1409.68|fp16|78.7%|94.4%|
|cspresnet50|1|1874.4|2197.48|int8|71.4%|90.5%|

