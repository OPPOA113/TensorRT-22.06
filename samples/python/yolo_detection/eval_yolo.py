import argparse
import sys
import os
# import numpy as np
# from collections import OrderedDict, namedtuple

# import torch
# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit


# categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
#             "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
#             "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
#             "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
#             "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
#             "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
#             "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
#             "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
#             "hair drier", "toothbrush"]

# def DetectMultiBackend_trt(weights, data=data, fp16=half):

#     device = torch.device('cuda:0')
#     Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
#     logger = trt.Logger(trt.Logger.INFO)
#     with open(weights, 'rb') as f, trt.Runtime(logger) as runtime:
#         model = runtime.deserialize_cuda_engine(f.read())
#     context = model.create_execution_context()
#     bindings = OrderedDict()
#     fp16 = False  # default updated below
#     dynamic = False
#     for index in range(model.num_bindings):
#         is_input = False
#         if model.binding_is_input(index):
#             is_input = True
#         name = model.get_binding_name(index)
#         if name not in ["images","output"]:
#             continue
#         dtype = trt.nptype(model.get_binding_dtype(index))

#         # if model.binding_is_input(index):
#         #     if -1 in tuple(model.get_binding_shape(index)):  # dynamic
#         #         dynamic = True
#         #         context.set_binding_shape(index, tuple(model.get_profile_shape(0, index)[2]))
#         #     if dtype == np.float16:
#         #         fp16 = True
#         shape = tuple(context.get_binding_shape(index))
#         size = dtype.itemsize
#         for s in shape:
#             size *= s
#         allocation = cuda.mem_alloc(size)
#         host_allocation = np.zeros(shape, dtype)
#         # im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
#         # print(f"index:{index},name:{name},im.dtype:{im.dtype},im.device:{im.device},model.binding_is_input(index):{model.binding_is_input(index)}")

#         bindings[name] = Binding(name, dtype, shape, host_allocation, allocation)

#     binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
#     batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size


# def map_acc(
#         weights,
#         input_data,
#         imgsz=(640, 640), 
#         conf_thres=0.25, 
#         iou_thres=0.45,
#         device='0',
#         outputs="outputs"
#         # save_txt=True,  # save results to *.txt
#         # nosave=False,  # do not save images/videos
#         # classes=None,  # filter by class: --class 0, or --class 0 2 3
#         # agnostic_nms=False,  # class-agnostic NMS
#         # augment=False,  # augmented inference
#         # visualize=False,  # visualize features
#         # line_thickness=3,  # bounding box thickness (pixels)
#         # hide_labels=False,  # hide labels
#         # hide_conf=False,  # hide confidences
#         # half=False,  # use FP16 half-precision inference
#         # dnn=False,  # use OpenCV DNN for ONNX inference
#         # txtpath=None
#     ):

#     # Directories
#     # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
#     save_dir = outputs
#     print(f"save_dir:{outputs}")
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#     # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

#     # Load model
#     model = DetectMultiBackend_trt(weights, data=input_data)
    
#     stride, names, pt = model.stride, model.names, model.pt
    
#     imgsz = check_img_size(imgsz, s=stride)  # check image size
#     print("pt",pt)
#     # Dataloader
#     if webcam:
#         view_img = check_imshow()
#         cudnn.benchmark = True  # set True to speed up constant image size inference
#         dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
#         bs = len(dataset)  # batch_size
#     else:
#         dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
#         bs = 1  # batch_size
#     # vid_path, vid_writer = [None] * bs, [None] * bs
    
#     # Run inference
#     model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
#     seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
#     for path, im, im0s, vid_cap, s in dataset:
#         with dt[0]:
#             im = torch.from_numpy(im).to(device)
#             im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
#             im /= 255  # 0 - 255 to 0.0 - 1.0
#             if len(im.shape) == 3:
#                 im = im[None]  # expand for batch dim

#         # Inference
#         with dt[1]:
#             visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
#             pred = model(im, augment=augment, visualize=visualize)
        
#         # NMS
#         with dt[2]:
#             pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
#         # Process predictions
#         for i, det in enumerate(pred):  # per image
#             seen += 1
#             if webcam:  # batch_size >= 1
#                 p, im0, frame = path[i], im0s[i].copy(), dataset.count
#                 s += f'{i}: '
#             else:
#                 p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

#             p = Path(p)  # to Path
            
#             txt_path = os.path.join(txtpath, p.name)
#             s += '%gx%g ' % im.shape[2:]  # print string
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#             # imc = im0.copy() if save_crop else im0  # for save_crop
#             # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

#                 # Print results
#                 for c in det[:, -1].unique():
#                     n = (det[:, -1] == c).sum()  # detections per class
#                     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

#                 # Write results
#                 for *xyxy, conf, cls in reversed(det):
#                     if save_txt:  # Write to file
#                         # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                         # line = (cls, conf, *xywh, ) if save_conf else (cls, *xywh)  # label format
#                         # line = (categories[int(cls)], conf, *xyxy) if save_conf else (cls, *xyxy)
#                         tpath = txt_path.replace('.jpg','')
                        
#                         xy = (torch.tensor(xyxy).view(1,4)).view(-1).tolist()
#                         sss = str(xy).replace(',','').replace(']','').replace('[','')
                        
#                         with open(f'{tpath}.txt', 'a') as f:
#                             f.write('{} {} {}\n'.format(categories[int(cls)], conf,sss))
                            
#                             # f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            


#     # Print results
#     t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
#     LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
#     if save_txt or save_img:
#         # s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
#         # LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
#         LOGGER.info(f"Results saved to {colorstr('bold', txtpath)}")
    






def run(args):
    try:
        # import detect
        from yolov5 import detect
    except ImportError:
        print("Could not import the 'detect' module ,check it in : {} ".format(yolov5))
        sys.exit(1)
    
    if not os.path.exists(args.engine):
        print("file not found error: {}".format(args.engine))
        exit(1)
    
    args.inputsize*=2 if len(args.inputsize)==1 else 1
    print("args.inputsize",args.inputsize)

    if args.include == 'map':
        detect.map(args.engine, args.input, imgsz=args.inputsize, conf_thres=args.conf_t, 
        iou_thres=args.iou_t, txtpath=args.txtpath)
        # map_acc(args.engine, args.input, imgsz=args.inputsize, conf_thres=args.conf_t, 
        # iou_thres=args.iou_t, txtpath=args.txtpath)
    elif args.include == 'infer':
        detect.run(args.engine, args.input, imgsz=args.inputsize, conf_thres=args.conf_t, 
        iou_thres=args.iou_t, project=args.outputs,nosave=True)
    else :
        print('check the task mode: topk, infer')                                                                     
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--engine",default=None,help="Set the path to the .engine or trt")
    parser.add_argument("-i","--input",default=None,help="the image path to detect")
    parser.add_argument("--include",default='infer',help="map, infer")
    parser.add_argument("--inputsize",nargs='+', type=int,default=[640],help="320,416,512,608,640")
    parser.add_argument("--txtpath",help="coco benchmark txt path")
    parser.add_argument("--outputs",type=str,default="./outputs",
                        help="The path where inference images are saved")
    parser.add_argument('--conf_t', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou_t', type=float, default=0.65, help='NMS IoU threshold')
    
    args = parser.parse_args()
    if not all([args.input,args.engine]):
        parser.print_help()
        print("\nThese arguments are required:  --input ,--yolov5path and --engine")
        sys.exit(1)
    run(args)