import argparse
import sys
import os
# from yolov5 import detect

def run(args):
    # cur_path=os.path.dirname(os.getcwd() + "/")
    # print("cur_path:",cur_path)
    # yolov5=os.path.join(cur_path,'yolov5')
    # sys.path.insert(1,yolov5)
    # print(sys.path)

    try:
        # import detect
        from yolov5 import detect
    except ImportError:
        print("Could not import the 'detect' module ,check it in : {} ".format(yolov5))
        sys.exit(1)
    if not os.path.exists(args.engine):
        print("file not found error: {}".format(args.engine))
        exit(1)
    end= args.engine.split('.')[-1]
    if end =='trt':
        new_engine = args.engine.replace('.trt','')+'.engine'
        print(args.engine)
        os.rename(args.engine,new_engine)
    else :
        new_engine=args.engine
    
    args.inputsize*=2 if len(args.inputsize)==1 else 1
    print("args.inputsize",args.inputsize)
    if args.include == 'map':
        detect.map(new_engine,args.input,imgsz=args.inputsize, conf_thres=args.conf_t, 
        iou_thres=args.iou_t, txtpath=args.txtpath)
    elif args.include == 'infer':
        detect.run(new_engine, args.input, imgsz=args.inputsize, conf_thres=args.conf_t, 
        iou_thres=args.iou_t, project=args.output)
    else :
        print('check the task mode: topk, infer')                                                                     
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--engine",default=None,help="Set the path to the .engine or trt")
    parser.add_argument("-i","--input",default=None,help="the image path to detect")
    # parser.add_argument("-p","--yolov5path",default="../../yolov5",help="Set the path of the dependency")
    parser.add_argument("--include",default='infer',help="map, infer")
    parser.add_argument("--inputsize",nargs='+', type=int,default=[640],help="320,416,512,608,640")
    parser.add_argument("-o","--output",default=None,help="The path where inference images are saved")
    parser.add_argument('--conf_t', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou_t', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument("--txtpath",help="coco benchmark txt path")
    args = parser.parse_args()
    if not all([args.input,args.engine]):
        parser.print_help()
        print("\nThese arguments are required:  --input ,--yolov5path and --engine")
        sys.exit(1)
    run(args)