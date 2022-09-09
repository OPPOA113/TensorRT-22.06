import argparse
import sys
import os


def main(args):
    cur_path=os.path.dirname(os.getcwd())
    yolov5=os.path.join(cur_path,'yolov5')
    sys.path.insert(1,yolov5)
    
    try:
        import export_onnx
    except ImportError:
        print("Could not import the 'export' module ,check the yolov5 path ")
        sys.exit(1)
    export_onnx.run(args.input,args.output,args.batch_size,args.input_size)
    print("saved onnx file in : {}".format(args.output))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input",default=None,help="Set the path to the .pt")
    parser.add_argument("-o","--output",default=None,help="Set the path to save the onnx file")
    # parser.add_argument("-p","--yolov5path",default="../../yolov5/",help="Set the path of the dependency")
    parser.add_argument("-b","--batch_size",default=1)
    parser.add_argument("-is","--input_size",default=[640])
    args = parser.parse_args()
    if not all([args.input,args.output]):
        parser.print_help()
        print("\nThese arguments are required:  --input and --output")
        sys.exit(1)
    main(args)