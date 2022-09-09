import onnx
import torch
from models.experimental import attempt_load
from utils.general import colorstr,check_img_size
from models.yolo import Detect
from utils.torch_utils import select_device

def export_onnx(model, im, file, opset, train, dynamic,simplify=False,  prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    
    print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
    f = file

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
        do_constant_folding=not train,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
            'images': {
                0: 'batch',
                2: 'height',
                3: 'width'},  # shape(1,3,640,640)
            'output': {
                0: 'batch',
                1: 'anchors'}  # shape(1,25200,85)
        } if dynamic else None)

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Metadata
    d = {'stride': int(max(model.stride)), 'names': model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
   
    onnx.save(model_onnx, f)

    
    return f, model_onnx



def run(pth,onnx,batch_size,input_size):
    
    device = select_device('0')
    
    model = attempt_load(pth, device=device, inplace=True, fuse=True)
    input_size*=2 if len(input_size)==1 else 1
    gs = int(max(model.stride))  # grid size (max stride)
    input_size = [check_img_size(x, gs) for x in input_size]
    im = torch.zeros(batch_size, 3, *input_size).to(device)
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = False
            m.onnx_dynamic = False
            m.export = True
    for _ in range(2):
        y = model(im)  # dry runs  
    export_onnx(model, im, onnx, 13, False, False,False)


if __name__ == "__main__":
    run()
    # export_onnx_old(model, im, file_old, 13, False, False)


