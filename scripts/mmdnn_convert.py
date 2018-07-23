import sys as _sys
import torch
from torch.autograd import Variable
import mmdnn
import mmdnn.conversion._script.convertToIR as start
import argparse
_sys.path.append("./")
from modules.models.pytorch import AlexNet, VGG19Net, Inceptionv3, Resnet, MobileNet, MobileNetV2, MobileNet_, MobileNet_2, MobileNet_3
from six import text_type as _text_type


def _get_parser():
    import argparse

    parser = argparse.ArgumentParser(description = 'Convert other model file formats to IR format.')

    parser.add_argument(
        '--srcFramework', '-f',
        type=_text_type,
        choices=["caffe", "caffe2", "cntk", "mxnet", "keras", "tensorflow", 'tf', 'torch', 'torch7', 'onnx', 'darknet', 'coreml', 'pytorch'],
        help="Source toolkit name of the model to be converted.")

    parser.add_argument(
        '--weights', '-w', '-iw',
        type=_text_type,
        default=None,
        help='Path to the model weights file of the external tool (e.g caffe weights proto binary, keras h5 binary')

    parser.add_argument(
        '--network', '-n', '-in',
        type=_text_type,
        default=None,
        help='Path to the model network file of the external tool (e.g caffe prototxt, keras json')

    parser.add_argument(
        '--dstPath', '-d', '-o',
        type=_text_type,
        required=True,
        help='Path to save the IR model.')

    parser.add_argument(
        '--inNodeName', '-inode',
        nargs='+',
        type=_text_type,
        default=None,
        help="[Tensorflow] Input nodes' name of the graph.")

    parser.add_argument(
        '--dstNodeName', '-node',
        nargs='+',
        type=_text_type,
        default=None,
        help="[Tensorflow] Output nodes' name of the graph.")

    parser.add_argument(
        '--inputShape',
        nargs='+',
        type=_text_type,
        default=None,
        help='[Tensorflow/MXNet/Caffe2/Torch7] Input shape of model (channel, height, width)')


    # Caffe
    parser.add_argument(
        '--caffePhase',
        type=_text_type,
        default='TRAIN',
        help='[Caffe] Convert the specific phase of caffe model.')


    # Darknet
    parser.add_argument(
        '--darknetYolo',
        type=_text_type,
        choices=["yolov3", "yolov2"],
        help='[Darknet] Convert the specific yolo model.')

    # resume-model
    parser.add_argument(
        '--resume-model', type=str, default=None,
        help='Load model definition file to use for resuming training \
        (it\'s necessary when you resume a training). \
        The file name is "epoch-{epoch number}.mode"')

    # NN
    parser.add_argument(
        '--NN',
        type=_text_type)
    return parser

def main():

    parser = _get_parser()
    args = parser.parse_args()
    args.inputShape = 3,224,224

    if args.NN == "VGG19":
        model = models.vgg19(pretrained=True)
    elif args.NN == "Inception3":
        args = Inceptionv3( aux_logits = False)
    elif args.NN == "ResNet":
        model = Resnet( )
    elif args.NN == "MobileNet":
        model = MobileNet( )
    elif args.NN == "MobileNet_":
        model = MobileNet_( )
    elif args.NN == "MobileNet_2":
        model = MobileNet_2( )
    elif args.NN == "MobileNet_3":
        model = MobileNet_3( )
    elif args.NN == "MobileNetV2":
        model = MobileNetV2( )
    else :
        model = AlexNet(args.Nj)
    
    model.load_state_dict(torch.load(args.resume_model))

    model.eval()
    dummy_input = Variable(torch.randn(1, 3, 224, 224))
    model(dummy_input)
    torch.save(model.cpu(), args.dstPath)

    mmdnn.conversion._script.convertToIR._convert(args)
    _sys.exit(int(ret)) # cast to int or else the exit code is always 1

if __name__ == '__main__':
    main()
