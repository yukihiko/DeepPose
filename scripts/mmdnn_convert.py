import os
import sys as _sys
import torch
from torch.autograd import Variable
import mmdnn
import mmdnn.conversion._script.convertToIR as start
import argparse
_sys.path.append("./")
from modules.models.pytorch import AlexNet, VGG19Net, Inceptionv3, Resnet, MobileNet, MobileNetV2, MobileNet_, MobileNet_2, MobileNet_3
from six import text_type as _text_type
from mmdnn.conversion.pytorch.pytorch_parser import PytorchParser
from mmdnn.conversion._script.IRToModel import _convert

'''
mmtocode -f tensorflow --IRModelPath ./result/coreml/MobileNet_2.pb --IRWeightPath ./result/coreml/MobileNet_2.npy --dstModelPath ./result/tensorflow/tf_MobileNet_2.py

mmtomodel -f tensorflow -in ./result/tensorflow/tf_MobileNet_2.py -iw ./result/coreml/MobileNet_2.npy -o ./result/tensorflow/tf_MobileNet_2 --dump_tag SERVING


Convert to CoreML
python -m mmdnn.conversion._script.IRToModel -f coreml -in MobileNetIR_.pb -iw MobileNetIR_.npy -o MobileNetIR_.mlmodel --scale 0.00392157 --redBias -0 --greenBias -0 --blueBias -0

    # For CoreML
    parser.add_argument('--inputNames', type=_text_type, nargs='*', help='Names of the feature (input) columns, in order (required for keras models).')
    parser.add_argument('--outputNames', type=_text_type, nargs='*', help='Names of the target (output) columns, in order (required for keras models).')
    parser.add_argument('--imageInputNames', type=_text_type, default=[], action='append', help='Label the named input as an image. Can be specified more than once for multiple image inputs.')
    parser.add_argument('--isBGR', action='store_true', default=False, help='True if the image data in BGR order (RGB default)')
    parser.add_argument('--redBias', type=float, default=0.0, help='Bias value to be added to the red channel (optional, default 0.0)')
    parser.add_argument('--blueBias', type=float, default=0.0, help='Bias value to be added to the blue channel (optional, default 0.0)')
    parser.add_argument('--greenBias', type=float, default=0.0, help='Bias value to be added to the green channel (optional, default 0.0)')
    parser.add_argument('--grayBias', type=float, default=0.0, help='Bias value to be added to the gray channel for Grayscale images (optional, default 0.0)')
    parser.add_argument('--scale', type=float, default=1.0, help='Value by which the image data must be scaled (optional, default 1.0)')
    parser.add_argument('--classInputPath', type=_text_type, default='', help='Path to class labels (ordered new line separated) for treating the neural network as a classifier')
    parser.add_argument('--predictedFeatureName', type=_text_type, default='class_output', help='Name of the output feature that captures the class name (for classifiers models).')
'''
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
        '--input_size',
        type=int,
        default=224,
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

    # path
    parser.add_argument(
        '--outpath',
        type=_text_type)

    parser.add_argument(
        '--framework', type=_text_type, choices=['coreml'], 
        help='Format of model at srcModelPath (default is to auto-detect).'
    )

    parser.add_argument(
        '--inputNetwork',
        type=_text_type,
        help='Path of the IR network architecture file.')

    parser.add_argument(
        '--inputWeight',
        type=_text_type,
        help='Path to the IR network weight file.')

    parser.add_argument(
        '--output',
        type=_text_type,
        help='Path to save the destination model')

    # For CoreML
    parser.add_argument('--inputNames', type=_text_type, nargs='*', help='Names of the feature (input) columns, in order (required for keras models).')
    parser.add_argument('--outputNames', type=_text_type, nargs='*', help='Names of the target (output) columns, in order (required for keras models).')
    parser.add_argument('--imageInputNames', type=_text_type, default=[], action='append', help='Label the named input as an image. Can be specified more than once for multiple image inputs.')
    parser.add_argument('--isBGR', action='store_true', default=False, help='True if the image data in BGR order (RGB default)')
    parser.add_argument('--redBias', type=float, default=0.0, help='Bias value to be added to the red channel (optional, default 0.0)')
    parser.add_argument('--blueBias', type=float, default=0.0, help='Bias value to be added to the blue channel (optional, default 0.0)')
    parser.add_argument('--greenBias', type=float, default=0.0, help='Bias value to be added to the green channel (optional, default 0.0)')
    parser.add_argument('--grayBias', type=float, default=0.0, help='Bias value to be added to the gray channel for Grayscale images (optional, default 0.0)')
    parser.add_argument('--scale', type=float, default=1.0, help='Value by which the image data must be scaled (optional, default 1.0)')
    parser.add_argument('--classInputPath', type=_text_type, default='', help='Path to class labels (ordered new line separated) for treating the neural network as a classifier')
    parser.add_argument('--predictedFeatureName', type=_text_type, default='class_output', help='Name of the output feature that captures the class name (for classifiers models).')
    return parser

def main():

    parser = _get_parser()
    args = parser.parse_args()

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
    
    if args.framework == None:
        args.framework = "coreml"
    
    if args.inputNetwork == None:
        args.inputNetwork = os.path.join(args.outpath, args.NN + ".pb")
    
    if args.inputWeight == None:
        args.inputWeight = os.path.join(args.outpath, args.NN + ".npy")
    
    if args.output == None:
        args.output = os.path.join(args.outpath, args.NN + ".mlmodel")

    IR_file = args.NN

    #model.load_state_dict(torch.load(args.resume_model))
    model = torch.load(args.resume_model)
    model.eval()
    model = model.cpu().float()

    '''
    model.eval()
    dummy_input = Variable(torch.randn(1, 3, args.input_size, args.input_size))
    model(dummy_input)
    '''

    pytorchparser = PytorchParser(model, [3, args.input_size, args.input_size])
    #dummy_input = Variable(torch.randn(1, 3, args.input_size, args.input_size))
    #model(dummy_input)

    pytorchparser.run(os.path.join(args.outpath, IR_file))

    _convert(args)

    #mmdnn.conversion._script.convertToIR._convert(args)
    #_sys.exit(int(ret)) # cast to int or else the exit code is always 1

if __name__ == '__main__':
    main()
