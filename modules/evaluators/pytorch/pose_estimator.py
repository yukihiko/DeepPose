# -*- coding: utf-8 -*-
""" Estimate pose by pytorch. """

import torch
from torch.autograd import Variable
from torchvision import transforms

from modules.errors import GPUNotFoundError
from modules.dataset_indexing.pytorch import PoseDataset, PoseDataset3D, Crop, RandomNoise, Scale
from modules.models.pytorch import AlexNet, VGG19Net, Inceptionv3, Resnet, MobileNet, MobileNet_, MobileNet_3, MobileNet_4, MobileNet__, MobileNet___, MnasNet, MnasNet_, MnasNet56_, MobileNet3D, MobileNet3D2, MnasNet3D


class PoseEstimator(object):
    """ Estimate pose using pose net trained by pytorch.

    Args:
        Nj (int): Number of joints.
        gpu (int): GPU ID (negative value indicates CPU).
        model_file (str): Model parameter file.
        filename (str): Image-pose list file.
    """

    def __init__(self, Nj, NN, gpu, model_file, filename, Dataset3D, isEval=True):
        # validate arguments.
        self.gpu = (gpu >= 0)
        self.NN = NN
        if self.gpu and not torch.cuda.is_available():
            raise GPUNotFoundError('GPU is not found.')
        # initialize model to estimate.
        if self.NN == "MobileNet_":
            self.model = MobileNet_()
        elif self.NN == "MobileNet__":
            self.model = MobileNet__()
        elif self.NN == "MobileNet_3":
            self.model = MobileNet_3()
        elif self.NN == "MobileNet_4":
            self.model = MobileNet_4()
        elif self.NN == "MobileNet___":
            self.model = MobileNet___()
        elif self.NN == "MobileNet":
            self.model = MobileNet()
        elif self.NN == "MnasNet":
            self.model = MnasNet()
        elif self.NN == "MnasNet_":
            self.model = MnasNet_()
        elif self.NN == "MnasNet56_":
            self.model = MnasNet56_()
        elif self.NN == "AlexNet":
            self.model = AlexNet(Nj)
        elif self.NN == "MobileNet3D":
            self.model = MobileNet3D()
        elif self.NN == "MnasNet3D":
            self.model = MnasNet3D()
        elif self.NN == "MobileNet3D2":
            self.model = MobileNet3D2()
        else:
            self.model = Resnet()

        self.model.load_state_dict(torch.load(model_file))
        if isEval == True:
            self.model.eval()
        # prepare gpu.
        if self.gpu:
            self.model.cuda()
        # load dataset to estimate.
        if Dataset3D:
            '''
            self.dataset = PoseDataset3D(
                filename,
                input_transform=transforms.ToTensor())
            '''
            self.dataset = PoseDataset3D(
                filename,
                input_transform=transforms.Compose([
                    transforms.ToTensor(),
                    RandomNoise()]))
            
        else:
            self.dataset = PoseDataset(
                filename,
                input_transform=transforms.Compose([
                    transforms.ToTensor(),
                    RandomNoise()]),
                output_transform=Scale(),
                transform=Crop(data_augmentation=False))

    def get_dataset_size(self):
        """ Get size of dataset. """
        return len(self.dataset)

    def estimate(self, index):
        """ Estimate pose of i-th image. """
        image, pose, _, _ = self.dataset[index]
        v_image = Variable(image.unsqueeze(0))
        if self.gpu:
            v_image = v_image.cuda()
        return image, self.model.forward(v_image), pose

    def estimate_(self, index):
        """ Estimate pose of i-th image. """
        image, pose, _, _ = self.dataset[index]
        v_image = Variable(image.unsqueeze(0))
        if self.gpu:
            v_image = v_image.cuda()
            offset, heatmap = self.model.forward(v_image)
        return image, offset, heatmap, pose

    def estimate224(self, index):
        """ Estimate pose of i-th image. """
        image, pose, _, _ = self.dataset[index]
        v_image = Variable(image.unsqueeze(0))
        if self.gpu:
            v_image = v_image.cuda()
            heatmap = self.model.forward(v_image)
        return image, heatmap, pose

    def estimate__(self, index):
        """ Estimate pose of i-th image. """
        image, pose, _, _ = self.dataset[index]
        v_image = Variable(image.unsqueeze(0))
        if self.gpu:
            v_image = v_image.cuda()
            offset, heatmap, output = self.model.forward(v_image)
        return image, offset, heatmap, output, pose
   
    def estimate3D(self, index):
        """ Estimate pose of i-th image. """
        image, dist, pose2D, pose3D, v, typ, path = self.dataset[index]
        v_image = Variable(image.unsqueeze(0))
        if self.gpu:
            v_image = v_image.cuda()
            offset, heatmap, offset3D, heatmap3D = self.model.forward(v_image)
        return image, offset, heatmap, offset3D, heatmap3D, pose2D, pose3D, path
     
