# -*- coding: utf-8 -*-
""" Estimate pose by pytorch. """

import torch
from torch.autograd import Variable
from torchvision import transforms

from modules.errors import GPUNotFoundError
from modules.dataset_indexing.pytorch import PoseDataset, Crop, RandomNoise, Scale
from modules.models.pytorch import AlexNet, VGG19Net, Inceptionv3, Resnet, MobileNet, MobileNet_, MobileNet__


class PoseEstimator(object):
    """ Estimate pose using pose net trained by pytorch.

    Args:
        Nj (int): Number of joints.
        gpu (int): GPU ID (negative value indicates CPU).
        model_file (str): Model parameter file.
        filename (str): Image-pose list file.
    """

    def __init__(self, Nj, NN, gpu, model_file, filename, isEval=True):
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
        elif self.NN == "MobileNet":
            self.model = MobileNet()
        elif self.NN == "AlexNet":
            self.model = AlexNet(Nj)
        else:
            self.model = Resnet()

        self.model.load_state_dict(torch.load(model_file))
        if isEval == True:
            self.model.eval()
        # prepare gpu.
        if self.gpu:
            self.model.cuda()
        # load dataset to estimate.
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
        
