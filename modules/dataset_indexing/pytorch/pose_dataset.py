# -*- coding: utf-8 -*-
""" Pose dataset indexing. """

from PIL import Image, ImageOps
import torch
import numpy
from torch.utils import data


class PoseDataset(data.Dataset):
    """ Pose dataset indexing.

    Args:
        path (str): A path to dataset.
        input_transform (Transform): Transform to input.
        output_transform (Transform): Transform to output.
        transform (Transform): Transform to both input and target.
    """

    def __init__(self, path, input_transform=None, output_transform=None, transform=None):
        self.path = path
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.transform = transform
        # load dataset.
        self.images, self.poses, self.visibilities, self.image_types = self._load_dataset()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """ Returns the i-th example. """
        image = self._read_image(self.images[index], self.image_types[index])
        pose = self.poses[index]
        visibility = self.visibilities[index]
        image_type = self.image_types[index]
        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.transform is not None:
            image, pose, visibility = self.transform(image, pose, visibility)
        if self.output_transform is not None:
            pose = self.output_transform(pose)
        return image, pose, visibility, image_type

    def _load_dataset(self):
        images = []
        poses = []
        visibilities = []
        image_types = []
        for line in open(self.path):
            """
            line_split = line[:-1].split(',')
            images.append(line_split[0])
            ls = list(map(float, line_split[1:]))
            x = torch.Tensor(ls)
            x = x.view(-1, 3)
            pose = x[:, :2]
            visibility = x[:, 2].clone().view(-1, 1).expand_as(pose)
            poses.append(pose)
            visibilities.append(visibility)
            """
            line_split = line[:-1].split(',')
            images.append(line_split[0])
            x = torch.Tensor(list(map(float, line_split[1:])))
            x = x.view(-1, 3)
            pose = x[:, :2]
            visibility = x[:, 2].clone().view(-1, 1).expand_as(pose)
            poses.append(pose)
            visibilities.append(visibility)
            image_types.append("N")

            # 画像の転置
            images.append(line_split[0])
            p = torch.stack([pose[:,1], pose[:,0]], dim=1)
            poses.append(p[[5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]])
            v = torch.stack([visibility[:,1], visibility[:,0]], dim=1)
            visibilities.append(v[[5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]])
            image_types.append("R")

            # 画像の反転
            images.append(line_split[0])
            p2 = torch.stack([256-pose[:,0], pose[:,1]], dim=1)
            poses.append(p2[[5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]])
            visibilities.append(v[[5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]])
            image_types.append("M")

            # 画像の上下反転
            images.append(line_split[0])
            p3 = torch.stack([pose[:,0], 256-pose[:,1]], dim=1)
            poses.append(p3[[5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]])
            visibilities.append(v[[5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]])
            image_types.append("F")

            # 画像の転置の反転
            images.append(line_split[0])
            p4 = torch.stack([256-p[:,0], p[:,1]], dim=1)
            poses.append(p4)
            visibilities.append(visibility.clone())
            image_types.append("T")

            # 画像の転置の上下反転
            images.append(line_split[0])
            p5 = torch.stack([p[:,0], 256-p[:,1]], dim=1)
            poses.append(p5)
            visibilities.append(visibility.clone())
            image_types.append("L")

            # 画像の転置の上下反転の反転
            images.append(line_split[0])
            p6 = torch.stack([256-p5[:,0], p5[:,1]], dim=1)
            poses.append(p6[[5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]])
            visibilities.append(v[[5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]])
            image_types.append("W")
            
            # 画像の上下反転の反転
            images.append(line_split[0])
            p7 = torch.stack([256-p3[:,0], p3[:,1]], dim=1)
            poses.append(p7)
            visibilities.append(visibility.clone())
            image_types.append("K")

        return images, poses, visibilities, image_types

    @staticmethod
    def _read_image(path, image_type):
        # return Image.open(path).convert('RGB')
        img = Image.open(path).convert('RGB')
        if image_type == "R":
            imgArray = numpy.asarray(img)
            R = imgArray[:,:,0].T
            G = imgArray[:,:,1].T
            B = imgArray[:,:,2].T
            image = numpy.stack([R, G, B], axis=2)
            img =Image.fromarray(numpy.uint8(image))
        elif image_type == "M":
            img =ImageOps.mirror(img)
        elif image_type == "F":
            img =ImageOps.flip(img)
        elif image_type == "T":
            imgArray = numpy.asarray(img)
            R = imgArray[:,:,0].T
            G = imgArray[:,:,1].T
            B = imgArray[:,:,2].T
            image = numpy.stack([R, G, B], axis=2)
            img =Image.fromarray(numpy.uint8(image))
            img =ImageOps.mirror(img)
        elif image_type == "L":
            imgArray = numpy.asarray(img)
            R = imgArray[:,:,0].T
            G = imgArray[:,:,1].T
            B = imgArray[:,:,2].T
            image = numpy.stack([R, G, B], axis=2)
            img =Image.fromarray(numpy.uint8(image))
            img =ImageOps.flip(img)
        elif image_type == "W":
            imgArray = numpy.asarray(img)
            R = imgArray[:,:,0].T
            G = imgArray[:,:,1].T
            B = imgArray[:,:,2].T
            image = numpy.stack([R, G, B], axis=2)
            img =Image.fromarray(numpy.uint8(image))
            img =ImageOps.flip(img)
            img =ImageOps.mirror(img)
        elif image_type == "K":
            img =ImageOps.flip(img)
            img =ImageOps.mirror(img)
        return img
