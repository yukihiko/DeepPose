# -*- coding: utf-8 -*-
""" Pose dataset indexing. """
import os
from PIL import Image, ImageOps
import torch
import numpy as np
from torch.utils import data
import cv2
import random

class PoseDataset3D(data.Dataset):
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
        self.images, self.dists, self.poses2D, self.poses3D, self.visibilities, self.image_types = self._load_dataset()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """ Returns the i-th example. """
        image_o = self._read_image(self.images[index], self.image_types[index])
        dist = self.dists[index]
        pose3D = self.poses3D[index]
        pose2D = self.poses2D[index]
        visibility = self.visibilities[index]
        image_type = self.image_types[index]
        path = self.images[index]

        data_numpy = cv2.imread(
            self.images[index], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        
        image_size = [224,224]
        rf = 30.0
        scale = np.array([1.0, 1.0])
        c = np.array([112.0,112.0])
        r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.9 else 0
        trans = get_affine_transform(c, scale, r, image_size)
        
        image = cv2.warpAffine(
            data_numpy,
            trans,
            (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)
        
        for i in range(visibility.shape[0]):
            if visibility[i, 0] > 0.0:
                pose2D[i] = affine_transform(pose2D[i]*224.0, trans)/224.0
                pose3D[i, 0:2] = affine_transform(pose3D[i, 0:2]*224.0, trans)/224.0
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2,0,1)/255.0
        
        '''
        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.transform is not None:
            image, pose2D, visibility = self.transform(image, pose2D, visibility)
        if self.output_transform is not None:
            pose2D = self.output_transform(pose2D)
        '''

        return torch.Tensor(image), dist, pose2D, pose3D, visibility, image_type, path

    def _load_dataset(self):
        images = []
        dists = []
        poses3D = []
        poses2D = []
        visibilities = []
        image_types = []
        for line in open(self.path):

            line_split = line[:-1].split(',')
            
            if not os.path.isfile(line_split[0]):
                continue

            x = torch.Tensor(list(map(float, line_split[2:])))
            x = x.view(-1, 6)
            pose3D = x[:, :3]
            if  pose3D[0, 0] == float('inf') or pose3D[0, 0] == float('-inf')  :
                continue

            pose2D = x[:, 3:5]
            visibility = x[:, 5].clone().view(-1, 1).expand_as(pose3D)

            # 通常の画像
            images.append(line_split[0])
            dists.append(float(line_split[1]))
            poses3D.append(pose3D)
            poses2D.append(pose2D)
            visibilities.append(visibility)
            image_types.append("N")

        return images, dists, poses2D, poses3D, visibilities, image_types

    @staticmethod
    def _read_image(path, image_type):
        # return Image.open(path).convert('RGB')
        img = Image.open(path).convert('RGB')

        return img

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    pt = pt.numpy()
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return torch.from_numpy(new_pt[:2])

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result
