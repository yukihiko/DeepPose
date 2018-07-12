# -*- coding: utf-8 -*-
""" Script for visualizing dataset. """

import sys
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from torchvision import transforms

sys.path.append('./')
from modules.dataset_indexing.pytorch import PoseDataset
from modules.dataset_indexing.pytorch import PoseDataset, Crop, Scale


def main():
    """ Main function. """
    # arg definition
    parser = argparse.ArgumentParser(
        description='Visualize image and its pose in dataset.')
    parser.add_argument(
        'path', type=str, help='Path to dataset (image-pose list file).')
    parser.add_argument(
        '--out', default='result/dataset', help='Output directory.')
    parser.add_argument(
        '--use-visibility', '-v', action='store_true', help='Use visibility to plot pose.')
    args = parser.parse_args()
    output_dir = os.path.join(args.out, os.path.basename(args.path))
    # create directory.
    try:
        os.makedirs(output_dir)
    except OSError:
        pass
    # get dataset.
    dataset = PoseDataset(args.path,
            input_transform=transforms.Compose([
                transforms.ToTensor()]),
            output_transform=Scale(),
            transform=Crop(data_augmentation=False))

    for index, (image, pose, visibility, image_types) in enumerate(tqdm(dataset, ascii=True)):
        # get data.
        #image = trans1(image)
        _, size, _ = image.shape
        #size = image.width
        if args.use_visibility:
            pose = pose[visibility.ravel().astype(bool)]
        else:
            pose = pose.abs()
        pose *= size
        pose_x, pose_y = zip(*pose)
        # plot image and pose.
        #fig = plt.figure()
        fig = plt.figure(figsize=(2.56, 2.56))
        img = image.numpy().transpose(1, 2, 0)
        plt.imshow(img, vmin=0., vmax=1.)
        plt.scatter(pose_x, pose_y, color="r", s=5)
        for i in range(14):   
            plt.scatter(pose_x[i], pose_y[i], color=cm.hsv(i/14.0),  s=15-(i+2))

        plt.axis("off")
        plt.savefig(os.path.join(output_dir, '{0}{1}.png'.format(image_types,index)))
        plt.close(fig)

if __name__ == '__main__':
    main()
