# -*- coding: utf-8 -*-
""" Evaluate estimating time. """

import os
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

from modules.evaluators import chainer, pytorch


class EstimatingTimeEvaluator(object):
    """ Evaluate estimating time of pose net by chainer and pytorch.

    Args:
        Nj (int): Number of joints.
        gpu (int): GPU ID (negative value indicates CPU).
        chainer_model_file (str): Chainer model parameter file.
        pytorch_model_file (str): Pytorch model parameter file.
        filename (str): Image-pose list file.
        output (str): Output directory.
        debug (bool): Debug mode.
    """

    def __init__(self, **kwargs):
        self.output = kwargs['output']
        self.Nj = kwargs['Nj']
        self.NN = kwargs['NN']
        self.col = 14
        try:
            os.makedirs(self.output)
        except OSError:
            pass
        self.estimator = {
            #'chainer': chainer.PoseEstimator(
            #    kwargs['Nj'], kwargs['gpu'], kwargs['chainer_model_file'], kwargs['filename']),
            'pytorch': pytorch.PoseEstimator(
                kwargs['Nj'], kwargs['NN'], kwargs['gpu'], kwargs['pytorch_model_file'], kwargs['filename'])}
        self.debug = kwargs['debug']

    def plot(self, samples, title):
        """ Plot estimating time of chainer and pytorch. """
        time_mean = []
        time_std = []
        for estimator in tqdm(self.estimator.values(), desc='testers'):
            random_index = np.random.randint(0, estimator.get_dataset_size(), samples)
            dataset_size = estimator.get_dataset_size()
            compute_time = []
            # get compute time.
            #for index in tqdm(random_index, desc='samples'):
            for index in tqdm(range(dataset_size), desc='samples'):
                start = time.time()


                if self.NN == "MobileNet_":
                    image, offset, heatmap, testPose = estimator.estimate_(index)
                    _, size, _ = image.shape
                    scale = float(size)/float(self.col)

                    reshaped = heatmap.view(-1, self.Nj, self.col*self.col)
                    _, argmax = reshaped.max(-1)
                    yCoords = argmax/self.col
                    xCoords = argmax - yCoords*self.col
                    xc = np.squeeze(xCoords.cpu().data.numpy()).astype(np.float32)
                    yc = np.squeeze(yCoords.cpu().data.numpy()).astype(np.float32)
                    dat_x = xc * scale
                    dat_y = yc * scale
               
                    # 最終
                    offset_reshaped = offset.view(-1, self.Nj * 2, self.col,self.col)
                    op = np.squeeze(offset_reshaped.cpu().data.numpy())
                    for j in range(self.Nj):
                            dat_x[j] = op[j, int(yc[j]), int(xc[j])] * scale + dat_x[j]
                            dat_y[j] = op[j + 14, int(yc[j]), int(xc[j])] * scale + dat_y[j]

                    fig = plt.figure(figsize=(2.24, 2.24))

                    img = image.numpy().transpose(1, 2, 0)
                    plt.imshow(img, vmin=0., vmax=1.)
                    for j in range(14):   
                        if heatmap[0, j, int(yc[j]), int(xc[j])] > 0.5:
                            plt.scatter(dat_x[j], dat_y[j], color=cm.hsv(j/14.0),  s=10)

                elif self.NN == "MobileNet__":
                    image, offset, heatmap, output, testPose = estimator.estimate__(index)
                    _, size, _ = image.shape
                    scale = float(size)/float(self.col)

                    reshaped = heatmap.view(-1, self.Nj, self.col*self.col)
                    _, argmax = reshaped.max(-1)
                    yCoords = argmax/self.col
                    xCoords = argmax - yCoords*self.col
                    xc = np.squeeze(xCoords.cpu().data.numpy()).astype(np.float32)
                    yc = np.squeeze(yCoords.cpu().data.numpy()).astype(np.float32)
                    dat_x = xc * scale
                    dat_y = yc * scale
               
                    # 最終
                    #offset_reshaped = offset.view(-1, self.Nj, 2)
                    offset_reshaped = offset.view(-1, self.Nj * 2, self.col,self.col)
                    op = np.squeeze(offset_reshaped.cpu().data.numpy())
                    for j in range(self.Nj):
                        dat_x[j] = op[j, int(yc[j]), int(xc[j])] * scale + dat_x[j]
                        dat_y[j] = op[j + 14, int(yc[j]), int(xc[j])] * scale + dat_y[j]
                        #dat_x[j] = op[j, 0] * scale + dat_x[j]
                        #dat_y[j] = op[j, 1] * scale + dat_y[j]
                    '''
                    pose = output.view(-1, self.Nj, 3)
                    pose = pose[:, :, 0:2]
                    dat = pose.data[0].cpu().numpy()
                    dat *= size
                    dat_x, dat_y = zip(*dat)
                    '''
                    fig = plt.figure(figsize=(2.24, 2.24))

                    img = image.numpy().transpose(1, 2, 0)
                    plt.imshow(img, vmin=0., vmax=1.)
                    for i in range(14):   
                        plt.scatter(dat_x[i], dat_y[i], color=cm.hsv(i/14.0),  s=8)

                elif self.NN == "MobileNet___":
                    image, offset, heatmap, testPose = estimator.estimate_(index)
                    _, size, _ = image.shape
                    scale = float(size)/float(self.col)

                    heatmap2 = heatmap[: , self.Nj:, :, :]
                    heatmap = heatmap[: , :self.Nj, :, :]
                    
                    h2_0 = heatmap2[:,0,:,:].view(-1, 1, self.col, self.col)
                    h2_1 = heatmap2[:,1,:,:].view(-1, 1, self.col, self.col)
                    h2_2 = heatmap2[:,2,:,:].view(-1, 1, self.col, self.col)
                    h2_3 = heatmap2[:,3,:,:].view(-1, 1, self.col, self.col)
                    z = torch.zeros(h2_0.size()).float().cuda()
                    h2_0 = torch.cat([h2_0, h2_0, h2_0,], dim=1)
                    h2_1 = torch.cat([h2_1, h2_1, h2_1], dim=1)
                    h2_2 = torch.cat([h2_2, h2_2, h2_2], dim=1)
                    h2_3 = torch.cat([h2_3, h2_3, h2_3], dim=1)

                    h3 = torch.cat([ h2_0, h2_1, h2_2, h2_3, z, z], dim=1)
                    heatmap = heatmap + h3

                    #for index in range(3):
                    #    heatmap = heatmap[:, index, :,]
                    reshaped = heatmap.view(-1, self.Nj, self.col*self.col)
                    _, argmax = reshaped.max(-1)
                    yCoords = argmax/self.col
                    xCoords = argmax - yCoords*self.col
                    xc = np.squeeze(xCoords.cpu().data.numpy()).astype(np.float32)
                    yc = np.squeeze(yCoords.cpu().data.numpy()).astype(np.float32)
                    dat_x = xc * scale
                    dat_y = yc * scale
               
                    # 最終
                    #offset_reshaped = offset.view(-1, self.Nj, 2)
                    offset_reshaped = offset.view(-1, self.Nj * 2, self.col,self.col)
                    op = np.squeeze(offset_reshaped.cpu().data.numpy())

                    fig = plt.figure(figsize=(2.24, 2.24))

                    img = image.numpy().transpose(1, 2, 0)
                    plt.imshow(img, vmin=0., vmax=1.)
                    m = torch.Tensor([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1.0,1.0]).cuda()

                    for j in range(self.Nj):
                        if heatmap[0, j, int(yc[j]), int(xc[j])] * m[j] > 0.5:
                            dx = op[j, int(yc[j]), int(xc[j])] * scale + dat_x[j]
                            dy = op[j + 14, int(yc[j]), int(xc[j])] * scale + dat_y[j]
                            #dx = dat_x[j]
                            #dy = dat_y[j]
                            plt.scatter(dx, dy, color=cm.hsv(j/14.0),  s=8)

                elif self.NN == "MnasNet":
                    image, pose, testPose = estimator.estimate(index)
                    _, size, _ = image.shape
               
                    pose = pose.view(-1, self.Nj, 3)
                    pose = pose[:, :, :2]
                    dat = pose.data[0].cpu().numpy()
                    dat *= size
                    dat_x = dat[:, 0]
                    dat_y = dat[:, 1]

                    testdat = testPose.cpu().numpy()
                    testdat *= size
                    testdat_x, testdat_y = zip(*testdat)

                    fig = plt.figure(figsize=(2.24, 2.24))
                    img = image.numpy().transpose(1, 2, 0)
                    plt.imshow(img, vmin=0., vmax=1.)
                    for i in range(14):   
                        #plt.scatter(testdat_x[i], testdat_y[i], color=cm.hsv(i/14.0), s=7)
                        plt.scatter(dat_x[i], dat_y[i], color=cm.hsv(i/14.0),  s=8)

                elif self.NN == "MnasNet_":
                    image, offset, heatmap, testPose = estimator.estimate_(index)
                    _, size, _ = image.shape
                    scale = float(size)/float(self.col)

                    reshaped = heatmap.view(-1, self.Nj, self.col*self.col)
                    _, argmax = reshaped.max(-1)
                    yCoords = argmax/self.col
                    xCoords = argmax - yCoords*self.col
                    xc = np.squeeze(xCoords.cpu().data.numpy()).astype(np.float32)
                    yc = np.squeeze(yCoords.cpu().data.numpy()).astype(np.float32)
                    dat_x = xc * scale
                    dat_y = yc * scale
               
                    # 最終
                    offset_reshaped = offset.view(-1, self.Nj * 2, self.col,self.col)
                    op = np.squeeze(offset_reshaped.cpu().data.numpy())
                    for j in range(self.Nj):
                        #dat_x[j] = op[j, int(yc[j]), int(xc[j])] * scale + dat_x[j]
                        #dat_y[j] = op[j + 14, int(yc[j]), int(xc[j])] * scale + dat_y[j]
                        dat_x[j] = dat_x[j]
                        dat_y[j] = dat_y[j]

                    fig = plt.figure(figsize=(2.24, 2.24))

                    img = image.numpy().transpose(1, 2, 0)
                    plt.imshow(img, vmin=0., vmax=1.)
                    for j in range(14):   
                        #if heatmap[0, j, int(yc[j]), int(xc[j])] > 0.5:
                        plt.scatter(dat_x[j], dat_y[j], color=cm.hsv(j/14.0),  s=10)

                elif self.NN == "MnasNet56_":
                    self.col = 56
                    image, offset, heatmap, testPose = estimator.estimate_(index)
                    _, size, _ = image.shape
                    scale = float(size)/float(self.col)

                    reshaped = heatmap.view(-1, self.Nj, self.col*self.col)
                    _, argmax = reshaped.max(-1)
                    yCoords = argmax/self.col
                    xCoords = argmax - yCoords*self.col
                    xc = np.squeeze(xCoords.cpu().data.numpy()).astype(np.float32)
                    yc = np.squeeze(yCoords.cpu().data.numpy()).astype(np.float32)
                    dat_x = xc * scale
                    dat_y = yc * scale
               
                    # 最終
                    offset_reshaped = offset.view(-1, self.Nj * 2, self.col,self.col)
                    op = np.squeeze(offset_reshaped.cpu().data.numpy())
                    for j in range(self.Nj):
                        #dat_x[j] = op[j, int(yc[j]), int(xc[j])] * scale + dat_x[j]
                        #dat_y[j] = op[j + 14, int(yc[j]), int(xc[j])] * scale + dat_y[j]
                        dat_x[j] = dat_x[j]
                        dat_y[j] = dat_y[j]

                    fig = plt.figure(figsize=(2.24, 2.24))

                    img = image.numpy().transpose(1, 2, 0)
                    plt.imshow(img, vmin=0., vmax=1.)
                    for j in range(14):   
                        #if heatmap[0, j, int(yc[j]), int(xc[j])] > 0.5:
                        plt.scatter(dat_x[j], dat_y[j], color=cm.hsv(j/14.0),  s=10)

                else:
                    image, pose, testPose = estimator.estimate(index)
                    _, size, _ = image.shape
               
                    pose = pose.view(-1, self.Nj, 2)
                    dat = pose.data[0].cpu().numpy()
                    dat *= size
                    dat_x, dat_y = zip(*dat)

                    testdat = testPose.cpu().numpy()
                    testdat *= size
                    testdat_x, testdat_y = zip(*testdat)

                    # pose *= size
                    # pose_x, pose_y = zip(*pose)
                    # plot image and pose.
                    fig = plt.figure(figsize=(2.24, 2.24))
                    #print(pose.data[0])
                    #print(pose.data[0][:, 0])
                    #print(pose.data[0][:, 1])
                    img = image.numpy().transpose(1, 2, 0)
                    plt.imshow(img, vmin=0., vmax=1.)
                    for i in range(14):   
                        #plt.scatter(testdat_x[i], testdat_y[i], color=cm.hsv(i/14.0), s=7)
                        plt.scatter(dat_x[i], dat_y[i], color=cm.hsv(i/14.0),  s=8)

                plt.axis("off")
                plt.savefig(os.path.join(self.output, '{}.png'.format(index)))
                plt.close(fig)

                compute_time.append(time.time() - start)
            # calculate mean and std.
            time_mean.append(np.mean(compute_time))
            time_std.append(np.std(compute_time))
        # plot estimating time.
        plt.bar(range(len(time_mean)), time_mean, yerr=time_std,
                width=0.3, align='center', ecolor='black', tick_label=self.estimator.keys())
        # plot settings.
        plt.title(title)
        plt.ylabel('estimating time [sec]')
        # save plot.
        if self.debug:
            plt.show()
        else:
            filename = '_'.join(title.split(' ')) + '.png'
            plt.savefig(os.path.join(self.output, filename))
