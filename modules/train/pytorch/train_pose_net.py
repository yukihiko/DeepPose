# -*- coding: utf-8 -*-
""" Train pose net. """

import os
import random
import time
from tqdm import tqdm, trange
import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn as nn
import subprocess
import scipy.ndimage.filters as fi

from modules.errors import FileNotFoundError, GPUNotFoundError, UnknownOptimizationMethodError, NotSupportedError
from modules.models.pytorch import AlexNet, VGG19Net, Inceptionv3, Resnet
from modules.models.pytorch import MobileNet, MobileNetV2, MobileNet_, MobileNet_2, MobileNet_3, MobileNet_4, MobileNet__, MobileNet___, MobileNet16_, MobileNet3D, MobileNet224HM
from modules.models.pytorch import MnasNet, MnasNet_, MnasNet56_, Discriminator, Discriminator2
from modules.dataset_indexing.pytorch import PoseDataset, Crop, RandomNoise, Scale
from modules.functions.pytorch import mean_squared_error, mean_squared_error2,mean_squared_error3, mean_squared_error2_, mean_squared_error2__, mean_squared_error_FC3, mean_squared_error2GAN, mean_squared_error224GAN, mean_squared_error224HM

class TrainLogger(object):
    """ Logger of training pose net.

    Args:
        out (str): Output directory.
    """

    def __init__(self, out):
        try:
            os.makedirs(out)
        except OSError:
            pass
        self.out = out
        self.logs = []

    def write(self, log, colab=False):
        """ Write log. """
        self.file = open(os.path.join(self.out, 'log.txt'), 'a')
        tqdm.write(log)
        tqdm.write(log, file=self.file)
        self.file.flush()
        self.file.close()
        self.logs.append(log)
        if colab == True:
            subprocess.run(["cp", "./result/pytorch/log", "../drive/result/pytorch/log.txt"])

    def write_oneDrive(self, log):
        """ Write log. """
        self.file = open('C:/Users/aoyag/OneDrive/pytorch/log_dp.txt', 'a')
        tqdm.write(log, file=self.file)
        self.file.flush()
        self.file.close()
        self.logs.append(log)

    def state_dict(self):
        """ Returns the state of the logger. """
        return {'logs': self.logs}

    def load_state_dict(self, state_dict):
        """ Loads the logger state. """
        self.logs = state_dict['logs']
        # write logs.
        tqdm.write(self.logs[-1])
        for log in self.logs:
            tqdm.write(log, file=self.file)


class TrainPoseNet(object):
    """ Train pose net of estimating 2D pose from image.

    Args:
        Nj (int): Number of joints.
        use_visibility (bool): Use visibility to compute loss.
        data-augmentation (bool): Crop randomly and add random noise for data augmentation.
        epoch (int): Number of epochs to train.
        opt (str): Optimization method.
        gpu (bool): Use GPU.
        seed (str): Random seed to train.
        train (str): Path to training image-pose list file.
        val (str): Path to validation image-pose list file.
        batchsize (int): Learning minibatch size.
        out (str): Output directory.
        resume (str): Initialize the trainer from given file.
            The file name is 'epoch-{epoch number}.iter'.
        resume_model (str): Load model definition file to use for resuming training
            (it\'s necessary when you resume a training).
            The file name is 'epoch-{epoch number}.model'.
        resume_opt (str): Load optimization states from this file
            (it\'s necessary when you resume a training).
            The file name is 'epoch-{epoch number}.state'.
    """

    def __init__(self, **kwargs):
        self.Nj = kwargs['Nj']
        self.use_visibility = kwargs['use_visibility']
        self.data_augmentation = kwargs['data_augmentation']
        self.epoch = kwargs['epoch']
        self.gpu = (kwargs['gpu'] >= 0)
        self.NN = kwargs['NN']
        self.opt = kwargs['opt']
        self.seed = kwargs['seed']
        self.train = kwargs['train']
        self.val = kwargs['val']
        self.batchsize = kwargs['batchsize']
        self.out = kwargs['out']
        self.resume = kwargs['resume']
        self.resume_model = kwargs['resume_model']
        self.resume_discriminator = kwargs['resume_discriminator']
        self.resume_discriminator2 = kwargs['resume_discriminator2']
        self.resume_opt = kwargs['resume_opt']
        self.colab = kwargs['colab']
        self.useOneDrive = kwargs['useOneDrive']
        # validate arguments.
        self._validate_arguments()

    def _validate_arguments(self):
        if self.seed is not None and self.data_augmentation:
            raise NotSupportedError('It is not supported to fix random seed for data augmentation.')
        if self.gpu and not torch.cuda.is_available():
            raise GPUNotFoundError('GPU is not found.')
        for path in (self.train, self.val):
            if not os.path.isfile(path):
                raise FileNotFoundError('{0} is not found.'.format(path))
        if self.opt not in ('MomentumSGD', 'Adam'):
            raise UnknownOptimizationMethodError(
                '{0} is unknown optimization method.'.format(self.opt))
        if self.resume is not None:
            for path in (self.resume, self.resume_model, self.resume_opt):
                if not os.path.isfile(path):
                    raise FileNotFoundError('{0} is not found.'.format(path))

    def _get_optimizer(self, model):
        if self.opt == 'MomentumSGD':
            optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        elif self.opt == "Adam":
            optimizer = optim.Adam(model.parameters())
        return optimizer

    def min_max(self, x, axis=None, maxV=1.0):
        min = x.min(axis=axis, keepdims=True)
        max = x.max(axis=axis, keepdims=True)
        result = (x-min)/(max-min)
        return torch.Tensor(result)

    def checkMatrix(self, xi, yi):
        f = False
        if xi >= 0 and xi <= 13 and yi >= 0 and yi <= 13:
            f = True
        return xi, yi, f

    def checkSize(self, xi, yi, size=224):
        f = False
        if xi >= 0 and xi < size and yi >= 0 and yi < size:
            f = True
        return xi, yi, f

    def _train(self, model, optimizer, train_iter, log_interval, logger, start_time, discriminator=None, optimizer_d=None, discriminator2=None, optimizer_d2=None):
        model.train()
        if discriminator != None:
            discriminator.train()
            loss_f = nn.BCEWithLogitsLoss()
            #loss_f = nn.BCELoss()
            
        if discriminator2 != None:
            discriminator2.train()
        lr = 0.1

        for iteration, batch in enumerate(tqdm(train_iter, desc='this epoch'), 1):
            image, pose, visibility = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
            if self.gpu:
                image, pose, visibility = image.cuda(), pose.cuda(), visibility.cuda()
            
            if discriminator != None:
                s = image.size()
                ones= Variable(torch.ones(s[0])).cuda()
                zero= Variable(torch.zeros(s[0])).cuda()

            optimizer.zero_grad()

            if self.NN == "MobileNet_":
                offset, heatmap = model(image)
                loss = mean_squared_error2(offset, heatmap, pose, visibility, self.use_visibility)
                loss.backward()
            elif self.NN == "MobileNet_3":
                offset, heatmap = model(image)
                loss = mean_squared_error2(offset, heatmap, pose, visibility, self.use_visibility)
                loss.backward()
            elif self.NN == "MobileNet_4":
                offset, heatmap = model(image)
                loss = mean_squared_error2(offset, heatmap, pose, visibility, self.use_visibility)
                loss.backward()
            elif self.NN == "MobileNet__":
                offset, heatmap, output = model(image)
                loss = mean_squared_error2_(offset, heatmap, output.view(-1, self.Nj, 3), pose, visibility, self.use_visibility)
                loss.backward()
            elif self.NN == "MobileNet___":
                offset, heatmap = model(image)
                loss = mean_squared_error2__(offset, heatmap, pose, visibility, self.use_visibility)
                loss.backward()
            elif self.NN == "MobileNet" or self.NN == "MobileNet_2":
                output = model(image)
                loss = mean_squared_error3(output, pose, visibility, self.use_visibility)
                loss.backward()
            elif self.NN == "MnasNet":
                output = model(image)
                loss = mean_squared_error_FC3(output.view(-1, self.Nj, 3), pose, visibility, self.use_visibility)
                loss.backward()
            elif self.NN == "MnasNet_":
                offset, heatmap = model(image)
                loss = mean_squared_error2(offset, heatmap, pose, visibility, self.use_visibility)
                loss.backward()
            elif self.NN == "MnasNet56_":
                offset, heatmap = model(image)
                loss = mean_squared_error2(offset, heatmap, pose, visibility, self.use_visibility, col=56)
                loss.backward()
            elif self.NN == "MnasNet_+Discriminator" or self.NN == "MobileNet_3+Discriminator":
                # fake data
                model.zero_grad()
                discriminator.zero_grad()
                optimizer_d.zero_grad()
                offset, heatmap = model(image)
                
                heatmap_tensor = heatmap.data
                out = discriminator(heatmap)
                #loss_m, tt = mean_squared_error2GAN(offset, Variable(heatmap.data), pose, visibility, self.use_visibility)
                #loss = loss_m + loss_f(out, ones)
                loss = loss_f(out, ones) 
                loss.backward()
                
                s = heatmap.data.size()
                tt = torch.zeros(s).float()
                ti = pose*14
                v = visibility
                for i in range(s[0]):
                    for j in range(14):

                        if int(v[i, j, 0]) == 1:
                            xi, yi, f = self.checkMatrix(int(ti[i, j, 0]), int(ti[i, j, 1]))
                            
                            if f == True:
                                # 正規分布に近似したサンプルを得る
                                # 平均は 100 、標準偏差を 1 
                                tt[i, j, yi, xi]  = 1
                                tt[i, j] = self.min_max(fi.gaussian_filter(tt[i, j], 1.0))
                tt = tt.cuda()
                diff1 = heatmap.data - tt
                cnt = 0
                for i in range(s[0]):
                    for j in range(self.Nj):
                        if int(v[i, j, 0]) == 0:
                            diff1[i, j].data[0] = diff1[i, j].data[0]*0
                        else:
                            cnt = cnt + 1
                diff1 = diff1.view(-1)
                loss_m = diff1.dot(diff1) / cnt
                
                '''
                s = heatmap.data.size()
                tt = torch.zeros(s).float()
                ti = pose*14
                v = visibility
                for i in range(s[0]):
                    for j in range(14):
                        if int(v[i, j, 0]) == 1:
                            
                            if xi >= 0 and xi <= 13 and yi >= 0 and yi <= 13:
                                xi = int(ti[i, j, 0])
                                yi = int(ti[i, j, 1])
                                tt[i, j, yi, xi]  = 1
                                gaussian = fi.gaussian_filter(tt[i, j], self.gaussian)
                                min = gaussian.min(axis=axis, keepdims=True)
                                max = gaussian.max(axis=axis, keepdims=True)
                                result = (x-min)/(max-min)
                                tt[i, j] = torch.Tensor(result)
                            #else:
                            #    v[i, j, 0] = 0
                            #    v[i, j, 1] = 0
                '''
            elif self.NN == "MobileNet_3+Discriminator2":
                # fake data
                model.zero_grad()
                discriminator.zero_grad()
                optimizer_d.zero_grad()
                offset, heatmap = model(image)
                heatmap_tensor = heatmap.data
                #loss_m, lossf1, tt, tt224, xx_tensor = mean_squared_error224GAN(offset, heatmap, pose, visibility, discriminator, discriminator2, self.use_visibility)
                #loss_m, lossf1, lossf2, tt, tt224, xx_tensor = mean_squared_error224GAN(offset, heatmap, pose, visibility, discriminator, discriminator2, self.use_visibility)
                loss_m = mean_squared_error224GAN(offset, heatmap, pose, visibility, discriminator, discriminator2, self.use_visibility)

                loss = loss_m
                #loss = loss_m + lossf1 + lossf2
                #loss = loss_m + lossf1 
                loss.backward()
                
            elif self.NN == "MobileNet224HM":
                heatmap = model(image)
                loss = mean_squared_error224HM(heatmap, pose, visibility, self.use_visibility, col=224)
                loss.backward()
            else :
                output = model(image)
                loss = mean_squared_error(output.view(-1, self.Nj, 2), pose, visibility, self.use_visibility)
                loss.backward()

            optimizer.step()
            
            if discriminator != None:
                model.zero_grad()
                discriminator.zero_grad()
                optimizer.zero_grad()
                optimizer_d.zero_grad()
                real_out = discriminator(tt224)
                loss_d_real = loss_f(real_out, ones[:heatmap.size()[0]]) 
                hm = Variable(xx_tensor)
                fake_out = discriminator(hm)
                loss_d_fake = loss_f(fake_out, zero[:heatmap.size()[0]]) 
                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                optimizer_d.step()

            if discriminator2 != None:
                model.zero_grad()
                discriminator.zero_grad()
                discriminator2.zero_grad()
                optimizer.zero_grad()
                optimizer_d2.zero_grad()
                real_out2 = discriminator2(tt.cuda())
                loss_d_real2 = loss_f(real_out2, ones[:heatmap.size()[0]]) 
                hm = Variable(heatmap_tensor)
                fake_out2 = discriminator2(hm)
                loss_d_fake2 = loss_f(fake_out2, zero[:heatmap.size()[0]]) 
                loss_d2 = loss_d_real2 + loss_d_fake2
                loss_d2.backward()
                optimizer_d2.step()

            if iteration % log_interval == 0:
                if discriminator2 != None:
                    log_d = 'elapsed_time: {0}, loss: {1}'.format(time.time() - start_time, loss)
                    #log_d = 'elapsed_time: {0:.3f}, loss_m: {1:.5f}, loss_f1: {2:.6f}, loss_f2: {3:.6f}, loss: {4:.5f}, _d1: {5:.6f}, _d2: {6:.6f}'.format(time.time() - start_time, loss_m, lossf1, lossf2, loss, loss_d, loss_d2)
                    #log_d = 'elapsed_time: {0:.3f}, loss_m: {1:.5f}, loss_f1: {2:.6f}, loss: {3:.5f}, _d1: {4:.6f}'.format(time.time() - start_time, loss_m, lossf1, loss, loss_d)
                    logger.write(log_d, self.colab)
                elif discriminator != None:
                    log_d = 'elapsed_time: {0:.3f}, loss_m: {1:.3f}, loss_f: {2:.3f}, loss: {3:.3f}, _d_real: {4:.5f}, _d_fake: {5:.5f}, _d: {6:.5f}'.format(time.time() - start_time, loss_m, lossf, loss, loss_d_real, loss_d_fake, loss_d)
                    #log_d = 'elapsed_time: {0}, loss: {1}'.format(time.time() - start_time, loss)
                    logger.write(log_d, self.colab)
                else:
                    log = 'elapsed_time: {0}, loss: {1}'.format(time.time() - start_time, loss)
                    logger.write(log, self.colab)
                """
                if loss.data[0] < 0.15 and lr > 0.001:
                    lr = 0.001
                    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                elif loss.data[0] < 0.05 and lr > 0.0005:
                    lr = 0.0005
                    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                elif loss.data[0] < 0.01 and lr > 0.0001:
                    lr = 0.0001
                    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                elif loss.data[0] < 0.005 and lr > 0.00001:
                    lr = 0.00001
                    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                """
                try:
                    torch.save(model, 'D:/github/DeepPose/result/pytorch/lastest.ptn.tar')
                    torch.save(model.state_dict(), 'D:/github/DeepPose/result/pytorch/lastest.model')
                    if self.useOneDrive == True:
                        if discriminator2 != None:
                            logger.write_oneDrive(log_d)
                        else:
                            torch.save(model.state_dict(), 'C:/Users/aoyag/OneDrive/pytorch/lastest.model')
                            logger.write_oneDrive(log)
                    if discriminator != None:
                        torch.save(discriminator.state_dict(), 'D:/github/DeepPose/result/pytorch/lastest_d.model')
                        #if self.useOneDrive == True:
                        #    torch.save(discriminator.state_dict(), 'C:/Users/aoyag/OneDrive/pytorch/lastest_d.model')
                        #    logger.write_oneDrive(log_d)
                except:
                    print("Unexpected error:")

    def _test(self, model, test_iter, logger, start_time):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_iter:
                image, pose, visibility = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True), Variable(batch[2], volatile=True)
                if self.gpu:
                    image, pose, visibility = image.cuda(), pose.cuda(), visibility.cuda()
                
                if self.NN == "MobileNet_":
                    offset, heatmap = model(image)
                    test_loss += mean_squared_error2(offset, heatmap, pose, visibility, self.use_visibility)
                elif self.NN == "MobileNet_3":
                    offset, heatmap = model(image)
                    test_loss += mean_squared_error2(offset, heatmap, pose, visibility, self.use_visibility)
                elif self.NN == "MobileNet_4":
                    offset, heatmap = model(image)
                    test_loss += mean_squared_error2(offset, heatmap, pose, visibility, self.use_visibility)
                elif self.NN == "MobileNet__":
                    offset, heatmap, output = model(image)
                    test_loss += mean_squared_error2_(offset, heatmap, output.view(-1, self.Nj, 3), pose, visibility, self.use_visibility)
                elif self.NN == "MobileNet___":
                    offset, heatmap = model(image)
                    test_loss += mean_squared_error2__(offset, heatmap, pose, visibility, self.use_visibility)
                elif self.NN == "MobileNet":
                    output = model(image)
                    test_loss += mean_squared_error3(output, pose, visibility, self.use_visibility)
                elif self.NN == "MnasNet":
                    output = model(image)
                    test_loss += mean_squared_error_FC3(output.view(-1, self.Nj, 3), pose, visibility, self.use_visibility)
                elif self.NN == "MnasNet_":
                    offset, heatmap = model(image)
                    test_loss += mean_squared_error2(offset, heatmap, pose, visibility, self.use_visibility)
                elif self.NN == "MnasNet56_":
                    offset, heatmap = model(image)
                    test_loss += mean_squared_error2(offset, heatmap, pose, visibility, self.use_visibility, col=56)
                elif self.NN == "MnasNet_+Discriminator" or self.NN == "MobileNet_3+Discriminator":
                    offset, heatmap = model(image)
                    loss, _ = mean_squared_error2GAN(offset, heatmap, pose, visibility, self.use_visibility)
                    test_loss += loss.data
                elif self.NN == "MobileNet_3+Discriminator2":
                    offset, heatmap = model(image)
                    loss = mean_squared_error2(offset, heatmap, pose, visibility, self.use_visibility)
                    test_loss += loss.data
                elif self.NN == "MobileNet224HM":
                    heatmap = model(image)
                    loss = mean_squared_error224HM(heatmap, pose, visibility, self.use_visibility, col=224)
                    test_loss += loss.data
                else :
                    output = model(image)
                    test_loss += mean_squared_error(output.view(-1, self.Nj, 2), pose, visibility, self.use_visibility)

        test_loss /= len(test_iter)
        log = 'elapsed_time: {0}, validation/loss: {1}'.format(time.time() - start_time, test_loss)
        logger.write(log, self.colab)
        if self.useOneDrive == True:
            logger.write_oneDrive(log)

    def _checkpoint(self, epoch, model, optimizer, logger, discriminator=None, discriminator2=None):
        filename = os.path.join(self.out, 'pytorch', 'epoch-{0}'.format(epoch + 1))
        torch.save({'epoch': epoch + 1, 'logger': logger.state_dict()}, filename + '.iter')
        torch.save(model.state_dict(), filename + '.model')
        torch.save(optimizer.state_dict(), filename + '.state')
                            
        if discriminator != None:
            torch.save(discriminator.state_dict(), filename + '_d.model')
                            
        if discriminator2 != None:
            torch.save(discriminator2.state_dict(), filename + '_d2.model')

        if self.colab == True:
            subprocess.run(["cp", "./result/pytorch/epoch-{0}.model".format(epoch + 1), "../drive/result/pytorch/epoch-{0}.model".format(epoch + 1)])


    def start(self):
        """ Train pose net. """
        discriminator = None
        optimizer_d = None
        discriminator2 = None
        optimizer_d2 = None

        # set random seed.
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            if self.gpu:
                torch.cuda.manual_seed(self.seed)
        # initialize model to train.
        if self.NN == "VGG19":
            model = models.vgg19(pretrained=True)
            # 学習済みデータは最後の層が1000なので、読込後入れ替える
            m3 = nn.Linear(4096, self.Nj*2)
            m3.weight.data.normal_(0, 0.01)
            m3.bias.data.zero_()
            removed = list(model.classifier.children())[:-1]
            model.classifier= torch.nn.Sequential(*removed)
            model.classifier = torch.nn.Sequential(model.classifier, m3)
        elif self.NN == "Inception3":
            model = Inceptionv3( aux_logits = False)
            # 学習済みデータは最後の層が1000なので、読込後入れ替える
            m3 = nn.Linear(2048, self.Nj*2)
            m3.weight.data.normal_(0, 0.01)
            m3.bias.data.zero_()
            # model.fc= m3
        elif self.NN == "ResNet":
            model = Resnet( )
        elif self.NN == "MobileNet":
            model = MobileNet( )
        elif self.NN == "MobileNet_":
            model = MobileNet_( )
        elif self.NN == "MobileNet__":
            model = MobileNet__( )
        elif self.NN == "MobileNet___":
            model = MobileNet___( )
        elif self.NN == "MobileNet_2":
            model = MobileNet_2( )
        elif self.NN == "MobileNet_3":
            model = MobileNet_3( )
        elif self.NN == "MobileNet_4":
            model = MobileNet_4( )
        elif self.NN == "MobileNetV2":
            model = MobileNetV2( )
        elif self.NN == "MnasNet":
            model = MnasNet( )
        elif self.NN == "MnasNet_":
            model = MnasNet_( )
        elif self.NN == "MnasNet56_":
            model = MnasNet56_( )
        elif self.NN == "MnasNet_+Discriminator":
            model = MnasNet_( )
            discriminator = Discriminator( )
        elif self.NN == "MobileNet_3+Discriminator":
            model = MobileNet_3( )
            discriminator = Discriminator( )
        elif self.NN == "MobileNet_3+Discriminator2":
            model = MobileNet_3( )
            discriminator = Discriminator2( )
            discriminator2 = Discriminator( )
        elif self.NN == "MobileNet224HM":
            model = MobileNet224HM( )
        else :
             model = AlexNet(self.Nj)
           
        if self.resume_model:
            model.load_state_dict(torch.load(self.resume_model))
            #model.fc2 = None
            #torch.save(model.state_dict(), 'del.model')
        if self.resume_discriminator:
            discriminator.load_state_dict(torch.load(self.resume_discriminator))
        if self.resume_discriminator2:
            discriminator2.load_state_dict(torch.load(self.resume_discriminator2))
        
        '''
        def conv_last(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            )
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        model.model2 = conv_dw(1024, 1024, 1)
        #model.output = None
        
        for p in model.model.parameters():
            p.requires_grad = False
        for p in model.heatmap.parameters():
            p.requires_grad = False
        for p in model.offset.parameters():
            p.requires_grad = False
        '''
        '''
        removed = list(model.model.children())[:-9]
        
        model.model1_1 = list(model.model.children())[5]
        model.model1_2 = list(model.model.children())[6]
        model.model1_3 = list(model.model.children())[7]
        model.model1_4 = list(model.model.children())[8]
        model.model1_5 = list(model.model.children())[9]
        model.model1_6 = list(model.model.children())[10]
        model.model1_7 = list(model.model.children())[11]
        model.model1_8 = list(model.model.children())[12]
        model.model1_9 = list(model.model.children())[13]
        model.model= torch.nn.Sequential(*removed)
        '''
        # prepare gpu.
        if self.gpu:
            model.cuda()
            if discriminator != None:
                discriminator.cuda()
            if discriminator2 != None:
                discriminator2.cuda()

        # load the datasets.
        input_transforms = [transforms.ToTensor()]
        if self.data_augmentation:
            input_transforms.append(RandomNoise())
        train = PoseDataset(
            self.train,
            input_transform=transforms.Compose(input_transforms),
            output_transform=Scale(),
            transform=Crop(data_augmentation=self.data_augmentation))
        val = PoseDataset(
            self.val,
            input_transform=transforms.Compose([
                transforms.ToTensor()]),
            output_transform=Scale(),
            transform=Crop(data_augmentation=False))
        # training/validation iterators.
        train_iter = torch.utils.data.DataLoader(train, batch_size=self.batchsize, shuffle=True)
        val_iter = torch.utils.data.DataLoader(val, batch_size=self.batchsize, shuffle=False)
        # set up an optimizer.
        if discriminator != None:
            optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        else:
            optimizer = self._get_optimizer(model)

        if self.resume_opt:
            optimizer.load_state_dict(torch.load(self.resume_opt))
        
        if discriminator != None:
            optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        if discriminator2 != None:
            optimizer_d2 = optim.Adam(discriminator2.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # set intervals.
        val_interval = 1
        #resume_interval = self.epoch/10
        resume_interval = 1
        log_interval = 10
        # set logger and start epoch.
        #logger = TrainLogger(os.path.join(self.out, 'pytorch'))
        logger = TrainLogger(self.out)
        start_epoch = 0
        if self.resume:
            resume = torch.load(self.resume)
            start_epoch = resume['epoch']
            logger.load_state_dict(resume['logger'])
        # start training.
        start_time = time.time()
        for epoch in trange(start_epoch, self.epoch, initial=start_epoch, total=self.epoch, desc='     total'):
            self._train(model, optimizer, train_iter, log_interval, logger, start_time, discriminator, optimizer_d, discriminator2, optimizer_d2)
            if (epoch + 1) % val_interval == 0:
                self._test(model, val_iter, logger, start_time)
            if (epoch + 1) % resume_interval == 0:
                self._checkpoint(epoch, model, optimizer, logger, discriminator, discriminator2)
