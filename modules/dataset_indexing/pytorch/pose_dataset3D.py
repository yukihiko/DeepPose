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

    def __init__(self, path, input_transform=None, output_transform=None, transform=None, lsp=False, affine=True, clop=True):
        self.path = path
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.transform = transform
        self.lsp = lsp
        self.affine = affine
        self.clop = clop
        # load dataset.
        self.images, self.dists, self.poses2D, self.poses3D, self.visibilities, self.image_types = self._load_dataset()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """ Returns the i-th example. """
        #image_o = self._read_image(self.images[index], self.image_types[index])
        dist = self.dists[index]
        pose3D = self.poses3D[index]
        pose2D = self.poses2D[index]
        visibility = self.visibilities[index]
        image_type = self.image_types[index]
        path = self.images[index]

        data_numpy = cv2.imread(
            self.images[index], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        
        image_size = [224,224]
        scale = np.array([1.0, 1.0])
        c = np.array([112.0,112.0])
        r = 0.0
        if self.affine == True:
            rf = 35.0
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                    if random.random() <= 0.7 else 0

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
        
        image = image.transpose(2,0,1)/255.0
        image = torch.Tensor(image)
        if self.clop == True and image_type == "N" and random.randrange(3) == 2:
            image = self.crop(image, pose2D, visibility[:,:2])

        return image, dist, pose2D, pose3D, visibility, image_type, path

    def crop(self, image, pose, visibility):
        visible_pose = torch.masked_select(pose * 224, visibility.byte()).view(-1, 2)
        p_min = visible_pose.min(0)[0].squeeze()
        p_max = visible_pose.max(0)[0].squeeze()

        p_min[0] = max(0, p_min[0] / 2)
        p_min[1] = max(0, p_min[1] / 2)
        p_max[0] = min(224, p_max[0] + (224 - p_max[0]) / 2)
        p_max[1] = min(224, p_max[1] + (224 - p_max[1]) / 2)

        p_min = p_min.int()
        p_max = p_max.int()

        baseImg = torch.zeros(image.size())
        baseImg[:, p_min[1]:p_max[1], p_min[0]:p_max[0]] = image[:, p_min[1]:p_max[1], p_min[0]:p_max[0]]
        return baseImg

    def _load_dataset(self):
        images = []
        dists = []
        poses3D = []
        poses2D = []
        visibilities = []
        image_types = []
        lsp_cnt = 0
        mpii_cnt = 0
        coco_cnt = 0
        myds_cnt = 0
        rn = 5
        if self.lsp == True:
            for line in open('data/train'):

                line_split = line[:-1].split(',')
                # 通常の画像
                pose2D = torch.zeros(24, 2).float()
                pose3D = torch.zeros(24, 3).float()
                visibility = torch.zeros(24, 3).float()
                x = torch.Tensor(list(map(float, line_split[1:])))
                x = x.view(-1, 3)
                p = x[:, :2]/float(256)
                vs = x[:, 2]
                
                '''
                for i in range(14):
                    if p[i, 0] == 0 and p[i, 1] == 0 and vs[i] == 0:
                        # only six files
                        continue
                '''
                if random.randrange(rn) != 0:
                    continue
                lsp_cnt = lsp_cnt + 1

                v = torch.stack([vs, vs, vs], dim=1)
                pose2D[0, :] = p[8, :]
                visibility[0, :] = v[8, :]
                pose2D[1, :] = p[7, :]
                visibility[1, :] = v[7, :]
                pose2D[2, :] = p[6, :]
                visibility[2, :] = v[6, :]
                pose2D[5, :] = p[9, :]
                visibility[5, :] = v[9, :]
                pose2D[6, :] = p[10, :]
                visibility[6, :] = v[10, :]
                pose2D[7, :] = p[11, :]
                visibility[7, :] = v[11, :]
                pose2D[15, :] = p[2, :]
                visibility[15, :] = v[2, :]
                pose2D[16, :] = p[1, :]
                visibility[16, :] = v[1, :]
                pose2D[17, :] = p[0, :]
                visibility[17, :] = v[0, :]
                pose2D[19, :] = p[3, :]
                visibility[19, :] = v[3, :]
                pose2D[20, :] = p[4, :]
                visibility[20, :] = v[4, :]
                pose2D[21, :] = p[5, :]
                visibility[21, :] = v[5, :]

                images.append(line_split[0])
                dists.append(float(-999))
                poses3D.append(pose3D)
                poses2D.append(pose2D)
                visibilities.append(visibility)
                image_types.append("L")

            '''
            joint id (
                0 - r ankle, 
                1 - r knee, 
                2 - r hip, 
                3 - l hip, 
                4 - l knee, 
                5 - l ankle, 
                6 - pelvis, 
                7 - thorax, 
                8 - upper neck, 
                9 - head top, 
                10 - r wrist, 
                11 - r elbow, 
                12 - r shoulder, 
                13 - l shoulder, 
                14 - l elbow, 
                15 - l wrist
                )
            '''

            for line in open('D:/work/3D_dataset/tran_mpii'):

                line_split = line[:-1].split(',')
                # 通常の画像
                pose2D = torch.zeros(24, 2).float()
                pose3D = torch.zeros(24, 3).float()
                visibility = torch.zeros(24, 3).float()
                x = torch.Tensor(list(map(float, line_split[1:])))
                x = x.view(-1, 3)
                p = x[:, :2]/float(224)
                vs = x[:, 2]
                
                #if vs.sum() != 16:
                #    continue

                if random.randrange(rn) != 0:
                    continue
                mpii_cnt = mpii_cnt + 1

                v = torch.stack([vs, vs, vs], dim=1)
                #v = torch.Tensor([1,1,1]).float()
                pose2D[0, :] = p[12, :]
                visibility[0, :] = v[12, :]
                pose2D[1, :] = p[11, :]
                visibility[1, :] = v[11, :]
                pose2D[2, :] = p[10, :]
                visibility[2, :] = v[10, :]
                pose2D[5, :] = p[13, :]
                visibility[5, :] = v[13, :]
                pose2D[6, :] = p[14, :]
                visibility[6, :] = v[14, :]
                pose2D[7, :] = p[15, :]
                visibility[7, :] = v[15, :]
                pose2D[15, :] = p[2, :]
                visibility[15, :] = v[2, :]
                pose2D[16, :] = p[1, :]
                visibility[16, :] = v[1, :]
                pose2D[17, :] = p[0, :]
                visibility[17, :] = v[0, :]
                pose2D[19, :] = p[3, :]
                visibility[19, :] = v[3, :]
                pose2D[20, :] = p[4, :]
                visibility[20, :] = v[4, :]
                pose2D[21, :] = p[5, :]
                visibility[21, :] = v[5, :]

                images.append(line_split[0])
                dists.append(float(-999))
                poses3D.append(pose3D)
                poses2D.append(pose2D)
                visibilities.append(visibility)
                image_types.append("M")

            '''
            "keypoints": {
                0: "nose",
                1: "left_eye",
                2: "right_eye",
                3: "left_ear",
                4: "right_ear",
                5: "left_shoulder",
                6: "right_shoulder",
                7: "left_elbow",
                8: "right_elbow",
                9: "left_wrist",
                10: "right_wrist",
                11: "left_hip",
                12: "right_hip",
                13: "left_knee",
                14: "right_knee",
                15: "left_ankle",
                16: "right_ankle"
            },
            "skeleton": [
                [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
                [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
            '''
            
            tw = 0
            for line in open('D:/work/3D_dataset/tran_coco'):
                #tw = tw + 1
                #if tw != 2:
                #    continue
                #tw = 0

                line_split = line[:-1].split(',')
                # 通常の画像
                pose2D = torch.zeros(24, 2).float()
                pose3D = torch.zeros(24, 3).float()
                visibility = torch.zeros(24, 3).float()
                x = torch.Tensor(list(map(float, line_split[1:])))
                x = x.view(-1, 3)
                p = x[:, :2]/float(224)
                vs = x[:, 2]
                if vs.sum() <= 6:
                    continue
                #if vs.sum() != 17:
                #    continue
                
                #if vs[:5].sum() < 3 or vs[5:].sum() < 9:
                #    continue
                #tw = tw + 1
                                
                if random.randrange(rn) != 0:
                    continue
                coco_cnt = coco_cnt + 1

                v = torch.stack([vs, vs, vs], dim=1)
                pose2D[0, :] = p[6, :]
                visibility[0, :] = v[6, :]
                pose2D[1, :] = p[8, :]
                visibility[1, :] = v[8, :]
                pose2D[2, :] = p[10, :]
                visibility[2, :] = v[10, :]
                pose2D[5, :] = p[5, :]
                visibility[5, :] = v[5, :]
                pose2D[6, :] = p[7, :]
                visibility[6, :] = v[7, :]
                pose2D[7, :] = p[9, :]
                visibility[7, :] = v[9, :]
                pose2D[10, :] = p[3, :]
                visibility[10, :] = v[3, :]
                pose2D[11, :] = p[1, :]
                visibility[11, :] = v[1, :]
                pose2D[12, :] = p[4, :]
                visibility[12, :] = v[4, :]
                pose2D[13, :] = p[2, :]
                visibility[13, :] = v[2, :]
                pose2D[14, :] = p[0, :]
                visibility[14, :] = v[0, :]
                pose2D[15, :] = p[12, :]
                visibility[15, :] = v[12, :]
                pose2D[16, :] = p[14, :]
                visibility[16, :] = v[14, :]
                pose2D[17, :] = p[16, :]
                visibility[17, :] = v[16, :]
                pose2D[19, :] = p[11, :]
                visibility[19, :] = v[11, :]
                pose2D[20, :] = p[13, :]
                visibility[20, :] = v[13, :]
                pose2D[21, :] = p[15, :]
                visibility[21, :] = v[15, :]

                images.append(line_split[0])
                dists.append(float(-999))
                poses3D.append(pose3D)
                poses2D.append(pose2D)
                visibilities.append(visibility)
                image_types.append("C")

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
            vs = x[:, 5]
            if vs.sum() <= 6:
                continue

            myds_cnt = myds_cnt + 1

            # 通常の画像
            images.append(line_split[0])
            dists.append(float(line_split[1]))
            poses3D.append(pose3D)
            poses2D.append(pose2D)
            visibilities.append(visibility)
            image_types.append("N")

        cnt = lsp_cnt + mpii_cnt + coco_cnt
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
