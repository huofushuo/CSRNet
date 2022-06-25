#coding=utf-8

import os
import cv2
import random
import numpy as np
try:
    from . import transform
except:
    import transform
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
# rgbt
mean_rgb = np.array([[[0.551, 0.619, 0.532]]])*255
mean_t =np.array([[[0.341,  0.360, 0.753]]])*255
std_rgb = np.array([[[0.241, 0.236, 0.244]]])*255
std_t = np.array([[[0.208, 0.269, 0.241]]])*255

#rgbd
# mean_rgb = np.array([[[0.43127787, 0.4015223, 0.44389117]]])*255
# std_rgb = np.array([[[0.25044188, 0.25923958, 0.25612995]]])*255
#
# mean_t = np.array([[[0.45592305, 0.45592305, 0.45592305]]])*255
# std_t = np.array([[[0.2845027, 0.2845027, 0.2845027]]])*255

#rgbtd
# mean_rgb = np.array([[[0.5100, 0.5730, 0.5120]]])*255
# mean_t =np.array([[[0.3649, 0.3662, 0.6973]]])*255
# std_rgb = np.array([[[0.2567, 0.2522, 0.2496]]])*255
# std_t = np.array([[[0.2246, 0.2768, 0.2767]]])*255

#DUTD
# mean_rgb = np.array([[[ 0.42454678, 0.3971446,0.44439128]]])*255
# std_rgb = np.array([[[ 0.25133714, 0.25833002,0.25390804]]])*255
#
# mean_t = np.array([[[0.5156196, 0.5156196, 0.5156196]]])*255
# std_t = np.array([[[0.27192974, 0.27192974, 0.27192974]]])*255

# [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# mean_rgb = np.array([[[0.551, 0.619, 0.532]]])*255
# mean_t =np.array([[[0.341,  0.360, 0.753]]])*255
# std_rgb = np.array([[[0.241, 0.236, 0.244]]])*255
# std_t = np.array([[[0.208, 0.269, 0.241]]])*255

# mean_rgb = np.array([[[0.485*255, 0.456*255, 0.406*255]]])
# mean_t = np.array([[[0.485*255, 0.456*255, 0.406*255]]])
# std_rgb = np.array([[[0.229 * 255, 0.224 * 255, 0.225 * 255]]])
# std_t = np.array([[[0.229 * 255, 0.224 * 255, 0.225 * 255]]])



def sp_noise(image,prob):


    output = np.zeros(image.shape, np.uint8)

    thres = 1 - prob

    for i in range(image.shape[0]):

        for j in range(image.shape[1]):

            rdn = random.random()

            if rdn < prob:

                output[i][j] = 0

            elif rdn > thres:

                output[i][j] = 255

            else:

                output[i][j] = image[i][j]

    return output

def gasuss_noise(image, mean=0, var=0.001):

    image = np.array(image/255, dtype=float)

    noise = np.random.normal(mean, var ** 0.5, image.shape)

    out = image + noise

    if out.min() < 0:

        low_clip = -1.

    else:

        low_clip = 0.

    out = np.clip(out, low_clip, 1.0)

    out = np.uint8(out*255)

    #cv.imshow("gasuss", out)

    return out



def random_noise(image,noise_num):

    # 参数image：，noise_num：
    # img = cv2.imread(image)
    img = np.array(image/255, dtype=float)
    img_noise = img
    # cv2.imshow("src", img)
    rows, cols, chn = img_noise.shape
    # 加噪声
    for i in range(noise_num):
        x = np.random.randint(0, rows)#随机生成指定范围的整数
        y = np.random.randint(0, cols)
        img_noise[x, y, :] = 255

    return img_noise
    # out = np.uint8(out*255)

def getRandomSample(rgb,t):
    n = np.random.randint(15)

    noise = np.random.randint(4)
    if n == 1:
        if noise == 0:
            # rgb = torch.from_numpy(np.zeros_like(rgb))
            rgb = np.zeros_like(rgb)
        elif noise == 1:
            # rgb = torch.from_numpy(random_noise(rgb, np.random.randint(90000, 130000)))
            rgb = random_noise(rgb, np.random.randint(60000, 100000))
        elif noise == 2:
            # rgb = torch.from_numpy(sp_noise(rgb, prob=np.random.uniform(0.1, 0.25)))
            rgb = sp_noise(rgb, prob=np.random.uniform(0.05, 0.15))
        elif noise == 3:
            # rgb = torch.from_numpy(gasuss_noise(rgb, mean=0, var=np.random.uniform(0.03, 0.06)))
            rgb = gasuss_noise(rgb, mean=0, var=np.random.uniform(0.02, 0.05))
    elif n == 2:
        if noise == 0:
            # t = torch.from_numpy(np.zeros_like(t))
            t = np.zeros_like(t)
        elif noise == 1:
            # t = torch.from_numpy(random_noise(t, np.random.randint(90000, 130000)))
            t = random_noise(t, np.random.randint(60000, 100000))
        elif noise == 2:
            # t = torch.from_numpy(sp_noise(t, prob=np.random.uniform(0.1, 0.25)))
            t = sp_noise(t, prob=np.random.uniform(0.05, 0.15))
        elif noise == 3:
            # t = torch.from_numpy(gasuss_noise(t, mean=0, var=np.random.uniform(0.03, 0.06)))
            gasuss_noise(t, mean=0, var=np.random.uniform(0.02, 0.05))
    return rgb, t

class Data(Dataset):
    def __init__(self, root,mode='train', dataset=''):
        self.samples = []
        self.mode = mode
        self.dataset = dataset
        lines = os.listdir(os.path.join(root, 'GT'))
        for line in lines:
            rgbpath = os.path.join(root, 'RGB', line[:-4]+'.jpg')
            tpath = os.path.join(root, 'T', line[:-4]+'.jpg')
            maskpath = os.path.join(root, 'GT', line)
            self.samples.append([rgbpath,tpath,maskpath])

        if mode == 'train':

            self.transform = transform.Compose( transform.Normalize(mean1=mean_rgb,mean2=mean_t,std1=std_rgb,std2=std_t),
                                                # transform.Resize(300, 300),
                                                transform.Resize(320, 320),#352_resnet50
                                                transform.RandomHorizontalFlip(),
                                                transform.ToTensor(),


                                               )

        elif mode == 'test':


            self.transform = transform.Compose(
                transform.Normalize(mean1=mean_rgb, mean2=mean_t, std1=std_rgb, std2=std_t),
                transform.Resize(320, 320),
                transform.ToTensor(),
                )

    def __getitem__(self, idx):
        # rgbpath,tpath,maskpath = self.samples[idx]
        # rgb = cv2.imread(rgbpath).astype(np.float32)
        # t = cv2.imread(tpath).astype(np.float32)
        # mask = cv2.imread(maskpath).astype(np.float32)
        #
        # H, W, C = mask.shape
        # rgb,t,mask = self.transform(rgb,t,mask)
        # if  self.mode == 'train':
        #     rgb,t =getRandomSample(rgb,t)
        # return rgb,t,mask, (H, W), maskpath.split('\\')[-1]

        rgbpath,tpath,maskpath = self.samples[idx]
        rgb = cv2.imread(rgbpath)
        t = cv2.imread(tpath)
        mask = cv2.imread(maskpath)

        H, W, C = mask.shape
        rgb = rgb.astype('float32')
        t = t.astype('float32')
        mask = mask.astype('float32')
        if self.mode == 'train':
          rgb, t =getRandomSample(rgb, t)
        rgb, t, mask = self.transform(rgb, t, mask)
        maskpath_short = maskpath.split('\\')[-1]

        return rgb, t, mask, (H, W), maskpath_short

    def __len__(self):
        return len(self.samples)