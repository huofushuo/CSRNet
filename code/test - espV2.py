#coding=utf-8
import os
from torch.utils.data import DataLoader
from lib.dataset import Data
import torch.nn.functional as F
import torch
import cv2
import time
import numpy as np
import warnings
from espnetV2 import ESPNET
import torchvision
from thop import profile

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
if __name__ == '__main__':

    model_path = ''

    test_datasets = ['821', '1000', '5000', 'DUT-RGBD','LFSD', 'NJUD','NLPR','SIP','SSD','STEREO', 'RGBD135']

    for dataset in test_datasets:
        out_path = './results/'
        dataset_path = './testall/' + dataset

        print('testing bach %s' % dataset)

        save_pred_path = out_path + dataset + '/'

        if not os.path.exists(save_pred_path):
            os.makedirs(save_pred_path)


        data = Data(root=dataset_path, mode='test', dataset = dataset)
        loader = DataLoader(data, batch_size=1,shuffle=False)

        net = ESPNET().cuda()

        print('loading model from %s...' % model_path)
        net.load_state_dict(torch.load(model_path))
        # if not os.path.exists(out_path): os.mkdir(out_path)
        time_s = time.time()
        img_num = len(loader)
        net.eval()

        time_sum = 0
        with torch.no_grad():
            for rgb, t, _ , (H, W), name in loader:
                print(name[0])
                # start = time.time()
                s1, s2, s3, s4, s5 = net(rgb.cuda().float(), t.cuda().float())
                torch.cuda.synchronize()
                # end = time.time()
                # time_sum = time_sum + (end-start)
                score = F.interpolate(s1, size=(H, W), mode='bilinear')
                pred = np.squeeze(score.cpu().data.numpy())
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)    #归一化
                multi_fuse = 255 * pred
                cv2.imwrite(os.path.join(save_pred_path, name[0][:-4] + '.jpg'), multi_fuse)
        # print('Running time {:.5f}'.format(time_sum))



