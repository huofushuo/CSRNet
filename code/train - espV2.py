import torch
import os
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lib.dataset import Data
from lib.data_prefetcher import DataPrefetcher

from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
import torch.optim as optim
import numpy as np
from espnetV2 import ESPNET

import warnings
from util import clip_gradient, muti_loss_fusion, adjust_learning_rate_poly
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
torch.backends.cudnn.benchmark = True




def main():

    # ------- 1. set the directory of training dataset --------
    # dataset
    img_root = './data/'
    # img_root = './datasets's
    save_path = './model'
    if not os.path.exists(save_path): os.mkdir(save_path)
    lr = 0.0001
    batch_size = 8
    epoch = 100
    test_loss = 0

    data = Data(img_root, mode='train')
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    net = ESPNET().cuda()

    optimizer = optim.Adam(net.parameters(), lr, betas=(0.9, 0.99), eps=1e-08,weight_decay= 0.0001)

    iter_num = len(loader)

    net.train()
    for epochi in range(1, epoch + 1):

        cur_lr = adjust_learning_rate_poly(optimizer, epochi, epoch, lr, 0.9)
        print("LearningRate {:.8f}".format(cur_lr))
        prefetcher = DataPrefetcher(loader)
        rgb, t, label = prefetcher.next()
        r_sal_loss = 0
        net.zero_grad()
        i = 0
        while rgb is not None:
            i += 1
            s1, s2, s3, s4, s5 = net(rgb,t)
            sal_loss = muti_loss_fusion(s1, s2, s3, s4, s5, label)
            sal_loss__ = sal_loss.data
            r_sal_loss = r_sal_loss + sal_loss__
            clip_gradient(optimizer, 0.5)
            sal_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            del s1, s2, s3, s4, s5

            if i % 100 == 0:
                print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f' % (
                    epochi, epoch, i, iter_num, r_sal_loss / 100))
                r_sal_loss = 0

            test_loss = test_loss + sal_loss__
            rgb, t, label = prefetcher.next()




if __name__ == "__main__":

    main()








