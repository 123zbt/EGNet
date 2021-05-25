# coding: utf-8

import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam, SGD
from torch.autograd import Variable
from torch.backends import cudnn
from model import build_model, weights_init
import scipy.misc as sm
import numpy as np
import os
import torchvision.utils as vutils
import cv2
import torch.nn.functional as F
import math
import time
import sys
import PIL.Image
import scipy.io
import os
import logging
EPSILON = 1e-8
p = OrderedDict() # 有序字典

from dataset import get_loader
base_model_cfg = 'resnet'
p['lr_bone'] = 5e-5  # Learning rate resnet:5e-5, vgg:2e-5
p['lr_branch'] = 0.025  # Learning rate
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.90  # Momentum
lr_decay_epoch = [15, 24] # [6, 9], now x3 #15
nAveGrad = 10  # Update the weights once in 'nAveGrad' forward passes
showEvery = 50
tmp_path = 'tmp_see'


class Solver(object):
    def __init__(self, train_loader, test_loader, config, save_fold=None):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.save_fold = save_fold
        self.mean = torch.Tensor([123.68, 116.779, 103.939]).view(3, 1, 1) / 255. # view()用来改变Tensor的size
        # inference: choose the side map (see paper)
        if config.visdom:
            self.visual = Viz_visdom("trueUnify", 1) # 好像是可视化啥的
        self.build_model()
        if self.config.pre_trained: self.net.load_state_dict(torch.load(self.config.pre_trained)) # 加载骨干网络参数
        if config.mode == 'train':
            self.log_output = open("%s/logs/log.txt" % config.save_fold, 'w')
        else:
            print('Loading pre-trained model from %s...' % self.config.model)
            self.net_bone.load_state_dict(torch.load(self.config.model))
            self.net_bone.eval()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel() # 返回参数的数目，这里就是计算总的参数和
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def get_params(self, base_lr):
        ml = []
        for name, module in self.net_bone.named_children():
            print(name)
            if name == 'loss_weight':
                ml.append({'params': module.parameters(), 'lr': p['lr_branch']})          
            else:
                ml.append({'params': module.parameters()})
        return ml

    # build the network
    def build_model(self): # 构建模型
        self.net_bone = build_model(base_model_cfg) # 骨干网络，这里使用的是resnet
        if self.config.cuda:
            self.net_bone = self.net_bone.cuda()
            
        self.net_bone.eval()  # use_global_stats = True ？，在训练的时候为啥用这个
        self.net_bone.apply(weights_init)
        if self.config.mode == 'train':
            if self.config.load_bone == '':
                if base_model_cfg == 'vgg':
                    self.net_bone.base.load_pretrained_model(torch.load(self.config.vgg))
                elif base_model_cfg == 'resnet':
                    self.net_bone.base.load_state_dict(torch.load(self.config.resnet))
            if self.config.load_bone != '': self.net_bone.load_state_dict(torch.load(self.config.load_bone))

        self.lr_bone = p['lr_bone']
        self.lr_branch = p['lr_branch']
        self.optimizer_bone = Adam(filter(lambda p: p.requires_grad, self.net_bone.parameters()), lr=self.lr_bone, weight_decay=p['wd'])

        self.print_network(self.net_bone, 'trueUnify bone part')

    # update the learning rate
    def update_lr(self, rate):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * rate


    def test(self, test_mode=0):
        EPSILON = 1e-8
        img_num = len(self.test_loader)
        time_t = 0.0
        name_t = 'EGNet_ResNet50/'

        if not os.path.exists(os.path.join(self.save_fold, name_t)):             
            os.mkdir(os.path.join(self.save_fold, name_t))
        for i, data_batch in enumerate(self.test_loader):
            self.config.test_fold = self.save_fold
            print(self.config.test_fold)
            images_, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            
            with torch.no_grad():
                
                images = Variable(images_)
                if self.config.cuda:
                    images = images.cuda()
                print(images.size())
                time_start = time.time()
                up_edge, up_sal, up_sal_f = self.net_bone(images)
                torch.cuda.synchronize()
                time_end = time.time()
                print(time_end - time_start)
                time_t = time_t + time_end - time_start                              
                pred = np.squeeze(torch.sigmoid(up_sal_f[-1]).cpu().data.numpy())             
                multi_fuse = 255 * pred
                

                
                cv2.imwrite(os.path.join(self.config.test_fold,name_t, name[:-4] + '.png'), multi_fuse)
          
        print("--- %s seconds ---" % (time_t))
        print('Test Done!')

   
    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size # 迭代次数
        aveGrad = 0
        F_v = 0
        if not os.path.exists(tmp_path): 
            os.mkdir(tmp_path)
        for epoch in range(self.config.epoch):                          
            r_edge_loss, r_sal_loss, r_sum_loss= 0,0,0
            self.net_bone.zero_grad()
            for i, data_batch in enumerate(self.train_loader): # enumerate()将可遍历的数组对象转换为索引序列，同时列出数据下标和数据
                sal_image, sal_label, sal_edge = data_batch['sal_image'], data_batch['sal_label'], data_batch['sal_edge']
                # 那这么看就是显著图的label是对边缘有单独label的
                # 下面的说法是错的，2: 应该表达的是第三维及之后，那应该还是h和w
                if sal_image.size()[2:] != sal_label.size()[2:]: # 前两维，那就是图片大小和label不同的时候，就跳过不处理(错误的)
                    print("Skip this batch")
                    continue
                sal_image, sal_label, sal_edge = Variable(sal_image), Variable(sal_label), Variable(sal_edge) # 给整成变量
                if self.config.cuda: 
                    sal_image, sal_label, sal_edge = sal_image.cuda(), sal_label.cuda(), sal_edge.cuda() # .cuda()是啥啊？ Returns a copy of this object in CUDA memory，就是给搞到CUDA中去

                up_edge, up_sal, up_sal_f = self.net_bone(sal_image) # 这个net_bone是骨干网络里来的，暂时还没有整明白
                # edge part，这也是论文的第一部分，主要搞的就是边缘
                edge_loss = [] # 应该是不同S，也就是不同旁路的loss
                for ix in up_edge: # up_edge里面存的是啥，下面算的是交叉熵
                    edge_loss.append(bce2d_new(ix, sal_edge, reduction='sum'))
                edge_loss = sum(edge_loss) / (nAveGrad * self.config.batch_size)
                r_edge_loss += edge_loss.data
                # sal part
                sal_loss1= []
                sal_loss2 = []
                for ix in up_sal:
                    sal_loss1.append(F.binary_cross_entropy_with_logits(ix, sal_label, reduction='sum'))

                for ix in up_sal_f: # 这个不知道是啥部分的loss，这个应该是从边整合过来的loss
                    sal_loss2.append(F.binary_cross_entropy_with_logits(ix, sal_label, reduction='sum'))
                sal_loss = (sum(sal_loss1) + sum(sal_loss2)) / (nAveGrad * self.config.batch_size)
              
                r_sal_loss += sal_loss.data
                loss = sal_loss + edge_loss
                r_sum_loss += loss.data
                loss.backward()
                aveGrad += 1

                if aveGrad % nAveGrad == 0: # nAveGrad表示的就应该是batch
       
                    self.optimizer_bone.step() # optimizer_bone就是Adama优化器，.step()应该就是一次优化
                    self.optimizer_bone.zero_grad()  # 把梯度置0
                    aveGrad = 0


                if i % showEvery == 0: # i是遍历到的图片的下标，那就是每50次打印下，但是为啥要把这三个loss置0，是因为这是一个batch？对的

                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Edge : %10.4f  ||  Sal : %10.4f  ||  Sum : %10.4f' % (
                        epoch, self.config.epoch, i, iter_num,  r_edge_loss*(nAveGrad * self.config.batch_size)/showEvery,
                                                                r_sal_loss*(nAveGrad * self.config.batch_size)/showEvery,
                                                                r_sum_loss*(nAveGrad * self.config.batch_size)/showEvery))

                    print('Learning rate: ' + str(self.lr_bone))
                    r_edge_loss, r_sal_loss, r_sum_loss= 0,0,0

                if i % 200 == 0:

                    vutils.save_image(torch.sigmoid(up_sal_f[-1].data), tmp_path+'/iter%d-sal-0.jpg' % i, normalize=True, padding = 0)

                    vutils.save_image(sal_image.data, tmp_path+'/iter%d-sal-data.jpg' % i, padding = 0)
                    vutils.save_image(sal_label.data, tmp_path+'/iter%d-sal-target.jpg' % i, padding = 0)
            
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net_bone.state_dict(), '%s/models/epoch_%d_bone.pth' % (self.config.save_fold, epoch + 1))
                
            if epoch in lr_decay_epoch:
                self.lr_bone = self.lr_bone * 0.1  
                self.optimizer_bone = Adam(filter(lambda p: p.requires_grad, self.net_bone.parameters()), lr=self.lr_bone, weight_decay=p['wd'])
                # requires_grad是pytorch中Tensor的一个通用属性，用于说明当前量是否需要在计算中保留对应的梯度信息


        torch.save(self.net_bone.state_dict(), '%s/models/final_bone.pth' % self.config.save_fold)
        
def bce2d_new(input, target, reduction=None): #
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float() # 比较两个张量，这里和1、0比，那就是比值么。含义就是正确估计位置;注意torch.eq()输出的是一个Boolean的Tensor，最后再.float()之后就转化为0和1构成的了
    neg = torch.eq(target, 0).float()
    # ing = ((torch.gt(target, 0) & torch.lt(target, 1))).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    # num_pos相比于num_neg来说会小很多
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total # 接近1
    beta = 1.1 * num_pos  / num_total # 很小，接近0. 最后乘neg加到到weight上，就说明更看重背景一些，背景值越多，权重就越大？？为啥呀
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg # weights最后就直接乘上损失函数

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction) # 计算二分类问题的交叉熵

