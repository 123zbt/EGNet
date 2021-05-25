# coding: utf-8
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np

from resnet import resnet50
from vgg import vgg16

# merge是长度为5的数组，最后两个是表示卷积核大小和步长
config_vgg = {'convert': [[128,256,512,512,512],[64,128,256,512,512]], 'merge1': [[128, 256, 128, 3,1], [256, 512, 256, 3, 1], [512, 0, 512, 5, 2], [512, 0, 512, 5, 2],[512, 0, 512, 7, 3]], 'merge2': [[128], [256, 512, 512, 512]]}  # no convert layer, no conv6

config_resnet = {'convert': [[64,256,512,1024,2048],[128,256,512,512,512]], 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]], 'score': 256, 'edgeinfo':[[16, 16, 16, 16], 128, [16,8,4,2]],'edgeinfoc':[64,128], 'block': [[512, [16]], [256, [16]], [256, [16]], [128, [16]]], 'fuse': [[16, 16, 16, 16], True], 'fuse_ratio': [[16,1], [8,1], [4,1], [2,1]],  'merge1': [[128, 256, 128, 3,1], [256, 512, 256, 3, 1], [512, 0, 512, 5, 2], [512, 0, 512, 5, 2],[512, 0, 512, 7, 3]], 'merge2': [[128], [256, 512, 512, 512]]}


class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up0, up1, up2 = [], [], []
        for i in range(len(list_k[0])):
          
            up0.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.ReLU(inplace=True)))


        self.convert0 = nn.ModuleList(up0)


    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl


        

class MergeLayer1(nn.Module): # list_k: [[64, 512, 64], [128, 512, 128], [256, 0, 256] ... ]
    def __init__(self, list_k):
        super(MergeLayer1, self).__init__()
        self.list_k = list_k # config['merge1']
        # vgg [[128, 256, 128, 3,1], [256, 512, 256, 3, 1], [512, 0, 512, 5, 2], [512, 0, 512, 5, 2],[512, 0, 512, 7, 3]]
        # resnet [[128, 256, 128, 3,1], [256, 512, 256, 3, 1], [512, 0, 512, 5, 2], [512, 0, 512, 5, 2],[512, 0, 512, 7, 3]] 都是一样的
        trans, up, score = [], [], []
        for ik in list_k:
            if ik[1] > 0: # 输入通道判断是不是0
                # conv2d函数中参数分别为：输入通道，输出通道，卷积核大小，步长
                trans.append(nn.Sequential(nn.Conv2d(ik[1], ik[0], 1, 1, bias=False), nn.ReLU(inplace=True))) # 用1*1卷积换通道数目

            # 卷积的四个参数：in_channels, out_channels, kernel_size, stride, padding
            # 怎么ik[0]和ik[2]值都是一样的，而且下面第一个卷积计算的输入输出的大小也是相同的；第二个和第三个卷积就是再卷一遍，上采样获得更多信息；
            up.append(nn.Sequential(nn.Conv2d(ik[0], ik[2], ik[3], 1, ik[4]), nn.ReLU(inplace=True), nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.ReLU(inplace=True), nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.ReLU(inplace=True)))
            score.append(nn.Conv2d(ik[2], 1, 3, 1, 1)) # 卷到一个通道上，应该是准备和针织图对比输出loss了
        trans.append(nn.Sequential(nn.Conv2d(512, 128, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.trans, self.up, self.score = nn.ModuleList(trans), nn.ModuleList(up), nn.ModuleList(score)
        self.relu =nn.ReLU()
        # 那最后 up就是4个输入输出尺寸和通道相同的三层上采样，那细分小层的话就是5*6=30，和base(也就是vgg没加ex之前的层数相同)
        # trans就是3个1*1卷积，但是输入输出通道怎么有点不对，2*2+2=6
        # score就是4个卷积，将每层卷到一个通道上输出, 仅有5个

    def forward(self, list_x, x_size): # 这个list_x就是给这层的输入x，若是resnet，通过convert转换下得到
        up_edge, up_sal, edge_feature, sal_feature = [], [], [], []
        # up_sal将每层的输出都放到和x_size一样大小了
        # sal_feature就是存放的是下层合并到当前层融合后的特征
        # edge_feature里面就1项，放的是最后的特征
        # up_edge里面也是一项，将edge_feature的那个值放缩到x_size大小
        # 总体来说，第一部分取得就是边的特征么所以，最后两个就存一个值
        
        
        num_f = len(list_x) # 得到的应该是b。不对 输入x经过了base处理，这里得到的就是网络的层数！！！   得到的num_f值是5
        tmp = self.up[num_f - 1](list_x[num_f-1]) # 就是将对应层的上采样卷积拿出来卷上采样，num_f - 1是最后一层
        sal_feature.append(tmp)
        U_tmp = tmp
        # F.interpolate上采样到输入的大小
        up_sal.append(F.interpolate(self.score[num_f - 1](tmp), x_size, mode='bilinear', align_corners=True)) # 算这层的loss
        
        for j in range(2, num_f ): # 为啥从2开始，因为刚才已经算过了num_f - 1位置，并且没有num_f - 0位置
            i = num_f - j # 从后面的层往前推
            # print("list_x和u_tmp : ")
            # print(list_x[i].size())
            # print(U_tmp.size())
            # print("--------------")

            # 比的通道数，为啥呀。。好像就是不为啥，不然不相同咋加，最后两层都是512，不用换，但是前面三层每层都不一样，得将下层换到和上层相同才行
            if list_x[i].size()[1] < U_tmp.size()[1]: # U_tmp最开始就是最后一层，那这个循环的过程就是对应PSFEM中将深层结果上采样和浅层加和
                U_tmp = list_x[i] + F.interpolate((self.trans[i](U_tmp)), list_x[i].size()[2:], mode='bilinear', align_corners=True)
            else:
                U_tmp = list_x[i] + F.interpolate((U_tmp), list_x[i].size()[2:], mode='bilinear', align_corners=True)
            
            
            
            # U_tmp就是合并上深层的当前层
            
            tmp = self.up[i](U_tmp)
            U_tmp = tmp
            sal_feature.append(tmp)
            up_sal.append(F.interpolate(self.score[i](tmp), x_size, mode='bilinear', align_corners=True)) # 将每层的score上采样到和输入相同大小

        U_tmp = list_x[0] + F.interpolate((self.trans[-1](sal_feature[0])), list_x[0].size()[2:], mode='bilinear', align_corners=True)
        tmp = self.up[0](U_tmp)
        edge_feature.append(tmp) # 融合到最后的边的显著性值
       
        up_edge.append(F.interpolate(self.score[0](tmp), x_size, mode='bilinear', align_corners=True)) 
        return up_edge, edge_feature, up_sal, sal_feature        
        
class MergeLayer2(nn.Module):
    def __init__(self, list_k): #
        # [[128], [256, 512, 512, 512]]
        super(MergeLayer2, self).__init__()
        self.list_k = list_k
        trans, up, score = [], [], []
        # up里面放的就是连续三个上采样的卷积，但是通道数和大小不变
        # score里面就是将通道数变成1，准备计算loss
        for i in list_k[0]: # 就是单独的一个128
            tmp = []
            tmp_up = []
            tmp_score = []
            feature_k = [[3,1],[5,2], [5,2], [7,3]] # 应该是卷积核大小和padding，这样的使用才能保存卷积后大小不变
            for idx, j in enumerate(list_k[1]): # 下面四层的通道数目
                tmp.append(nn.Sequential(nn.Conv2d(j, i, 1, 1, bias=False), nn.ReLU(inplace=True))) # 输入是各层的通道数，输出统一为128

                # 连续三次上采样，同时保持通道和图片大小不变
                tmp_up.append(nn.Sequential(nn.Conv2d(i , i, feature_k[idx][0], 1, feature_k[idx][1]), nn.ReLU(inplace=True), nn.Conv2d(i, i,  feature_k[idx][0],1 , feature_k[idx][1]), nn.ReLU(inplace=True), nn.Conv2d(i, i, feature_k[idx][0], 1, feature_k[idx][1]), nn.ReLU(inplace=True)))
                tmp_score.append(nn.Conv2d(i, 1, 3, 1, 1)) # 就是将通道数缩到1的卷积，准备和gt计算loss
            trans.append(nn.ModuleList(tmp)) # 这里是将4层里每层的通道转化为了128，i=128

            up.append(nn.ModuleList(tmp_up))
            score.append(nn.ModuleList(tmp_score))
            

        self.trans, self.up, self.score = nn.ModuleList(trans), nn.ModuleList(up), nn.ModuleList(score)
        # 最后输出的特征图还得先卷，再ReLu，再卷到1通道
        self.final_score = nn.Sequential(nn.Conv2d(list_k[0][0], list_k[0][0], 5, 1, 2), nn.ReLU(inplace=True), nn.Conv2d(list_k[0][0], 1, 3, 1, 1))
        self.relu =nn.ReLU()

    def forward(self, list_x, list_y, x_size): # 合并边和图特征 edge_feature, sal_feature, x_size
        # edge_feature里面就1项，sal_feature里面多是有每层融合后的特征(注意是从深层到浅层的顺序)
        up_score, tmp_feature = [], []
        list_y = list_y[::-1]

        
        for i, i_x in enumerate(list_x):
            for j, j_x in enumerate(list_y):
                # 先trans转化到128通道，在上采样到输入的大小，i_x是对饮的前面模块的输出值，就是论文中图示的棕色跳转线
                tmp = F.interpolate(self.trans[i][j](j_x), i_x.size()[2:], mode='bilinear', align_corners=True) + i_x                
                tmp_f = self.up[i][j](tmp) # 卷积，对应图中后面的conv卷积，就是上采样三次
                up_score.append(F.interpolate(self.score[i][j](tmp_f), x_size, mode='bilinear', align_corners=True)) # 就是将图像压到1通道，准备计算loss
                tmp_feature.append(tmp_f)
       
        tmp_fea = tmp_feature[0] # 最深层的上采样三次
        for i_fea in range(len(tmp_feature) - 1):
            # 从深到浅，一下下的合并特征图
            tmp_fea = self.relu(torch.add(tmp_fea, F.interpolate((tmp_feature[i_fea+1]), tmp_feature[0].size()[2:], mode='bilinear', align_corners=True)))
        up_score.append(F.interpolate(self.final_score(tmp_fea), x_size, mode='bilinear', align_corners=True))
      


        return up_score
       


# extra part
def extra_layer(base_model_cfg, vgg):
    if base_model_cfg == 'vgg':
        config = config_vgg
    elif base_model_cfg == 'resnet':
        config = config_resnet # 这里整出来的东西应该是算什么，算resnet的整体结构吗
    merge1_layers = MergeLayer1(config['merge1'])
    merge2_layers = MergeLayer2(config['merge2'])

    return vgg, merge1_layers, merge2_layers


# TUN network
class TUN_bone(nn.Module):
    def __init__(self, base_model_cfg, base, merge1_layers, merge2_layers):
        super(TUN_bone, self).__init__()
        self.base_model_cfg = base_model_cfg
        if self.base_model_cfg == 'vgg':

            self.base = base
            # self.base_ex = nn.ModuleList(base_ex)
            self.merge1 = merge1_layers
            self.merge2 = merge2_layers

        elif self.base_model_cfg == 'resnet':
            self.convert = ConvertLayer(config_resnet['convert']) # 根据给出的参数构建resnet网络结构
            self.base = base
            self.merge1 = merge1_layers
            self.merge2 = merge2_layers

    def forward(self, x):
        x_size = x.size()[2:] # 应该就是h*w
        print(x.size())
        print(x_size)
        conv2merge = self.base(x)        
        if self.base_model_cfg == 'resnet':            
            conv2merge = self.convert(conv2merge)
        up_edge, edge_feature, up_sal, sal_feature = self.merge1(conv2merge, x_size) # 那这个就是边和图片的特征都有
        up_sal_final = self.merge2(edge_feature, sal_feature, x_size) # 合并边到最终的loss上
        return up_edge, up_sal, up_sal_final


# build the whole network
def build_model(base_model_cfg='vgg'):
    if base_model_cfg == 'vgg':
        return TUN_bone(base_model_cfg, *extra_layer(base_model_cfg, vgg16())) # *及后面代表的应该是将函数作为参数传入
    elif base_model_cfg == 'resnet':
        return TUN_bone(base_model_cfg, *extra_layer(base_model_cfg, resnet50()))


# weight init
def xavier(param):
    # init.xavier_uniform(param)
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

if __name__ == '__main__':
    from torch.autograd import Variable
    # net = TUN(*extra_layer(vgg(base['tun'], 3), vgg(base['tun_ex'], 512), config['merge_block'], config['fuse'])).cuda()
    net = TUN_bone("vgg", *extra_layer("vgg", vgg16()))
    img = Variable(torch.randn((1, 3, 256, 256))).cuda()
    out = net(img)
    print(len(out))
    print(len(out[0]))
    print(out[0].shape)
    print(len(out[1]))
    # print(net)
    input('Press Any to Continue...')
