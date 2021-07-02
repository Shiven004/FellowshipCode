# coding: utf-8

# In[1]:

import time
import os
import random
import math

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from PIL import Image
from image_loader import ImageDataset
from utils.func import *
from utils.IoU import *
from models.models import *
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
import argparse

# In[2]:

### Some utilities


# In[3]:
#bb_path = 'CUB_200_2011/bounding_boxes.txt'
#img_list_path = 'CUB_200_2011/images.txt'

def make_bb_dict(img_list_path, bb_path):
    with open(img_list_path, 'r') as file:
        img_txt = file.readlines()
    with open(bb_path, 'r') as file:
        bb_txt = file.readlines()
    bb_dict = {}
    for line in img_txt:
        indx = int(float(line.split('\n')[0].split(' ')[0]))
        img_name = line.split('\n')[0].split(' ')[1]
        bb = [float(x) for x in bb_txt[indx - 1].split('\n')[0].split(' ')[1:5]]
        bb_dict[img_name] = bb
    return bb_dict

def compute_reg_acc(preds, targets, theta=0.5):
    # preds = box_transform_inv(preds.clone(), im_sizes)
    # preds = crop_boxes(preds, im_sizes)
    # targets = box_transform_inv(targets.clone(), im_sizes)
    IoU = compute_IoU(preds, targets)
    # print(preds, targets, IoU)
    corr = (IoU >= theta).sum()
    return float(corr) / float(preds.size(0))


def compute_cls_acc(preds, targets):
    pred = torch.max(preds, 1)[1]
    # print(preds, pred)
    num_correct = (pred == targets).sum()
    return float(num_correct) / float(preds.size(0))


def compute_acc(reg_preds, reg_targets, cls_preds, cls_targets, theta=0.5):
    IoU = compute_IoU(reg_preds, reg_targets)
    reg_corr = (IoU >= theta)

    pred = torch.max(cls_preds, 1)[1]
    cls_corr = (pred == cls_targets)

    corr = (reg_corr & cls_corr).sum()

    return float(corr) / float(reg_preds.size(0))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# ### Visualize training data

# In[8]:
def train_psol(data, ddt_path, savepath, locmodel='resnet50', input_size=256, crop_size=224, batch_size=32, epochs=6):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_transfrom = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    # ### Training

    # In[10]:

    # prepare data
    '''parser = argparse.ArgumentParser(description='Parameters for PSOL evaluation')
    parser.add_argument('--loc-model', metavar='locarg', type=str, default='resnet50',dest='locmodel')
    parser.add_argument('--input_size',default=256,dest='input_size')
    parser.add_argument('--crop_size',default=224,dest='crop_size')
    parser.add_argument('--epochs',default=6,dest='epochs')
    parser.add_argument('--gpu',help='which gpu to use',default='4,5,6,7',dest='gpu')
    parser.add_argument('--ddt_path',help='generated ddt path',default='ImageNet/Projection/VGG16-448',dest="ddt_path")
    parser.add_argument('--gt_path',help='validation groundtruth path',default='ImageNet_gt/',dest="gt_path")
    parser.add_argument('--save_path',help='model save path',default='ImageNet_checkpoint',dest='save_path')
    parser.add_argument('--batch_size',default=256,dest='batch_size')
    parser.add_argument('data',metavar='DIR',help='path to imagenet dataset')
    '''


    # args = parser.parse_args()
    #lr = 1e-3 * (batch_size / 64)
    lr = 1e-3 * (batch_size / 256)
    # lr = 3e-4
    momentum = 0.9
    weight_decay = 1e-4
    print_freq = 10
    root = data

    #bb_dict = make_bb_dict(imgs_list_path, bb_path)

    MyTrainData = ImageDataset(root=root, ddt_path=ddt_path,train=True, input_size=input_size,crop_size = crop_size,
                                     transform=train_transform)
    MyTestData = ImageDataset(root=root, ddt_path=ddt_path, train=False, input_size=input_size,crop_size = crop_size,
                                    transform=test_transfrom)



    train_loader = torch.utils.data.DataLoader(dataset=MyTrainData,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=20, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=MyTestData, batch_size=batch_size,
                                              num_workers=8, pin_memory=True)
    dataloaders = {'train': train_loader, 'test': test_loader}

    # construct model
    model = choose_locmodel(locmodel)
    print(model)
    model = torch.nn.DataParallel(model).cuda()
    reg_criterion = nn.MSELoss().cuda()
    dense1_params = list(map(id, model.module.fc.parameters()))
    rest_params = filter(lambda x: id(x) not in dense1_params, model.parameters())
    param_list = [{'params': model.module.fc.parameters(), 'lr': 2 * lr},
                  {'params': rest_params,'lr': 1 * lr}]
    optimizer = torch.optim.SGD(param_list, lr, momentum=momentum,
                                weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=0.1)
    torch.backends.cudnn.benchmark = True
    best_model_state = model.state_dict()
    best_epoch = -1
    best_acc = 0.0

    epoch_loss = {'train': [], 'test': []}
    epoch_acc = {'train': [], 'test': []}
    lambda_reg = 0
    for epoch in range(epochs):
        lambda_reg = 50
        for phase in ('train', 'test'):
            reg_accs = AverageMeter()
            accs = AverageMeter()
            reg_losses = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            if phase == 'train':
                if epoch >0:
                    scheduler.step()
                model.train()
            else:
                model.eval()

            end = time.time()
            cnt = 0
            for ims, labels, boxes in dataloaders[phase]:
                data_time.update(time.time() - end)
                inputs = Variable(ims.cuda())
                boxes = Variable(boxes.cuda())
                labels = Variable(labels.cuda())

                optimizer.zero_grad()

                # forward
                if phase == 'train':
                    if 'inception' in locmodel:
                        reg_outputs1,reg_outputs2 = model(inputs)
                        reg_loss1 = reg_criterion(reg_outputs1, boxes)
                        reg_loss2 = reg_criterion(reg_outputs2, boxes)
                        reg_loss = 1 * reg_loss1 + 0.3 * reg_loss2
                        reg_outputs = reg_outputs1
                    else:
                        reg_outputs = model(inputs)
                        reg_loss = reg_criterion(reg_outputs, boxes)
                            #_,reg_loss = compute_iou(reg_outputs,boxes)
                else:
                    with torch.no_grad():
                        reg_outputs = model(inputs)
                        reg_loss = reg_criterion(reg_outputs, boxes)
                loss = lambda_reg * reg_loss
                reg_acc = compute_reg_acc(reg_outputs.data.cpu(), boxes.data.cpu())

                nsample = inputs.size(0)
                reg_accs.update(reg_acc, nsample)
                reg_losses.update(reg_loss.item(), nsample)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                batch_time.update(time.time() - end)
                end = time.time()
                if cnt % print_freq == 0:
                    print(
                            '[{}]\tEpoch: {}/{}\t Iter: {}/{} Time {:.3f} ({:.3f})\t Data {:.3f} ({:.3f})\tLoc Loss: {:.4f}\tLoc Acc: {:.2%}\t'.format(
                                phase, epoch + 1, epochs, cnt, len(dataloaders[phase]), batch_time.val,batch_time.avg,data_time.val,data_time.avg,lambda_reg * reg_losses.avg, reg_accs.avg))
                cnt += 1
            if phase == 'test' and reg_accs.avg > best_acc:
                best_acc = reg_accs.avg
                best_epoch = epoch
                best_model_state = model.state_dict()

            elapsed_time = time.time() - end
            print(
                '[{}]\tEpoch: {}/{}\tLoc Loss: {:.4f}\tLoc Acc: {:.2%}\tTime: {:.3f}'.format(
                    phase, epoch + 1, epochs, lambda_reg * reg_losses.avg, reg_accs.avg,elapsed_time))
            epoch_loss[phase].append(reg_losses.avg)
            epoch_acc[phase].append(reg_accs.avg)

        print('[Info] best test acc: {:.2%} at {}th epoch'.format(best_acc, best_epoch + 1))
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        torch.save(model.state_dict(), os.path.join(savepath,'checkpoint_localization_imagenet_ddt_' + locmodel + "_"  + str(epoch) + '.pth.tar'))
        torch.save(best_model_state, os.path.join(savepath,'best_cls_localization_imagenet_ddt_' + locmodel + "_"  + str(epoch) + '.pth.tar'))


