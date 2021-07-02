import os
import sys
import json
import copy
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.backends import cudnn
import torch.nn as nn
import torchvision
from PIL import Image
from utils.func import *
from utils.vis import *
from utils.IoU import *
from models.models import choose_locmodel,choose_clsmodel
from utils.augment import *
import argparse
from image_loader import get_bbox_dict
'''
parser = argparse.ArgumentParser(description='Parameters for PSOL evaluation')
parser.add_argument('--loc-model', metavar='locarg', type=str, default='vgg16',dest='locmodel')
parser.add_argument('--cls-model', metavar='clsarg', type=str, default='vgg16',dest='clsmodel')
parser.add_argument('--input_size',default=256,dest='input_size')
parser.add_argument('--crop_size',default=224,dest='crop_size')
parser.add_argument('--ten-crop', help='tencrop', action='store_true',dest='tencrop')
parser.add_argument('--gpu',help='which gpu to use',default='4',dest='gpu')
parser.add_argument('data',metavar='DIR',help='path to imagenet dataset')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['OMP_NUM_THREADS'] = "4"
os.environ['MKL_NUM_THREADS'] = "4"
'''

def copy_parameters(model, pretrained_dict):
    model_dict = model.state_dict()

    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict and pretrained_dict[k].size()==model_dict[k[7:]].size()}
    for k, v in pretrained_dict.items():
        print(k)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def choose_locmodel(model_name, model_path=None, num_classes=1000):#, pretrained=False):
    if model_name == 'densenet161':
        model = torchvision.models.densenet161(pretrained=True)

        model.classifier = nn.Sequential(
            nn.Linear(2208, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        if model_path:
            model = copy_parameters(model, torch.load(model_path))
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False, num_classes=num_classes)
        model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        if model_path:
            model = copy_parameters(model, torch.load(model_path))
    elif model_name == 'vgggap':
        model = VGGGAP(pretrained=True,num_classes=num_classes)
        if model_path:
            model = copy_parameters(model, torch.load(model_path))
    elif model_name == 'vgg16':
        model = VGG16(pretrained=True,num_classes=num_classes)
        if model_path:
            model = copy_parameters(model, torch.load(model_path))
    elif model_name == 'inceptionv3':
        #need for rollback inceptionv3 official code
        pass
    else:
        raise ValueError('Do not have this model currently!')
    return model

def psol_infer(data, locmodel, model_path, input_size=256, crop_size=224, gt=True, visualize=True, visualize_pct=0.01, colab=False):
    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
    ])
    cls_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
    ])
    ten_crop_aug = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.TenCrop(crop_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
    ])

    root = data
    val_imagedir = os.path.join(root, 'val')
    classes = os.listdir(val_imagedir)
    classes.sort()
    temp_softmax = nn.Softmax()

    locname = locmodel
    model = choose_locmodel(locname, model_path, len(classes))

    print(model)
    model = model.to(0)
    model.eval()
    class_to_idx = {classes[i]:i for i in range(len(classes))}

    savepath = os.path.join(root, 'annoted')
    result = {}

    accs = []
    accs_top5 = []
    ious = []
    cls_accs = []
    final_cls = []
    final_iou = []
    final_clsloc = []
    final_clsloctop5 = []
    final_ind = []
    for k in range(len(classes)):
        cls = classes[k]

        total = 0
        IoUSet = []
        IoUSetTop5 = []
        LocSet = []
        ClsSet = []

        files = os.listdir(os.path.join(val_imagedir, cls))
        files.sort()
        if gt:
            gt_dict = get_bbox_dict(root)
        for (i, name) in enumerate(files):
            # raw_img = cv2.imread(os.path.join(imagedir, cls, name))
            '''
            now_index = int(name.split('_')[-1].split('.')[0])
            final_ind.append(now_index-1)
            xmlfile = os.path.join(val_annodir, cls, name.split('.')[0] + '.xml')
            '''
            if gt:
                gt_bbox = gt_dict[os.path.join(cls, name)]

            raw_img = Image.open(os.path.join(val_imagedir, cls, name)).convert('RGB')
            

            with torch.no_grad():
                img = transform(raw_img)
                img = torch.unsqueeze(img, 0)
                img = img.to(0)
                reg_outputs = model(img)

                bbox = to_data(reg_outputs)
                bbox = torch.squeeze(bbox)
                bbox = bbox.numpy()
            w, h = crop_size, crop_size
            bbox[0] = bbox[0] * w
            bbox[2] = bbox[2] * w + bbox[0]
            bbox[1] = bbox[1] * h
            bbox[3] = bbox[3] * h + bbox[1]
            #handle resize and centercrop for gt_boxes
            #for j in range(len(gt_boxes)):
            if gt:
                temp_list = list(gt_bbox)
                raw_img_i, gt_bbox_i = ResizedBBoxCrop((input_size, input_size))(raw_img, temp_list)
                raw_img_i, gt_bbox_i = CenterBBoxCrop((crop_size))(raw_img_i, gt_bbox_i)
                w, h = raw_img_i.size

                gt_bbox_i[0] = gt_bbox_i[0] * w
                gt_bbox_i[2] = gt_bbox_i[2] * w
                gt_bbox_i[1] = gt_bbox_i[1] * h
                gt_bbox_i[3] = gt_bbox_i[3] * h

                #raw_img = raw_img_i
                gt_box = gt_bbox_i
                iou = IoU(bbox, gt_box)
                LocSet.append(iou)
                
                result[os.path.join(cls, name)] = iou
                IoUSet.append(iou)
            else:
                raw_img_i = raw_img

            #visualization code
            if visualize:
                opencv_image = copy.deepcopy(np.array(raw_img_i))
                opencv_image = opencv_image[:, :, ::-1].copy()
                if gt:
                    cv2.rectangle(opencv_image, (int(gt_box[0]), int(gt_box[1])),
                        (int(gt_box[2]), int(gt_box[3])), (0, 255, 0), 4)
                cv2.rectangle(opencv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              (0, 255, 255), 4)
                if np.random.binomial(1, visualize_pct, 1):
                    if colab:
                        from google.colab.patches import cv2_imshow
                        cv2_imshow(opencv_image)
                    else:
                        cv2.imshow(opencv_image)
                cv2.imwrite(os.path.join(savepath, str(name) + '.jpg'), np.asarray(opencv_image))
            
        if gt:
            cls_loc_acc = np.sum(np.array(IoUSet) > 0.5) / len(IoUSet)
            final_clsloc.extend(IoUSet)
            iou_avg = np.sum(np.array(IoUSet)) / len(IoUSet)
            #final_iou.extend(LocSet)
            
            print('{} cls-loc acc is {}, avg iou is {}'.format(cls, cls_loc_acc, iou_avg))
            with open('inference_CorLoc.txt', 'a+') as corloc_f:
                corloc_f.write('{} {}\n'.format(cls, iou_avg))
            accs.append(cls_loc_acc)
            ious.append(iou_avg)
            if (k+1) %100==0:
                print(k)


    if gt:
        print(accs)
        print('Cls-Loc acc {}'.format(np.mean(accs)))

        print('Avg IoU {}'.format(np.mean(ious)))
        #with open('Corloc_result.txt', 'w') as f:
         #   for k in sorted(result.keys()):
          #      f.write('{} {}\n'.format(k, str(result[k])))
