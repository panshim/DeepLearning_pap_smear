# --------------------------------------------------
# Input:  an overlapped cell image
# Output: detected cell image
# --------------------------------------------------
import argparse # argument parser
import torch
import tqdm
import numpy as np
import metrics
from model import FusionNet, DilationCNN, UNet
from torch.utils.data import DataLoader
from loss import dice_loss
import glob
import os
import PIL
from imgaug import augmenters as iaa
from typing import List

# ----------function: loadImage-------------
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def loadImage(ImgPATH):
    # step1: construct transformation
    img = PIL.Image.open(ImgPATH)
    trans1 = transforms.ToTensor()
    trans2 = transforms.ToPILImage()
    
    # step2: get imgTensor
    imgTensor = trans1(img)
    # imgTensor = imgTensor.reshape(imgTensor.shape[1],imgTensor.shape[2])
    # print(imgTensor.shape)

    # step3: show the original image
    imgPIL = trans2(imgTensor).convert('RGB')
    imgPIL.show()
    imgplot = plt.imshow(imgPIL)
    
    return imgTensor.unsqueeze(0)
# ----------------------------------------------


def main(args):

    # get image data
    x_test = loadImage(args.imgdata)
    print("---------Finish loading dataset---------")

    # device
    device = torch.device(args.device)

    # model
    dim=1
    target_dim=1
    if args.model == "fusion":
        model = FusionNet(args, dim)
    elif args.model == "dilation":
        model = DilationCNN(dim)
    elif args.model == "unet":
        model = UNet(args.num_kernel, args.kernel_size, dim, target_dim)
    print("---------Finish creating model---------")

    if args.device == "cuda":
        # parse gpu_ids for data paralle
        if ',' in args.gpu_ids:
            gpu_ids = [int(ids) for ids in args.gpu_ids.split(',')]
        else:
            gpu_ids = int(args.gpu_ids)

        # parallelize computation
        if type(gpu_ids) is not int:
            model = nn.DataParallel(model, gpu_ids)
    model.to(device)
    print("---------Finish move to cuda---------")

    # load the saved model
    pretrained_dict = torch.load(args.load_model)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict['model_state'].items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    print("---------Finish loading model---------")

    # optimizer
    parameters = model.parameters()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, args.lr)
    else:
        optimizer = torch.optim.SGD(parameters, args.lr)

    # loss 
    loss_function = dice_loss
    print("---------Finish optimizer & loss---------")
    
    # Model Prediction
    model.eval() # without BatchNormalization and Dropout
    total_loss = []
    total_iou = []
    total_pred = []

    print("Start the predicting!")
    # send data and label to device
    x = torch.tensor(x_test, requires_grad=False, device='cuda', dtype=torch.float32)
    # predict segmentation
    with torch.no_grad():
        pred = model.forward(x)
        total_pred.append(pred)
    # No need to calculate loss
    print('Finished predicting')
    print("")
    
    # log result
    print(pred.shape)
    torch.save(pred, args.save_dir + args.experiment_name + '.pt')
    print('done!')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_kernel', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--test_dataset', type=str, default="PATH_TO_TRAIN_DATA")
    parser.add_argument('--load_model', type=str, default="PATH_TO_SAVED_MODEL")
    parser.add_argument('--save_dir', type=str, default="./")
    parser.add_argument('--imgdata', type=str, default="IMAGE_PATH")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--max_mean', type=str, default='max')
    parser.add_argument('--target_channels', type=str, default='0,2,3')
    parser.add_argument('--batch_size', type=int, default='8')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default='16')
    parser.add_argument('--experiment_name', type=str, default='MODEL_DATASET_ID')
    # agumentations
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    parser.add_argument('--transform', type=boolean_string, default="False")

    args = parser.parse_args()
    main(args)