# Load the model and Crop the Images (1.Nuclei 2.Pap Smear)
import argparse # argument parser
import torch
import tqdm
import numpy as np
import torch.nn as nn
import pickle
import metrics
from skimage import io
from skimage import transform
from model import FusionNet, DilationCNN, UNet
from dataset import NucleiDataset, HPADataset, NeuroDataset, HPASingleDataset,get_augmenter
from torch.utils.data import DataLoader
from loss import dice_loss
import imageio
import torchvision
import glob
import os
import PIL
from imgaug import augmenters as iaa

from typing import List
def write_prediction(prediction: List[int], filename: str):
    with open(filename, 'w') as fp:
        print('Id,Category', file=fp)
        for i, pred in enumerate(prediction):
            print(f'{i},{pred}', file=fp)

def main(args):

    # get dataset
    if args.dataset == "nuclei":
        test_dataset = NucleiDataset(args.test_dataset, 'train', args.transform, args.target_channels)
    elif args.dataset == "hpa":
        test_dataset = HPADataset(args.test_dataset, 'train', args.transform, args.max_mean, args.target_channels)
    elif args.dataset == "hpa_single":
        test_dataset = HPASingleDataset(args.test_dataset, 'train', args.transform)
    else:
        test_dataset = NeuroDataset(args.test_dataset, 'train', args.transform)
    print("---------Finish get dataset---------")
    
    # create dataloader
    test_params = {'batch_size': args.batch_size,
                    'shuffle': False,
                    'num_workers': args.num_workers}
    test_dataloader = DataLoader(test_dataset, **test_params)
    print("---------Finish create dataloader---------")

    # device
    device = torch.device(args.device)

    # model
    if args.model == "fusion":
        model = FusionNet(args, test_dataset.dim)
    elif args.model == "dilation":
        model = DilationCNN(test_dataset.dim)
    elif args.model == "unet":
        model = UNet(args.num_kernel, args.kernel_size, test_dataset.dim, test_dataset.target_dim)
    print("---------Finish create model---------")

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

    # optimizer
    parameters = model.parameters()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, args.lr)
    else:
        optimizer = torch.optim.SGD(parameters, args.lr)

    # loss 
    loss_function = dice_loss
    print("---------Finish optimizer & loss---------")

    # load the saved model
    pretrained_dict = torch.load('/content/drive/My Drive/Segmentation/DilationCNN_nuclei_1c_16.pth')
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict['model_state'].items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    print("---------Finish load model---------")
    
    # Model Prediction
    model.eval() # without BatchNormalization and Dropout
    total_loss = []
    total_iou = []
    total_pred = []

    print("Start the loop!")
    for i, (x_test, y_test) in enumerate(test_dataloader):
        # send data and label to device
        x = torch.tensor(x_test, requires_grad=False, device='cuda', dtype=torch.float32)
        y = torch.tensor(y_test, requires_grad=False, device='cuda')
        
        # predict segmentation
        with torch.no_grad():
            pred = model.forward(x)
            total_pred.append(pred)

        # calculate loss
        loss = loss_function(pred, y)
        total_loss.append(loss.item()) 

        # calculate IoU precision
        predictions = pred.clone().squeeze().detach().cpu().numpy()
        gt = y.clone().squeeze().detach().cpu().numpy()
        ious = [metrics.get_ious(p, g, 0.5) for p,g in zip(predictions, gt)]
        total_iou.append(np.mean(ious))

        # print('Batch {:5d}: Obtained an ious of {:.3f}.'.format(i, ious))
        print('Batch', i, ', Obtained an ious of', ious)
        print("")
    
    # log loss and iou
    # write_prediction(total_pred, 'ShiminNucleiPred')
    print('Tensor size:', np.shape(total_pred))
    torch.save(total_pred, 'NeucleiMaskTensor.pt')
    avg_loss = np.mean(total_loss)
    avg_iou = np.mean(total_iou)
    print('done!')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_kernel', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--test_dataset', type=str, default="PATH_TO_TRAIN_DATA")
    parser.add_argument('--save_dir', type=str, default="./")
    parser.add_argument('--dataset', type=str, default="Hpa")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--max_mean', type=str, default='max')
    parser.add_argument('--target_channels', type=str, default='0,2,3')
    parser.add_argument('--batch_size', type=int, default='8')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default='16')
    parser.add_argument('--experiment_name', type=str, default='test')
    # agumentations
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    parser.add_argument('--transform', type=boolean_string, default="False")

    args = parser.parse_args()
    main(args)