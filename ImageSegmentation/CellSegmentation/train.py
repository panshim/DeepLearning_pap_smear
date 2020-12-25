import argparse
import torch
import tqdm
import logger
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
import matplotlib.pyplot as plt

def main(args):

    # tensorboard
    logger_tb = logger.Logger(log_dir=args.experiment_name)

    # get dataset
    if args.dataset == "nuclei":
        train_dataset = NucleiDataset(args.train_data, 'train', args.transform, args.target_channels)
    elif args.dataset == "hpa":
        train_dataset = HPADataset(args.train_data, 'train', args.transform, args.max_mean, args.target_channels)
    elif args.dataset == "hpa_single":
        train_dataset = HPASingleDataset(args.train_data, 'train', args.transform)
    else:
        train_dataset = NeuroDataset(args.train_data, 'train', args.transform)

    # split dataset into training and validation sets
    # temporary no need
    
    # create dataloader

    train_params = {'batch_size': args.batch_size,
                    'shuffle': False,
                    'num_workers': args.num_workers}
    train_dataloader = DataLoader(train_dataset, **train_params)

    # device
    device = torch.device(args.device)

    # model
    if args.model == "fusion":
        model = FusionNet(args, train_dataset.dim)
    elif args.model == "dilation":
        model = DilationCNN(train_dataset.dim)
    elif args.model == "unet":
        model = UNet(args.num_kernel, args.kernel_size, train_dataset.dim, train_dataset.target_dim)

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

    # optimizer
    parameters = model.parameters()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, args.lr)
    else:
        optimizer = torch.optim.SGD(parameters, args.lr)

    # loss 
    loss_function = dice_loss

    count = 0
    info = []
    fig, ax = plt.subplots(2, 1, sharex=True)
    # train model
    for epoch in range(args.epoch):
        # part1: train process
        with tqdm.tqdm(total=len(train_dataloader.dataset), unit=f"epoch {epoch} itr") as progress_bar:
            total_loss = []
            total_iou = []
            total_precision = []
            dev_total_loss = []
            dev_total_iou = []
            for i, (x_train, y_train) in enumerate(train_dataloader):
                if i%5!=1 :
                    model.train()
                    with torch.set_grad_enabled(True):

                        # send data and label to device
                        x = torch.Tensor(x_train.float()).to(device)
                        y = torch.Tensor(y_train.float()).to(device)

                        # predict segmentation
                        pred = model.forward(x)

                        # calculate loss
                        loss = loss_function(pred, y)
                        total_loss.append(loss.item()) 

                        # calculate IoU precision
                        predictions = pred.clone().squeeze().detach().cpu().numpy()
                        gt = y.clone().squeeze().detach().cpu().numpy()
                        ious = [metrics.get_ious(p, g, 0.5) for p,g in zip(predictions, gt)]
                        total_iou.append(np.mean(ious))

                        # back prop
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # log loss and iou 
                    avg_loss = np.mean(total_loss)
                    avg_iou = np.mean(total_iou)

                    logger_tb.update_value('train loss', avg_loss, count)
                    logger_tb.update_value('train iou', avg_iou, count)

                    # display segmentation on tensorboard 
                    if i == 0:
                        original = x_train[0].squeeze()
                        truth = y_train[0].squeeze()
                        seg = pred[0].cpu().squeeze().detach().numpy()

                        # TODO display segmentations based on number of ouput
                        logger_tb.update_image("truth", truth, count)
                        logger_tb.update_image("segmentation", seg, count)
                        logger_tb.update_image("original", original, count)

                        count += 1
                        progress_bar.update(len(x))
        
                #part2: evaluate process
                else: # i%5==1
                    model.eval()
                    x = torch.Tensor(x_train.float()).to(device)
                    y = torch.Tensor(y_train.float()).to(device)
                    dev_pred = model.forward(x)
                    
                    dev_loss = loss_function(dev_pred, y)
                    dev_total_loss.append(dev_loss.item())

                    dev_predictions = dev_pred.clone().squeeze().detach().cpu().numpy()
                    dev_gt = y.clone().squeeze().detach().cpu().numpy()
                    dev_ious = [metrics.get_ious(p, g, 0.5) for p,g in zip(dev_predictions, dev_gt)]
                    dev_total_iou.append(np.mean(dev_ious))
        
        dev_avg_loss = np.mean(dev_total_loss)
        dev_avg_iou = np.mean(dev_total_iou)
        
        print('Epoch {:5d}: Obtained a validation dev_avg_iou of {:.3f}.'.format(epoch, dev_avg_iou))
        print('Train_loss: ', avg_loss)
        print('Val_loss: ', dev_avg_loss)
        print('Train_IoU: ', avg_iou)
        print('Val IoU: ', dev_avg_iou)
        print('------------------------------------------------')
        # recording data for plot
        info.append([epoch, avg_loss, dev_avg_loss, avg_iou, dev_avg_iou])
        x, y11, y12, y21, y22 = zip(*info)
        ax[0].plot(x, y11, color='red')
        ax[0].plot(x, y12, color='blue')
        ax[0].legend(['Train loss', 'Val loss'])
        ax[1].plot(x, y21, color='red')
        ax[1].plot(x, y22, color='blue')
        ax[1].legend(['Train IoU', 'Valid IoU'])

    # save model 
    ckpt_dict = {'model_name': model.__class__.__name__, 
                 'model_args': model.args_dict(), 
                 'model_state': model.to('cpu').state_dict()}
    experiment_name = f"{model.__class__.__name__}_{args.dataset}_{train_dataset.target_dim}c"
    if args.dataset == "hpa":
        experiment_name += f"_{args.max_mean}"
    experiment_name += f"_{args.num_kernel}"
    ckpt_path = os.path.join(args.save_dir, f"{experiment_name}.pth")
    torch.save(ckpt_dict, ckpt_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_kernel', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--train_data', type=str, default="/content/drive/My Drive/Segmentation/HPA_HDF5/data.hdf5")
    parser.add_argument('--save_dir', type=str, default="/content/drive/My Drive/Segmentation")
    parser.add_argument('--dataset', type=str, default="hpa")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--max_mean', type=str, default='max')
    parser.add_argument('--target_channels', type=str, default='0,2,3')
    parser.add_argument('--batch_size', type=int, default='8')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default='16')
    parser.add_argument('--experiment_name', type=str, default='HPA_model1')

    # agumentations
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    parser.add_argument('--transform', type=boolean_string, default="False")

    args = parser.parse_args()

    main(args)
