from configs.MDTconfig import cfg
from dataloaders.deepfake_data import Deepfake_Loader
from datetime import datetime
from metrics.metrics import calc_sensitivity, save_roc_curve
from models.batch_GAIN_Deepfake import batch_GAIN_Deepfake
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50
from torchvision.transforms import Resize, Normalize, ToTensor
from utils.image import show_cam_on_image, denorm, Deepfake_preprocess_image
import PIL.Image
import argparse
import math
from main_detection_with_GAIN import (
    my_collate,
    monitor_test_epoch,
    monitor_test_viz,
    test,
    handle_AM_loss,
    handle_EX_loss,
)
import numpy as np
import os
import pathlib
import torch

parser = argparse.ArgumentParser(description='PyTorch GAIN Training')
parser.add_argument('--batchsize', type=int, default=cfg.BATCHSIZE, help='batch size')
parser.add_argument('--total_epochs', type=int, default=35, help='total number of epoch to train')
parser.add_argument('--nepoch', type=int, default=6000, help='number of iterations per epoch')
parser.add_argument('--nepoch_am', type=int, default=100, help='number of epochs to train without am loss')
parser.add_argument('--nepoch_ex', type=int, default=1, help='number of epochs to train without ex loss')
parser.add_argument('--masks_to_use', type=float, default=0.1, help='the relative number of masks to use in ex-supevision training')

parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--manualSeed', default=cfg.RNG_SEED, type=int, help='manual seed')
parser.add_argument('--net', dest='torchmodel', help='path to the pretrained weights file', default=None, type=str)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--level', default=0, type=int, help='epoch to resume from')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--deviceID', type=int, help='deviceID', default=0)
parser.add_argument('--num_workers', type=int, help='workers number for the dataloaders', default=3)

parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--pos_to_write_train', type=int, help='train positive samples visualizations to monitor in tb', default=50)
parser.add_argument('--neg_to_write_train', type=int, help='train negative samples visualizations to monitor in tb', default=20)
parser.add_argument('--pos_to_write_test', type=int, help='test positive samples visualizations to monitor in tb', default=50)
parser.add_argument('--neg_to_write_test', type=int, help='test negative samples visualizations to monitor in tb', default=20)
parser.add_argument('--write_every', type=int, help='how often write to tensorboard numeric measurement', default=100)
parser.add_argument('--log_name', type=str, help='identifying name for storing tensorboard logs')
parser.add_argument('--test_before_train', type=int, default=0, help='test before train epoch')
parser.add_argument('--batch_pos_dist', type=float, help='positive relative amount in a batch', default=0.25)
parser.add_argument('--fill_color', type=list, help='fill color of masked area in AM training', default=[0.4948,0.3301,0.16])
parser.add_argument('--grad_layer', help='path to the input idr', type=str, default='features')
parser.add_argument('--grad_magnitude', help='grad magnitude of second path', type=int, default=1)
parser.add_argument('--cl_weight', default=1, type=int, help='classification loss weight')
parser.add_argument('--am_weight', default=1, type=int, help='attention-mining loss weight')
parser.add_argument('--ex_weight', default=1, type=int, help='extra-supervision loss weight')
parser.add_argument('--am_on_all', default=0, type=int, help='train am on positives and negatives')

parser.add_argument('--input_dir', help='path to the input idr', type=str)
parser.add_argument('--output_dir', help='path to the outputdir', type=str)
parser.add_argument('--checkpoint_file_path_load', help='checkpoint name', type=str)

def main(args):
    categories = [
        'Neg', 'Pos'
    ]
    # python main_detection_with_GAIN_test.py --input_dir= output_dir=logs_deepfake_s2f_psi_1 log_name=test1 checkpoint_file_path_load= batchsize= nepoch=
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    num_classes = len(categories)
    device = torch.device('cuda:' + str(args.deviceID))
    model = resnet50(pretrained=False).train().to(device)
    # source code of resnet50: https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet50
    # change the last layer for finetuning

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes).to(device)
    )
    checkpoint = torch.load(args.checkpoint_file_path_load)
    model.load_state_dict(checkpoint['model_state_dict'])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    batch_size = args.batchsize
    epoch_size = args.nepoch
    roc_log_path = os.path.join(args.output_dir, "roc_log_2f_psi_1")
    pathlib.Path(roc_log_path).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(args.output_dir + args.log_name + '_stylegan2_f_psi_1_000000_test' +
                           datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    deepfake_loader = Deepfake_Loader(args.input_dir, [1 - args.batch_pos_dist, args.batch_pos_dist],
                                      batch_size=batch_size, steps_per_epoch=epoch_size,
                                      masks_to_use=args.masks_to_use, mean=mean, std=std,
                                      transform=Deepfake_preprocess_image,
                                      collate_fn=my_collate)

    test(args, cfg, model, device, deepfake_loader.datasets['test'],
         deepfake_loader.test_dataset, writer, 0, 0, roc_log_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

