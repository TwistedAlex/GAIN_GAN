from configs.MDTconfig import cfg
from dataloaders.deepfake_data import DeepfakeLoader
from dataloaders.deepfake_data import DeepfakeTestingOnlyLoader
from datetime import datetime
from metrics.metrics import calc_sensitivity, save_roc_curve, save_roc_curve_with_threshold, roc_curve
from models.batch_GAIN_Deepfake import batch_GAIN_Deepfake
from sklearn.metrics import accuracy_score, average_precision_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from models.resnet import resnet50
from torchvision.transforms import Resize, Normalize, ToTensor
from utils.image import show_cam_on_image, denorm, Deepfake_CVPR_preprocess_image, Deepfake_preprocess_image
import PIL.Image
import argparse
import logging
import math
import numpy as np
import os
import pathlib
import torch


def my_collate(batch):
    orig_imgs, preprocessed_imgs, agumented_imgs, masks, preprocessed_masks, \
    used_masks, labels, datasource, file, background_mask, indices = zip(*batch)
    used_masks = [mask for mask, used in zip(preprocessed_masks, used_masks) if used == True]
    preprocessed_masks = [mask for mask in preprocessed_masks if mask.size > 1]
    res_dict = {'orig_images': orig_imgs,
                'preprocessed_images': preprocessed_imgs,
                'augmented_images': agumented_imgs, 'orig_masks': masks,
                'preprocessed_masks': preprocessed_masks,
                'used_masks': used_masks,
                'labels': labels, 'source': datasource, 'filename': file, 'bg_mask': background_mask, 'idx': indices}
    return res_dict


def monitor_test_epoch(writer, test_dataset, pos_count, y_pred, y_true, epoch,
                       path, mode, logger):
    num_test_samples = len(test_dataset)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    print('Average epoch single test accuracy: {:.3f}'.format(acc))
    logger.warning('Average epoch single test accuracy: {:.3f}'.format(acc))
    writer.add_text('Test/' + mode + '/Accuracy/cl_accuracy', 'Accuracy: {:.3f}'.format(acc), global_step=epoch)
    writer.add_scalar('Test/' + mode + '/Accuracy/cl_accuracy_only_pos', f_acc, epoch)
    writer.add_text('Test/' + mode + '/Accuracy/cl_accuracy_only_pos', 'Accuracy: {:.3f}'.format(f_acc),
                    global_step=epoch)
    writer.add_scalar('Test/' + mode + '/Accuracy/cl_accuracy_only_neg', r_acc, epoch)
    writer.add_text('Test/' + mode + '/Accuracy/cl_accuracy_only_neg', 'Accuracy: {:.3f}'.format(r_acc),
                    global_step=epoch)
    writer.add_scalar('Test/' + mode + '/Accuracy/cl_accuracy', acc, epoch)
    writer.add_scalar('Test/' + mode + '/Accuracy/ap', ap, epoch)
    writer.add_text('Test/' + mode + '/Accuracy/ap', 'Accuracy: {:.3f}'.format(ap),
                    global_step=epoch)

    fpr, tpr, auc, threshold = roc_curve(y_true, y_pred)
    writer.add_scalar('ROC/Test/AUC', auc, epoch)
    with open(path + f'/{mode}test_res.txt', 'w') as f:
        f.write(
            mode + ': AP: {:2.2f}, AUC: {:2.2f}, Acc: {:2.2f}, Acc (real): {:2.2f}, Acc (fake): {:2.2f}'.format(
                ap * 100.,
                auc * 100,
                acc * 100.,
                r_acc * 100.,
                f_acc * 100.))
    save_roc_curve(y_true, y_pred, epoch, path)
    save_roc_curve_with_threshold(y_true, y_pred, epoch, path)


def viz_test_heatmap(index_img, heatmaps, sample, label_idx_list, logits_cl, cfg, path):
    htm = np.uint8(heatmaps[0].squeeze().cpu().detach().numpy() * 255)
    resize = Resize(size=224)
    orig = sample['orig_images'][0].permute([2, 0, 1])
    orig = resize(orig).permute([1, 2, 0])
    np_orig = orig.cpu().detach().numpy()
    visualization, heatmap = show_cam_on_image(np_orig, htm, True)
    viz = torch.from_numpy(visualization).unsqueeze(0)
    orig = orig.unsqueeze(0)

    orig_viz = torch.cat((orig, viz), 1)
    gt = [cfg['categories'][x] for x in label_idx_list][0]
    y_scores = logits_cl.sigmoid().flatten().tolist()
    if not args.heatmap_output:
        if gt in ['Neg']:
            PIL.Image.fromarray(orig_viz[0].cpu().numpy(), 'RGB').save(
                path + "/Neg/{:.7f}".format(y_scores[0]) + '_' + str(sample['filename'][0][:-4]) + '_' + str(
                    index_img) + '_gt_' + gt + '.png')
        else:
            PIL.Image.fromarray(orig_viz[0].cpu().numpy(), 'RGB').save(
                path + "/Pos/{:.7f}".format(y_scores[0]) + '_' + str(sample['filename'][0][:-4]) + '_' + str(
                    index_img) + '_gt_' + gt + '.png')

    # writer.add_text('Test_Heatmaps_Description/image_' + str(j) + '_' + gt, cl_text + am_text,
    #                 global_step=epoch)


def test(cfg, model, device, test_loader, test_dataset, writer, epoch, output_path, batchsize, mode, logger):
    print("******** Test ********")
    logger.warning('******** Test ********')
    print(mode)
    logger.warning(mode)

    model.eval()
    output_path_heatmap = output_path + "/test_heatmap/"
    print(output_path_heatmap)
    logger.warning(output_path_heatmap)
    output_path_heatmap_pos = output_path_heatmap + "/Pos/"
    output_path_heatmap_neg = output_path_heatmap + "/Neg/"
    pathlib.Path(output_path_heatmap_pos).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path_heatmap_neg).mkdir(parents=True, exist_ok=True)
    j = 0

    y_true, y_pred = [], []

    # iterate all samples in test_loader
    for sample in test_loader:
        # print("image number: "+str(j))
        label_idx_list = sample['labels']  # size bach_size list of label idx of the samples in this
        batch = torch.stack(sample['preprocessed_images'], dim=0).squeeze()  # dim=0 stores the res in the 1st dimension
        if len(batch.size()) == 3:
            batch = []
            batch.append(torch.stack(sample['preprocessed_images'], dim=0).squeeze())
            batch = torch.stack(batch, dim=0)
        batch = batch.to(device)  # a list of images
        labels = torch.Tensor(label_idx_list).to(device).float()  # a list of label idx
        # output of the model based on the input images and labels

        logits_cl, logits_am, heatmaps, masks, masked_images, logits_em = model(batch, labels)

        # Single label evaluation
        y_pred.extend(logits_cl.sigmoid().flatten().tolist())
        y_true.extend(label_idx_list)

        # related to am text info in viz
        am_scores = logits_am.sigmoid()
        am_labels = [1 if x > 0.5 else 0 for x in am_scores]
        pos_count = test_dataset.positive_len()

        viz_test_heatmap(j, heatmaps, sample, label_idx_list, logits_cl, cfg, output_path_heatmap)
        j += 1
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    monitor_test_epoch(writer, test_dataset, pos_count, y_pred, y_true, epoch,
                       output_path, mode, logger)


def select_clo_far_heatmaps(heatmap_home_dir, input_path_heatmap, log_name, mode):
    input_path_heatmap_pos = input_path_heatmap + "/Pos/"
    input_path_heatmap_neg = input_path_heatmap + "/Neg/"
    heatmap_home_dir = heatmap_home_dir + f"{datetime.now().strftime('%Y%m%d')}_heatmap_output_" + log_name + "/" + mode
    output_path_heatmap_pos_cl = heatmap_home_dir + "/Pos_Fake_1/" + "/50_closest/"
    output_path_heatmap_pos_fa = heatmap_home_dir + "/Pos_Fake_1/" + "/50_farthest/"
    output_path_heatmap_neg_cl = heatmap_home_dir + "/Neg_Real_0/" + "/50_closest/"
    output_path_heatmap_neg_fa = heatmap_home_dir + "/Neg_Real_0/" + "/50_farthest/"
    pathlib.Path(output_path_heatmap_pos_cl).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path_heatmap_pos_fa).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path_heatmap_neg_cl).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path_heatmap_neg_fa).mkdir(parents=True, exist_ok=True)

    pos_heatmaps = os.listdir(input_path_heatmap_pos)
    neg_heatmaps = os.listdir(input_path_heatmap_neg)
    pos_heatmaps.sort()
    neg_heatmaps.sort()

    for file in pos_heatmaps[-50:]:
        command = 'cp ' + input_path_heatmap_pos + file + ' ' + output_path_heatmap_pos_cl
        os.system(command)
    for file in pos_heatmaps[0:50]:
        command = 'cp ' + input_path_heatmap_pos + file + ' ' + output_path_heatmap_pos_fa
        os.system(command)

    for file in neg_heatmaps[-50:]:
        command = 'cp ' + input_path_heatmap_neg + file + ' ' + output_path_heatmap_neg_fa
        os.system(command)
    for file in neg_heatmaps[0:50]:
        command = 'cp ' + input_path_heatmap_neg + file + ' ' + output_path_heatmap_neg_cl
        os.system(command)
    if not args.save_all_heatmap:
        command = 'rm -rf ' + input_path_heatmap_pos
        os.system(command)
        command = 'rm -rf ' + input_path_heatmap_neg
        os.system(command)


parser = argparse.ArgumentParser(description='PyTorch GAIN Training')
parser.add_argument('--batchsize', type=int, default=cfg.BATCHSIZE, help='batch size')
parser.add_argument('--total_epochs', type=int, default=35, help='total number of epoch to train')
parser.add_argument('--nepoch', type=int, default=6000, help='number of iterations per epoch')
parser.add_argument('--nepoch_am', type=int, default=100, help='number of epochs to train without am loss')
parser.add_argument('--nepoch_ex', type=int, default=1, help='number of epochs to train without ex loss')
parser.add_argument('--masks_to_use', type=float, default=0.1,
                    help='the relative number of masks to use in ex-supevision training')

parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--manualSeed', default=cfg.RNG_SEED, type=int, help='manual seed')
parser.add_argument('--net', dest='torchmodel', help='path to the pretrained weights file', default=None, type=str)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--level', default=0, type=int, help='epoch to resume from')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--deviceID', type=int, help='deviceID', default=0)
parser.add_argument('--num_workers', type=int, help='workers number for the dataloaders', default=3)

parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--pos_to_write_train', type=int, help='train positive samples visualizations to monitor in tb',
                    default=50)
parser.add_argument('--neg_to_write_train', type=int, help='train negative samples visualizations to monitor in tb',
                    default=20)
parser.add_argument('--pos_to_write_test', type=int, help='test positive samples visualizations to monitor in tb',
                    default=50)
parser.add_argument('--neg_to_write_test', type=int, help='test negative samples visualizations to monitor in tb',
                    default=20)
parser.add_argument('--write_every', type=int, help='how often write to tensorboard numeric measurement', default=100)
parser.add_argument('--log_name', type=str, help='identifying name for storing tensorboard logs')
parser.add_argument('--test_before_train', type=int, default=0, help='test before train epoch')
parser.add_argument('--batch_pos_dist', type=float, help='positive relative amount in a batch', default=0.25)
parser.add_argument('--fill_color', type=list, help='fill color of masked area in AM training',
                    default=[0.4948, 0.3301, 0.16])
parser.add_argument('--grad_layer', help='path to the input idr', type=str, default='features')
parser.add_argument('--grad_magnitude', help='grad magnitude of second path', type=int, default=1)
parser.add_argument('--cl_weight', default=1, type=int, help='classification loss weight')
parser.add_argument('--am_weight', default=1, type=int, help='attention-mining loss weight')
parser.add_argument('--ex_weight', default=1, type=int, help='extra-supervision loss weight')
parser.add_argument('--am_on_all', default=0, type=int, help='train am on positives and negatives')
parser.add_argument('--heatmap_output', '-o', action='store_true', help='not output heatmaps')
parser.add_argument('--save_all_heatmap', '-s', action='store_true', help='not output heatmaps')
parser.add_argument('--input_dir', help='path to the input idr', type=str)
parser.add_argument('--output_dir', help='path to the outputdir', type=str)
parser.add_argument('--checkpoint_file_path_load', help='checkpoint name', type=str)
parser.add_argument('--model', '-m', help='path to the outputdir', type=str)


def main(args):
    heatmap_home_dir = "/server_data/image-research/"
    categories = [
        'Neg', 'Pos'
    ]
    test_psi05_batchsize = 1
    test_psi1_batchsize = 1
    test_psi05_nepoch = 2000
    test_psi1_nepoch = 2000

    heatmap_paths, input_dirs, modes = [], [], []

    heatmap_home_dir = "/server_data/image-research/"
    all_test_path = "/home/shuoli/attention_env/GAIN_GAN/deepfake_data/test/"
    if args.model == 's':
        psi_05_heatmap_path = args.output_dir + "/test_" + args.log_name + f'_{args.model}' + "_PSI_0.5/"  # '_ffhq' # "_PSI_0.5/"
        psi_1_heatmap_path = args.output_dir + "/test_" + args.log_name + f'_{args.model}' + "_PSI_1/"  # '_CeleAHQ' # "_PSI_1/"
        psi_1_input_dir = "/home/shuoli/attention_env/GAIN_GAN/deepfake_data/s_psi1/"  # "/home/shuoli/GAIN_GAN/deepfake_data/test_VQGAN/celeahq/" # "/home/shuoli/deepfake_test_data/s2f_psi_1/"
        psi_05_input_dir = "/home/shuoli/attention_env/GAIN_GAN/deepfake_data/s_psi05/"  # "/home/shuoli/attention_env/GAIN_GAN/deepfake_data/test/P2_weighting/" # "deepfake_data/data_s2_20kT/"
    elif args.model == 's2':
        psi_05_heatmap_path = args.output_dir + "/test_" + args.log_name + f'_{args.model}' + "_PSI_0.5/"
        psi_1_heatmap_path = args.output_dir + "/test_" + args.log_name + f'_{args.model}' + "_PSI_1/"
        psi_1_input_dir = "/home/shuoli/deepfake_test_data/s2f_psi_1/"
        psi_05_input_dir = "deepfake_data/data_s2_20kT/"
    elif args.model == 'new':
        heatmap_paths = [  # args.output_dir + "/test_" + args.log_name + '_NVAE_celebahq/',
            args.output_dir + "/test_" + args.log_name + '_NVAE_ffhq/',
            # args.output_dir + "/test_" + args.log_name + '_P2_celebahq/',
            args.output_dir + "/test_" + args.log_name + '_P2_ffhq/',
            # args.output_dir + "/test_" + args.log_name + '_VQGAN_celebahq/',
            args.output_dir + "/test_" + args.log_name + '_VQGAN_ffhq/',
            args.output_dir + "/test_" + args.log_name + '_diffae_ffhq/', ]
        input_dirs = [  # all_test_path + "NVAE_celebahq/",
            all_test_path + "NVAE_ffhq/",
            # all_test_path + "P2_celebahq/",
            all_test_path + "P2_ffhq/",
            # all_test_path + "VQGAN_celebahq/",
            all_test_path + "VQGAN_ffhq/",
            all_test_path + "diffae_ffhq/", ]
        modes = ["NVAE_ffhq", "P2_ffhq", "VQGAN_ffhq", "diffae_ffhq"]
        for mydir in input_dirs:
            pathlib.Path(mydir).mkdir(parents=True, exist_ok=True)
        for path in heatmap_paths:
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    elif args.model == 's3':
        heatmap_paths = [args.output_dir + "/test_" + args.log_name + '_s3/', ]
        input_dirs = [all_test_path + "s3/", ]
        modes = ["s3", ]
        for mydir in input_dirs:
            pathlib.Path(mydir).mkdir(parents=True, exist_ok=True)
        for path in heatmap_paths:
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    elif args.model == '300s1p0.5ex':
        heatmap_paths = [args.output_dir + "/test_" + args.log_name + '_300s1p0.5ex/', ]
        input_dirs = [all_test_path + "300s1p0.5ex/", ]
        modes = ["300s1p0.5ex", ]
        for mydir in input_dirs:
            pathlib.Path(mydir).mkdir(parents=True, exist_ok=True)
        for path in heatmap_paths:
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    elif args.model == '500s2p1ex':
        heatmap_paths = [args.output_dir + "/test_" + args.log_name + '_500s2p1ex/', ]
        input_dirs = [all_test_path + "500s2p1ex/", ]
        modes = ["500s2p1ex", ]
        for mydir in input_dirs:
            pathlib.Path(mydir).mkdir(parents=True, exist_ok=True)
        for path in heatmap_paths:
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    elif args.model == 'debg':
        heatmap_paths = [
            args.output_dir + "/test_" + args.log_name + '_s1p05_debg/',
            args.output_dir + "/test_" + args.log_name + '_s1p1_debg/',
            args.output_dir + "/test_" + args.log_name + '_s2p05_debg/',
            args.output_dir + "/test_" + args.log_name + '_s2p1_debg/',
            args.output_dir + "/test_" + args.log_name + '_nvae_ffhq_debg/',
            args.output_dir + "/test_" + args.log_name + '_vqgan_ffhq_debg/',
            args.output_dir + "/test_" + args.log_name + '_P2_ffhq_debg/',
            args.output_dir + "/test_" + args.log_name + '_diffae_ffhq_debg/',
        ]
        input_dirs = ["/home/shuoli/attention_env/GAIN_GAN/deepfake_data/test/s1/s1p05_debg/",
                      "/home/shuoli/attention_env/GAIN_GAN/deepfake_data/test/s1/s1p1_debg/",
                      "/home/shuoli/attention_env/GAIN_GAN/deepfake_data/test/s2/s2p05_debg/",
                      "/home/shuoli/attention_env/GAIN_GAN/deepfake_data/test/s2/s2p1_debg/",
                      "/home/shuoli/attention_env/GAIN_GAN/deepfake_data/test/nvae/nvae_ffhq_debg/",
                      "/home/shuoli/attention_env/GAIN_GAN/deepfake_data/test/vqgan/vqgan_ffhq_debg/",
                      "/home/shuoli/attention_env/GAIN_GAN/deepfake_data/test/p2/P2_ffhq_debg/",
                      "/home/shuoli/attention_env/GAIN_GAN/deepfake_data/test/diffae/diffae_ffhq_debg/",
                      ]
        modes = ["s1p05_debg", "s1p1_debg", "s2p05_debg", "s2p1_debg",
                 "nvae_ffhq_debg", "vqgan_ffhq_debg", "P2_ffhq_debg", "diffae_ffhq_debg", ]
        for mydir in input_dirs:
            pathlib.Path(mydir).mkdir(parents=True, exist_ok=True)
        for path in heatmap_paths:
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    if args.model == "s" or args.model == "s2":
        psi_05_input_path_heatmap = psi_05_heatmap_path + "/test_heatmap/"
        psi_1_input_path_heatmap = psi_1_heatmap_path + "/test_heatmap/"
        pathlib.Path(psi_05_heatmap_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(psi_1_heatmap_path).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.DEBUG,
                        filename=args.output_dir + "/std.log",
                        format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)

    # python main_detection_with_GAIN_test.py --input_dir= output_dir=logs_deepfake_s2f_psi_1 log_name=test1 checkpoint_file_path_load= batchsize= nepoch=
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    num_classes = len(categories)
    device = torch.device('cuda:' + str(args.deviceID))
    model = resnet50(num_classes=1).train().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    cfg = {'categories': categories}

    batch_size = args.batchsize
    epoch_size = args.nepoch
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(args.output_dir + args.log_name +
                           datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    norm = Normalize(mean=mean, std=std)
    fill_color = norm(torch.tensor(args.fill_color).view(1, 3, 1, 1)).cuda()
    grad_layer = ["layer4"]
    if len(args.checkpoint_file_path_load) > 0:
        checkpoint = torch.load('/home/shuoli/attention_env/CNNDetection/weights/blur_jpg_prob0.5.pth',
                                map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        chkpnt_epoch = checkpoint['total_steps'] + 1
        chkpnt_epoch = 0
        model.cur_epoch = chkpnt_epoch
    model = batch_GAIN_Deepfake(model=model, grad_layer=grad_layer, num_classes=num_classes,
                                am_pretraining_epochs=args.nepoch_am,
                                ex_pretraining_epochs=args.nepoch_ex,
                                fill_color=fill_color,
                                test_first_before_train=1,
                                grad_magnitude=args.grad_magnitude)

    if len(args.checkpoint_file_path_load) > 0 and 'blur' not in args.checkpoint_file_path_load:
        checkpoint = torch.load(args.checkpoint_file_path_load, map_location='cpu')
        if 'model_state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            chkpnt_epoch = checkpoint['epoch'] + 1
            model.cur_epoch = chkpnt_epoch
        else:
            model.load_state_dict(checkpoint['model'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            chkpnt_epoch = checkpoint['total_steps'] + 1
            model.cur_epoch = chkpnt_epoch

    epoch = chkpnt_epoch

    if args.model == 'new':
        print("begin")
        for idx in range(len(modes)):
            print(idx)
            deepfake_psi0_loader = DeepfakeTestingOnlyLoader(input_dirs[idx],
                                                             batch_size=test_psi05_batchsize,
                                                             mean=mean, std=std,
                                                             transform=Deepfake_preprocess_image,
                                                             collate_fn=my_collate)
            test(cfg, model, device, deepfake_psi0_loader.datasets['test'],
                 deepfake_psi0_loader.test_dataset, writer, epoch, heatmap_paths[idx], test_psi05_batchsize, modes[idx],
                 logger)
            if not args.heatmap_output:
                select_clo_far_heatmaps(heatmap_home_dir, heatmap_paths[idx] + "/test_heatmap/", args.log_name,
                                        args.model + modes[idx])
    elif args.model == 's' or args.model == 's2':
        deepfake_psi0_loader = DeepfakeTestingOnlyLoader(psi_05_input_dir,
                                                         batch_size=test_psi05_batchsize,
                                                         mean=mean, std=std,
                                                         transform=Deepfake_preprocess_image,
                                                         collate_fn=my_collate)
        test(cfg, model, device, deepfake_psi0_loader.datasets['test'],
             deepfake_psi0_loader.test_dataset, writer, epoch, psi_05_heatmap_path, test_psi05_batchsize, "psi05",
             logger)
        if not args.heatmap_output:
            select_clo_far_heatmaps(heatmap_home_dir, psi_05_input_path_heatmap, args.log_name, args.model + "_psi05")
        # test psi 1 dataset
        deepfake_psi1_loader = DeepfakeTestingOnlyLoader(psi_1_input_dir,
                                                         batch_size=test_psi1_batchsize,
                                                         mean=mean, std=std,
                                                         transform=Deepfake_preprocess_image,
                                                         collate_fn=my_collate)
        test(cfg, model, device, deepfake_psi1_loader.datasets['test'],
             deepfake_psi1_loader.test_dataset, writer, epoch, psi_1_heatmap_path, test_psi1_batchsize, "psi1",
             logger)
        if not args.heatmap_output:
            select_clo_far_heatmaps(heatmap_home_dir, psi_1_input_path_heatmap, args.log_name, args.model + "_psi1")

    elif args.model == 'debg':
        print("begin")
        for idx in range(len(modes)):
            print(idx)
            deepfake_psi0_loader = DeepfakeTestingOnlyLoader(input_dirs[idx],
                                                             batch_size=test_psi05_batchsize,
                                                             mean=mean, std=std,
                                                             transform=Deepfake_preprocess_image,
                                                             collate_fn=my_collate)
            test(cfg, model, device, deepfake_psi0_loader.datasets['test'],
                 deepfake_psi0_loader.test_dataset, writer, epoch, heatmap_paths[idx], test_psi05_batchsize, modes[idx],
                 logger)
            if not args.heatmap_output:
                select_clo_far_heatmaps(heatmap_home_dir, heatmap_paths[idx] + "/test_heatmap/", args.log_name,
                                        args.model + modes[idx])


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
