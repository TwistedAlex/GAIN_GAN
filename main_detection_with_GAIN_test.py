from configs.MDTconfig import cfg
from dataloaders.deepfake_data import DeepfakeTestingOnlyLoader
from datetime import datetime
from metrics.metrics import calc_sensitivity, save_roc_curve, save_roc_curve_with_threshold
from models.batch_GAIN_Deepfake import batch_GAIN_Deepfake
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50
from torchvision.transforms import Resize, Normalize, ToTensor
from utils.image import show_cam_on_image, denorm, Deepfake_preprocess_image
import PIL.Image
import argparse
import math
import numpy as np
import os
import pathlib
import torch
import logging


def my_collate(batch):
    orig_imgs, preprocessed_imgs, agumented_imgs, masks, preprocessed_masks, \
    used_masks, labels, datasource, file, indices = zip(*batch)
    used_masks = [mask for mask, used in zip(preprocessed_masks, used_masks) if used == True]
    preprocessed_masks = [mask for mask in preprocessed_masks if mask.size > 1]
    res_dict = {'orig_images': orig_imgs,
                'preprocessed_images': preprocessed_imgs,
                'augmented_images': agumented_imgs, 'orig_masks': masks,
                'preprocessed_masks': preprocessed_masks,
                'used_masks': used_masks,
                'labels': labels, 'source': datasource, 'filename': file, 'idx': indices}
    return res_dict


def monitor_test_epoch(writer, test_dataset, pos_count, test_differences, test_total_pos_correct, epoch,
                       total_test_single_accuracy, test_total_neg_correct, path, mode, logger):
    num_test_samples = len(test_dataset)
    print('Average epoch single test accuracy: {:.3f}'.format(total_test_single_accuracy / num_test_samples))
    logger.warning('Average epoch single test accuracy: {:.3f}'.format(total_test_single_accuracy / num_test_samples))
    writer.add_text('Test/' + mode + '/Accuracy/cl_accuracy', 'Accuracy: {:.3f}'.format(
        total_test_single_accuracy / num_test_samples
    ),
                    global_step=epoch)

    writer.add_scalar('Test/' + mode + '/Accuracy/cl_accuracy_only_pos',
                      (test_total_pos_correct / pos_count) if pos_count != 0 else -1, epoch)
    writer.add_text('Test/' + mode + '/Accuracy/cl_accuracy_only_pos', 'Accuracy: {:.3f}'.format((test_total_pos_correct / pos_count)
                                                                                    if pos_count != 0
                                                                                    else -1),
                    global_step=epoch)
    writer.add_scalar('Test/' + mode + '/Accuracy/cl_accuracy_only_neg',
                      (test_total_neg_correct / (num_test_samples - pos_count))
                      if (num_test_samples - pos_count) != 0
                      else -1,
                      epoch)
    writer.add_text('Test/' + mode + '/Accuracy/cl_accuracy_only_neg', 'Accuracy: {:.3f}'.format(
        (test_total_neg_correct / (num_test_samples - pos_count))
        if (num_test_samples - pos_count) != 0
        else -1),
                    global_step=epoch)
    writer.add_scalar('Test/' + mode + '/Accuracy/cl_accuracy',
                      total_test_single_accuracy / num_test_samples,
                      epoch)

    ones = torch.ones(pos_count)
    test_labels = torch.zeros(num_test_samples)
    test_labels[0:len(ones)] = ones
    test_labels = test_labels.int()
    all_sens, auc, fpr, tpr = calc_sensitivity(test_labels.cpu().numpy(), test_differences)
    writer.add_scalar('Test/' + mode + '/ROC/ROC_0.1', all_sens[0], epoch)
    writer.add_scalar('Test/' + mode + '/ROC/ROC_0.05', all_sens[1], epoch)
    writer.add_scalar('Test/' + mode + '/ROC/ROC_0.3', all_sens[2], epoch)
    writer.add_scalar('Test/' + mode + '/ROC/ROC_0.5', all_sens[3], epoch)
    writer.add_scalar('Test/' + mode + '/ROC/AUC', auc, epoch)

    save_roc_curve(test_labels.cpu().numpy(), test_differences, epoch, path)
    save_roc_curve_with_threshold(test_labels.cpu().numpy(), test_differences, epoch, path)


def viz_test_heatmap(index_img, heatmaps, sample, masked_images, test_dataset,
                     label_idx_list, logits_cl, am_scores, am_labels, writer,
                     epoch, cfg, path):

    htm = np.uint8(heatmaps[0].squeeze().cpu().detach().numpy() * 255)
    resize = Resize(size=224)
    # test preprocessed_images, visual orig_images

    orig = sample['orig_images'][0].permute([2, 0, 1])
    orig = resize(orig).permute([1, 2, 0])
    np_orig = orig.cpu().detach().numpy()


    visualization, heatmap = show_cam_on_image(np_orig, htm, True)
    viz = torch.from_numpy(visualization).unsqueeze(0)
    orig = orig.unsqueeze(0)

    masked_image = denorm(masked_images[0].detach().squeeze(),
                          test_dataset.mean, test_dataset.std)
    masked_image = (masked_image.squeeze().permute([1, 2, 0]).cpu().detach().numpy() * 255).round().astype(
        np.uint8)
    masked_image = torch.from_numpy(masked_image).unsqueeze(0)
    orig_viz = torch.cat((orig, viz, masked_image), 1)

    # PIL.Image.fromarray(orig_viz[0].cpu().numpy(), 'RGB').save(path + "/orig_viz0.png")
    # PIL.Image.fromarray(orig[0].cpu().numpy(), 'RGB').save(path + "/orig0.png")
    # PIL.Image.fromarray(viz[0].cpu().numpy(), 'RGB').save(path + "/viz0.png")
    # PIL.Image.fromarray(masked_image[0].cpu().numpy(), 'RGB').save(path + "/masked_image0.png")

    gt = [cfg['categories'][x] for x in label_idx_list][0]
    # writer.add_images(tag='Test_Heatmaps/image_' + str(j) + '_' + gt,
    #                   img_tensor=orig_viz, dataformats='NHWC', global_step=epoch)
    y_scores = nn.Softmax(dim=1)(logits_cl.detach())
    predicted_categories = y_scores[0].unsqueeze(0).argmax(dim=1)  # 'Neg', 'Pos': 0, 1
    predicted_cl = [(cfg['categories'][x], format(y_scores[0].view(-1)[x], '.4f')) for x in
                    predicted_categories.view(-1)]
    labels_cl = [(cfg['categories'][x], format(y_scores[0].view(-1)[x], '.4f')) for x in [(label_idx_list[0])]]
    import itertools
    predicted_cl = list(itertools.chain(*predicted_cl))
    labels_cl = list(itertools.chain(*labels_cl))
    cl_text = 'cl_gt_' + '_'.join(labels_cl) + '_pred_' + '_'.join(predicted_cl)

    predicted_am = [(cfg['categories'][x], format(am_scores[0].view(-1)[x], '.4f')) for x in am_labels[0].view(-1)]
    labels_am = [(cfg['categories'][x], format(am_scores[0].view(-1)[x], '.4f')) for x in [label_idx_list[0]]]
    import itertools
    predicted_am = list(itertools.chain(*predicted_am))
    labels_am = list(itertools.chain(*labels_am))
    am_text = '_am_gt_' + '_'.join(labels_am) + '_pred_' + '_'.join(predicted_am)
    # print("**       save heatmap         **")
    # print(y_scores[0].unsqueeze(0).cpu())
    # print(y_scores[0].unsqueeze(0)[0].cpu())
    # print('pic: {:.7f}'.format(y_scores[0].unsqueeze(0)[0][0].cpu()))


    if gt in ['Neg']:
        PIL.Image.fromarray(orig_viz[0].cpu().numpy(), 'RGB').save(
            path + "/Neg/{:.7f}".format(y_scores[0].unsqueeze(0)[0][0]) + '_' + str(sample['filename'][0][:-4]) + '_gt_'+ gt + '.png')
    else:
        PIL.Image.fromarray(orig_viz[0].cpu().numpy(), 'RGB').save(
            path + "/Pos/{:.7f}".format(y_scores[0].unsqueeze(0)[0][0].cpu())+ '_' + str(sample['filename'][0][:-4]) + '_gt_'+ gt + '.png')

    # writer.add_text('Test_Heatmaps_Description/image_' + str(j) + '_' + gt, cl_text + am_text,
    #                 global_step=epoch)


def test(cfg, model, device, test_loader, test_dataset, writer, epoch, output_path, batchsize, mode, logger):
    print("******** Test ********")
    logger.warning('******** Test ********')
    print(mode)
    logger.warning(mode)

    model.eval()
    output_path_heatmap = output_path+"/test_heatmap/"
    print(output_path_heatmap)
    logger.warning(output_path_heatmap)
    output_path_heatmap_pos = output_path_heatmap+"/Pos/"
    output_path_heatmap_neg = output_path_heatmap+"/Neg/"
    pathlib.Path(output_path_heatmap_pos).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path_heatmap_neg).mkdir(parents=True, exist_ok=True)
    j = 0
    test_total_pos_correct, test_total_neg_correct = 0, 0
    total_test_single_accuracy = 0
    test_differences = np.zeros(len(test_dataset))

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
        labels = torch.Tensor(label_idx_list).to(device).long()  # a list of label idx
        # output of the model based on the input images and labels

        logits_cl, logits_am, heatmaps, masks, masked_images = model(batch, labels)

        # Single label evaluation
        y_pred = logits_cl.detach().argmax(dim=1)
        y_pred = y_pred.view(-1)
        gt = labels.view(-1)
        acc = (y_pred == gt).sum()
        total_test_single_accuracy += acc.detach().cpu()  # acc accumulation. final acc=total/size_of_test_dataset

        pos_correct = (y_pred == gt).logical_and(gt == 1).sum()  # num of positive correct tests
        neg_correct = (y_pred == gt).logical_and(gt == 0).sum()  # num of negative correct tests
        test_total_neg_correct += neg_correct  # sum num of positive correct tests
        test_total_pos_correct += pos_correct  # sum num of negative correct tests

        difference = (logits_cl[:, 1] - logits_cl[:, 0]).cpu().detach().numpy()  # to cal fpr, tpr
        test_differences[j * batchsize: j * batchsize + len(difference)] = difference

        # related to am text info in viz
        am_scores = nn.Softmax(dim=1)(logits_am)
        am_labels = am_scores.argmax(dim=1)
        pos_count = test_dataset.positive_len()

        viz_test_heatmap(j, heatmaps, sample, masked_images, test_dataset,
                         label_idx_list, logits_cl, am_scores, am_labels, writer,
                         epoch, cfg, output_path_heatmap)
        j += 1

    monitor_test_epoch(writer, test_dataset, pos_count, test_differences, test_total_pos_correct, epoch,
                       total_test_single_accuracy, test_total_neg_correct, output_path, mode, logger)


def select_clo_far_heatmaps(heatmap_home_dir, input_path_heatmap, log_name, mode):
    input_path_heatmap_pos = input_path_heatmap + "/Pos/"
    input_path_heatmap_neg = input_path_heatmap + "/Neg/"
    heatmap_home_dir = heatmap_home_dir + f"{datetime.now().strftime('%Y%m%d')}_heatmap_output_" + log_name + "/" + mode
    output_path_heatmap_pos_cl = heatmap_home_dir + "/Pos_Fake_0/" + "/50_closest/"
    output_path_heatmap_pos_fa = heatmap_home_dir + "/Pos_Fake_0/" + "/50_farthest/"
    output_path_heatmap_neg_cl = heatmap_home_dir + "/Neg_Real_1/" + "/50_closest/"
    output_path_heatmap_neg_fa = heatmap_home_dir + "/Neg_Real_1/" + "/50_farthest/"
    pathlib.Path(output_path_heatmap_pos_cl).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path_heatmap_pos_fa).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path_heatmap_neg_cl).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path_heatmap_neg_fa).mkdir(parents=True, exist_ok=True)

    pos_heatmaps = os.listdir(input_path_heatmap_pos)
    neg_heatmaps = os.listdir(input_path_heatmap_neg)
    pos_heatmaps.sort()
    neg_heatmaps.sort()

    for file in pos_heatmaps[0:50]:
        command = 'cp ' + input_path_heatmap_pos + file + ' ' + output_path_heatmap_pos_cl
        os.system(command)
    for file in pos_heatmaps[-50:]:
        command = 'cp ' + input_path_heatmap_pos + file + ' ' + output_path_heatmap_pos_fa
        os.system(command)

    for file in neg_heatmaps[0:50]:
        command = 'cp ' + input_path_heatmap_neg + file + ' ' + output_path_heatmap_neg_fa
        os.system(command)
    for file in neg_heatmaps[-50:]:
        command = 'cp ' + input_path_heatmap_neg + file + ' ' + output_path_heatmap_neg_cl
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
parser.add_argument('--input_dir', help='path to the input idr', type=str)
parser.add_argument('--output_dir', help='path to the outputdir', type=str)
parser.add_argument('--checkpoint_file_path_load', help='checkpoint name', type=str)


def main(args):
    heatmap_home_dir = "/server_data/image-research/"
    categories = [
        'Neg', 'Pos'
    ]
    test_psi05_batchsize = 1
    test_psi1_batchsize = 1
    test_psi05_nepoch = 1000
    test_psi1_nepoch = 1000
    heatmap_home_dir = "/server_data/image-research/"
    psi_05_heatmap_path = args.output_dir + "/test_" + args.log_name + "_PSI_0.5/"
    psi_1_heatmap_path = args.output_dir + "/test_" + args.log_name + "_PSI_1/"
    psi_1_input_dir = "/home/shuoli/deepfake_test_data/s2f_psi_1/"
    psi_05_input_dir = "deepfake_data/data_s2_20kT/"
    psi_05_input_path_heatmap = psi_05_heatmap_path + "/test_heatmap/"
    psi_1_input_path_heatmap = psi_1_heatmap_path + "/test_heatmap/"
    roc_log_path = args.output_dir + "roc_log"
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(psi_05_heatmap_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(psi_1_input_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(roc_log_path).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.DEBUG,
                        filename=args.output_dir + "/std.log",
                        format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)

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
    model.train()

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
    model = batch_GAIN_Deepfake(model=model, grad_layer=grad_layer, num_classes=num_classes,
                                am_pretraining_epochs=args.nepoch_am,
                                ex_pretraining_epochs=args.nepoch_ex,
                                fill_color=fill_color,
                                test_first_before_train=1,
                                grad_magnitude=args.grad_magnitude)

    checkpoint = torch.load(args.checkpoint_file_path_load)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch=49

    deepfake_psi0_loader = DeepfakeTestingOnlyLoader(psi_05_input_dir,
                                                     batch_size=test_psi05_batchsize,
                                                     mean=mean, std=std,
                                                     transform=Deepfake_preprocess_image,
                                                     collate_fn=my_collate)
    test(cfg, model, device, deepfake_psi0_loader.datasets['test'],
         deepfake_psi0_loader.test_dataset, writer, epoch, psi_05_heatmap_path, test_psi05_batchsize, "PSI_0.5", logger)
    if not args.heatmap_output:
        select_clo_far_heatmaps(heatmap_home_dir, psi_05_input_path_heatmap, args.log_name, "psi_0.5")
    # test psi 1 dataset
    deepfake_psi1_loader = DeepfakeTestingOnlyLoader(psi_1_input_dir,
                                                     batch_size=test_psi1_batchsize,
                                                     mean=mean, std=std,
                                                     transform=Deepfake_preprocess_image,
                                                     collate_fn=my_collate)
    test(cfg, model, device, deepfake_psi1_loader.datasets['test'],
         deepfake_psi1_loader.test_dataset, writer, epoch, psi_1_heatmap_path, test_psi1_batchsize, "PSI_1", logger)
    if not args.heatmap_output:
        select_clo_far_heatmaps(heatmap_home_dir, psi_1_input_path_heatmap, args.log_name, "psi_1")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
