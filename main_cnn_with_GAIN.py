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
from utils.image import show_cam_on_image, denorm, Deepfake_preprocess_image
import PIL.Image
import argparse
import logging
import math
import numpy as np
import os
import pathlib
import torch

torch.multiprocessing.set_sharing_strategy('file_system')


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

        logits_cl, logits_am, heatmaps, masks, masked_images = model(batch, labels)

        # Single label evaluation
        y_pred.extend(logits_cl.sigmoid().flatten().tolist())
        y_true.extend(label_idx_list)

        # related to am text info in viz
        am_scores = nn.Softmax(dim=1)(logits_am)
        am_labels = am_scores.argmax(dim=1)
        pos_count = test_dataset.positive_len()

        viz_test_heatmap(j, heatmaps, sample, label_idx_list, logits_cl, cfg, output_path_heatmap)
        j += 1

    monitor_test_epoch(writer, test_dataset, pos_count, y_pred, y_true, epoch,
                       output_path, mode, logger)


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


def monitor_validation_epoch(writer, validation_dataset, args, pos_count,
                             epoch_test_am_loss, y_pred, y_true, epoch, logger):
    num_test_samples = len(validation_dataset)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    print('Average epoch single validation accuracy: {:.3f}'.format(acc))
    logger.warning('Average epoch single validation accuracy: {:.3f}'.format(acc))
    writer.add_scalar('Loss/validation/am_total_loss', epoch_test_am_loss, epoch)
    writer.add_scalar('Accuracy/validation/cl_accuracy_only_pos', f_acc, epoch)
    writer.add_scalar('Accuracy/validation/cl_accuracy_only_neg', r_acc, epoch)
    writer.add_scalar('Accuracy/validation/cl_accuracy', acc, epoch)

    fpr, tpr, auc, threshold = roc_curve(y_true, y_pred)
    writer.add_scalar('ROC/validation/AUC', auc, epoch)
    # if epoch == last_epoch:
    #     save_roc_curve(test_labels.cpu().numpy(), test_differences, epoch, path)
    return acc, ap, f_acc, r_acc


def monitor_validation_viz(j, t, heatmaps, sample, masked_images, test_dataset,
                           label_idx_list, logits_cl, am_scores, am_labels, writer,
                           epoch, cfg):
    if (j < args.pos_to_write_test and j % t == 0) or (
            j > args.pos_to_write_test and j % (args.pos_to_write_test * 3) == 0):
        htm = np.uint8(heatmaps[0].squeeze().cpu().detach().numpy() * 255)
        resize = Resize(size=224)
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
        orig_viz = torch.cat((orig, viz, masked_image), 0)
        gt = [cfg['categories'][x] for x in label_idx_list][0]
        writer.add_images(tag='Validation_Heatmaps/image_' + str(j) + '_' + gt,
                          img_tensor=orig_viz, dataformats='NHWC', global_step=epoch)
        y_scores = logits_cl.sigmoid().flatten().tolist()
        predicted_categories = [1 if y_scores[i] > 0.5 else 0 for i in range(len(y_scores))]
        predicted_cl = [(cfg['categories'][x], format(y_scores[0], '.4f')) for x in
                        predicted_categories[0]]
        labels_cl = [(cfg['categories'][x], format(y_scores[0], '.4f')) for x in [(label_idx_list[0])]]
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

        writer.add_text('Validation_Heatmaps_Description/image_' + str(j) + '_' + gt, cl_text + am_text,
                        global_step=epoch)


def train_validate(args, cfg, model, device, validation_loader, validation_dataset, writer, epoch, last_epoch,
                   output_path, logger):
    model.eval()

    j = 0
    epoch_validation_am_loss = 0
    y_true, y_pred = [], []

    for sample in validation_loader:
        label_idx_list = sample['labels']
        batch = torch.stack(sample['preprocessed_images'], dim=0).squeeze()
        batch = batch.to(device)
        labels = torch.Tensor(label_idx_list).to(device).float()

        logits_cl, logits_am, heatmaps, masks, masked_images = model(batch, labels)

        # Single label evaluation
        y_pred.extend(logits_cl.sigmoid().flatten().tolist())
        y_true.extend(label_idx_list)

        am_scores = nn.Softmax(dim=1)(logits_am)
        am_labels = am_scores.argmax(dim=1)
        pos_indices = [idx for idx, x in enumerate(sample['labels']) if x == 1]
        cur_pos_num = len(pos_indices)
        # The code to replace to train on positives and negatives
        if cur_pos_num > 1:
            am_labels_scores = am_scores[pos_indices, torch.ones(cur_pos_num).long()]
            am_loss = am_labels_scores.sum() / am_labels_scores.size(0)
            epoch_validation_am_loss += (am_loss * args.am_weight).detach().cpu().item()

        pos_count = validation_dataset.positive_len()
        t = math.ceil(pos_count / (args.batchsize * args.pos_to_write_test))
        monitor_validation_viz(j, t, heatmaps, sample, masked_images, validation_dataset,
                               label_idx_list, logits_cl, am_scores, am_labels, writer,
                               epoch, cfg)
        j += 1

    return monitor_validation_epoch(writer, validation_dataset, args, pos_count, epoch_validation_am_loss,
                                    y_pred, y_true, epoch, logger)


def monitor_train_epoch(args, writer, count_pos, count_neg, c_psi1, c_psi05, c_ffhq, epoch, am_count,
                        ex_count, epoch_train_ex_loss,
                        epoch_train_am_loss, epoch_train_cl_loss,
                        num_train_samples, epoch_train_total_loss,
                        batchsize, epoch_IOU, IOU_count, train_labels,
                        total_train_single_accuracy, test_before_train,
                        train_total_pos_correct, train_total_pos_seen,
                        train_total_neg_correct, train_total_neg_seen,
                        train_differences, have_mask_indices, logger):
    print("pos = {} neg = {}".format(count_pos, count_neg))
    print(f"psi 0.5 = {c_psi05} psi 1 = {c_psi1} ffhq = {c_ffhq}")
    logger.warning(
        "pos = {} neg = {}".format(count_pos, count_neg))
    logger.warning(
        f"psi 0.5 = {c_psi05} psi 1 = {c_psi1} ffhq = {c_ffhq}")
    writer.add_scalar('Loss/train/Epoch_total_loss', epoch_train_total_loss, epoch)
    writer.add_scalar('Loss/train/Epoch_cl_total_loss', epoch_train_cl_loss, epoch)
    writer.add_scalar('Loss/train/Epoch_am_total_loss', epoch_train_am_loss, epoch)
    writer.add_scalar('Loss/train/Epoch_ex_total_loss', epoch_train_ex_loss, epoch)
    writer.add_scalar('Loss/train/Epoch_ex_1weight_total_loss',
                      (epoch_train_ex_loss if args.ex_weight == 0 else epoch_train_ex_loss / args.ex_weight), epoch)
    if (test_before_train and epoch > 0) or test_before_train == False:
        print('Average epoch train am loss: {:.3f}'.format(epoch_train_am_loss /
                                                           am_count))
        logger.warning(
            'Average epoch train am loss: {:.3f}'.format(epoch_train_am_loss /
                                                         am_count))
        print('Average epoch train ex loss: {:.3f}'.format(0 if ex_count == 0 else (epoch_train_ex_loss / ex_count)))
        logger.warning(
            'Average epoch train ex loss: {:.3f}'.format(0 if ex_count == 0 else (epoch_train_ex_loss / ex_count)))
        print('Average epoch train cl loss: {:.3f}'.format(
            epoch_train_cl_loss / (num_train_samples * batchsize)))
        logger.warning('Average epoch train cl loss: {:.3f}'.format(
            epoch_train_cl_loss / (num_train_samples * batchsize)))
        print('Average epoch train total loss: {:.3f}'.format(
            epoch_train_total_loss / (num_train_samples * batchsize)))
        logger.warning('Average epoch train total loss: {:.3f}'.format(
            epoch_train_total_loss / (num_train_samples * batchsize)))
        print('Average epoch single train accuracy: {:.3f}'.format(
            total_train_single_accuracy))
        logger.warning('Average epoch single train accuracy: {:.3f}'.format(
            total_train_single_accuracy))
    if (test_before_train and epoch > 0) or test_before_train == False:
        writer.add_scalar('Loss/train/Average_cl_total_loss', epoch_train_cl_loss /
                          (num_train_samples * batchsize), epoch)
        writer.add_scalar('Loss/train/Average_am_total_loss', epoch_train_am_loss /
                          am_count, epoch)
        writer.add_scalar('Loss/train/Average_ex_total_loss', (
            0 if ex_count == 0 else (epoch_train_ex_loss / ex_count)),
                          epoch
                          )
        writer.add_scalar('IOU/train/average_IOU_per_sample', epoch_IOU /
                                                              IOU_count if IOU_count != 0 else 0, epoch)
        writer.add_scalar('Accuracy/train/cl_accuracy',
                          total_train_single_accuracy / (num_train_samples *
                                                         batchsize), epoch)
        writer.add_scalar('Accuracy/train/cl_accuracy_only_pos',
                          train_total_pos_correct / train_total_pos_seen,
                          epoch)
        writer.add_scalar('Accuracy/train/cl_accuracy_only_neg',
                          train_total_neg_correct / train_total_neg_seen,
                          epoch)
        all_sens, auc, _, _ = calc_sensitivity(train_labels, train_differences)
        writer.add_scalar('ROC/train/ROC_0.1', all_sens[0], epoch)
        writer.add_scalar('ROC/train/ROC_0.05', all_sens[1], epoch)
        writer.add_scalar('ROC/train/ROC_0.3', all_sens[2], epoch)
        writer.add_scalar('ROC/train/ROC_0.5', all_sens[3], epoch)
        writer.add_scalar('ROC/train/AUC', auc, epoch)


def monitor_train_iteration(sample, writer, logits_cl, cl_loss,
                            cl_loss_fn, total_loss, epoch, args, cfg, labels,
                            train_total_pos_seen, train_total_pos_correct,
                            train_total_neg_correct, train_total_neg_seen):
    if cfg['i'] % 100 == 0:
        pos_indices = [idx for idx, x in enumerate(sample['labels']) if x == 1]
        neg_indices = [idx for idx, x in enumerate(sample['labels']) if x == 0]
        pos_correct = len(
            [pos_idx for pos_idx in pos_indices if logits_cl[pos_idx, 1] > logits_cl[pos_idx, 0]])
        neg_correct = len(
            [neg_idx for neg_idx in neg_indices if logits_cl[neg_idx, 1] <= logits_cl[neg_idx, 0]])
        train_total_pos_seen += len(pos_indices)
        train_total_pos_correct += pos_correct
        train_total_neg_correct += neg_correct
        train_total_neg_seen += len(neg_indices)
        if len(pos_indices) > 0:
            cl_loss_only_on_pos_samples = cl_loss_fn(
                logits_cl.detach()[pos_indices], labels.detach()[pos_indices])
            weighted_cl_pos = cl_loss_only_on_pos_samples * args.cl_weight
            writer.add_scalar('Loss/train/cl_loss_only_on_pos_samples',
                              weighted_cl_pos.detach().cpu().item(), cfg['am_i'])

        writer.add_scalar('Loss/train/cl_loss',
                          (cl_loss * args.cl_weight).detach().cpu().item(),
                          cfg['i'])
        writer.add_scalar('Loss/train/total_loss',
                          total_loss.detach().cpu().item(), cfg['total_i'])
        cfg['total_i'] += 1

    cfg['i'] += 1

    if epoch == 0 and args.test_before_train == False:
        cfg['num_train_samples'] += 1
    if epoch == 1 and args.test_before_train == True:
        cfg['num_train_samples'] += 1

    return train_total_pos_seen, train_total_pos_correct, \
           train_total_neg_correct, train_total_neg_seen


def monitor_train_viz(writer, records_indices, heatmaps, augmented_batch,
                      sample, masked_images, train_dataset, label_idx_list,
                      epoch, logits_cl, am_scores, gt, cfg, cl_loss, am_loss, ex_loss):
    for idx in records_indices:
        htm = np.uint8(heatmaps[idx].squeeze().cpu().detach().numpy() * 255)
        visualization, _ = show_cam_on_image(np.asarray(augmented_batch[idx]), htm, True)
        viz = torch.from_numpy(visualization).unsqueeze(0)
        augmented = torch.tensor(np.asarray(augmented_batch[idx])).unsqueeze(0)
        resize = Resize(size=224)
        orig = sample['orig_images'][idx].permute([2, 0, 1])
        orig = resize(orig).permute([1, 2, 0]).unsqueeze(0)
        masked_img = denorm(masked_images[idx].detach().squeeze(),
                            train_dataset.mean, train_dataset.std)
        masked_img = (masked_img.squeeze().permute([1, 2, 0]).cpu().detach().numpy() * 255).round().astype(
            np.uint8)
        masked_img = torch.from_numpy(masked_img).unsqueeze(0)
        if gt[idx] == 1 and sample['orig_masks'][idx].numel() != 1:
            orig_mask = sample['orig_masks'][idx].permute([2, 0, 1])
            orig_mask = resize(orig_mask).permute([1, 2, 0])
            orig_mask[orig_mask == 255] = 30
            orig_masked = orig + orig_mask.unsqueeze(0)
            orig_viz = torch.cat((orig, orig_masked, augmented, viz, masked_img), 0)
        else:
            orig_viz = torch.cat((orig, augmented, viz, masked_img), 0)

        groundtruth = [cfg['categories'][x] for x in label_idx_list][idx]
        img_idx = sample['idx'][idx]
        writer.add_images(
            tag='Epoch_' + str(epoch) + '/Train_Heatmaps/image_' + str(sample['filename'][idx]) + '_' + groundtruth +
                f'_{cl_loss:.4f}' + f'_{am_loss:.4f}' + f'_{ex_loss:.4f}',
            img_tensor=orig_viz, dataformats='NHWC', global_step=cfg['counter'][img_idx])
        y_scores = nn.Softmax(dim=1)(logits_cl.detach())
        predicted_categories = y_scores[idx].unsqueeze(0).argmax(dim=1)
        predicted_cl = [(cfg['categories'][x], format(y_scores[idx].view(-1)[x], '.4f')) for x in
                        predicted_categories.view(-1)]
        labels_cl = [(cfg['categories'][x], format(y_scores[idx].view(-1)[x], '.4f')) for x in
                     [(label_idx_list[idx])]]
        import itertools
        predicted_cl = list(itertools.chain(*predicted_cl))
        labels_cl = list(itertools.chain(*labels_cl))
        cl_text = 'cl_gt_' + '_'.join(labels_cl) + '_pred_' + '_'.join(predicted_cl)
        am_labels = am_scores.argmax(dim=1)
        predicted_am = [(cfg['categories'][x], format(am_scores[idx].view(-1)[x], '.4f')) for x in
                        am_labels[idx].view(-1)]
        labels_am = [(cfg['categories'][x], format(am_scores[idx].view(-1)[x], '.4f')) for x in
                     [label_idx_list[idx]]]
        import itertools
        predicted_am = list(itertools.chain(*predicted_am))
        labels_am = list(itertools.chain(*labels_am))
        am_text = '_am_gt_' + '_'.join(labels_am) + '_pred_' + '_'.join(predicted_am)
        writer.add_text('Train_Heatmaps_Description/image_' + str(img_idx) + '_' + groundtruth,
                        cl_text + am_text,
                        global_step=cfg['counter'][img_idx])
        cfg['counter'][img_idx] += 1


def handle_AM_loss(cur_pos_num, am_scores, pos_indices, model, total_loss,
                   epoch_train_am_loss, am_count, writer, cfg, args, labels):
    iter_am_loss = 0
    if not args.am_on_all and cur_pos_num > 1:
        am_labels_scores = am_scores[pos_indices,
                                     torch.ones(cur_pos_num).long()]
        am_loss = am_labels_scores.sum() / am_labels_scores.size(0)
        iter_am_loss = (am_loss * args.am_weight).detach().cpu().item()
        if model.AM_enabled():
            total_loss += (am_loss * args.am_weight)
        epoch_train_am_loss += iter_am_loss
        am_count += 1
        if cfg['i'] % 100 == 0:
            writer.add_scalar('Loss/train/am_loss',
                              iter_am_loss,
                              cfg['am_i'])
        cfg['am_i'] += 1
        return total_loss, epoch_train_am_loss, am_count, iter_am_loss
    if args.am_on_all:
        am_labels_scores = am_scores[list(range(args.batchsize)), labels]
        am_loss = am_labels_scores.sum() / am_labels_scores.size(0)
        iter_am_loss = (am_loss * args.am_weight).detach().cpu().item()
        if model.AM_enabled():
            total_loss += am_loss * args.am_weight
        if cfg['i'] % 100 == 0:
            writer.add_scalar('Loss/train/am_loss',
                              iter_am_loss,
                              cfg['am_i'])
        epoch_train_am_loss += iter_am_loss
        cfg['am_i'] += 1
        am_count += 1
    return total_loss, epoch_train_am_loss, am_count, iter_am_loss


def handle_EX_loss(model, used_mask_indices, augmented_masks, heatmaps,
                   writer, total_loss, cfg, logger, epoch_train_ex_loss, ex_mode, ex_count):
    ex_loss = 0
    iter_ex_loss = 0
    if model.EX_enabled() and len(used_mask_indices) > 0:
        # print("External Supervision started")
        augmented_masks = [ToTensor()(x).cuda() for x in augmented_masks]
        augmented_masks = torch.cat(augmented_masks, dim=0)

        if ex_mode:
            # new logic: Image Max
            # external masks = Image_Max(heatmaps, pixel level masks)
            # Equation 7: L_e = (A - Image_Max(A, H))^2
            # idx_augmented = augmented_masks < heatmaps[used_mask_indices].squeeze()
            # augmented_masks[idx_augmented] = heatmaps[used_mask_indices].squeeze()[idx_augmented]

            augmented_masks = torch.maximum(augmented_masks, heatmaps[used_mask_indices].squeeze())
            # new logic 2: Image Addition
            # external masks = Image_Addition(heatmaps, pixel level masks)
            # Equation 7: L_e = 1/n sum_c (A^c - Image_Addition(A^c, H^c))^2
            # augmented_masks = heatmaps[used_mask_indices].squeeze() + augmented_masks
            # idx_augmented = augmented_masks > 255
            # augmented_masks[idx_augmented] = 255
        else:
            # e_loss calculation: equation 7
            # caculate (A^c - H^c) * (A^c - H^c): just a pixel-wise square error between the original mask and the returned from the model
            # augmented_masks: H^c; the heatmap returned from the model: heatmaps[used_mask_indices]
            pass

        squared_diff = torch.pow(heatmaps[used_mask_indices].squeeze() - augmented_masks, 2)
        flattened_squared_diff = squared_diff.view(len(used_mask_indices), -1)
        flattned_sum = flattened_squared_diff.sum(dim=1)
        flatten_size = flattened_squared_diff.size(1)
        ex_loss = (flattned_sum / flatten_size).sum() / len(used_mask_indices)
        iter_ex_loss = (ex_loss * args.ex_weight).detach().cpu().item()
        writer.add_scalar('Loss/train/ex_loss',
                          iter_ex_loss,
                          cfg['ex_i'])
        print(f"ex loss: {iter_ex_loss}")
        logger.warning(f"ex loss: {iter_ex_loss}")
        total_loss += args.ex_weight * ex_loss
        cfg['ex_i'] += 1
        ex_count += 1
    epoch_train_ex_loss += args.ex_weight * ex_loss
    return total_loss, epoch_train_ex_loss, ex_count, iter_ex_loss


def train(args, cfg, model, device, train_loader, train_dataset, optimizer,
          writer, epoch, logger, y_cl_loss_exsup_img, y_am_loss_exsup_img, y_ex_loss_exsup_img, x_epoch_exsup_img,
          img_idx, iter_num_list, noex_y_cl_loss_exsup_img, noex_y_am_loss_exsup_img, noex_y_ex_loss_exsup_img,
          noex_x_epoch_exsup_img, noex_img_idx, noex_iter_num_list):
    # switching model to train mode
    print('*****Training Begin*****')
    logger.warning('*****Training Begin*****')
    model.train()
    # initializing all required variables
    count_pos, count_neg, c_psi1, c_psi05, c_ffhq, dif_i, epoch_IOU, am_count, ex_count = 0, 0, 0, 0, 0, 0, 0, 0, 0
    train_differences = np.zeros(args.epochsize)
    train_labels = np.zeros(args.epochsize)
    total_train_single_accuracy = 0
    epoch_train_cl_loss, epoch_train_am_loss, epoch_train_ex_loss = 0, 0, 0
    epoch_train_total_loss = 0
    IOU_count = 0
    train_total_pos_correct, train_total_pos_seen = 0, 0
    train_total_neg_correct, train_total_neg_seen = 0, 0
    # defining classification loss function
    cl_loss_fn = torch.nn.BCEWithLogitsLoss()
    # data loading loop
    print(f"masks picked: {len(train_dataset.used_masks)}")
    logger.warning(f"masks picked: {len(train_dataset.used_masks)}")

    for sample in train_loader:
        # preparing all required data
        label_idx_list = sample['labels']
        augmented_batch = sample['augmented_images']
        augmented_masks = sample['used_masks']
        all_augmented_masks = sample['preprocessed_masks']
        datasource_list = sample['source']
        batch = torch.stack(sample['preprocessed_images'], dim=0).squeeze()
        batch = batch.to(device)
        # starting the forward, backward, optimzer.step process
        optimizer.zero_grad()
        labels = torch.Tensor(label_idx_list).to(device).float()
        # sanity check for batch pos and neg distribution (for printing) #TODO: can be removed as it checked and it is ok
        count_pos += (labels == 1).int().sum()
        count_neg += (labels == 0).int().sum()
        # print(type(datasource_list)) : tuple
        c_psi1 += datasource_list.count('psi_1')
        c_psi05 += datasource_list.count('psi_0.5')
        c_ffhq += datasource_list.count('ffhq')
        # one_hot transformation
        # lb1 = labels.unsqueeze(0)
        # lb2 = 1 - lb1
        # lbs = torch.cat((lb2, lb1), dim=0).transpose(0, 1).float()
        # model forward

        logits_cl, logits_am, heatmaps, masks, masked_images = \
            model(batch, labels)
        # cl_loss and total loss computation
        cl_loss = cl_loss_fn(logits_cl.squeeze(1), labels)

        total_loss = 0
        total_loss += cl_loss * args.cl_weight
        # AM loss computation and monitoring
        pos_indices = [idx for idx, x in enumerate(sample['labels']) if x == 1]
        cur_pos_num = len(pos_indices)
        am_scores = nn.Softmax(dim=1)(logits_am)
        total_loss, epoch_train_am_loss, am_count, iter_am_loss = handle_AM_loss(
            cur_pos_num, am_scores, pos_indices, model, total_loss,
            epoch_train_am_loss, am_count, writer, cfg, args, labels)
        # monitoring cl_loss per epoch
        epoch_train_cl_loss += (cl_loss * args.cl_weight).detach().cpu().item()
        # saving logits difference for ROC monitoring
        difference = (logits_cl[:, 1] - logits_cl[:, 0]).cpu().detach().numpy()
        train_differences[dif_i: dif_i + len(difference)] = difference
        train_labels[dif_i: dif_i + len(difference)] = labels.squeeze().cpu().detach().numpy()
        dif_i += len(difference)

        # IOU monitoring
        all_masks = train_dataset.get_masks_indices()
        have_mask_indices = [sample['idx'].index(x) for x in sample['idx']
                             if x in all_masks]

        # Ex loss computation and monitoring
        used_mask_indices = [sample['idx'].index(x) for x in sample['idx']
                             if x in train_dataset.used_masks]
        total_loss, epoch_train_ex_loss, ex_count, iter_ex_loss = handle_EX_loss(model, used_mask_indices,
                                                                                 augmented_masks,
                                                                                 heatmaps, writer, total_loss, cfg,
                                                                                 logger, epoch_train_ex_loss,
                                                                                 args.ex_mode, ex_count)
        # optimization
        total_loss.backward()
        optimizer.step()
        # Single label evaluation
        epoch_train_total_loss += total_loss.detach().cpu().item()
        # epoch_train_ex_loss += total_ex_loss.detach().cpu().item()
        y_pred = logits_cl.detach().argmax(dim=1)
        y_pred = y_pred.view(-1)
        gt = labels.view(-1)
        acc = (y_pred == gt).sum()
        total_train_single_accuracy += acc.detach().cpu()
        # monitoring per iteration measurements
        train_total_pos_seen, train_total_pos_correct, \
        train_total_neg_correct, train_total_neg_seen = \
            monitor_train_iteration(
                sample, writer, logits_cl, cl_loss, cl_loss_fn,
                total_loss, epoch, args, cfg, labels, train_total_pos_seen,
                train_total_pos_correct, train_total_neg_correct,
                train_total_neg_seen)
        # monitoring visualizations which were choosen for recording
        pos_neg = cfg['counter'].keys()
        records_indices = [sample['idx'].index(x) for x in sample['idx'] if x in pos_neg]

        # record losses of the image with mask f'_{cl_loss:.4f}' + f'_{am_loss:.4f}' + f'_{ex_loss:.4f}'
        if model.EX_enabled() and len(used_mask_indices) > 0:
            y_cl_loss_exsup_img.append(cl_loss * args.cl_weight)
            y_am_loss_exsup_img.append(iter_am_loss)
            y_ex_loss_exsup_img.append(iter_ex_loss)
            x_epoch_exsup_img.append(epoch)
            img_idx.append(sample['filename'])
            iter_num_list.append(cfg['i'])
            writer.add_scalar('Loss/train/Exsup_cl_loss', cl_loss * args.cl_weight, cfg['ex_i'] - 1)
            writer.add_scalar('Loss/train/Exsup_am_loss', iter_am_loss, cfg['ex_i'] - 1)
            writer.add_scalar('Loss/train/Exsup_ex_loss', iter_ex_loss, cfg['ex_i'] - 1)
        else:
            noex_y_cl_loss_exsup_img.append(cl_loss * args.cl_weight)
            noex_y_am_loss_exsup_img.append(iter_am_loss)
            noex_y_ex_loss_exsup_img.append(iter_ex_loss)
            noex_x_epoch_exsup_img.append(epoch)
            noex_img_idx.append(sample['filename'])
            noex_iter_num_list.append(cfg['i'])

        monitor_train_viz(writer, records_indices, heatmaps, augmented_batch,
                          sample, masked_images, train_dataset, label_idx_list,
                          epoch, logits_cl, am_scores, gt, cfg, cl_loss * args.cl_weight, iter_am_loss, iter_ex_loss)
    # monitoring per epoch measurements
    monitor_train_epoch(
        args, writer, count_pos, count_neg, c_psi1, c_psi05, c_ffhq, epoch, am_count, ex_count,
        epoch_train_ex_loss, epoch_train_am_loss, epoch_train_cl_loss, cfg['num_train_samples'],
        epoch_train_total_loss, args.batchsize, epoch_IOU, IOU_count,
        train_labels, total_train_single_accuracy, args.test_before_train,
        train_total_pos_correct, train_total_pos_seen,
        train_total_neg_correct, train_total_neg_seen, train_differences, have_mask_indices, logger)


# Parse all the input argument
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
parser.add_argument('--pretrain', '-p', action='store_true', help='use pretrained model')
parser.add_argument('--level', default=0, type=int, help='epoch to resume from')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--deviceID', type=int, help='deviceID', default=0)
parser.add_argument('--num_workers', type=int, help='workers number for the dataloaders', default=3)
parser.add_argument('--num_masks', type=int, help='number of masks to pick for each epoch', default=250)
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
parser.add_argument('--ex_weight', default=1, type=float, help='extra-supervision loss weight')
parser.add_argument('--ex_mode', '-e', action='store_true', help='use new external supervision logic')
parser.add_argument('--heatmap_output', '-o', action='store_true', help='not output heatmaps')
parser.add_argument('--am_on_all', default=0, type=int, help='train am on positives and negatives')
parser.add_argument('--customize_num_masks', action='store_true', help='resume from checkpoint')
parser.add_argument('--input_dir', help='path to the input idr', type=str)
parser.add_argument('--output_dir', help='path to the outputdir', type=str)
parser.add_argument('--checkpoint_name', help='checkpoint name', type=str)
parser.add_argument('--checkpoint_file_path_load', type=str, default='',
                    help='a full path including the name of the checkpoint_file to load from, empty otherwise')
parser.add_argument('--writer_file_load', type=str, default='',
                    help='a full path including the name of the writer_file to load from, empty otherwise')


def main(args):
    categories = [
        'Neg', 'Pos'
    ]
    test_psi05_batchsize = 1
    test_psi1_batchsize = 1
    test_psi05_nepoch = 2000
    test_psi1_nepoch = 2000
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
    logger = logging.getLogger('PIL')
    logger.setLevel(logging.WARNING)

    # num_classes = len(categories)
    num_classes = 1
    device = torch.device('cuda:' + str(args.deviceID))
    model = resnet50(num_classes=1).train().to(device)
    # source code of resnet50: https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet50
    # change the last layer for finetuning

    model.train()
    # target_layer = model.features[-1]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    batch_size = args.batchsize
    epoch_size = args.nepoch
    print('loader creating...')
    logger.warning('loader creating...')
    deepfake_loader = DeepfakeLoader(args.input_dir, [1 - args.batch_pos_dist, args.batch_pos_dist],
                                     batch_size=batch_size, steps_per_epoch=epoch_size,
                                     masks_to_use=args.masks_to_use, mean=mean, std=std,
                                     transform=Deepfake_preprocess_image,
                                     collate_fn=my_collate, customize_num_masks=args.customize_num_masks,
                                     num_masks=args.num_masks)
    print('loader created')
    logger.warning('loader created')
    # if True test epoch will run first
    test_first_before_train = bool(args.test_before_train)

    epochs = args.total_epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    norm = Normalize(mean=mean, std=std)
    fill_color = norm(torch.tensor(args.fill_color).view(1, 3, 1, 1)).cuda()
    grad_layer = ["layer4"]  # , "layer3", "layer2", "layer1", "maxpool", "relu", "bn1", "conv1"
    print('model creating...')
    logger.warning('model creating...')
    i = 0
    num_train_samples = 0
    args.epochsize = epoch_size * batch_size
    am_i = 0
    ex_i = 0
    total_i = 0
    IOU_i = 0

    # load from existing model
    if len(args.checkpoint_file_path_load) > 0:
        checkpoint = torch.load(args.checkpoint_file_path_load, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        chkpnt_epoch = checkpoint['total_steps'] + 1
        chkpnt_epoch = 0
        model.cur_epoch = chkpnt_epoch
        # if model.cur_epoch > model.am_pretraining_epochs:
        #     model.enable_am = True
        # if model.cur_epoch > model.ex_pretraining_epochs:
        #     model.enable_ex = True
        # if 'i' in checkpoint:
        #     i = checkpoint['i']
        # else:
        #     i = chkpnt_epoch * args.nepoch
        # if 'total_i' in checkpoint:
        #     total_i = checkpoint['total_i']
        # else:
        #     total_i = 1 / 100
        # if 'am_i' in checkpoint:
        #     am_i = checkpoint['am_i']
        # else:
        #     am_i = (chkpnt_epoch - args.nepoch_am) * args.nepoch if args.nepoch_am >= chkpnt_epoch else 0
        # if 'ex_i' in checkpoint:
        #     ex_i = checkpoint['ex_i']
        # else:
        #     ex_i = (chkpnt_epoch - args.nepoch_ex) * args.nepoch if args.nepoch_ex >= chkpnt_epoch else 0
        # if 'num_train_samples' in checkpoint:
        #     num_train_samples = checkpoint['num_train_samples']
        # else:
        #     num_train_samples = 1

    model = batch_GAIN_Deepfake(model=model, grad_layer=grad_layer, num_classes=num_classes,
                                am_pretraining_epochs=args.nepoch_am,
                                ex_pretraining_epochs=args.nepoch_ex,
                                fill_color=fill_color,
                                test_first_before_train=test_first_before_train,
                                grad_magnitude=args.grad_magnitude)
    print('mode created')
    logger.warning('model created')
    chkpnt_epoch = 0

    if len(args.writer_file_load) > 1:
        writer = SummaryWriter(args.output_dir + args.writer_file_load)
    else:
        writer = SummaryWriter(args.output_dir + args.log_name + '_' +
                               datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


    pos_to_write = args.pos_to_write_train
    neg_to_write = args.neg_to_write_train
    pos_idx = list(range(pos_to_write))
    pos_count = deepfake_loader.get_train_pos_count()
    neg_idx = list(range(pos_count, pos_count + neg_to_write))
    idx = pos_idx + neg_idx
    counter = dict({x: 0 for x in idx})

    cfg = {'categories': categories, 'i': i, 'num_train_samples': num_train_samples,
           'am_i': am_i, 'ex_i': ex_i, 'total_i': total_i,
           'IOU_i': IOU_i, 'counter': counter}

    y_cl_loss_exsup_img = list()
    y_am_loss_exsup_img = list()
    y_ex_loss_exsup_img = list()
    x_epoch_exsup_img = list()
    img_idx = list()
    iter_num_list = list()

    noex_y_cl_loss_exsup_img = list()
    noex_y_am_loss_exsup_img = list()
    noex_y_ex_loss_exsup_img = list()
    noex_x_epoch_exsup_img = list()
    noex_img_idx = list()
    noex_iter_num_list = list()

    for epoch in range(chkpnt_epoch, epochs):
        if not test_first_before_train or \
                (test_first_before_train and epoch != 0):
            train(args, cfg, model, device, deepfake_loader.datasets['train'],
                  deepfake_loader.train_dataset, optimizer, writer, epoch, logger,
                  y_cl_loss_exsup_img, y_am_loss_exsup_img, y_ex_loss_exsup_img, x_epoch_exsup_img, img_idx,
                  iter_num_list,
                  noex_y_cl_loss_exsup_img, noex_y_am_loss_exsup_img, noex_y_ex_loss_exsup_img, noex_x_epoch_exsup_img,
                  noex_img_idx, noex_iter_num_list)

        acc_val, ap_val = train_validate(args, cfg, model, device, deepfake_loader.datasets['validation'],
                                         deepfake_loader.validation_dataset, writer, epoch, (args.total_epochs - 1),
                                         roc_log_path, logger)[:2]
        if epoch == (epochs - 1):
            print("********Testing module starts********")
            logger.warning("********Testing module starts********")
            # test psi 0.5 dataset
            deepfake_psi0_loader = DeepfakeTestingOnlyLoader(psi_05_input_dir,
                                                             batch_size=test_psi05_batchsize,
                                                             mean=mean, std=std,
                                                             transform=Deepfake_preprocess_image,
                                                             collate_fn=my_collate)
            test(cfg, model, device, deepfake_psi0_loader.datasets['test'],
                 deepfake_psi0_loader.test_dataset, writer, epoch, psi_05_heatmap_path, test_psi05_batchsize, "PSI_0.5",
                 logger)
            if not args.heatmap_output:
                select_clo_far_heatmaps(heatmap_home_dir, psi_05_input_path_heatmap, args.log_name, "psi_0.5")
            # test psi 1 dataset
            deepfake_psi1_loader = DeepfakeTestingOnlyLoader(psi_1_input_dir,
                                                             batch_size=test_psi1_batchsize,
                                                             mean=mean, std=std,
                                                             transform=Deepfake_preprocess_image,
                                                             collate_fn=my_collate)
            test(cfg, model, device, deepfake_psi1_loader.datasets['test'],
                 deepfake_psi1_loader.test_dataset, writer, epoch, psi_1_heatmap_path, test_psi1_batchsize, "PSI_1",
                 logger)
            if not args.heatmap_output:
                select_clo_far_heatmaps(heatmap_home_dir, psi_1_input_path_heatmap, args.log_name, "psi_1")

        print("finished epoch number:")
        logger.warning("finished epoch number:")
        print(epoch)
        logger.warning(str(epoch))
        model.increase_epoch_count()

        chkpt_path = args.output_dir + '/checkpoints/'
        pathlib.Path(chkpt_path).mkdir(parents=True, exist_ok=True)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'counter': cfg['counter'],
            'i': cfg['i'],
            'num_train_samples': cfg['num_train_samples'],
            'am_i': cfg['am_i'],
            'ex_i': cfg['ex_i'],
            'total_i': cfg['total_i'],
            'IOU_i': cfg['IOU_i'],
        }, chkpt_path + args.checkpoint_name + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        if args.customize_num_masks:
            print('*****customize_num_masks*****')
            logger.warning('*****customize_num_masks*****')
            deepfake_loader = DeepfakeLoader(args.input_dir, [1 - args.batch_pos_dist, args.batch_pos_dist],
                                             batch_size=batch_size, steps_per_epoch=epoch_size,
                                             masks_to_use=args.masks_to_use, mean=mean, std=std,
                                             transform=Deepfake_preprocess_image,
                                             collate_fn=my_collate, customize_num_masks=args.customize_num_masks,
                                             num_masks=args.num_masks)

    # output losses for exsup images
    heatmap_home_dir = heatmap_home_dir + f"{datetime.now().strftime('%Y%m%d')}_heatmap_output_" + args.log_name + "/"
    np.save(heatmap_home_dir + 'y_cl_loss_exsup_img', np.array(y_cl_loss_exsup_img))
    np.save(heatmap_home_dir + 'y_am_loss_exsup_img', np.array(y_am_loss_exsup_img))
    np.save(heatmap_home_dir + 'y_ex_loss_exsup_img', np.array(y_ex_loss_exsup_img))
    np.save(heatmap_home_dir + 'x_epoch_exsup_img', np.array(x_epoch_exsup_img))
    np.save(heatmap_home_dir + 'img_idx', np.array(img_idx))
    np.save(heatmap_home_dir + 'iter_num_list', np.array(iter_num_list))

    np.save(heatmap_home_dir + 'noex_y_cl_loss_exsup_img', np.array(noex_y_cl_loss_exsup_img))
    np.save(heatmap_home_dir + 'noex_y_am_loss_exsup_img', np.array(noex_y_am_loss_exsup_img))
    np.save(heatmap_home_dir + 'noex_y_ex_loss_exsup_img', np.array(noex_y_ex_loss_exsup_img))
    np.save(heatmap_home_dir + 'noex_x_epoch_exsup_img', np.array(noex_x_epoch_exsup_img))
    np.save(heatmap_home_dir + 'noex_img_idx', np.array(noex_img_idx))
    np.save(heatmap_home_dir + 'noex_iter_num_list', np.array(noex_iter_num_list))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
