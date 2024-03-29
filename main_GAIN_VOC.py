import argparse

import torch
import torchvision
from torch import nn

from torchvision.models import vgg19
import numpy as np

from torchvision.transforms import Resize, Normalize, ToTensor

from configs.VOCconfig import cfg
from dataloaders import data

from utils.image import show_cam_on_image, preprocess_image, deprocess_image, denorm

from models.batch_GAIN_VOC import batch_GAIN_VOC

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Parse all the input argument
parser = argparse.ArgumentParser(description='PyTorch GAIN Training')
parser.add_argument('--device', type=str, help='device:ID', default='cuda:1') #for gpu use the string format of 'cuda:#', e.g: cuda:0
parser.add_argument('--batch_size', type=int, help='batch size', default='1')
parser.add_argument('--epoch_size', type=int, help='number of iterations per epoch', default='6000')
parser.add_argument('--dataset_path', type=str, help='the path to your local VOC 2012 dataset directory')
parser.add_argument('--logging_path', type=str, help='the path to your local logging directory') #recommended a big enough storage for many runs
parser.add_argument('--logging_name', type=str, help='a name for the current run for logging purposes') #recommended a big enough storage for many runs
parser.add_argument('--checkpoint_file_path_load', type=str, default='', help='a full path including the name of the checkpoint_file to load from, empty otherwise')
parser.add_argument('--checkpoint_file_path_save', type=str, default='', help='a full path including the name of the checkpoint_file to save to, empty otherwise')
parser.add_argument('--checkpoint_nepoch', type=int, help='each how much to save a checkpoint', default=0)
parser.add_argument('--workers_num', type=int, help='number of threads for data loading', default=0)
parser.add_argument('--test_first', type=int, help='0 for not testing before training in the beginning, 1 otherwise', default=0)
parser.add_argument('--cl_loss_factor', type=float, help='a parameter for the classification loss magnitude, constant 1 in the paper', default=1)
parser.add_argument('--am_loss_factor', type=float, help='a parameter for the AM (Attention-Mining) loss magnitude, alpha in the paper', default=1)
parser.add_argument('--e_loss_factor', type=float, help='a parameter for the extra supervision loss magnitude, omega in the paper', default=1)
parser.add_argument('--nepoch', type=int, help='number of epochs to train', default=50)
parser.add_argument('--nepoch_am', type=int, default=1, help='number of epochs to train without am loss')
parser.add_argument('--nepoch_ex', type=int, default=1, help='number of epochs to train without ex loss')
parser.add_argument('--lr', type=float, help='learning rate', default=0.00001) #recommended 0.0001 for batchsiuze > 1, 0.00001 otherwise
parser.add_argument('--npretrain', type=int, help='number of epochs to pretrain before using AM', default=5) #recommended at least 1
parser.add_argument('--record_itr_train', type=int, help='each which number of iterations to log images in training mode', default=1000)
parser.add_argument('--record_itr_test', type=int, help='each which number of iterations to log images in test mode', default=100)
parser.add_argument('--nrecord', type=int, help='how much images of a batch to record', default=1)

parser.add_argument('--grads_off', type=int, default=0, help='mode: \
                with gradients (as in the paper) or without \
                (a novel approach of ours)')
parser.add_argument('--grads_magnitude', type=int, default=1, 
                help='mode with gradients: set the magnitude of the \
                backpropogated gradients (default 1 as in the paper, \
                otherwise as a hyperparameter to play with)')







def main(args):
    categories = cfg.CATEGORIES
    num_classes = len(categories)
    picked_categories = [0] # TODO: assign index of the category rather than the name of the category
    device = torch.device(args.device)
    model = vgg19(pretrained=True).train().to(device)


    # change the last layer for finetuning
    classifier = model.classifier
    num_ftrs = classifier[-1].in_features
    new_classifier = torch.nn.Sequential(*(list(model.classifier.children())[:-1]),
                                         nn.Linear(num_ftrs, num_classes).to(device))
    model.classifier = new_classifier
    model.train()

    batch_size = args.batch_size
    epoch_size = args.epoch_size

    #use values to normiulize the input as in torchvision ImageNet guide
    mean = cfg.MEAN
    std = cfg.STD
    
    batch_size_dict = {'train': batch_size, 'test': batch_size}
    rds = data.RawDataset(root_dir=args.dataset_path,
                          num_workers=args.workers_num,
                          output_dims=cfg.INPUT_DIMS,
                          batch_size_dict=batch_size_dict)

    test_first = bool(args.test_first)

    # weight parameters for the final loss
    cl_factor = args.cl_loss_factor
    am_factor = args.am_loss_factor
    # TODO:
    e_factor = args.e_loss_factor

    epochs = args.nepoch
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # a GAIN model which saves the chosen classification model and calculates
    # the gradients w.r.t the grad_layer and performs GAIN algorithm
    gain = batch_GAIN_VOC(model=model, grad_layer='features',
                          num_classes=num_classes,
                          am_pretraining_epochs=args.nepoch_am,
                          ex_pretraining_epochs=args.nepoch_ex,
                          test_first_before_train=test_first,
                          grads_off=bool(args.grads_off),
                          grads_magnitude=args.grads_magnitude,
                          device=device)

    i = 0
    num_train_samples = 0
    chkpnt_epoch = 0

    # load intermediate model from existing checkpoint file
    if len(args.checkpoint_file_path_load) > 0:
        checkpoint = torch.load('args.checkpoint_file_path_load')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        chkpnt_epoch = checkpoint['epoch']+1
        i = checkpoint['iteration'] + 1
        num_train_samples = checkpoint['num_train_samples']

    # profiling
    writer = SummaryWriter(
        args.logging_path + args.logging_name + '_' +
                datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
                
    writer.add_text('Start', 'start')
    
    print('Started')
    for epoch in range(chkpnt_epoch, epochs):
        print("enter epoch")

        total_train_single_accuracy = 0
        total_test_single_accuracy = 0

        epoch_train_cl_loss = 0

        model.train(True)

        # Training Block
        # if test first, skip this block only for the 1st epoch
        if not test_first or (test_first and epoch != 0):

            total_train_single_accuracy = 0
            total_test_single_accuracy = 0


            epoch_train_am_loss = 0
            epoch_train_cl_loss = 0
            # TODO:
            epoch_train_e_loss = 0
            epoch_train_total_loss = 0

            # Sampling training batch
            for sample in rds.datasets['rnd_train']:
                # TODO:
                augmented_masks = sample[3]
                mask_flag = sample[4] # data.py my_collate mask_flags

                # Get batch and augmented batch for the 1st img in the 1st array of sample
                augmented_batch = []
                batch, augmented = preprocess_image(sample[0][0].squeeze().cpu().detach().numpy(), train=True, mean=mean, std=std)
                augmented_batch.append(augmented)

                # for img starting from the 2nd img in the 1st array
                for img in sample[0][1:]:
                    # Concatenate the batch and augmented data for the 1st img and the rest imgs from sample
                    input_tensor, augmented_image = preprocess_image(img.squeeze().cpu().detach().numpy(), train=True, mean=mean, std=std)
                    batch = torch.cat((batch,input_tensor), dim=0)
                    augmented_batch.append(augmented_image)
                batch = batch.to(device)

                optimizer.zero_grad()

                # Lable: the 3nd array of sample
                labels = sample[2]

                # logits_am: onehot score of classification for the sample
                logits_cl, logits_am, heatmap, scaled_ac, masked_image, mask = gain(batch, sample[1])

                # Class onehot: the 2nd array of sample
                class_onehot = torch.stack(sample[1]).float()

                # cl_loss classification loss calculation: multi-label soft margin loss(paper); torch.nn.BCEWithLogitsLoss(implemented)
                cl_loss = loss_fn(logits_cl, class_onehot)

                # Attention map scores: softmax score of logists_am, S (I^{*}) in paper equation 5, logists_am is I^{*}
                am_scores = nn.Softmax(dim=1)(logits_am)
                batch_am_lables = []
                batch_am_labels_scores = []

                # am_loss calculation: equation 5
                #   num_of_labels(multiclass): num of labels for image k in sample
                #   am_labels: index of the top num_of_labels labels based on am_scores of image k
                #   am_labels_scores: get the L_{am} for image k
                #   batch_am_labels_scores: store am_labels_scores for each item in the batch
                for k in range(len(batch)):
                    num_of_labels = len(sample[2][k])
                    _, am_labels = am_scores[k].topk(num_of_labels)
                    batch_am_lables.append(am_labels)
                    # equation 5
                    am_labels_scores = am_scores[k].view(-1)[labels[k]].sum() / num_of_labels
                    batch_am_labels_scores.append(am_labels_scores)
                am_loss = sum(batch_am_labels_scores) / batch_size

                # TODO: am_loss calculation: equation 5
                # TODO: caculate (A^c - H^c) * (A^c - H^c): just a pixel-wise sqeared error between the original mask and the returned from the model
                #  (trainable attention map - extraSupervision),
                #  trainable attention map A, logits_am is I^{*}
                # ToTensor()(x).cuda()

                # augmented_masks = torch.cat(augmented_masks, dim=0)
                # train_dataset -> medt_loader.datasets['train']
                e_loss = 0
                print(mask_flag[0])
                if mask_flag[0]:
                    print("***********External supervision added***********")
                    print(sample[5])
                    augmented_masks = [ToTensor()(x).cuda() for x in augmented_masks]
                    squared_diff = torch.pow(scaled_ac.squeeze() - augmented_masks[0], 2)
                    flattened_squared_diff = squared_diff.view(1, -1)
                    flattned_sum = flattened_squared_diff.sum(dim=1)
                    flatten_size = flattened_squared_diff.size(1)
                    e_loss = (flattned_sum / flatten_size).sum() / 1
                    total_loss = cl_loss * cl_factor + am_loss * am_factor + e_loss * e_factor

                    epoch_train_e_loss += (e_loss * e_factor) # detach().cpu().item()


                # g = make_dot(am_loss, dict(gain.named_parameters()), show_attrs = True, show_saved = True)
                # g.save('grad_viz', train_path)



                else:
                    total_loss = cl_loss * cl_factor + am_loss * am_factor
                epoch_train_am_loss += (am_loss * am_factor).detach().cpu().item()
                epoch_train_cl_loss += (cl_loss * cl_factor).detach().cpu().item()
                # TODO:

                epoch_train_total_loss += total_loss.detach().cpu().item()
                writer.add_scalar('Per_Step/train/e_loss', (e_loss * e_factor), i)
                writer.add_scalar('Per_Step/train/cl_loss', (cl_loss * cl_factor).detach().cpu().item(), i)
                writer.add_scalar('Per_Step/train/am_loss', (am_loss * am_factor).detach().cpu().item(), i)
                # TODO:
                writer.add_scalar('Per_Step/train/total_loss', total_loss.detach().cpu().item(), i)
                
                
                loss = cl_loss * cl_factor
                if gain.AM_enabled():
                    loss += am_loss * am_factor
                    # TODO:
                if gain.EX_enabled():
                    loss += e_loss * e_factor
                loss.backward()
                optimizer.step()

                # Single label evaluation
                for k in range(len(batch)):
                    num_of_labels = len(sample[2][k])
                    _, y_pred = logits_cl[k].detach().topk(k=num_of_labels)
                    y_pred = y_pred.view(-1)
                    gt = torch.tensor(sorted(sample[2][k]), device=device)

                    correct_label_counter = 0
                    total_picked_gt_label = 0
                    if picked_categories:
                        # print(picked_categories)
                        for gt_label in gt:
                            if gt_label in picked_categories:
                                total_picked_gt_label += 1
                            if gt_label in y_pred and gt_label in picked_categories:
                                correct_label_counter += 1
                        if total_picked_gt_label == 0:
                            # print("train: divided by 0")
                            # print(correct_label_counter)
                            # print(correct_label_counter != 0)
                            if correct_label_counter != 0:
                                total_train_single_accuracy += 0
                            else:
                                total_train_single_accuracy += 1
                        else:
                            print("train: acc")
                            print((correct_label_counter / total_picked_gt_label))
                            total_train_single_accuracy += (correct_label_counter / total_picked_gt_label)
                    else:
                        acc = acc.detach().cpu()/len(gt)
                        total_train_single_accuracy += acc
                        if acc.detach().cpu() > 1:
                            print("Invalid acc")
                            print(acc.detach().cpu())
                            # print(gt)
                            # print(y_pred)

                # Multi label evaluation
                #_, y_pred_multi = logits_cl.detach().topk(num_of_labels)
                #y_pred_multi = y_pred_multi.view(-1)
                #acc_multi = (y_pred_multi == gt).sum() / num_of_labels
                #total_train_multi_accuracy += acc_multi.detach().cpu()

                if i % args.record_itr_train == 0:
                    for t in range(args.nrecord):
                        num_of_labels = len(sample[2][t])
                        one_heatmap = heatmap[t].squeeze().cpu().detach().numpy()

                        one_augmented_im = torch.tensor(np.array(augmented_batch[t])).to(device).unsqueeze(0)
                        one_masked_image = masked_image[t].detach().squeeze()
                        htm = deprocess_image(one_heatmap)
                        visualization, red_htm = show_cam_on_image(one_augmented_im.cpu().detach().numpy(), htm, True)

                        viz = torch.from_numpy(visualization).to(device)
                        masked_im = denorm(one_masked_image, mean, std)
                        masked_im = (masked_im.squeeze().permute([1, 2, 0])
                            .cpu().detach().numpy() * 255).round()\
                            .astype(np.uint8)

                        orig = sample[0][t].unsqueeze(0)
                        masked_im = torch.from_numpy(masked_im).unsqueeze(0).to(device)
                        orig_viz = torch.cat((orig, one_augmented_im, viz, masked_im), 0)
                        grid = torchvision.utils.make_grid(orig_viz.permute([0, 3, 1, 2]))
                        gt = [categories[x] for x in labels[t]]
                        writer.add_image(tag='Train_Heatmaps/image_' + str(i) +
                                             '_' + str(t) + '_'  +'_'.join(gt),
                                         img_tensor=grid, global_step=epoch,
                                         dataformats='CHW')
                        y_scores = nn.Softmax()(logits_cl[t].detach())
                        _, predicted_categories = y_scores.topk(num_of_labels)
                        predicted_cl = [(categories[x], format(y_scores.view(-1)[x], '.4f')) for x in
                                        predicted_categories.view(-1)]
                        labels_cl = [(categories[x], format(y_scores.view(-1)[x], '.4f')) for x in labels[t]]
                        import itertools
                        predicted_cl = list(itertools.chain(*predicted_cl))
                        labels_cl = list(itertools.chain(*labels_cl))
                        cl_text = 'cl_gt_' + '_'.join(labels_cl) + '_pred_' + '_'.join(predicted_cl)

                        predicted_am = [(categories[x], format(am_scores[t].view(-1)[x], '.4f')) for x in
                                        batch_am_lables[t].view(-1)]
                        labels_am = [(categories[x], format(am_scores.view(-1)[x], '.4f')) for x in labels[t]]
                        import itertools
                        predicted_am = list(itertools.chain(*predicted_am))
                        labels_am = list(itertools.chain(*labels_am))
                        am_text = '_am_gt_' + '_'.join(labels_am) + '_pred_' + '_'.join(predicted_am)

                        writer.add_text('Train_Heatmaps_Description/image_' + str(i) + '_' + str(t) + '_' +
                                        '_'.join(gt), cl_text + am_text, global_step=epoch)
                i += 1

                if epoch == 0 and test_first == False:
                    num_train_samples += 1
                if epoch == 1 and test_first == True:
                    num_train_samples += 1

                if i % epoch_size == 0:
                    break


        model.train(False)
        j = 0

        for sample in rds.datasets['seq_test']:

            batch, _ = preprocess_image(sample[0][0].squeeze().cpu().detach().numpy(), train=False, mean=mean, std=std)
            for img in sample[0][1:]:
                input_tensor, input_image = preprocess_image(img.squeeze().cpu().detach().numpy(), train=False,
                                                             mean=mean, std=std)
                batch = torch.cat((batch, input_tensor), dim=0)
            batch = batch.to(device)
            labels = sample[2]

            logits_cl, logits_am, heatmap, scaled_ac, masked_image, mask = gain(batch, sample[1])

            am_scores = nn.Softmax(dim=1)(logits_am)
            batch_am_lables = []
            for k in range(len(batch)):
                num_of_labels = len(sample[2][k])
                _, am_labels = am_scores[k].topk(num_of_labels)
                batch_am_lables.append(am_labels)


            # Single label evaluation


            for k in range(len(batch)):
                num_of_labels = len(sample[2][k])
                _, y_pred = logits_cl[k].detach().topk(k=num_of_labels)
                y_pred = y_pred.view(-1)
                gt = torch.tensor(sorted(sample[2][k]), device=device)

                correct_label_counter = 0
                total_picked_gt_label = 0
                if picked_categories:
                    for gt_label in gt:
                        if gt_label in picked_categories:
                            total_picked_gt_label += 1
                        if gt_label in y_pred and gt_label in picked_categories:
                            correct_label_counter += 1
                    if total_picked_gt_label == 0:
                        print("train: divided by 0")
                        print(correct_label_counter)
                        print(correct_label_counter != 0)
                        if correct_label_counter != 0:
                            total_train_single_accuracy += 0
                        else:
                            total_test_single_accuracy += 1
                    else:
                        print("train: acc")
                        print((correct_label_counter / total_picked_gt_label))
                        total_test_single_accuracy += (correct_label_counter / total_picked_gt_label)
                else:
                    acc = acc.detach().cpu() / len(gt)
                    total_test_single_accuracy += acc
                    if acc.detach().cpu() > 1:
                        print("Invalid acc")
                        print(acc.detach().cpu())

            if j % args.record_itr_test == 0:
                for t in range(args.nrecord):
                    num_of_labels = len(sample[2][t])
                    one_heatmap = heatmap[t].squeeze().cpu().detach().numpy()
                    one_input_image = sample[0][t].cpu().detach().numpy()
                    one_masked_image = masked_image[t].detach().squeeze()
                    htm = deprocess_image(one_heatmap)
                    visualization, heatmap = show_cam_on_image(one_input_image, htm, True)
                    viz = torch.from_numpy(visualization).unsqueeze(0).to(device)
                    augmented = torch.tensor(one_input_image).unsqueeze(0).to(device)
                    masked_image = denorm(one_masked_image, mean, std)
                    masked_image = (
                            masked_image.squeeze().permute([1, 2, 0]).cpu().detach().numpy() * 255).round().astype(
                        np.uint8)
                    orig = sample[0][t].unsqueeze(0)
                    masked_image = torch.from_numpy(masked_image).unsqueeze(0).to(device)
                    orig_viz = torch.cat((orig, augmented, viz, masked_image), 0)
                    grid = torchvision.utils.make_grid(orig_viz.permute([0, 3, 1, 2]))
                    gt = [categories[x] for x in labels[t]]
                    writer.add_image(tag='Test_Heatmaps/image_' + str(j) + '_' + '_'.join(gt),
                                     img_tensor=grid, global_step=epoch,
                                     dataformats='CHW')
                    y_scores = nn.Softmax()(logits_cl[0].detach())
                    _, predicted_categories = y_scores.topk(num_of_labels)
                    predicted_cl = [(categories[x], format(y_scores.view(-1)[x], '.4f')) for x in
                                    predicted_categories.view(-1)]
                    labels_cl = [(categories[x], format(y_scores.view(-1)[x], '.4f')) for x in labels[t]]
                    import itertools
                    predicted_cl = list(itertools.chain(*predicted_cl))
                    labels_cl = list(itertools.chain(*labels_cl))
                    cl_text = 'cl_gt_' + '_'.join(labels_cl) + '_pred_' + '_'.join(predicted_cl)

                    predicted_am = [(categories[x], format(am_scores[0].view(-1)[x], '.4f')) for x in
                                    batch_am_lables[0].view(-1)]
                    labels_am = [(categories[x], format(am_scores.view(-1)[x], '.4f')) for x in labels[t]]
                    import itertools
                    predicted_am = list(itertools.chain(*predicted_am))
                    labels_am = list(itertools.chain(*labels_am))
                    am_text = '_am_gt_' + '_'.join(labels_am) + '_pred_' + '_'.join(predicted_am)

                    writer.add_text('Test_Heatmaps_Description/image_' + str(j) + '_' + '_'.join(gt),
                                    cl_text + am_text, global_step=epoch)



            j += 1

        num_test_samples = len(rds.datasets['seq_test'])*batch_size
        # print("finished epoch number:")
        # print(epoch)
        # print(num_train_samples)
        # print(batch_size)
        if (test_first and epoch > 0) or test_first == False:
            writer.add_scalar('Loss/train/cl_total_loss', epoch_train_cl_loss / (num_train_samples*batch_size), epoch)
            writer.add_scalar('Loss/train/am_tota_loss', epoch_train_am_loss / num_train_samples, epoch)
            writer.add_scalar('Loss/train/combined_total_loss', epoch_train_total_loss / num_train_samples, epoch)
            writer.add_scalar('Accuracy/train/cl_accuracy', total_train_single_accuracy / (num_train_samples*batch_size), epoch)


        writer.add_scalar('Accuracy/test/cl_accuracy', total_test_single_accuracy / num_test_samples, epoch)
        
        gain.increase_epoch_count()

        if len(args.checkpoint_file_path_save) > 0 and epoch % args.checkpoint_nepoch == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration' : i,
                'num_train_samples': num_train_samples
            }, args.checkpoint_file_path_save + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
