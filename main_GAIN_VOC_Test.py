import argparse
import torch
import torchvision
from torch import nn
from torchvision.models import vgg19
import numpy as np
import os
from configs.VOCconfig import cfg
from utils.image import show_cam_on_image, preprocess_image, deprocess_image, denorm
from models.batch_GAIN_VOC import batch_GAIN_VOC
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Parse all the input argument
parser = argparse.ArgumentParser(description='PyTorch GAIN Single Image Testing')
parser.add_argument('--file_name', type=str, help='file name', default='')
parser.add_argument('--device', type=str, help='device:ID', default='cuda:1') #for gpu use the string format of 'cuda:#', e.g: cuda:0
parser.add_argument('--dataset_path', type=str, help='the path to your local VOC 2012 dataset directory')
parser.add_argument('--logging_path', type=str, help='the path to your local logging directory') #recommended a big enough storage for many runs
parser.add_argument('--logging_name', type=str, help='a name for the current run for logging purposes') #recommended a big enough storage for many runs
parser.add_argument('--checkpoint_file_path_load', type=str, default='', help='a full path including the name of the checkpoint_file to load from, empty otherwise')
parser.add_argument('--test_first', type=int, help='0 for not testing before training in the beginning, 1 otherwise', default=0)
parser.add_argument('--cl_loss_factor', type=float, help='a parameter for the classification loss magnitude, constant 1 in the paper', default=1)
parser.add_argument('--am_loss_factor', type=float, help='a parameter for the AM (Attention-Mining) loss magnitude, alpha in the paper', default=1)
TODO: parser.add_argument('--e_loss_factor', type=float, help='a parameter for the extra supervision loss magnitude, omega in the paper', default=1)
parser.add_argument('--nepoch_am', type=int, default=1, help='number of epochs to train without am loss')
parser.add_argument('--nepoch_ex', type=int, default=1, help='number of epochs to train without ex loss')
parser.add_argument('--lr', type=float, help='learning rate', default=0.00001) #recommended 0.0001 for batchsiuze > 1, 0.00001 otherwise

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

    device = torch.device(args.device)
    model = vgg19(pretrained=True).train().to(device)

    # change the last layer for finetuning
    classifier = model.classifier
    num_ftrs = classifier[-1].in_features
    new_classifier = torch.nn.Sequential(*(list(model.classifier.children())[:-1]),
                                         nn.Linear(num_ftrs, num_classes).to(device))
    model.classifier = new_classifier
    model.train()

    # use values to normiulize the input as in torchvision ImageNet guide
    mean = cfg.MEAN
    std = cfg.STD


    test_first = bool(args.test_first)

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


    # load intermediate model from existing checkpoint file
    if len(args.checkpoint_file_path_load) > 0:
        checkpoint = torch.load(args.checkpoint_file_path_load)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # profiling
    writer = SummaryWriter(
        args.logging_path + args.logging_name + '_' +
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    writer.add_text('Start', 'start')

    print('Started')

    model.train(False)
    from PIL import Image
    from os.path import join as pjoin

    test_img = args.file_name  # file name
    dataset_path = os.path.join(args.dataset_path, 'VOCdevkit', 'VOC2012')
    # get sample[0]: padded_images->img->im
    im_path = pjoin(dataset_path, "JPEGImages", test_img + ".jpg")
    img = Image.open(pjoin(im_path))
    img = np.asarray(img)
    img = torch.tensor(img)
    im = torch.zeros((img.shape[0], img.shape[1])).unsqueeze(2).repeat([1, 1, 3])
    im[0:img.size()[0], 0:img.size()[1]] = img
    im = im.type(torch.ByteTensor)

    # get sample[2]: labels->sample['label/idx']->labels
    segm_path = pjoin(args.dataset_path, "SegmentationClass/pre_encoded")
    segm = Image.open(pjoin(segm_path, test_img + ".png"))
    truth = np.asarray(segm)
    # pixel values in truth array skipping pixel value 0(black color): value - 1 == pixel value mapping to its corresponding category
    classes = [categories[l] for l in np.unique(truth)[1:] - 1]
    class_onehot = np.array(
        [[1 if (c == class_) else 0 for c in categories]
         for class_ in classes])
    class_idx = [np.nonzero(c_onehot)[0][0] for c_onehot in class_onehot]
    labels = class_idx

    # get sample[1]: labels_ohe -> sample['label/onehot'] np array image->sample1
    sample1 = sum(class_onehot).to(device)

    # Single Image Testing

    padded = []
    padded.append(im)
    batch, augmented = preprocess_image(padded[0].squeeze().cpu().detach().numpy(), train=True, mean=mean, std=std)
    batch = batch.to(device)
    logits_cl, logits_am, heatmap, scaled_ac, masked_image, mask = gain(batch, sample1)
    am_scores = nn.Softmax(dim=1)(logits_am)
    batch_am_lables = []
    for k in range(len(batch)):
        num_of_labels = len(labels[k])
        _, am_labels = am_scores[k].topk(num_of_labels)
        batch_am_lables.append(am_labels)
    num_of_labels = len(labels)
    one_heatmap = heatmap[0].squeeze().cpu().detach().numpy()
    one_input_image = im.cpu().detach().numpy()
    one_masked_image = masked_image[0].detach().squeeze()
    htm = deprocess_image(one_heatmap)
    visualization, heatmap = show_cam_on_image(one_input_image, htm, True)
    viz = torch.from_numpy(visualization).unsqueeze(0).to(device)
    augmented = torch.tensor(one_input_image).unsqueeze(0).to(device)
    masked_im = denorm(one_masked_image, mean, std)
    masked_im = (masked_im.squeeze().permute([1, 2, 0])
                 .cpu().detach().numpy() * 255).round() \
        .astype(np.uint8)
    orig = im.unsqueeze(0)
    masked_im = torch.from_numpy(masked_im).unsqueeze(0).to(device)
    orig_viz = torch.cat((orig, augmented, viz, masked_im), 0)
    grid = torchvision.utils.make_grid(orig_viz.permute([0, 3, 1, 2]))
    # ground truth
    gt = [categories[x] for x in labels]
    orig = im.unsqueeze(0)
    orig_viz = torch.cat((orig, augmented, viz, masked_image), 0)
    grid = torchvision.utils.make_grid(orig_viz.permute([0, 3, 1, 2]))
    writer.add_image(tag='Test_Heatmaps/image_' + '_' + '_'.join(gt),
                     img_tensor=grid,
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

    writer.add_text('Test_Heatmaps_Description/image_' + '_' + '_'.join(gt),
                    cl_text + am_text)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
