from random import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image import denorm


def is_bn(m):
    return isinstance(m, nn.modules.batchnorm.BatchNorm2d) | isinstance(m, nn.modules.batchnorm.BatchNorm1d)

def take_bn_layers(model):
    for m in model.modules():
        if is_bn(m):
            yield m
    return []

class FreezedBnModel(nn.Module):
    def __init__(self, model, is_train=True):
        super(FreezedBnModel, self).__init__()
        self.model = model
        self.bn_layers = list(take_bn_layers(self.model))


    def forward(self, x):
        is_train = len(self.bn_layers) > 0 and self.bn_layers[0].training
        if is_train:
            self.set_bn_train_status(is_train=False)
        predicted = self.model(x)
        if is_train:
            self.set_bn_train_status(is_train=True)

        return predicted

    def set_bn_train_status(self, is_train: bool):
        for layer in self.bn_layers:
            layer.train(mode=is_train)
            layer.weight.requires_grad = is_train #TODO: layer.requires_grad = is_train - check is its OK
            layer.bias.requires_grad = is_train


class batch_GAIN_VOC(nn.Module):
    def __init__(self, model, grad_layer, device, num_classes, am_pretraining_epochs=1, ex_pretraining_epochs=1,
                 test_first_before_train=False, grads_off=False, grads_magnitude=1):
        super(batch_GAIN_VOC, self).__init__()

        self.model = model

        self.device = device

        self.freezed_bn_model = FreezedBnModel(model)

        # print(self.model)
        self.grad_layer = grad_layer

        self.num_classes = num_classes

        self.grads_off = grads_off
        self.grads_magnitude = grads_magnitude
        if grads_off:
            self.grads_magnitude = 1

        # Feed-forward features
        self.feed_forward_features = None
        # Backward features
        self.backward_features = None

        # Register hooks
        self._register_hooks(grad_layer)

        # sigma, omega for making the soft-mask
        self.sigma = 0.5
        self.omega = 10

        self.am_pretraining_epochs = am_pretraining_epochs
        self.ex_pretraining_epochs = ex_pretraining_epochs
        self.cur_epoch = 0
        if test_first_before_train == True:
            self.cur_epoch = -1
        self.enable_am = False
        self.enable_ex = False
        if self.am_pretraining_epochs == 0:
            self.enable_am = True
        if self.ex_pretraining_epochs == 0:
            self.enable_ex = True

    def _register_hooks(self, grad_layer):
        def forward_hook(module, input, output):
            self.feed_forward_features = output

        def backward_hook(module, grad_input, grad_output):
            self.backward_features = grad_output[0]

        gradient_layer_found = False
        print("print named modules")
        for idx, m in self.model.named_modules():
            print("idx")
            print(idx)
            print("m")
            print(m)
            if idx == grad_layer:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                print("Register forward hook !")
                print("Register backward hook !")
                gradient_layer_found = True
                # break
        exit(0)
        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

    def _to_ohe(self, labels):

        ohe = torch.nn.functional.one_hot(labels, self.num_classes).sum(dim=0).unsqueeze(0).float()
        ohe = torch.autograd.Variable(ohe)

        return ohe

    def _to_ohe_multibatch(self, labels):

        ohe = torch.nn.functional.one_hot(labels, self.num_classes).float()
        ohe = torch.autograd.Variable(ohe)

        return ohe

    def forward(self, images, labels): #TODO: no need for saving the hook results ; Put Nan

        # Remember, only do back-probagation during the training. During the validation, it will be affected by bachnorm
        # dropout, etc. It leads to unstable validation score. It is better to visualize attention maps at the testset

        is_train = self.model.training

        with torch.enable_grad():
            # labels_ohe = self._to_ohe(labels).cuda()
            # labels_ohe.requires_grad = True

            _, _, img_h, img_w = images.size()

            self.model.train(is_train)  # TODO: use is_train
            logits_cl = self.model(images)  # BS x num_classes
            self.model.zero_grad()

            if not is_train:
                pred = F.softmax(logits_cl).argmax(dim=1)
                labels_ohe = self._to_ohe_multibatch(pred).to(self.device)
            else:
                if type(labels) is tuple or type(labels) is list:
                    labels_ohe = torch.stack(labels)
                else:
                    labels_ohe = labels

            # gradient = logits * labels_ohe
            grad_logits = (logits_cl * labels_ohe).sum(dim=1)  # BS x num_classes
            grad_logits.backward(retain_graph=True, gradient=torch.ones_like(grad_logits))
            self.model.zero_grad()

        backward_features = self.backward_features  # BS x C x H x W
        fl = self.feed_forward_features  # BS x C x H x W
        weights = F.adaptive_avg_pool2d(backward_features, 1)
        Ac = torch.mul(fl, weights).sum(dim=1, keepdim=True)
        Ac = F.relu(Ac)
        # Ac = F.interpolate(Ac, size=images.size()[2:], mode='bilinear', align_corners=False)
        Ac = F.interpolate(Ac, size=images.size()[2:], mode='bilinear')
        heatmap = Ac

        Ac_min, _ = Ac.view(len(images), -1).min(dim=1)
        Ac_max, _ = Ac.view(len(images), -1).max(dim=1)
        import sys
        eps = torch.tensor(sys.float_info.epsilon).to(self.device)
        scaled_ac = (Ac - Ac_min.view(-1, 1, 1, 1)) / \
                    (Ac_max.view(-1, 1, 1, 1) - Ac_min.view(-1, 1, 1, 1)
                     + eps.view(1, 1, 1, 1))
        mask = torch.sigmoid(self.omega * (scaled_ac - self.sigma))
        masked_image = images - images * mask
        
        #masked_image.register_hook(lambda grad: grad * self.grad_magnitude)

        if self.grads_off:
            for param in self.model.parameters():
                param.requires_grad = False

        logits_am = self.freezed_bn_model(masked_image)

        if self.grads_off:
            for param in self.model.parameters():
                param.requires_grad = True
        
        #masked_image.register_hook(lambda grad: grad / self.grad_magnitude)

        return logits_cl, logits_am, heatmap, scaled_ac, masked_image, mask

    def increase_epoch_count(self):
        self.cur_epoch += 1
        if self.cur_epoch >= self.am_pretraining_epochs:
            self.enable_am = True
        if self.cur_epoch >= self.ex_pretraining_epochs:
            self.enable_ex = True

    def AM_enabled(self):
        return self.enable_am

    def EX_enabled(self):
        return self.enable_ex
