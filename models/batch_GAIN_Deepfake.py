import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
import numpy as np
import PIL.Image

def is_bn(m):
    return isinstance(m, nn.modules.batchnorm.BatchNorm2d) | isinstance(m, nn.modules.batchnorm.BatchNorm1d)


def take_bn_layers(model):
    for m in model.modules():
        if is_bn(m):
            yield m


class FreezedBnModel(nn.Module):
    def __init__(self, model, is_train=True):
        super(FreezedBnModel, self).__init__()
        self.model = model
        self.bn_layers = list(take_bn_layers(self.model))

    def forward(self, x):
        is_train = self.bn_layers[0].training
        if is_train:
            self.set_bn_train_status(is_train=False)
        predicted = self.model(x)
        if is_train:
            self.set_bn_train_status(is_train=True)

        return predicted

    def set_bn_train_status(self, is_train: bool):
        for layer in self.bn_layers:
            layer.train(mode=is_train)
            layer.weight.requires_grad = is_train  # TODO: layer.requires_grad = is_train - check is its OK
            layer.bias.requires_grad = is_train


class batch_GAIN_Deepfake(nn.Module):
    def __init__(self, model, grad_layer, num_classes, fill_color,
                 am_pretraining_epochs=1, ex_pretraining_epochs=1,
                 test_first_before_train=False, grad_magnitude=1, last_ex_epoch=500):
        super(batch_GAIN_Deepfake, self).__init__()

        self.model = model

        # using freezed BN model configuration on the second path in AM training to not influence the statistics
        self.freezed_bn_model = FreezedBnModel(model)

        # print(self.model)
        self.grad_layer = grad_layer

        self.num_classes = num_classes
        self.fill_color = fill_color
        # print("before norm")
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        norm = Normalize(mean=mean, std=std)
        # print("before em fill")
        self.em_fill_color = torch.tensor([228.0 / 255.0, 249.0 / 255.0, 52.0 / 255.0]).view(1, 3, 1, 1).cuda()
        # Feed-forward features
        self.feed_forward_features = None
        # Backward features
        self.backward_features = None

        # Register hooks
        self._register_hooks(grad_layer)

        # sigma, omega for making the soft-mask
        self.sigma = 0.5 # 0.5
        self.omega = 30
        self.grad_magnitude = grad_magnitude

        self.am_pretraining_epochs = am_pretraining_epochs
        self.ex_pretraining_epochs = ex_pretraining_epochs
        self.last_ex_epoch = last_ex_epoch
        self.cur_epoch = 0
        if test_first_before_train == True:
            self.cur_epoch = -1
        self.enable_am = False
        self.enable_ex = False
        if self.am_pretraining_epochs == 0:
            self.enable_am = True
        if self.ex_pretraining_epochs == 0:
            self.enable_ex = True

        self.use_hook = 0

    def _register_hooks(self, grad_layer):
        def forward_hook(module, input, output):
            self.feed_forward_features = output

        def backward_hook(module, grad_input, grad_output):
            self.backward_features = grad_output[0]

        gradient_layer_found = False
        # print("print named modules")
        for idx, m in self.model.named_modules():
            # print("idx")
            # print(idx)
            # print("m")
            # print(m)
            if idx in self.grad_layer:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                print("Register forward hook !")
                print("Register backward hook !")
                gradient_layer_found = True
                # break
        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

    def _to_ohe_multibatch(self, labels):

        ohe = torch.nn.functional.one_hot(labels, self.num_classes).float()
        ohe = torch.autograd.Variable(ohe)

        return ohe

    def forward(self, images, labels, train_flag=False, image_with_masks=None,
                e_masks=None, has_mask_indexes=None):  # TODO: no need for saving the hook results ; Put Nan

        # Remember, only do back-probagation during the training. During the validation, it will be affected by bachnorm
        # dropout, etc. It leads to unstable validation score. It is better to visualize attention maps at the testset

        is_train = self.model.training

        with torch.enable_grad():
            # labels_ohe = self._to_ohe(labels).cuda()
            # labels_ohe.requires_grad = True

            _, _, img_h, img_w = images.size()

            logits_cl = self.model(images)  # BS x num_classes
            self.model.zero_grad()

            if not is_train:
                pred = F.softmax(logits_cl).argmax(dim=1)
                # print("pred")
                # print(pred)
                # print(F.softmax(logits_cl))
                labels_ohe = self._to_ohe_multibatch(pred).cuda()
                # print("labels_ohe")
                # print(labels_ohe)
            else:
                if type(labels) is tuple:
                    labels_ohe = torch.stack(labels)
                else:
                    labels_ohe = labels

            # gradient = logits * labels_ohe
            # old version: grad_logits = (logits_cl * labels_ohe).sum(dim=1)  # BS x num_classes
            # print("old grad_logits")
            # print((logits_cl * labels_ohe).sum(dim=1))
            # new version:
            grad_logits = logits_cl.sum(dim=1)  # BS x num_classes
            # print("new grad_logits")
            # print(grad_logits)
            # exit(1)
            grad_logits.backward(retain_graph=True, gradient=torch.ones_like(grad_logits))
            self.model.zero_grad()

        backward_features = self.backward_features  # BS x C x H x W
        fl = self.feed_forward_features  # BS x C x H x W
        weights = F.adaptive_avg_pool2d(backward_features, 1)
        Ac = torch.mul(fl, weights).sum(dim=1, keepdim=True)
        # print("Ac.shape")
        # print(Ac.shape)
        Ac = F.relu(Ac)
        # Ac = F.interpolate(Ac, size=images.size()[2:], mode='bilinear', align_corners=False)
        Ac = F.interpolate(Ac, size=images.size()[2:], mode='bilinear')
        # print("interpolate.shape")
        # print(Ac.shape)
        Ac_min, _ = Ac.view(len(images), -1).min(dim=1)
        Ac_max, _ = Ac.view(len(images), -1).max(dim=1)
        import sys
        eps = torch.tensor(sys.float_info.epsilon).cuda()
        scaled_ac = (Ac - Ac_min.view(-1, 1, 1, 1)) / \
                    (Ac_max.view(-1, 1, 1, 1) - Ac_min.view(-1, 1, 1, 1)
                     + eps.view(1, 1, 1, 1))
        # print("scaled_ac.shape")
        # print(scaled_ac.shape)
        mask = torch.sigmoid(self.omega * (scaled_ac - self.sigma))
        torch.set_printoptions(linewidth=200)
        print(torch.unique(mask))

        # print("mask.shape")
        # print(mask.shape) # 20, 1, 224, 224
        # print("images.shape")
        # print(images.shape) # 20, 3, 224, 224
        masked_image = images - images * mask + mask * self.fill_color
        PIL.Image.fromarray(
            (masked_image[0].cpu().detach().numpy() * 255).round().astype(
                np.uint8), 'RGB').save("masked_em3.png")
        exit(1)
        # print("masked_image.shape")
        # print(masked_image.shape)
        # masked_image.register_hook(lambda grad: grad * self.grad_magnitude)
        # #TODO: use this to control gradient magnitude

        # for param in self.model.parameters(): #TODO: use this to control set gradients on/off
        #    param.requires_grad = False

        logits_am = self.freezed_bn_model(masked_image)

        logits_em = 0
        if train_flag and len(e_masks) > 0:
            e_masks = tuple(map(torch.stack, zip(e_masks)))
            torch_masks = torch.stack(e_masks, dim=0).squeeze(1).to(torch.device('cuda:' + str(0)))
            # em_mask size [2, 1, 224, 224]
            em_mask = torch.sigmoid(self.omega * (torch_masks - self.sigma))
            # Option 1: use only H
            merged_mask = em_mask
            # Option 2: merge A and H:
            merged_mask = torch.maximum(em_mask, mask[has_mask_indexes, :, :, :])
            # merged_mask2 = em_mask + mask[has_mask_indexes, :, :, :]

            # em_masked_image size [2, 3, 224, 224]
            # em_masked_image = image_with_masks * em_mask + (torch.ones(em_mask.shape).to(torch.device('cuda:' + str(0)))
            #                                                 - em_mask) * self.em_fill_color
            em_masked_image = image_with_masks * merged_mask + (torch.ones(merged_mask.shape).to(
                torch.device('cuda:' + str(0))) - merged_mask) * self.em_fill_color
            # import PIL.Image
            # import numpy as np
            # PIL.Image.fromarray((image_with_masks[0].permute([1, 2, 0]).cpu().detach().numpy() * 255).round().astype(
            #     np.uint8), 'RGB').save('/home/shuoli/image.png')
            # PIL.Image.fromarray(((image_with_masks * em_mask)[0].cpu().detach().numpy() * 255).round().astype(
            #     np.uint8), 'RGB').save('/home/shuoli/image_times_mask.png')
            # PIL.Image.fromarray(((self.fill_color * em_mask)[0].cpu().detach().numpy() * 255).round().astype(
            #     np.uint8), 'RGB').save('/home/shuoli/fill_times_mask.png')
            # PIL.Image.fromarray(((merged_mask2[0][0]).cpu().detach().numpy() * 255).round().astype(
            #     np.uint8), 'L').save('/home/shuoli/merged_mask_add.png')
            # PIL.Image.fromarray(((merged_mask[0][0]).cpu().detach().numpy() * 255).round().astype(
            #     np.uint8), 'L').save('/home/shuoli/merged_mask_max.png')
            # PIL.Image.fromarray((em_masked_image1[0].permute([1, 2, 0]).cpu().detach().numpy() * 255)
            #                     .round().astype(np.uint8), 'RGB').save('/home/shuoli/maskedNew.png')
            # PIL.Image.fromarray((em_masked_image[0].permute([1, 2, 0]).cpu().detach().numpy() * 255)
            #                     .round().astype(np.uint8), 'RGB').save('/home/shuoli/maskedOld.png')
            # exit(0)
            logits_em = self.model(em_masked_image)  # [2, 1]

            # for param in self.model.parameters(): #TODO: use this to control set gradients on/off
        #    param.requires_grad = True

        # logits_am.register_hook(lambda grad: grad / self.grad_magnitude) #TODO: use this to control gradient magnitude

        return logits_cl, logits_am, scaled_ac, mask, masked_image, logits_em

    def increase_epoch_count(self):
        self.cur_epoch += 1
        if self.cur_epoch >= self.am_pretraining_epochs:
            self.enable_am = True
        if self.cur_epoch >= self.ex_pretraining_epochs:
            self.enable_ex = True
        if self.cur_epoch >= self.last_ex_epoch:
            self.enable_ex = False

    def AM_enabled(self):
        return self.enable_am

    def EX_enabled(self):
        return self.enable_ex
