from torch.utils import data
from torch.utils.data import SequentialSampler, RandomSampler
import PIL.Image
import torch
import os
import numpy as np
import random

def build_balanced_dataloader(dataset, labels, collate_fn, target_weight=None, batch_size=1, steps_per_epoch=500, num_workers=1):
    assert len(dataset) == len(labels)
    labels = np.asarray(labels)
    ulabels, label_count = np.unique(labels, return_counts=True)
    assert (ulabels == list(range(len(ulabels)))).all()
    balancing_weight = 1 / label_count
    target_weight = target_weight if target_weight is not None else np.ones(len(ulabels))
    assert len(target_weight) == len(ulabels)

    from torch.utils.data import WeightedRandomSampler
    num_samples = steps_per_epoch * batch_size
    weighted_sampler = WeightedRandomSampler(
        weights=(target_weight * balancing_weight)[labels],
        num_samples=num_samples,
        replacement=False
    )
    loader = torch.utils.data.DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        sampler=weighted_sampler,
                        collate_fn=collate_fn)
    return loader


def load_func(path, file, all_files):
    label = 0 if 'Neg' in path else 1
    source = 'ffhq' if label == 0 else 'psi_1' if 'psi1' in file else 'psi_0.5'
    path_to_file = os.path.join(path, file)
    p_image = PIL.Image.open(path_to_file)
    np_image = np.asarray(p_image)
    tensor_image = torch.tensor(np_image)
    img_name, format = str(file).split('.')
    mask_file = img_name+'m'+'.'+format
    if all_files is not None and label == 1 and mask_file in all_files:
        path_to_mask = os.path.join(path, mask_file)
        p_mask = PIL.Image.open(path_to_mask).convert('RGB')
        np_mask = np.asarray(p_mask)
        tensor_mask = torch.tensor(np_mask)
        return tensor_image, tensor_mask, label, source, file
    return tensor_image, torch.tensor(-1), label, source, file


def load_tuple_func(path, file, all_files):
    label = 0 if 'Neg' in path else 1
    path_to_file = os.path.join(path, file)
    p_image = PIL.Image.open(path_to_file)
    np_image = np.asarray(p_image)
    tensor_image = torch.tensor(np_image)
    img_name, format = str(file).split('.')
    mask_file = img_name+'m'+'.'+format
    if all_files is not None and label == 1 and mask_file in all_files:
        path_to_mask = os.path.join(path, mask_file)
        p_mask = PIL.Image.open(path_to_mask).convert('RGB')
        np_mask = np.asarray(p_mask)
        tensor_mask = torch.tensor(np_mask)
        return tensor_image, tensor_mask, label
    return tensor_image, torch.tensor(-1), label

# return a list of (path, filename) tuple under softlinks
def get_files_under_folder(dir):
    path_file_tuple_under_folder = []
    files_under_folder = []
    list_softlinks = os.listdir(dir)
    for softlink in list_softlinks:
        if '.png' in softlink:
            path_file_tuple_under_folder += (dir, softlink)
            files_under_folder.append(softlink)
        if os.path.islink(softlink):
            abs_path_softlink = os.readlink(softlink)
            files = os.listdir(abs_path_softlink)
            for file in files:
                path_file_tuple_under_folder += (abs_path_softlink, file)
                files_under_folder.append(file)

    return path_file_tuple_under_folder, files_under_folder


class DeepfakeTrainData(data.Dataset):
    def __init__(self, masks_to_use, mean, std, transform, batch_size, steps_per_epoch, target_weight, customize_num_masks, num_masks, root_dir='train', loader=load_func):
        self.pos_root_dir = root_dir+'Pos/'
        self.neg_root_dir = root_dir + 'Neg/'
        all_neg_files = os.listdir(self.neg_root_dir)
        all_pos_files = os.listdir(self.pos_root_dir)

        # all_neg_files_tupes, all_neg_files = get_files_under_folder(self.neg_root_dir)
        # all_pos_files_tupes, all_pos_files = get_files_under_folder(self.pos_root_dir)

        pos_cl_images = [file for file in all_pos_files if 'm' not in file]

        # target_weight[1] -> positive ratio

        # if customize the num of masks to be picked in each epoch
        if customize_num_masks:
            mask_images = [(file[:-5] + '.png') for file in all_pos_files if 'm' in file]
            pos_cl_images_without_masks = [file for file in pos_cl_images if file not in mask_images]
            total_num_images = batch_size * steps_per_epoch
            total_num_masks = len(mask_images)
            total_num_pos_cl = (int)(total_num_images * target_weight[1])
            total_num_neg_files = total_num_images - total_num_pos_cl
            picked_mask_images = random.sample(mask_images, num_masks)
            all_pos_files = random.sample(pos_cl_images_without_masks, total_num_pos_cl - num_masks)
            all_neg_files = random.sample(all_neg_files, total_num_neg_files)
            pos_cl_images = picked_mask_images + all_pos_files
            picked_cl_with_masks = [(file[:-4] + 'm.png') for file in picked_mask_images]
            all_pos_files = pos_cl_images + picked_cl_with_masks

        # dummy masks creation:
        path_to_file = os.path.join(self.pos_root_dir, pos_cl_images[0])
        p_image = PIL.Image.open(path_to_file)
        np_image = np.asarray(p_image)

        tensor_image = torch.tensor(np_image)
        self.masks_indices = [idx for idx,pos in enumerate(pos_cl_images) if pos.split('.')[0]+'m'+'.png' in all_pos_files]
        self.all_files = all_pos_files + all_neg_files
        self.all_cl_images = pos_cl_images+all_neg_files
        self.pos_num_of_samples = len(pos_cl_images)
        self.loader = loader
        mask_max_idx = int(self.pos_num_of_samples * masks_to_use) # maximum num of masks ready to use, masks_to_use is the ratio of masked image to use over all pos cl images
        self.used_masks = self.masks_indices[:mask_max_idx]
        self.mean = mean
        self.std = std
        self.transform = transform
        self.dummy_mask = torch.tensor(np.zeros_like(np_image))


    def __len__(self):
        return len(self.all_cl_images)

    def __getitem__(self, index):
        if index < self.pos_num_of_samples:
            res = list(self.loader(self.pos_root_dir,
                                   self.all_cl_images[index], self.all_files))
            # original: mask=res[1].unsqueeze(0)
            if res[1].numel() > 1:
                preprocessed, augmented, augmented_mask = \
                    self.transform(img=res[0].squeeze().permute([2, 0, 1]),
                                   mask=res[1].squeeze().permute([2, 0, 1]), train=True,
                                   mean=self.mean, std=self.std)
            else:
                preprocessed, augmented, augmented_mask = \
                    self.transform(img=res[0].squeeze().permute([2, 0, 1]),
                                             mask=self.dummy_mask.squeeze().permute([2, 0, 1]), train=True,
                                             mean=self.mean, std=self.std)
            if index in self.used_masks:
                res = [res[0]] + [preprocessed] + [augmented] + [res[1]]+ \
                      [augmented_mask]+[True] + [res[2]] + [res[3]] + [res[4]]
            else:
                res = [res[0]] + [preprocessed] + [augmented] + [res[1]] +\
                      [augmented_mask]+[False] + [res[2]] + [res[3]] + [res[4]]
        else:
            res = list(self.loader(self.neg_root_dir,
                                   self.all_cl_images[index], None))
            preprocessed, augmented, augmented_mask = \
                self.transform(img=res[0].squeeze().permute([2, 0, 1]),
                                         mask=self.dummy_mask.squeeze().permute([2, 0, 1]), train=True,
                                         mean=self.mean, std=self.std)
            res = [res[0]] + [preprocessed] + [augmented] + [res[1]] + \
                  [np.array(-1)] + [False] + [res[2]] + [res[3]] + [res[4]]
        res.append(index)
        return res

    def positive_len(self):
        return self.pos_num_of_samples

    def get_masks_indices(self):
        return self.masks_indices


class DeepfakeValidationData(data.Dataset):
    def __init__(self, mean, std, transform, root_dir='validation', loader=load_func):
        self.pos_root_dir = root_dir+'Pos/'
        self.neg_root_dir = root_dir + 'Neg/'
        self.all_files = os.listdir(self.pos_root_dir) + \
                         os.listdir(self.neg_root_dir)
        self.pos_num_of_samples = len(os.listdir(self.pos_root_dir))
        self.loader = loader
        self.mean = mean
        self.std = std
        self.transform = transform

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        if index < self.pos_num_of_samples:
            res = list(self.loader(self.pos_root_dir,
                                   self.all_files[index], None))
        else:
            res = list(self.loader(self.neg_root_dir,
                                   self.all_files[index], None))
        preprocessed, augmented, _ = \
            self.transform(img=res[0].squeeze().numpy(),
                           train=False, mean=self.mean, std=self.std)
        res = [res[0]] + [preprocessed] + [augmented] + [res[1]] + [np.array(-1)] +\
              [False] + [res[2]] + [res[3]] + [res[4]]
        res.append(index)
        return res

    def positive_len(self):
        return self.pos_num_of_samples


class DeepfakeTestData(data.Dataset):
    def __init__(self, mean, std, transform, root_dir='test', loader=load_func):
        self.pos_root_dir = root_dir+'Pos/'
        self.neg_root_dir = root_dir + 'Neg/'
        self.all_files = os.listdir(self.pos_root_dir) + \
                         os.listdir(self.neg_root_dir)
        self.pos_num_of_samples = len(os.listdir(self.pos_root_dir))
        self.loader = loader
        self.mean = mean
        self.std = std
        self.transform = transform

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        if index < self.pos_num_of_samples:
            res = list(self.loader(self.pos_root_dir,
                                   self.all_files[index], None))
        else:
            res = list(self.loader(self.neg_root_dir,
                                   self.all_files[index], None))
        preprocessed, augmented, _ = \
            self.transform(img=res[0].squeeze().numpy(),
                           train=False, mean=self.mean, std=self.std)
        res = [res[0]] + [preprocessed] + [augmented] + [res[1]] + [np.array(-1)] +\
              [False] + [res[2]] + [res[3]] + [res[4]]
        res.append(index)
        return res

    def positive_len(self):
        return self.pos_num_of_samples


class DeepfakeLoader():
    def __init__(self, root_dir, target_weight, masks_to_use, mean, std,
                 transform, collate_fn, customize_num_masks, num_masks, batch_size=1, steps_per_epoch=6000,
                 num_workers=3):

        self.train_dataset = DeepfakeTrainData(root_dir=root_dir + 'training/',
                                               masks_to_use=masks_to_use,
                                               mean=mean, std=std,
                                               transform=transform, batch_size=batch_size, steps_per_epoch=steps_per_epoch, target_weight=target_weight, customize_num_masks=customize_num_masks, num_masks=num_masks)
        self.validation_dataset = DeepfakeValidationData(root_dir=root_dir + 'validation/',
                                                         mean=mean, std=std,
                                                         transform=transform)

        self.test_dataset = DeepfakeTestData(root_dir=root_dir + 'testing/',
                                             mean=mean, std=std,
                                             transform=transform)

        #train_sampler = RandomSampler(self.train_dataset, num_samples=maxint,
        #                              replacement=True)
        test_sampler = SequentialSampler(self.test_dataset)

        validation_sampler = SequentialSampler(self.validation_dataset)

        train_as_test_sampler = SequentialSampler(self.train_dataset)

        '''
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=1,
            num_workers=0,
            sampler=train_sampler)
        '''
        ones = torch.ones(self.train_dataset.positive_len())
        labels = torch.zeros(len(self.train_dataset))
        labels[0:len(ones)] = ones

        train_loader = build_balanced_dataloader(
                    self.train_dataset, labels.int(),
                    target_weight=target_weight, batch_size=batch_size,
                    steps_per_epoch=steps_per_epoch, num_workers=num_workers,
                    collate_fn=collate_fn)

        validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler=validation_sampler,
            collate_fn=collate_fn)

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler=test_sampler,
            collate_fn=collate_fn)

        train_as_test_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler=train_as_test_sampler)

        self.datasets = {'train': train_loader, 'validation': validation_loader, 'test': test_loader, 
                         'train_as_test': train_as_test_loader }

    def get_test_pos_count(self, train_as_test=False):
        if train_as_test:
            return self.train_dataset.pos_num_of_samples
        return self.validation_dataset.pos_num_of_samples

    def get_train_pos_count(self):
        return self.train_dataset.pos_num_of_samples


class DeepfakeTestingOnlyLoader():
    def __init__(self, root_dir, target_weight, masks_to_use, mean, std,
                 transform, collate_fn, batch_size=1, steps_per_epoch=6000,
                 num_workers=3):
        self.test_dataset = DeepfakeTestData(root_dir=root_dir + 'testing/',
                                             mean=mean, std=std,
                                             transform=transform)

        test_sampler = SequentialSampler(self.test_dataset)

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler=test_sampler,
            collate_fn=collate_fn)

        self.datasets = {'test': test_loader}
