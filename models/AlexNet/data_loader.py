import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import OriginalDataset
from utils.transforms import ImageTransfrom

DEFAULT_DATASET_SAVE_DIR = Path('D:/workspace/ObjectDetection/dataset/CIFAR-10')
CIFAR10_LABELS = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

DEFAULT_ILSVRC_DIR = Path('D:/workspace/ObjectDetection/dataset/ILSVRC/')

IMAGE_SIZE = 224
IMAGE_NET_MEAN = (103.6660007 , 116.34035908, 122.08825076)
IMAGE_NET_STD = (0.95279436, 0.87966153, 0.9571214)


def create_dataset_CIFAR10(dataset_save_dir=DEFAULT_DATASET_SAVE_DIR, download=True, batch_size=256, num_workers=2):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    # transform = transforms.ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(root=dataset_save_dir, train=True, download=download, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(root=dataset_save_dir, train=False, download=download, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, test_loader, CIFAR10_LABELS


def create_dataset_ILSVRC(data_dir=DEFAULT_ILSVRC_DIR, batch_size=256, num_workers=2, pin_memory=True, pre_download_annotation=True, pre_download_image=False):
    train_dataset = ILSVRC(data_dir=data_dir, mode='train', pre_download_image=pre_download_image, pre_download_annotation=pre_download_annotation)
    validation_dataset = ILSVRC(data_dir=data_dir, mode='validation', pre_download_image=pre_download_image, pre_download_annotation=pre_download_annotation)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=pin_memory)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=pin_memory)

    return train_dataloader, validation_dataloader


class ILSVRC(OriginalDataset):
    def __init__(
            self,
            data_dir=DEFAULT_ILSVRC_DIR,
            mode='train',
            transform=None,
            image_suffix='.npy',
            image_size=IMAGE_SIZE,
            image_mean=IMAGE_NET_MEAN,
            image_std=IMAGE_NET_STD,
            pre_download_annotation=False,
            pre_download_image=False
        ) -> None:
        """
        Args:
            data_dir[Path]: datasetへのパス
            mode[str]: train, validation or test
                       
        """
        print(f'{mode} loader')
        super().__init__()

        if mode == 'validation':
            mode = 'val'

        if transform is None:
            self.transform = ImageTransfrom(size=image_size, mean=image_mean, std=image_std, mode=mode)
        else:
            self.transform = transform

        self.image_suffix = image_suffix

        self.pre_download_annotation = pre_download_annotation
        self.annotation_paths = self._get_annotation_paths(data_dir=data_dir / mode, suffix='.json')
        if self.pre_download_annotation:
            self._get_annotations()

        self.pre_download_image = pre_download_image
        if self.pre_download_image:
            self.image_paths = self._get_image_paths(data_dir=data_dir / mode, suffix='.jpg')
            self._get_images()

        self.data_dir = str(data_dir / mode)
        if self.data_dir[-1] != '/':
            self.data_dir += '/'

    def __len__(self):
        return len(self.annotation_paths)
    
    def _get_annotations(self):
        self.annotation_list = []
        for annotation_path in tqdm(self.annotation_paths, desc='Getting annotations'):
            with open(annotation_path) as f:
                annotation = json.load(f)

            self.annotation_list.append(annotation['category_id'])

        self.annotation_list = torch.tensor(self.annotation_list)

    def _get_images(self):
        self.images = [cv2.imread(str(image_path)).transpose(2, 0, 1) for image_path in tqdm(self.image_paths, desc='Getting images')]

    def __getitem__(self, index):
        str_index = str(index).zfill(8)
        if self.pre_download_annotation:
            annotation = self.annotation_list[index]
        else:
            annotation_path = self.data_dir + str_index + '.json'
            with open(annotation_path) as f:
                annotation = json.load(f)['category_id']

        if self.pre_download_image:
            image = self.images[index]
        else:
            image_path = self.data_dir + str_index + self.image_suffix
            if self.image_suffix == '.npy':
                image = torch.tensor(np.load(image_path).transpose(2, 0, 1), dtype=torch.float32)
            elif self.image_suffix == '.jpg':
                image = torch.tensor(cv2.imread(image_path).transpose(2, 0, 1), dtype=torch.float32)
        transformed_image = self.transform(image)
        
        return transformed_image, annotation
