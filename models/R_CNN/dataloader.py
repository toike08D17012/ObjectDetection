import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

DEFAULT_DATASET_SAVE_DIR = Path('../../dataset')


def create_dataset_VOC(dataset_save_dir=DEFAULT_DATASET_SAVE_DIR, download=True, batch_size=256, num_workers=2, year='2007', transform=None):
    """
    Pascal VOCのデータを扱うdatasetクラスを作成する関数
    Args:
        dataset_save_dir[Path or str]: datasetが保存される/されているパス
        download[bool]: データセットをダウンロードするかどうかを切り替えるフラグ
        batch_size[int]: バッチサイズ
        num_workers[int]: workerの数
        year[str]: どの年のデータセットか(2007, 2008, 2009, 2010, 2011, 2012)
        transform[torchvision.transforms]: transforms
    """
    train_dataset = torchvision.datasets.VOCDetection(root=dataset_save_dir, year=year, train=True, download=download, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    test_dataset = torchvision.datasets.VOCDetection(root=dataset_save_dir, year=year, train=False, download=download, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, test_loader
