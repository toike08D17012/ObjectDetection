from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms


class OriginalDataset(Dataset):
    def __init__(self):
        super().__init__()

    def _get_image_paths(self, data_dir: Path, suffix: str = '.jpg') -> list[Path]:
        """
        指定した階層より下位のディレクトリ内の全画像のパスを取得
        Args:
            dir_path[Path]: 探索するディレクトリへのパス
            suffix[str]: 画像の拡張子
        Returns:
            ...[list[Path]]: 画像のパスのリスト
        """
        return self._get_paths(data_dir=data_dir, suffix=suffix)
    
    def _get_annotation_paths(self, data_dir: Path, suffix: str = '.jpg') -> list[Path]:
        """
        指定した階層より下位のディレクトリ内の全アノテーションデータのパスを取得
        Args:
            data_dir[Path]: 探索するディレクトリへのパス
            suffix[str]: アノテーションファイルの拡張子
        Returns:
            ...[list[Path]]: アノテーションデータのパスのリスト
        """
        return self._get_paths(data_dir=data_dir, suffix=suffix)


    @staticmethod
    def _get_paths(data_dir: Path, suffix: str) -> list[Path]:
        """
        指定した階層より下位のディレクトリ内の指定拡張子のファイルパスをすべて取得
        Args:
            data_dir[Path]: 探索するディレクトリへのパス
            suffix[str]: 探索するファイルの拡張子
        Returns:
            ...[list[Path]]: 指定拡張子のファイルのパスリスト
        """
        # 指定した拡張子に"."がついていなければつける
        if '.' not in suffix:
            suffix = '.' + suffix

        return sorted(data_dir.glob(f'**/*{suffix}'))