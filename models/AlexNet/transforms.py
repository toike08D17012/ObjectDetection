from PIL import Image
from torchvision import transforms


class ImageTransformRecognision():
    """
    画像認識タスクに対するtransform
    ImageTransformObjectDetectionとの違いは、画像を切り抜くかどうか
    """
    def __init__(self, size, mean, std, mode='train'):
        if mode == 'train':
            self.data_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomResizedCrop(size=size, scale=(0.5, 1)),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean=mean, std=std)
                ]
            )
        elif mode == 'validation' or mode == 'val':
            self.data_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(size),
                    transforms.CenterCrop(size),    # RandomCropでもよいかも？
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

    def __call__(self, image: Image):
        """
        Args:
            image[Image or np.ndarray]: pillowのImage型 or np.ndarray形式の画像
        """
        return self.data_transform(image)


class ImageTransformObjectDetection():
    def __init__(self, size, mean, std) -> None:
        self.data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=mean, std=std)
            ]
        )

    def __call__(self, image: Image):
        """
        Args:
            image[Image or np.ndarray]: pillowのImage型 or np.ndarray形式の画像
        """
        return self.data_transform(image)
