from torchvision import transforms


class ImageTransfrom():
    def __init__(self, size, mean, std, mode='train'):
        if mode == 'train':
            self.data_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=size, scale=(0.5, 1)),
                    transforms.RandomHorizontalFlip(),
                    # transforms.ToTensor(),
                    # transforms.Normalize(mean=mean, std=std)
                ]
            )
        elif mode == 'validation' or mode == 'val':
            self.data_transform = transforms.Compose(
                [
                    # transforms.Resize(size),
                    # transforms.CenterCrop(size),    # RandomCropでもよいかも？
                    # transforms.ToTensor(),
                    # transforms.Normalize(mean=mean, std=std),
                ]
            )

    def __call__(self, image):
        # print(self.data_transform)
        return self.data_transform(image)
