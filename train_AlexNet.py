import torch.nn as nn
import torch.optim as optim

from models.AlexNet.AlexNet import AlexNet
from models.AlexNet.data_loader import create_dataset_ILSVRC
from utils.trainer import Trainer

TRAIN_EPOCH = 250
BATCH_SIZE = 512


def train():
    model = AlexNet(num_classes=1000)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RAdam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    train_dataloader, validation_dataloader = create_dataset_ILSVRC(batch_size=BATCH_SIZE, pin_memory=False, num_workers=20, pre_download_annotation=True)

    trainer = Trainer(model_name='AlexNet', model=model, visualize_interval=1, save_model_interval=1)
    trainer.train_model(
        train_epoch=TRAIN_EPOCH,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_dataloader,
        validation_loader=validation_dataloader
    )


if __name__ == '__main__':
    train()
