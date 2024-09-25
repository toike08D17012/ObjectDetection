from logging import getLogger, StreamHandler, Formatter,INFO, DEBUG
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

DEFAULT_LOG_DIR = Path('./logs')

class Trainer():
    def __init__(self, model_name:str, model: torch.nn.Module, resume=False, checkpoint_path=None, log_dir=None, visualize_interval=1, save_model_interval=1, logger=None) -> None:
        self.model_name = model_name
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = model.to(device=self.device)

        assert not resume or checkpoint_path is not None, 'If resume is True, must be set checkpoint path'
        if resume:
            weights = torch.load(checkpoint_path)
            model.load_state_dict(weights)

        if log_dir is None:
            log_dir = DEFAULT_LOG_DIR / model_name
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.checkpoint_dir = self.log_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.visualized_result_dir = self.log_dir / 'figures'
        self.visualized_result_dir.mkdir(exist_ok=True)

        self.visualize_interval = visualize_interval
        self.save_model_interval = save_model_interval

        if logger is None:
            logger = getLogger(__name__)
            logger.setLevel(INFO)
            # logger.setLevel(DEBUG)
            if not logger.hasHandlers():
                handler = StreamHandler()
                formatter = Formatter('[%(asctime)s] %(levelname)s : %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
        self.logger = logger

        self.train_loss_list = []
        self.train_acc_list = []
        self.validation_loss_list = []
        self.validation_acc_list = []

    def train_model(
            self,
            train_epoch: int,
            optimizer: torch.optim.Optimizer,
            criterion,
            train_loader,
            validation_loader
        ):
        train_data_num = len(train_loader.dataset)
        validation_data_num = len(validation_loader.dataset)

        self.logger.info('Start Training.')
        for epoch in range(train_epoch):
            train_loss = 0
            train_acc = 0
            validation_loss = 0
            validation_acc = 0

            # train
            self.model.train()
            for images, category_ids in tqdm(train_loader):
                self.logger.debug('Load to GPU.')
                # GPUメモリに移動
                images = images.to(self.device)
                category_ids = category_ids.to(self.device)

                optimizer.zero_grad()
                self.logger.debug('Calculation Model Output.')
                outputs = self.model(images)

                self.logger.debug('Calculation Loss.')
                loss = criterion(outputs, category_ids)

                train_loss += loss.item()
                train_acc += (outputs.max(dim=1)[1] == category_ids).sum().item()

                self.logger.debug('Back Propagation')
                loss.backward()
                optimizer.step()

            avg_train_loss = train_loss / train_data_num
            avg_train_acc = train_acc / train_data_num

            # validation
            self.model.eval()
            with torch.no_grad():
                for images, category_ids in validation_loader:
                    # GPUメモリに移動
                    images = images.to(self.device)
                    category_ids = category_ids.to(self.device)

                    outputs = self.model(images)

                    loss = criterion(outputs, category_ids)

                    validation_loss += loss.item()
                    validation_acc += (outputs.max(dim=1)[1] == category_ids).sum().item()

            avg_validation_loss = validation_loss / validation_data_num
            avg_validation_acc = validation_acc / validation_data_num

            self.logger.info(f'Epoch [{epoch+1}/{train_epoch}], Loss: {avg_train_loss:.4f}, val_loss: {avg_validation_loss:.4f}, val_acc: {avg_validation_acc:.4f}')

            self.train_loss_list.append(avg_train_loss)
            self.train_acc_list.append(avg_train_acc)
            self.validation_loss_list.append(avg_validation_loss)
            self.validation_acc_list.append(avg_validation_acc)

            if epoch % self.save_model_interval == 0:
                torch.save(self.model.state_dict(), self.checkpoint_dir / f'{epoch}.pth')

            if epoch % self.visualize_interval == 0:
                save_result_path = self.visualized_result_dir / f'{epoch}.jpg'
                self._visualize(save_result_path)
        
        torch.save(self.model.state_dict(), self.checkpoint_dir / f'{self.model_name}.pth')

    def _visualize(self, save_result_path):
        """
        左にloss, 右にaccuracyを可視化
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        
        # 左にlossを描画
        ax = axes[0]
        ax.plot(self.train_loss_list, label='train loss')
        ax.plot(self.validation_loss_list, label='validation loss')

        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title('Training and validation loss')

        ax.grid()
        ax.legend()

        # 右にaccuracyを描画
        ax = axes[1]
        ax.plot(self.train_acc_list, label='train accuracy')
        ax.plot(self.validation_acc_list, label='validation accuracy')

        ax.set_xlabel('epoch')
        ax.set_ylabel('accuracy')
        ax.set_title('Training and validation accuracy')
        ax.grid()
        ax.legend()

        fig.tight_layout()

        fig.savefig(save_result_path)

        plt.clf()
        plt.close()
