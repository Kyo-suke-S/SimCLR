import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class Scratch(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def train(self, train_loader, val_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start Scratch training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            top1_train_accuracy = 0
            for counter, (x_batch, y_batch) in tqdm(enumerate(train_loader), disable=self.args.no_tqdm):
                x_batch = x_batch.to(self.args.device)
                y_batch = y_batch.to(self.args.device)

                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)
                top1 = accuracy(logits, y_batch, topk=(1,))
                top1_train_accuracy += top1[0]

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

            top1_train_accuracy /= (counter + 1)
            top1_accuracy = 0
            top5_accuracy = 0

            for counter, (x_batch, y_batch) in tqdm(enumerate(val_loader), disable=self.args.no_tqdm):
                x_batch = x_batch.to(self.args.device)
                y_batch = y_batch.to(self.args.device)

                logits = self.model(x_batch)
  
                top1, top5 = accuracy(logits, y_batch, topk=(1,5))
                top1_accuracy += top1[0]
                top5_accuracy += top5[0]
  
            top1_accuracy /= (counter + 1)
            top5_accuracy /= (counter + 1)              

            #if n_iter % self.args.log_every_n_steps == 0:
                #top1, top5 = accuracy(logits, labels, topk=(1, 5))
            self.writer.add_scalar('loss', loss, global_step=n_iter)
            self.writer.add_scalar('acc/top1', top1_accuracy, global_step=n_iter)
            self.writer.add_scalar('acc/top5', top5_accuracy, global_step=n_iter)
            self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

            #n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
