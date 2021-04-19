import os
import sys
import time
import json
import logging
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from model import GestureTranslator
from data import AirTypingDataset
from options import AirTypingOptions
from utils import WarmupMultiStepLR, Lamb, sec_to_hm_str, cer, calc_seq_len, calc_seq_len_mc3, calc_seq_len_rmc3, seq_len_r3d_kor, seq_len_mc3_kor, seq_len_rmc3_kor


options = AirTypingOptions()
opts = options.parse()


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    if opts.data_type == 'english' or (opts.data_type == 'korean' and opts.num_res_layer == 1):
        if opts.model_type == 'r3d' or opts.model_type == 'twoplusone':
            x_lens = torch.LongTensor([calc_seq_len(len(x)) for x in xx])
        elif opts.model_type == 'rmc3':
            x_lens = torch.LongTensor([calc_seq_len_rmc3(len(x)) for x in xx])
        elif opts.model_type == 'mc3':
            x_lens = torch.LongTensor([calc_seq_len_mc3(len(x)) for x in xx])

    elif opts.data_type == 'korean' and opts.num_res_layer == 2:
        if opts.model_type == 'r3d':
            x_lens = torch.LongTensor([seq_len_r3d_kor(len(x)) for x in xx])
        elif opts.model_type == 'twoplusone':
            x_lens = torch.LongTensor([calc_seq_len(len(x)) for x in xx])
        elif opts.model_type == 'rmc3':
            x_lens = torch.LongTensor([seq_len_rmc3_kor(len(x)) for x in xx])
        elif opts.model_type == 'mc3':
            x_lens = torch.LongTensor([seq_len_mc3_kor(len(x)) for x in xx])

    y_lens = torch.LongTensor([len(y) for y in yy])
    return xx_pad, yy_pad, x_lens, y_lens


class Trainer:
    def __init__(self):
        self.opts = options.parse()
        self.save_dir = os.path.join(self.opts.save_dir, self.opts.model_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # set a logger
        logging.basicConfig(
            filename=os.path.join(self.save_dir, "log-train.log"),
            format='%(asctime)s %(message)s',
            datefmt='%m/%d/%Y %p %I:%M:%S',
            level=logging.INFO
        )
        logging.getLogger().setLevel(logging.INFO)
        self.logger = logging.getLogger("trainLogger")
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        # set device for training/validation
        use_cuda = torch.cuda.is_available()  # check if GPU exists
        self.device = torch.device(
            "cuda" if use_cuda else "cpu")    # use CPU or GPU

        # train, val, test datasets
        data_train = AirTypingDataset(self.opts, self.opts.data_path_train)
        data_val   = AirTypingDataset(self.opts, self.opts.data_path_val)
        data_test  = AirTypingDataset(self.opts, self.opts.data_path_test)
        self.data_loader_train = DataLoader(
            data_train,
            batch_size=self.opts.batch_size,
            shuffle=True,
            num_workers=self.opts.num_workers,
            collate_fn=pad_collate,
            drop_last=True
        )
        self.data_loader_val = DataLoader(
            data_val,
            batch_size=self.opts.batch_size,
            shuffle=False,
            num_workers=self.opts.num_workers,
            collate_fn=pad_collate,
            drop_last=True
        )
        self.data_loader_test = DataLoader(
            data_test,
            batch_size=self.opts.batch_size,
            shuffle=False,
            num_workers=self.opts.num_workers,
            collate_fn=pad_collate,
            drop_last=True
        )

        # create a model
        self.model = GestureTranslator(self.opts).to(self.device)

        # parallelize the model to multiple GPUs
        if torch.cuda.device_count() > 1:
            self.logger.info("We're Using {} GPUs!".format(
                torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)
            self.model_without_dp = self.model.module
        elif torch.cuda.device_count() == 1:
            self.logger.info("We're Using {} GPU!".format(
                torch.cuda.device_count()))
            self.model_without_dp = self.model

        # define an optimizer
        self.params_to_train = list(self.model_without_dp.parameters())
        if self.opts.optimizer_type == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(
                self.params_to_train, lr=self.opts.learning_rate)
        elif self.opts.optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.params_to_train, lr=self.opts.learning_rate)
        elif self.opts.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.params_to_train, lr=self.opts.learning_rate)
        elif self.opts.optimizer_type == 'adamW':
            self.optimizer = torch.optim.AdamW(
                self.params_to_train, lr=self.opts.learning_rate)
        elif self.opts.optimizer_type == 'lamb':
            self.optimizer = Lamb(
                self.params_to_train, lr=self.opts.learning_rate)

        # define a scheduler
        iter_size = len(self.data_loader_train)
        scheduler_step_size = self.opts.scheduler_step_size * iter_size
        if self.opts.scheduler_type == 'warmup':
            self.scheduler = WarmupMultiStepLR(
                self.optimizer,
                [scheduler_step_size * (i + 1)
                 for i in range(self.opts.num_epochs)],
                gamma=self.opts.scheduler_gamma,
                warmup_iters=500)
        elif self.opts.scheduler_type == 'steplr':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_step_size,
                gamma=self.opts.scheduler_gamma)
        else:
            self.scheduler = None

        # load pretrained weights if available
        if self.opts.load_dir:
            self.load_model()

        # define a loss
        self.ctc_loss = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)

        # set train variables and save options
        self.epoch = 0
        self.step = 0
        self.start_step = 0
        self.start_time = time.time()
        self.num_total_steps = iter_size * self.opts.num_epochs
        self.best_val_loss = float('Inf')
        self.best_val_cer = float('Inf')

        self.save_opts()

    def train(self):
        for self.epoch in range(self.opts.num_epochs):
            print("epoch: ", self.epoch)
            self.model.train()
            self.run_one_epoch()
            is_best = self.validate(self.data_loader_val, False)
            if is_best or (self.opts.save_frequency > 0 and (self.epoch + 1) % self.opts.save_frequency == 0):
                self.save_model(is_best)

    def run_one_epoch(self):
        losses = []
        for batch_idx, (xx_pad, yy_pad, x_lens, y_lens) in enumerate(self.data_loader_train):
            time_before_process = time.time()
            # distribute data to device
            xx_pad, yy_pad = xx_pad.to(self.device), yy_pad.to(self.device)
            x_lens, y_lens = x_lens.to(self.device), y_lens.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(xx_pad, x_lens)
            output = output.permute(1, 0, 2).log_softmax(2)
            loss = self.ctc_loss(output, yy_pad, x_lens, y_lens)
            y_pred = torch.max(output, 2)[1]
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            duration = abs(time_before_process - time.time())
            # log information
            if (batch_idx + 1) % self.opts.log_interval == 0:
                self.log_time(duration, batch_idx, loss.item())
                self.logger.info('\tGround Truth: {}'.format(
                    self.data_loader_train.dataset.converter.decode(yy_pad[0, :y_lens[0]], y_lens[:1])))
                self.logger.info('\tModel Output: {}'.format(
                    self.data_loader_train.dataset.converter.decode(y_pred[:x_lens[0], 0], x_lens[:1])))
                if self.scheduler:
                    self.logger.info("\tCurrent LR: {:.6f}".format(
                        self.scheduler.get_lr()[0]))
            self.step += 1
        return losses

    def validate(self, data_loader, load=False):
        self.logger.info('---------------Validation------------------')
        if load:
            self.load_model()
        self.model.eval()
        losses = []
        with torch.no_grad():
            total_err, total_len = 0, 0
            for batch_idx, (xx_pad, yy_pad, x_lens, y_lens) in enumerate(data_loader):
                # distribute data to device
                xx_pad, yy_pad = xx_pad.to(self.device), yy_pad.to(self.device)
                x_lens, y_lens = x_lens.to(self.device), y_lens.to(self.device)

                output = self.model(xx_pad, x_lens)
                output = output.permute(1, 0, 2).log_softmax(2)
                loss = self.ctc_loss(output, yy_pad, x_lens, y_lens)
                y_pred = torch.max(output, 2)[1]
                losses.append(loss.item())

                # to compute accuracy
                # TODO: need to work for batch_size > 1
                gt = data_loader.dataset.converter.decode(
                    yy_pad[0, :y_lens[0]], y_lens[:1])
                pred = data_loader.dataset.converter.decode(
                    y_pred[:x_lens[0], 0], x_lens[:1])
                err, length = cer(gt, pred)
                if err > length:
                    err = length
                total_err += err
                total_len += length
                
                # log intermediate results
                if (batch_idx + 1) % self.opts.log_interval == 0:
                    self.logger.info('\tGround Truth: {}'.format(gt))
                    self.logger.info('\tModel Output: {}'.format(pred))

            cur_val_loss = sum(losses) / len(losses)
            cur_val_cer = total_err / total_len
            
            self.logger.info('\tcurrent_validation_loss: {}'.format(cur_val_loss))
            self.logger.info('\tcurrent_validation_cer: {}'.format(cur_val_cer))

            if self.best_val_cer > cur_val_cer:
                self.best_val_cer = cur_val_cer
                return True
            return False

    def log_time(self, duration, batch_idx, loss):
        samples_per_sec = self.opts.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = ((self.num_total_steps - self.step) / (
                    self.step - self.start_step)) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch [{:>4}/{:>4}] | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        self.logger.info(print_string.format(
            self.epoch,
            batch_idx*self.opts.batch_size,
            len(self.data_loader_train.dataset),
            samples_per_sec,
            loss,
            sec_to_hm_str(time_sofar),
            sec_to_hm_str(training_time_left)))

    def save_opts(self):
        to_save = self.opts.__dict__.copy()
        with open(os.path.join(self.save_dir, 'opts.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, is_best):
        save_folder = os.path.join(
            self.save_dir,
            "models",
            "weights_{}{}".format(self.epoch, "_best" if is_best else "")
        )
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, "{}.pth".format("model"))
        to_save = self.model_without_dp.state_dict()
        torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("optimizer"))
        torch.save(self.optimizer.state_dict(), save_path)

        if self.scheduler:
            save_path = os.path.join(save_folder, "{}.pth".format("scheduler"))
            torch.save(self.scheduler.state_dict(), save_path)

    def load_model(self):
        self.opts.load_dir = os.path.expanduser(self.opts.load_dir)

        assert os.path.isdir(self.opts.load_dir), \
            "Cannot find directory {}".format(self.opts.load_dir)
        print("loading model from directory {}".format(self.opts.load_dir))

        # loading model state
        path = os.path.join(self.opts.load_dir, "model.pth")
        pretrained_dict = torch.load(path)
        self.model_without_dp.load_state_dict(pretrained_dict)

        # loading optimizer state
        optimizer_load_path = os.path.join(self.opts.load_dir, "optimizer.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Optimizer weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Optimizer weights so Optimizer is randomly initialized")

        # loading scheduler state
        if self.scheduler:
            scheduler_load_path = os.path.join(
                self.opts.load_dir, "scheduler.pth")
            if os.path.isfile(scheduler_load_path):
                print("Loading Scheduler weights")
                scheduler_dict = torch.load(scheduler_load_path)
                self.scheduler.load_state_dict(scheduler_dict)
            else:
                print(
                    "Cannot find Scheduler weights so Scheduler is initialized as arranged")


if __name__ == "__main__":
    os.environ['PYTHONHASHSEED'] = str(opts.seed_number)
    random.seed(opts.seed_number)
    np.random.seed(opts.seed_number)
    torch.manual_seed(opts.seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    trainer = Trainer()
    trainer.train()
