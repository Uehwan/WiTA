import os
import torch
import random
import numpy as np

from torch.utils.data import DataLoader

from data import AirTypingDataset
from options import AirTypingOptions
from train import Trainer, pad_collate

options = AirTypingOptions()
opts = options.parse()


class Tester(Trainer):
    def __init__(self):
        super(Tester, self).__init__()

        # test dataset
        data_test = AirTypingDataset(self.opts, self.opts.data_path_test)
        self.data_loader_test = DataLoader(
            data_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.opts.num_workers,
            collate_fn=pad_collate,
            drop_last=True
        )


if __name__ == "__main__":
    os.environ['PYTHONHASHSEED'] = str(opts.seed_number)
    random.seed(opts.seed_number)
    np.random.seed(opts.seed_number)
    torch.manual_seed(opts.seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tester = Tester()
    tester.validate(tester.data_loader_test)
