from __future__ import absolute_import, division, print_function

import os
import argparse

# the directory that options.py resides in
file_dir = os.path.dirname(__file__)
print(file_dir)


class AirTypingOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Air Typing Options")

        # EXPERIMENT Name
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="model name",
                                 default="wita-resnet3d")

        # PATHS
        self.parser.add_argument("--data_path_train",
                                 type=str,
                                 help="path to the training data",
                                 default='/root/dataset/wita/RawData/data/english/train')
        self.parser.add_argument("--data_path_val",
                                 type=str,
                                 help="path to the validation data",
                                 default = '/root/dataset/wita/RawData/data/english/val')
        self.parser.add_argument("--data_path_test",
                                 type=str,
                                 help="path to the test data",
                                 default='/root/dataset/wita/RawData/data/english/test')
        self.parser.add_argument("--save_dir",
                                 type=str,
                                 help="directory to save logs, models, etc.",
                                 default=file_dir)
        self.parser.add_argument("--load_dir",
                                 type=str,
                                 help="directory of models to load",
                                 default='/root/TypingInTheAir14/r3d_10_avg_eng_aug_l2/models/weights_129_best')
        
        # MODEL options
        self.parser.add_argument("--model_type",
                                 type=str,
                                 help="model type ['r3d', 'mc3', 'rmc3', 'twoplusone']",
                                 default='r3d')
        self.parser.add_argument("--num_res_layer",
                                 type=int,
                                 help="the number of resnet layer",
                                 default=1)
        self.parser.add_argument("--pretrained",
                                 type=bool,
                                 help="load pretrained model for resnet18",
                                 default=False)
        self.parser.add_argument("--recurrent_type",
                                 type=str,
                                 help="type of recurrent network to use [gru, lstm, none]",
                                 default="none")
        self.parser.add_argument("--input_size",
                                 type=int,
                                 help="the number of features in rnn input",
                                 default=256)
        self.parser.add_argument("--hidden_size",
                                 type=int,
                                 help="the dimension of rnn hidden states",
                                 default=256)
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="the number of recurrent layers",
                                 default=2)
        self.parser.add_argument("--hidden_size_fc",
                                 type=int,
                                 help="the number of features in hidden states of fc layer",
                                 default=128)
        self.parser.add_argument("--img_size",
                                 type=int,
                                 help="image width and height",
                                 default=112)
        self.parser.add_argument("--pooling_type",
                                type=str,
                                help="type of pooling to use [max, average]",
                                default="average")
        
        # TRAINING options
        self.parser.add_argument('--track_running',
                                 type=str,
                                 help="tracking mean and variance for batchnorms",
                                 default=True)
        self.parser.add_argument('--seed_number',
                                 type=int,
                                 help="fix randomness",
                                 default=0)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=50)
        self.parser.add_argument("--log_interval",
                                 type=int,
                                 help="log interval during the training procedure",
                                 default=100)
        self.parser.add_argument("--tensorboard_path",
                                 type=str,
                                 help="path to write summary for tensorboard")

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=1)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-3)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=175)
        self.parser.add_argument("--optimizer_type",
                                 type=str,
                                 help="type of optimizer for training ['rmsprop', 'sgd', 'lamb', 'adam', 'adamW']",
                                 default='adam')
        self.parser.add_argument("--scheduler_type",
                                 type=str,
                                 help="type of scheduler for training ['warmup', 'steplr', 'none']",
                                 default='warmup')
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=5)
        self.parser.add_argument("--scheduler_gamma",
                                 type=float,
                                 help="learning rate",
                                 default=0.9)

        # DATASET options
        self.parser.add_argument("--data_type",
                                 type=str,
                                 help="data type: [english, korean, or eng_kor]",
                                 default='english')
        self.parser.add_argument("--data_augment",
                                 type=bool,
                                 help="augment data if True",
                                 default=False)

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=2)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
