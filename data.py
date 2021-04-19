from __future__ import absolute_import, division, print_function

import os
import glob
import hgtk
import utils

from os.path import dirname
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.transforms as transforms
from torch.utils.data import Dataset

from video_transforms import ColorJitter, RandomRotation, ClipToTensor, Compose, Normalize


class AirTypingDataset(Dataset):
    def __init__(self, opts, data_path):
        self.data_path = data_path
        self.data_type = opts.data_type
        self.data_augment = opts.data_augment
        if self.data_type == "english":
            self.converter = utils.StrLabelConverter(utils.ALPHABET)
        elif self.data_type == "korean":
            self.converter = utils.StrLabelConverter(utils.HANGUL, False)
        else:
            self.converter = utils.StrLabelConverter(utils.ALPHA_HAN, False)
        self.video_list = []
        for x in glob.glob(os.path.join(self.data_path, '*/*/*')):
            if x.split('_')[-1].split('/')[-1] not in ['gt.txt', 'gt2.txt']:
                self.video_list.append(x)
        self.img_size = opts.img_size

        self.video_transform_aug_list = [
            ColorJitter(0.5, 0.5, 0.5, 0.5),
            RandomRotation(5),
            ClipToTensor(channel_nb=3),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        self.video_transform_list = [
            ClipToTensor(channel_nb=3),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        self.resize = transforms.Resize(112)
        self.video_transform = Compose(self.video_transform_list)
        self.video_transform_aug = Compose(self.video_transform_aug_list)
        self.labels = []

        # collect labels for each video
        if self.data_type == "english":
            for video_name in self.video_list:
                text_list = video_name.split('_')
                text_idx = int(text_list[-1].split('/')[-1])
                text_file = open(os.path.join(dirname(video_name), 'gt.txt'), 'r')
                text_line = text_file.readlines()
                self.labels.append(text_line[text_idx][:-1])
                text_file.close()
        elif self.data_type == "korean":
            for video_name in self.video_list:
                text_list = video_name.split('_')
                text_idx = int(text_list[-1].split('/')[-1])
                text_file = open(os.path.join(dirname(video_name), 'gt.txt'), 'rb')
                text_line = text_file.readlines()
                self.labels.append(text_line[text_idx][:-2].decode('cp949'))
                text_file.close()
        else:
            pass

    def __getitem__(self, index):
        video_tensor = self.read_images(index).permute(1, 0, 2, 3)
        video_label = self.read_labels(index)
        return video_tensor, video_label

    def __len__(self):
        return len(self.video_list)

    def read_images(self, index):
        selected_video = self.video_list[index]
        frames = sorted(os.listdir(selected_video))
        list_of_images = []
        for frame_name in frames:
            frame = Image.open(os.path.join(selected_video, frame_name))
            if self.img_size == 112:
                frame = self.resize(frame)
            list_of_images.append(frame)
        if self.data_augment:
            list_of_images = self.video_transform_aug(list_of_images)
        else:
            list_of_images = self.video_transform(list_of_images)
        return list_of_images

    def read_labels(self, index):
        if self.data_type == "korean":
            labels_raw = self.labels[index]
            labels_raw = hgtk.text.decompose(labels_raw)
        else:
            labels_raw = self.labels[index]
        labels_enc, _ = self.converter.encode(labels_raw)
        return labels_enc


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from options import AirTypingOptions
    
    
    options = AirTypingOptions()
    opts = options.parse()

    train_data = AirTypingDataset(opts, '/root/dataset/wita/RawData/english/train')

    """
    Visualize data augmentation
    """
    columns = 5
    rows = 5
    img_seq = train_data[0][0]
    fig = plt.figure(figsize=(5,5), constrained_layout=False)
    gs = gridspec.GridSpec(rows, columns, figure=fig,
                           wspace=0.0, hspace=0.0, top=0.98, bottom=0.01,
                           left=0.015, right=1-0.015)
    for i in range(rows):
        for j in range(columns):
            im = img_seq.numpy()[i*columns+j, :, :, :].transpose(1, 2, 0)
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(im)
            ax.set(xticks=[], yticks=[])

    for ax in fig.get_axes():
        ax.spines['top'].set_visible(ax.is_first_row())
        ax.spines['bottom'].set_visible(ax.is_last_row())
        ax.spines['left'].set_visible(ax.is_first_col())
        ax.spines['right'].set_visible(ax.is_last_col())
    plt.tight_layout()
    plt.savefig("test2.png", bbox_inches='tight', pad_inches=0.0)
    plt.show()
