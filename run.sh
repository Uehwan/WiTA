#!/bin/sh

# Best performing configuration for English
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --model_name=r3d_10_avg_eng_aug --model_type=r3d \
                --pooling_type=average --img_size=112 \
                --data_type=english --optimizer_type=adam  --num_res_layer=1 \
                --data_augment=True --batch_size=8 --num_workers=8 \
                --data_path_train=../wita/english/train \
                --data_path_val=../wita/english/val \
                --data_path_test=../wita/english/test

# Best performing configuration for Korean
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --model_name=r3d_18_max_kor_pre --model_type=r3d \
                --pooling_type=max --img_size=112 \
                --data_type=korean --optimizer_type=adam  --num_res_layer=2 \
                --pretrained=True --batch_size=4 --num_workers=4 \
                --data_path_train=../wita/korean/train \
                --data_path_val=../wita/korean/val \
                --data_path_test=../wita/korean/test
