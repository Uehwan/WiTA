# Writing in the Air

## Introduction
We introduce a new benchmark dataset for the writing in the air (WiTA) task. Our dataset consists of five sub-datasets in two languages (Korean and English) and amounts to 209,926 video instances from 122 participants. The proposed spatio-temporal residual networks perform unconstrained text recognition from finger movement while guaranteeing a real-time operation.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Requirements

* Ubuntu 16.04+
* CUDA = 10.1+
* Python 3.6+
* Pytorch = 1.2+
* torchvision = 0.4.2+

### Installation

We tested the code in the following environments: 1) CUDA 10.1 on Ubuntu 16.04 and 2) CUDA 11.0 on Ubuntu 18.04. WiTA may work in other environments, but you might need to modify a part of the code. We recommend you using Anaconda for the environment setup.

```bash
conda create --name wita python=3.6
conda activate wita
conda install -c pytorch pytorch=1.2 torchvision cudatoolkit=10.1

git clone https://github.com/Uehwan/WiTA.git
cd WiTA
```

### Data Preparation
The full data can be access through [this link](https://kaistackr-my.sharepoint.com/:f:/g/personal/ykh5013_kaist_ac_kr/Eo8xavj1KFpOpq9pZ3sa4aIBnP-2-V8h9s8AI5QqD-cuFA?e=e4dsJq).  
The dataset in the link is constructed as follows:
```bash
|---- WITA
|     |---- english
|           |---- train
|                 |---- lex
|                 |---- nonlex
|           |---- val
|                 |---- lex
|                 |---- nonlex
|           |---- test
|                 |---- lex
|                 |---- nonlex
|     |---- korean
|           |---- train
|                 |---- lex
|                 |---- nonlex
|           |---- val
|                 |---- lex
|                 |---- nonlex
|           |---- test
|                 |---- lex
|                 |---- nonlex
|     |---- kor_eng
|           |---- train
|                 |---- lex
|                 |---- nonlex
|           |---- val
|                 |---- lex
|                 |---- nonlex
|           |---- test
|                 |---- lex
|                 |---- nonlex
```
When downloading the dataset, make sure to follow the same structure as above.

You could also collect your own dataset using [collector.py](collector.py). Simply run the file and the usage is quite straight-forward.

## Training
To train the model, we recommend utilizing [run.sh](run.sh)  
Current [run.sh](run.sh) file shows how we trained the models to obtain the best performance for both English and Korean dataset.  
Feel free to alter the file according to your needs.


## Evaluation
To evaluate the model, simply run [test.py](test.py)

## Citation
Please consider citing this paper if you use our model or dataset in your work:
```bash
@inproceedings{kim2021,
  title={Writing in The Air: Unconstrained Text Recognition from Finger Movement Using Spatio-Temporal Convolution},
  author={Ue-Hwan Kim*, Yewon Hwang*, Sun-Kyung Lee and Jong-Hwan Kim},
  journal={arXiv preprint arXiv:09021},
  year={2021}
}
```

## Acknowledgment
We referred to the following repositories when developing WiTA. We appreciate the authors for their work.
* [CRNN.Pytorch](https://github.com/meijieru/crnn.pytorch/)
* [torch_videovision](https://github.com/hassony2/torch_videovision/blob/master/torchvideotransforms/video_transforms.py)

This work was supported by Institute for Information & communications Technology Promotion (IITP) grant funded by the Korea government (MSIT) (No.2020-0-00440, Development of Artificial Intelligence Technology that Continuously Improves Itself as the Situation Changes in the Real World)
