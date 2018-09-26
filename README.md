# siamese-pose-estimation
Code for the paper "3D Human Pose Estimation with Siamese Equivariant Embedding", <http://arxiv.org/abs/1809.07217>

## Prerequisites

The code was created on Python 2.7 using the following libraries:

- Tensorflow 1.7
- Keras 2.1.5
- h5py 2.7

Other versions should work though note that these were the versions the code was tested with.

## Installation

First clone this repository. Then download the [preprocessed dataset](https://drive.google.com/open?id=1_1_DjwyCxEYVVnxc7u7mVj6EwiCnnsWT) and the [pretrained model](https://drive.google.com/open?id=1RsDpUmR7t3IBPhUbi-4wC3bjw3LrJgoE). The latter is only needed if you don't want to retrain the model. 

```bash
git clone https://github.com/vegesm/siamese-pose-estimation
cd siamese-pose-estimation
mkdir data
cd data
unzip h36m.zip
rm h36m.zip

# Only if you need the pretrained model
unzip pretrained.zip
rm pretrained.zip
```



## Usage

### Training

To quickly train a model using Protocol 3, use the following command:

``python run.py --epochs 100 --use-augmentation --cross-camera``

### Evaluating

To evaluate the pretrained model, run the command below:

``python run.py --eval --cross-camera --model-folder ../data/pretrained``

If you evaluate your own model, make sure it is using the same training protocol  (``--cross-camera`` argument) in testing and training. The pretrained model was created using Protocol 3 (``--cross-camera`` turned on).

### Command line arguments

The main script is ``run.py`` in the ``src`` folder. It is designed to run with the working directory being ``src``. The parameters of the script:

```
--eval                evaluate a model instead of training
--use-augmentation    use additional augmented camera views, has no effect when --eval
                      is set
--cross-camera        use Protocol 3
--epochs EPOCHS       number of epochs to train the model
--model-folder PATH   folder where the model is saved/loaded from
```



## Citing

If you used this code or found our paper useful please cite the following paper:

```
@article{veges2018siamese,
  title={3D Human Pose Estimation with Siamese Equivariant Embedding},
  author={V\'eges, M\'arton and Varga, Viktor and L\H{o}rincz, Andr\'as},
  journal={arXiv preprint arXiv:1809.07217},
  year={2018}
}
```