# YOLOv3
Especially modified for the assignment of lesson Machine Learning.

Group member: Junhao Zhao, Liwei Zuo, Huijie Qi, Shiwei Yu, Bo Wang.

## Requirements
Test passed with Python 3.7.5 + Pytorch 1.3.1 + CUDA 10.1

## Download
Pretrained weights file ```yolov3.pth```:

https://bhpan.buaa.edu.cn:443/link/703CF22437E9DDF7F207191BC6A8857C

## Usage
0. train: place dataset (two folders: annos and images) under ```data/custom```.
1. Place weights file ```yolov3.pth``` under the folder ```checkpoints```.
2. Edit path codes in ```__main__``` of ```test.py```.
3. Run ```test.py``` and wait for results.

## Reference
Main codes shamelessly taken from:

https://github.com/eriklindernoren/PyTorch-YOLOv3
