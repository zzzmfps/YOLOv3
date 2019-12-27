from models import Darknet
from utils.datasets import ImageFolder
from utils.utils import load_classes, non_max_suppression, rescale_boxes

import argparse
import numpy as np
import os
import random as rd
import tqdm

from PIL import Image

import torch as tc
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def detect(model, path, conf_thres, nms_thres, img_size, batch_size, device, n_cpu) -> tuple:
    model.eval()  # Set in evaluation mode
    dataloader = DataLoader(
        ImageFolder(folder_path=path, img_size=img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
    )
    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print()
    tq = tqdm.tqdm(dataloader, desc='Performing object detection', ncols=100)
    for img_paths, input_imgs in tq:
        # Configure input
        input_imgs = input_imgs.to(device)
        # Get detections
        with tc.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres)
        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    return imgs, img_detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='data/custom/images')
    parser.add_argument('--model_def', type=str, default='config/custom.cfg')
    parser.add_argument('--weights_path', type=str, default='checkpoints/yolov3.pth')
    parser.add_argument('--class_path', type=str, default='data/custom/classes.names')
    parser.add_argument('--conf_thres', type=float, default=0.8)
    parser.add_argument('--nms_thres', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_cpu', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=416)
    parser.add_argument('--print_progress', type=bool, default=False)
    opt = parser.parse_args()
    print(opt)

    os.makedirs('output', exist_ok=True)

    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
    classes = load_classes(opt.class_path)  # Extracts class labels from file

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    if opt.weights_path.endswith('.weights'):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(tc.load(opt.weights_path))

    # detect targets
    imgs, img_detections = detect(
        model=model,
        path=opt.image_folder,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        n_cpu=opt.n_cpu,
        device=device,
    )

    # Bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print('\nSaving images...')
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections), 1):
        if opt.print_progress: print('{0:>4} Image: "{1}"'.format(img_i, path))
        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = rd.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                if opt.print_progress:
                    print('\t+ Label: {0}, Conf: {1:.5f}'.format(classes[int(cls_pred)], cls_conf.item()))
                # width and height
                box_w = x2 - x1
                box_h = y2 - y1
                # color
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1 - 50,
                    s=classes[int(cls_pred)],
                    color='white',
                    verticalalignment='top',
                    bbox={
                        'color': color,
                        'pad': 0
                    },
                )
        # Save generated image with detections
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split('\\')[-1].split('.')[0]
        plt.savefig(f'output/{filename}.png', bbox_inches='tight', pad_inches=0.0)
        plt.close()
    print('Completed.')
