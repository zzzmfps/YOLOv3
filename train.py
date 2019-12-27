from models import Darknet
from utils.datasets import ListDataset
from utils.parse_config import parse_data_config
from utils.utils import xywh2xyxy, non_max_suppression, get_batch_statistics, ap_per_class
from utils.utils import load_classes, weights_init_normal

import argparse
import datetime
import numpy as np
import os
import time
import torch as tc
import tqdm

from terminaltables import AsciiTable


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, device):
    model.eval()
    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = tc.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=1,
                                          collate_fn=dataset.collate_fn)
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    try:
        print()
        tq = tqdm.tqdm(dataloader, desc='Detecting objects', ncols=100)
        for _, imgs, targets in tq:
            imgs = imgs.to(device)
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= img_size
            with tc.no_grad():
                outputs = model(imgs)
                outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
    finally:
        tq.close()
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    return precision, recall, AP, f1, ap_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulations', type=int, default=2)
    parser.add_argument('--model_def', type=str, default='config/custom.cfg')
    parser.add_argument('--data_config', type=str, default='config/custom.data')
    parser.add_argument('--pretrained_weights', type=str, default=None)
    parser.add_argument('--n_cpu', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=416)
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--evaluation_interval', type=int, default=10)
    parser.add_argument('--multiscale_training', type=bool, default=True)
    opt = parser.parse_args()
    print(opt)

    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
    os.makedirs('checkpoints', exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config['train']
    valid_path = data_config['valid']
    class_names = load_classes(data_config['names'])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith('.pth'):
            model.load_state_dict(tc.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = tc.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = tc.optim.Adam(model.parameters())

    metrics = [
        'grid_size',
        'loss',
        'x',
        'y',
        'w',
        'h',
        'conf',
        'cls',
        'cls_acc',
        'recall50',
        'recall75',
        'precision',
        'conf_obj',
        'conf_noobj',
    ]

    for epoch in range(1, opt.epochs + 1):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            batches_done = len(dataloader) * (epoch - 1) + batch_i

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------
            if batch_i % 100 == 0:
                log_str = '\n---- [Epoch {0}/{1}, Batch {2}/{3}] ----\n'.format(epoch, opt.epochs, batch_i,
                                                                                len(dataloader))
                metric_table = [['Metrics', *[f'YOLO Layer {i}' for i in range(len(model.yolo_layers))]]]
                # Log metrics at each YOLO layer
                for i, metric in enumerate(metrics):
                    formats = {m: '%.6f' for m in metrics}
                    formats['grid_size'] = '%2d'
                    formats['cls_acc'] = '%.2f%%'
                    row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                    metric_table += [[metric, *row_metrics]]
                log_str += AsciiTable(metric_table).table
                log_str += f'\nTotal loss {loss.item()}'
                # Determine approximate time left for epoch
                epoch_batches_left = len(dataloader) - (batch_i + 1)
                time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                log_str += f'\n---- ETA {time_left}'
                print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.checkpoint_interval == 0:
            tc.save(model.state_dict(), f'checkpoints/yolov3_ckpt_{epoch}.pth')

        if epoch % opt.evaluation_interval == 0:
            print('\n---- Evaluating Model ----')
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model=model,
                path=valid_path,
                iou_thres=0.5,  # four arguments same with test.py
                conf_thres=0.01,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=opt.batch_size << 1,
                device=device,
            )

            # Print class APs and mAP
            ap_table = [['Index', 'Class name', 'AP']]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], '{0:.5f}'.format(AP[i])]]
            print(AsciiTable(ap_table).table)
            print(f'---- mAP = {AP.mean()} ----')
