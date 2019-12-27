from draw import detect
from initialize import initialize
from models import Darknet
from train import evaluate
from utils.utils import load_classes, rescale_boxes

import os
import torch as tc

from PIL import Image


def test(img_path: str = 'data/custom/images/', anno_path: str = 'data/custom/annos/', action: int = 1) -> None:
    # transform something
    initialize(img_path=img_path, anno_path=anno_path, split_ratio=1.0)

    # some common config
    iou_thres = 0.5
    conf_thres = 0.01
    nms_thres = 0.5
    img_size = 416
    batch_size = 16
    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')

    # other paths
    weights_path = 'checkpoints/yolov3.pth'
    model_def = 'config/custom.cfg'
    test_path = 'data/custom/train.txt'
    class_path = 'data/custom/classes.names'

    # load model
    class_names = load_classes(class_path)
    model = Darknet(model_def).to(device)
    model.load_state_dict(tc.load(weights_path))

    if action & 1:  # 1 or 3
        # compute APs and mAP
        _, _, AP, _, ap_class = evaluate(
            model=model,
            path=test_path,
            iou_thres=iou_thres,
            conf_thres=conf_thres,
            nms_thres=nms_thres,
            img_size=img_size,
            batch_size=batch_size,
            device=device,
        )
        # print results
        print('Average Precisions:')
        for i, c in enumerate(ap_class):
            print(f'+ Class "{c}" ({class_names[c]}) - AP = {AP[i]}')
        print(f' ---- mAP = {AP.mean()}\n')

    if action > 1:  # 2 or 3
        # compute coordinates of boxes
        imgs, img_detections = detect(
            model=model,
            path=img_path,
            conf_thres=conf_thres,
            nms_thres=nms_thres,
            img_size=img_size,
            batch_size=batch_size,
            n_cpu=8,
            device=device,
        )

        os.makedirs('predicted_file', exist_ok=True)

        class1 = open('predicted_file/det_test_带电芯充电宝.txt', 'w')
        class2 = open('predicted_file/det_test_不带电芯充电宝.txt', 'w')
        for path, boxes in zip(imgs, img_detections):
            w, h = Image.open(path).size
            boxes = rescale_boxes(boxes, img_size, (h, w))
            for box in boxes:
                line = [
                    # os.path.split(path)[1].split('_')[1].split('.')[0],  # no prefix
                    os.path.split(path)[1].split('.')[0],  # with prefix 'core_' or 'coreless_'
                    f'{box[4].tolist():.3f}',  # conf
                    f'{box[0].tolist():.1f}',
                    f'{box[1].tolist():.1f}',
                    f'{box[2].tolist():.1f}',
                    f'{box[3].tolist():.1f}',
                ]
                if box[-1] == 0.0:
                    class1.write(' '.join(line) + '\n')
                elif box[-1] == 1.0:
                    class2.write(' '.join(line) + '\n')
        class1.close()
        class2.close()
        print('Output file saved.\n')


if __name__ == "__main__":
    # trailing '/' in path is necessary
    img_path = 'D:/BUAA/data/Image_test/'.replace('\\', '/')
    anno_path = 'D:/BUAA/data/Anno_test/'.replace('\\', '/')
    action = 3  # 1: mAP (internal), 2: create TXTs, 3: both, respectively
    test(img_path, anno_path, action=action)
