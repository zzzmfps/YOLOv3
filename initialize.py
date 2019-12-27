import os
import random as rd

from PIL import Image


def initialize(img_path: str = 'data/custom/images/',
               anno_path: str = 'data/custom/annos/',
               lbl_path: str = 'data/custom/labels/',
               split_ratio: float = 0.8) -> None:
    # -----------------
    #   remake labels
    # -----------------
    os.makedirs(lbl_path, exist_ok=True)
    file_list = os.listdir(anno_path)
    file_nums = len(file_list)
    correct_anno, wrong_anno, empty_file = 0, 0, []
    for _file in file_list:
        with open(anno_path + _file, 'r', encoding='utf8') as f:
            rows = [row for row in f.read().split('\n') if row]
        w, h = Image.open(img_path + _file.replace('txt', 'jpg')).size
        for i in range(len(rows) - 1, -1, -1):
            rows[i] = rows[i].split(' ')[1:]
            lu_x, lu_y, rb_x, rb_y = map(int, rows[i][1:])
            if lu_x < 0 or lu_y < 0 or rb_x >= w or rb_y >= h or rows[i][0] not in '不带电芯充电宝':  # wrong annos
                rows.pop(i)
                wrong_anno += 1
            else:
                rows[i][0] = 0 if rows[i][0] == '带电芯充电宝' else 1
                rows[i][1], rows[i][2] = (lu_x + rb_x) / 2 / w, (lu_y + rb_y) / 2 / h  # center_x, center_y
                rows[i][3], rows[i][4] = (rb_x - lu_x) / w, (rb_y - lu_y) / h  # width, height
                correct_anno += 1
        if not rows:
            empty_file.append(_file)  # the anno file that contains no correct anno
        else:
            with open(lbl_path + _file, 'w') as f:
                for row in rows:
                    if isinstance(row[0], int): f.write(' '.join(list(map(str, row)) + ['\n']))
    if wrong_anno:
        print(f'\nWarning: {wrong_anno} wrong anno(s) detected,',
              f'{len(empty_file)} anno file(s) is/are thus empty and omitted:')
        for f in empty_file:
            print(' -', f)
    print(f'\nFound {correct_anno} correct anno(s), with {file_nums-len(empty_file)} anno file(s) created.')

    # -------------------
    #   divide data set
    # -------------------
    file_list = [path.replace('.txt', '.jpg') for path in os.listdir(lbl_path)]
    file_nums = len(file_list)
    rd.shuffle(file_list)
    # train set
    with open('data/custom/train.txt', 'w', encoding='utf8') as f:
        for i in range(int(split_ratio * file_nums)):
            f.write(img_path + file_list[i] + '\n')
    # validation set
    with open('data/custom/valid.txt', 'w', encoding='utf8') as f:
        for i in range(int(split_ratio * file_nums), file_nums):
            f.write(img_path + file_list[i] + '\n')


if __name__ == "__main__":
    initialize()
