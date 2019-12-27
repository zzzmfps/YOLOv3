import torch as tc


def horisontal_flip(images, targets):
    images = tc.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets
