import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import deeppy as dp


def img_tile(imgs):
    return dp.misc.img_tile(dp.misc.img_stretch(imgs))


def plot_img(img, title, filename=None):
    plt.close('all')
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)


def one_hot(labels, n_classes):
    onehot = np.zeros((labels.size, n_classes,), dtype=labels.dtype)
    onehot[np.arange(labels.size), labels] = 1
    return onehot


def random_walk(start_pos, n_steps, step_std):
    pos = np.copy(start_pos)
    for i in range(n_steps):
        if i % 10 == 0:
            step = np.random.normal(scale=step_std, size=pos.shape)
            sign_change = np.logical_and(np.abs(pos) > 0.7,
                                         np.sign(pos) == np.sign(step))
            step[sign_change] *= -1
        pos += step
        yield pos
