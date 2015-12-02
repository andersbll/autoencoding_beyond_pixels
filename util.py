import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import deeppy as dp


def img_tile(imgs):
    if imgs.dtype not in [np.int_, np.uint8]:
        imgs = dp.misc.img_stretch(imgs)
    return dp.misc.img_tile(imgs)


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


def random_walk(start_pos, n_steps, n_dir_steps=10, mean=0.0, std=1.0):
    pos = np.copy(start_pos)
    for i in range(n_steps):
        if i % n_dir_steps == 0:
            if isinstance(mean, float):
                next_point = np.random.normal(
                    scale=std, loc=mean, size=pos.shape
                )
            else:
                next_point = np.random.multivariate_normal(
                    mean=mean, cov=std, size=pos.shape[0]
                )
            step = (next_point - pos)
            step /= n_dir_steps
        pos += step
        yield pos
