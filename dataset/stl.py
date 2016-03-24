import os
import numpy as np
import deeppy as dp
from skimage import transform
import joblib

from .augment import (img_augment, sample_img_augment_params, AugmentedFeed)
from .util import img_transform, ShuffledSupervisedFeed


cachedir = os.getenv('CACHE_HOME', './cache')
mem = joblib.Memory(cachedir=os.path.join(cachedir, 'stl'))


def arrays(name, normalize=True):
    dataset = dp.dataset.STL10()
    x_train, y_train, x_test, y_test, x_unlabeled = dataset.arrays()
    if name == 'unlabeled':
        x = x_unlabeled
        y = np.array([])
    elif name == 'train':
        x = x_train
        y = y_train
    elif name == 'test':
        x = x_test
        y = y_test
    y = y.astype(dp.int_)
    x = np.transpose(x, (0, 3, 2, 1))
    x = np.ascontiguousarray(x)
    return x, y


def _resize(args):
    img, rescale_size = args
    img = transform.resize(img, (rescale_size, rescale_size, 3), order=3)
    img = (img*255).astype(np.uint8)
    return img


def _resize_augment(args):
    img, rescale_size = args
    augment_params = sample_img_augment_params(
        translation_sigma=2.0, scale_sigma=0.025, rotation_sigma=0.015,
        gamma_sigma=0.07, contrast_sigma=0.075, hue_sigma=0.02
    )
    img = img_augment(img, *augment_params, border_mode='nearest')
    img = _resize((img, rescale_size))
    return img


@mem.cache
def resize_imgs(imgs, rescale_size, n_augment=0):
    if n_augment == 0:
        if rescale_size == 96:
            return imgs
        preprocess_fun = _resize
        n_imgs = len(imgs)
    else:
        preprocess_fun = _resize_augment
        n_imgs = n_augment

    def img_iter():
        for i in range(n_imgs):
            yield imgs[i % len(imgs)]

    with joblib.Parallel(n_jobs=-2) as parallel:
        imgs = parallel(joblib.delayed(preprocess_fun)
                        ((img, rescale_size)) for img in img_iter())
    imgs = np.array(imgs)
    return imgs


@mem.cache
def unlabeled_feed(img_size, batch_size=128, epoch_size=250,
                   n_augment=0):
    x_unlabeled, _ = arrays('unlabeled')
    x_unlabeled = resize_imgs(x_unlabeled, img_size, n_augment)
    if n_augment == 0:
        x_unlabeled = img_transform(x_unlabeled, to_bc01=True)
        unlabeled_feed = dp.Feed(x_unlabeled, batch_size=batch_size,
                                 epoch_size=epoch_size)
    else:
        x_unlabeled = np.transpose(x_unlabeled, (0, 3, 1, 2))
        unlabeled_feed = AugmentedFeed(x_unlabeled, batch_size=batch_size,
                                       epoch_size=epoch_size)
    return unlabeled_feed


def supervised_feed(img_size, batch_size=128, epoch_size=250, val_fold=None):
    x_train, y_train = arrays('train')
    x_test, y_test = arrays('test')
    x_train = resize_imgs(x_train, img_size)
    x_test = resize_imgs(x_test, img_size)
    x_train = img_transform(x_train, to_bc01=True)
    x_test = img_transform(x_test, to_bc01=True)

    # TODO use folds
    train_feed = ShuffledSupervisedFeed(
        x_train, y_train, batch_size=batch_size, epoch_size=epoch_size
    )
    test_feed = dp.SupervisedFeed(x_test, y_test, batch_size=batch_size)
    return train_feed, test_feed
