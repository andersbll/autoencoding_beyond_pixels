import os
import numpy as np
import deeppy as dp
import joblib

from .util import img_transform, ShuffledFeed, ShuffledSupervisedFeed

cachedir = os.getenv('CACHE_HOME', './cache')
mem = joblib.Memory(cachedir=os.path.join(cachedir, 'cifar'))


@mem.cache
def arrays(split='test'):
    dataset = dp.dataset.CIFAR10()
    x_train, y_train, x_test, y_test = dataset.arrays()
    x_train = np.copy(x_train)
    x_test = np.copy(x_test)
    x_train = x_train.astype(dp.float_)
    x_test = x_test.astype(dp.float_)
    y_train = y_train.astype(dp.int_)
    y_test = y_test.astype(dp.int_)

    if split == 'test':
        x_val = np.copy(x_test)
        y_val = np.copy(y_test)
    else:
        n_val = 10000
        x_val = np.copy(x_train[-n_val:])
        y_val = np.copy(y_train[-n_val:])
        x_train = x_train[:-n_val]
        y_train = y_train[:-n_val]
    if 'perclasstrain' in split:
        n_imgs_per_class = int(split[:split.find('per')])
        x_train_subset = []
        y_train_subset = []
        for i in range(dataset.n_classes):
            idxs = np.nonzero(y_train == i)[0]
            idxs = idxs[:n_imgs_per_class]
            x_train_subset.append(x_train[idxs])
            y_train_subset.append(y_train[idxs])
        n_imgs = n_imgs_per_class*dataset.n_classes
        idxs = np.random.permutation(np.arange(n_imgs))
        x_train = np.vstack(x_train_subset)[idxs]
        y_train = np.hstack(y_train_subset)[idxs]

    x_train = img_transform(x_train, to_bc01=False)
    x_val = img_transform(x_val, to_bc01=False)
    x_test = img_transform(x_test, to_bc01=False)
    x_train = x_train.astype(dp.float_)
    x_val = x_val.astype(dp.float_)
    x_test = x_test.astype(dp.float_)
    return x_train, y_train, x_val, y_val, x_test, y_test


def feeds(split='test', batch_size=128, epoch_size=None, preprocessing='',
          augmentation='', supervised=False):
    x_train, y_train, x_val, y_val, x_test, y_test = arrays(split)
    if supervised:
        train_feed = ShuffledSupervisedFeed(
            x_train, y_train, batch_size=batch_size, epoch_size=epoch_size
        )
        val_feed = dp.SupervisedFeed(x_val, y_val, batch_size=batch_size)
        test_feed = dp.SupervisedFeed(x_test, y_test, batch_size=batch_size)
    else:
        train_feed = ShuffledFeed(
            x_train, batch_size=batch_size, epoch_size=epoch_size
        )
        val_feed = dp.Feed(x_val, batch_size=batch_size)
        test_feed = dp.Feed(x_test, batch_size=batch_size)
    return train_feed, val_feed, test_feed
