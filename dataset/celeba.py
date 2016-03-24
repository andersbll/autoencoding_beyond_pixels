import os
import numpy as np
import joblib
from skimage import transform, filters
import deeppy as dp

from .augment import (img_augment, sample_img_augment_params, AugmentedFeed,
                      SupervisedAugmentedFeed)
from .util import img_transform


cachedir = os.getenv('CACHE_HOME', './cache')
mem = joblib.Memory(cachedir=os.path.join(cachedir, 'celeba'))


def _resize(args):
    img, rescale_size, bbox = args
    img = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    # Smooth image before resize to avoid moire patterns
    scale = img.shape[0] / float(rescale_size)
    sigma = np.sqrt(scale) / 2.0
    img = filters.gaussian_filter(img, sigma=sigma, multichannel=True)
    img = transform.resize(img, (rescale_size, rescale_size, 3), order=3)
    img = (img*255).astype(np.uint8)
    return img


def _resize_augment(args):
    img, rescale_size, bbox = args
    augment_params = sample_img_augment_params(
        translation_sigma=2.00, scale_sigma=0.01, rotation_sigma=0.01,
        gamma_sigma=0.05, contrast_sigma=0.05, hue_sigma=0.01
    )
    img = img_augment(img, *augment_params, border_mode='nearest')
    img = _resize((img, rescale_size, bbox))
    return img


@mem.cache
def celeba_imgs(img_size=64, bbox=(40, 218-30, 15, 178-15), img_idxs=None,
                n_augment=0):
    if bbox[1] - bbox[0] != bbox[3] - bbox[2]:
        raise ValueError('Image is not square')
    dataset = dp.dataset.CelebA()
    if img_idxs is None:
        img_idxs = list(range(dataset.n_imgs))
    if n_augment == 0:
        preprocess_fun = _resize
        n_imgs = len(img_idxs)
    else:
        preprocess_fun = _resize_augment
        n_imgs = n_augment

    def img_iter():
        for i in range(n_imgs):
            yield dataset.img(img_idxs[i % len(img_idxs)])

    with joblib.Parallel(n_jobs=-2) as parallel:
        imgs = parallel(joblib.delayed(preprocess_fun)
                        ((img, img_size, bbox)) for img in img_iter())
    imgs = np.array(imgs)
    return imgs


def feeds(img_size, batch_size, epoch_size, n_augment=int(6e5),
          with_attributes=False, split='val'):
    dataset = dp.dataset.CelebA()
    if split == 'val':
        train_idxs = dataset.train_idxs
        test_idxs = dataset.val_idxs
    elif split == 'test':
        train_idxs = np.hstack((dataset.train_idxs, dataset.val_idxs))
        test_idxs = dataset.test_idxs
    x_train = celeba_imgs(img_size, img_idxs=train_idxs, n_augment=n_augment)
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = celeba_imgs(img_size, img_idxs=test_idxs)
    x_test = img_transform(x_test, to_bc01=True)
    attributes = dataset.attributes.astype(dp.float_)
    y_train = attributes[train_idxs]
    y_test = attributes[test_idxs]
    if n_augment > 0:
        y_train = y_train[np.arange(n_augment) % len(y_train)]

    if with_attributes:
        train_feed = SupervisedAugmentedFeed(
            x_train, y_train, batch_size=batch_size, epoch_size=epoch_size
        )
        test_feed = dp.SupervisedFeed(
            x_test, y_test, batch_size=batch_size
        )
    else:
        train_feed = AugmentedFeed(x_train, batch_size, epoch_size)
        test_feed = dp.Feed(x_test, batch_size)

    return train_feed, test_feed
