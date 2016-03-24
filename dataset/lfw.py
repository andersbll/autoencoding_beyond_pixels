import os
import numpy as np
import joblib
from skimage import transform
import deeppy as dp

from .augment import (img_augment, sample_img_augment_params, AugmentedFeed,
                      SupervisedAugmentedFeed)
from .util import img_transform


cachedir = os.getenv('CACHE_HOME', './cache')
mem = joblib.Memory(cachedir=os.path.join(cachedir, 'lfw'))


@mem.cache
def lfw_imgs(alignment):
    if alignment == 'landmarks':
        dataset = dp.dataset.LFW('original')
        imgs = dataset.imgs
        landmarks = dataset.landmarks('68')
        n_landmarks = 68
        landmarks_mean = np.mean(landmarks, axis=0)
        landmarks_mean = np.array([landmarks_mean[:n_landmarks],
                                   landmarks_mean[n_landmarks:]])
        aligned_imgs = []
        for img, points in zip(imgs, landmarks):
            points = np.array([points[:n_landmarks], points[n_landmarks:]])
            transf = transform.estimate_transform('similarity',
                                                  landmarks_mean.T, points.T)
            img = img / 255.
            img = transform.warp(img, transf, order=3)
            img = np.round(img*255).astype(np.uint8)
            aligned_imgs.append(img)
        imgs = np.array(aligned_imgs)
    else:
        dataset = dp.dataset.LFW(alignment)
        imgs = dataset.imgs
    return imgs


def lfw_imgs_split(alignment, split_name, with_attributes=True, test_fold=0):
    imgs = lfw_imgs(alignment)
    dataset = dp.dataset.LFW()
    if split_name == 'testtrain':
        all_persons = list(dataset.index.keys())
        test_persons = dataset.people_splits['test'][test_fold]
        persons = [p for p in all_persons if p not in test_persons]
    if split_name == 'valtrain':
        test_persons = dataset.people_splits['train']
    elif split_name == 'val':
        persons = dataset.people_splits[split_name]
    elif split_name == 'test':
        persons = dataset.people_splits[split_name][test_fold]

    if not with_attributes:
        new_imgs = []
        for person_id in persons:
            for img_idx in dataset.index[person_id]:
                new_imgs.append(imgs[img_idx])
        imgs = np.array(new_imgs)
        return imgs

    # Extract attributes vectors and discard images without attributes
    new_imgs = []
    attrs = []
    for person_id in persons:
        if person_id in dataset.attributes:
            for img_no in range(1, len(dataset.index[person_id])+1):
                if img_no in dataset.attributes[person_id]:
                    new_imgs.append(imgs[dataset.index[person_id][img_no-1]])
                    attrs.append(dataset.attributes[person_id][img_no])
    imgs = np.array(new_imgs)
    attrs = np.array(attrs).astype(dp.float_)
    return imgs, attrs


def _resize(args):
    img, crop_size, rescale_size = args
    crop = (img.shape[0] - crop_size) // 2
    img = img[crop:-crop, crop:-crop]
    img = transform.resize(img, (rescale_size, rescale_size, 3), order=3)
    img = (img*255).astype(np.uint8)
    return img


def _resize_augment(args):
    img, crop_size, rescale_size = args
    augment_params = sample_img_augment_params(
        translation_sigma=1.0, scale_sigma=0.01, rotation_sigma=0.01,
        gamma_sigma=0.07, contrast_sigma=0.07, hue_sigma=0.0125
    )
    img = img_augment(img, *augment_params)
    img = _resize((img, crop_size, rescale_size))
    return img


@mem.cache
def resize_imgs(imgs, crop_size, rescale_size, n_augment=0):
    if n_augment == 0:
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
                        ((img, crop_size, rescale_size)) for img in img_iter())
    imgs = np.array(imgs)
    return imgs


@mem.cache
def feeds(alignment, crop_size, rescale_size, batch_size, epoch_size,
          n_augment=int(1e5), with_attributes=False, split='val'):
    if split == 'val':
        train_split = 'valtrain'
        test_split = 'val'
    elif split == 'test':
        train_split = 'testtrain'
        test_split = 'test'
    x_train, y_train = lfw_imgs_split(alignment, train_split)

    # Shuffle training images
    idxs = np.random.permutation(len(x_train))
    x_train = x_train[idxs]
    y_train = y_train[idxs]

    if n_augment > 0:
        y_train = y_train[np.arange(n_augment) % len(x_train)]
    x_train = resize_imgs(x_train, crop_size, rescale_size, n_augment)
    x_train = np.transpose(x_train, (0, 3, 1, 2))

    x_test, y_test = lfw_imgs_split(alignment, test_split)
    x_test = resize_imgs(x_test, crop_size, rescale_size)
    x_test = img_transform(x_test, to_bc01=True)

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
