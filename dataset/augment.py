import random
import numpy as np
from skimage import transform
from skimage import color
import cudarray as ca
import deeppy as dp

from util import img_transform


def sample_img_augment_params(translation_sigma=1.0, scale_sigma=0.01,
                              rotation_sigma=0.01, gamma_sigma=0.07,
                              contrast_sigma=0.07, hue_sigma=0.0125):
    translation = np.random.normal(scale=translation_sigma, size=2)
    scale = np.random.normal(loc=1.0, scale=scale_sigma)
    rotation = np.random.normal(scale=rotation_sigma)
    mu = gamma_sigma**2
    gamma = np.random.normal(loc=mu, scale=gamma_sigma)
    gamma = np.exp(gamma/np.log(2))
    mu = contrast_sigma**2
    contrast = np.random.normal(loc=mu, scale=contrast_sigma)
    contrast = np.exp(contrast/np.log(2))
    hue = np.random.normal(scale=hue_sigma)
    return translation, scale, rotation, gamma, contrast, hue


def img_augment(img, translation=0.0, scale=1.0, rotation=0.0, gamma=1.0,
                contrast=1.0, hue=0.0, border_mode='constant'):
    if not (np.all(np.isclose(translation, [0.0, 0.0])) and
            np.isclose(scale, 1.0) and
            np.isclose(rotation, 0.0)):
        img_center = np.array(img.shape[:2]) / 2.0
        scale = (scale, scale)
        transf = transform.SimilarityTransform(translation=-img_center)
        transf += transform.SimilarityTransform(scale=scale, rotation=rotation)
        translation = img_center + translation
        transf += transform.SimilarityTransform(translation=translation)
        img = transform.warp(img, transf, order=3, mode=border_mode)
    if not np.isclose(gamma, 1.0):
        img **= gamma
    colorspace = 'rgb'
    if not np.isclose(contrast, 1.0):
        img = color.convert_colorspace(img, colorspace, 'hsv')
        colorspace = 'hsv'
        img[..., 1:] **= contrast
    if not np.isclose(hue, 0.0):
        img = color.convert_colorspace(img, colorspace, 'hsv')
        colorspace = 'hsv'
        img[..., 0] += hue
        img[img[..., 0] > 1.0, 0] -= 1.0
        img[img[..., 0] < 0.0, 0] += 1.0
    img = color.convert_colorspace(img, colorspace, 'rgb')
    if np.min(img) < 0.0 or np.max(img) > 1.0:
        raise ValueError('Invalid values in output image.')
    return img


class AugmentedFeed(dp.Feed):
    def batches(self):
        x = ca.empty(self.x_shape, dtype=dp.float_)
        for start, stop in self._batch_slices():
            if stop > start:
                x_np = self.x[start:stop]
            else:
                x_np = np.concatenate((self.x[start:], self.x[:stop]))
            if random.randint(0, 1) == 0:
                x_np = x_np[:, :, :, ::-1]
            x_np = img_transform(x_np, to_bc01=False)
            x_np = np.ascontiguousarray(x_np)
            ca.copyto(x, x_np)
            yield x,


class SupervisedAugmentedFeed(dp.SupervisedFeed):
    def batches(self):
        x = ca.empty(self.x_shape, dtype=dp.float_)
        y = ca.empty(self.y_shape, dtype=dp.float_)
        for start, stop in self._batch_slices():
            if stop > start:
                x_np = self.x[start:stop]
                y_np = self.y[start:stop]
            else:
                x_np = np.concatenate((self.x[start:], self.x[:stop]))
                y_np = np.concatenate((self.y[start:], self.y[:stop]))
            if random.randint(0, 1) == 0:
                x_np = x_np[:, :, :, ::-1]
            x_np = img_transform(x_np, to_bc01=False)
            x_np = np.ascontiguousarray(x_np)
            ca.copyto(x, x_np)
            ca.copyto(y, y_np)
            yield x, y
