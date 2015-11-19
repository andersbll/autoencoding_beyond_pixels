import numpy as np
import joblib
from skimage.transform import resize
import deeppy as dp


mem = joblib.Memory(cachedir='./cache/lfw',)


@mem.cache
def lfw_imgs(alignment, size, crop, shuffle):
    imgs, names_idx, names = dp.dataset.LFW(alignment).arrays()
    new_imgs = []
    for img in imgs:
        img = img[crop:-crop, crop:-crop]
        img = resize(img, (size, size, 3), order=3)
        new_imgs.append(img)
    imgs = np.array(new_imgs)
    if shuffle:
        idxs = np.random.permutation(np.arange(len(imgs)))
        imgs = imgs[idxs]
    imgs = np.ascontiguousarray(dp.misc.to_bc01(imgs)).astype(dp.float_)
    return imgs
