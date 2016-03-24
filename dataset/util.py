import numpy as np
import deeppy as dp
import scipy as sp


def img_transform(imgs, to_bc01=True):
    imgs = imgs.astype(dp.float_)
    imgs /= 127.5
    imgs -= 1.0
    if to_bc01:
        imgs = np.transpose(imgs, (0, 3, 1, 2))
    return imgs


def img_inverse_transform(imgs, to_b01c=True):
    imgs = ((imgs+1)*0.5*255).astype(np.uint8)
    if to_b01c:
        imgs = np.transpose(imgs, (0, 2, 3, 1))
    return imgs


def contrast_normalize(data, scale=10.0, eps=1e-5, copy=True):
    if copy:
        data = np.copy(data)
    orig_shape = data.shape
    data = np.reshape(data, (data.shape[0], -1))
    data -= data.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(data**2, axis=1, keepdims=True))
    print(np.min(norms), np.max(norms))
    data /= np.maximum(norms, eps)
    return data.reshape(orig_shape)


class ZCA():
    def __init__(self, bias=0.1):
        self.bias = bias

    def _flat(self, x):
        return np.reshape(x, (x.shape[0], -1))

    def _normalize(self, x):
        return (x.astype(np.float_) - self._mean) / self._std

    def fit(self, x):
        x = self._flat(x)
        self._mean = np.mean(x, axis=0, dtype=np.float64).astype(x.dtype)
        self._std = np.std(x, axis=0, dtype=np.float64).astype(x.dtype)
        x = self._normalize(x)
        try:
            # Perform dot product on GPU
            import cudarray as ca
            x_ca = ca.array(x)
            cov = np.array(ca.dot(x_ca.T, x_ca)).astype(np.float_)
        except:
            cov = np.dot(x.T, x)
        cov = cov / x.shape[0] + self.bias * np.identity(x.shape[1])
        s, v = np.linalg.eigh(cov)
        s = np.diag(1.0 / np.sqrt(s))
        self.whitener = np.dot(np.dot(v, s), v.T)
        return self

    def transform(self, x):
        shape = x.shape
        x = self._flat(x)
        x = self._normalize(x)
        x_white = np.dot(x, self.whitener.T)
        return np.reshape(x_white, shape)


class LCN():
    @staticmethod
    def gaussian_kernel(sigma, size=None):
        if size is None:
            size = int(np.ceil(sigma*2.))
            if size % 2 == 0:
                size += 1
        xs = np.linspace(-size/2., size/2., size)
        kernel = 1/(np.sqrt(2*np.pi))*np.exp(-xs**2/(2*sigma**2))/sigma
        return kernel/np.sum(kernel)

    def __init__(self, sigma, eps=1e-1, subtractive=False):
        self.sigma = sigma
        self.eps = eps
        self.subtractive = subtractive
        self.kernel = LCN.gaussian_kernel(sigma, size=int(sigma*3))
        self.kernel = np.outer(self.kernel, self.kernel)

    def _normalize(self, x):
        return (x.astype(np.float_) - self._mean) / self._std

    def fit(self, x):
        self._mean = np.mean(x, dtype=np.float64).astype(x.dtype)
        self._std = np.std(x, dtype=np.float64).astype(x.dtype)
        return self

    def _transform_img(self, img):
        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        n_channels = img.shape[0]

        # Calculate local mean
        mean = np.zeros((1,) + img.shape[1:])
        for i in range(n_channels):
            mean += sp.ndimage.filters.convolve(img[i], self.kernel,
                                                mode='nearest')
        mean /= n_channels

        # Center input with local mean
        centered = img - mean
        if self.subtractive:
            return centered

        # Calculate local standard deviation
        centered_sqr = centered**2
        std = np.zeros((1,) + img.shape[1:])
        for i in range(n_channels):
            std += sp.ndimage.filters.convolve(centered_sqr[i], self.kernel,
                                               mode='nearest')
        std /= n_channels
        std = np.sqrt(std)

        # Scale centered input with standard deviation
        return centered / (std + self.eps)

    def transform(self, x):
        x = self._normalize(x)
        return np.array([self._transform_img(img) for img in x])


class ShuffleMixin(object):
    y = None

    def batches(self):
        idxs = np.random.permutation(np.arange(len(self.x)))
        self.x = self.x[idxs]
        if self.y is not None:
            self.y = self.y[idxs]
        for batch in super(ShuffleMixin, self).batches():
            yield batch


class ShuffledFeed(dp.Feed, ShuffleMixin):
    pass


class ShuffledSupervisedFeed(dp.SupervisedFeed, ShuffleMixin):
    pass
