import os
import numpy as np
import scipy as sp
import deeppy as dp

from video import Video


def _norm(arr):
    return np.sqrt(np.sum(arr**2, axis=1, keepdims=True))


def random_walk(start_pos, n_shifts=15, n_steps_per_shift=10,
                change_beta_a=0.25, change_beta_b=0.25):
    pos = np.copy(start_pos)
    for shift_idx in range(n_shifts):
        norm = _norm(pos)
        next_point = np.random.normal(size=pos.shape)
        next_norm = _norm(next_point)
        weights = np.random.beta(change_beta_a, change_beta_b, size=pos.shape)
        next_point = pos*(1.0 - weights) + next_point*weights
        next_point *= next_norm / _norm(next_point)
        for step_idx in range(n_steps_per_shift):
            step = (next_point - pos)
            pos += step / (n_steps_per_shift - step_idx)
            norm_step = (next_norm - norm)
            norm += norm_step / (n_steps_per_shift - step_idx)
            pos *= norm / _norm(pos)
            yield pos


def samples(model, samples_z, out_dir, inv_transform=None):
    print('Outputting samples')
    model.phase = 'test'
    samples_x = model.decode(samples_z.astype(dp.float_))
    if inv_transform is not None:
        samples_x = inv_transform(samples_x)
    samples_dir = os.path.join(out_dir, 'samples')
    if not os.path.exists(samples_dir):
        os.mkdir(samples_dir)
    for i in range(len(samples_z)):
        sp.misc.imsave(os.path.join(samples_dir, '%.3d.png' % i), samples_x[i])


def reconstructions(model, original_x, out_dir, inv_transform=None):
    print('Outputting reconstructions')
    model.phase = 'test'
    recon_x = model.decode(model.encode(original_x))
    if inv_transform is not None:
        original_x = inv_transform(original_x)
        recon_x = inv_transform(recon_x)
    recon_dir = os.path.join(out_dir, 'reconstructions')
    if not os.path.exists(recon_dir):
        os.mkdir(recon_dir)
    for i in range(len(original_x)):
        sp.misc.imsave(os.path.join(recon_dir, '%.3d.png' % i), original_x[i])
        sp.misc.imsave(os.path.join(recon_dir, '%.3d_recon.png' % i),
                       recon_x[i])


def walk(model, samples_z, out_dir, inv_transform=None):
    print('Outputting walk video')
    model.phase = 'test'
    walk_video = Video(os.path.join(out_dir, 'walk.mp4'))
    for z in random_walk(samples_z):
        x = model.decode(z)
        if inv_transform is not None:
            x = inv_transform(x)
        walk_video.append(dp.misc.img_tile(x))
