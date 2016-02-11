import os
import numpy as np
import scipy as sp
import deeppy as dp

from video import Video


def random_walk(start_pos, n_steps, n_dir_steps=10, change_fraction=1.0):
    pos = np.copy(start_pos)
    for i in range(n_steps):
        if i % n_dir_steps == 0:
            next_point = np.random.normal(size=pos.shape)
            if change_fraction < 1.0:
                mask = np.random.uniform(low=0, high=1, size=pos.shape)
                mask = mask < change_fraction
                next_point[mask] = pos[mask]
            step = (next_point - pos)
            step /= n_dir_steps
        pos += step
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
    for z in random_walk(samples_z, 150, n_dir_steps=10, change_fraction=0.1):
        x = model.decode(z)
        if inv_transform is not None:
            x = inv_transform(x)
        walk_video.append(dp.misc.img_tile(x))
