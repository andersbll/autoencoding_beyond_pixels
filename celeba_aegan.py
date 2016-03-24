#!/usr/bin/env python

import os
import pickle
import numpy as np
import scipy as sp
import deeppy as dp

import dataset.celeba
import aegan
from dataset.util import img_transform, img_inverse_transform


def run():
    experiment_name = 'celeba'

    img_size = 64
    epoch_size = 250
    batch_size = 64
    n_augment = int(6e5)
    train_feed, test_feed = dataset.celeba.feeds(
        img_size, split='test', batch_size=batch_size, epoch_size=epoch_size,
        n_augment=n_augment,
    )
    n_hidden = 128
    model, experiment_name = aegan.build_model(
        experiment_name, img_size, n_hidden=n_hidden, recon_depth=9,
        recon_vs_gan_weight=1e-6, real_vs_gen_weight=0.5,
        discriminate_ae_recon=False, discriminate_sample_z=True,
    )
    print('experiment_name: %s' % experiment_name)
    output_dir = os.path.join('out', experiment_name)
    aegan.train(
        model, output_dir, train_feed, test_feed, n_epochs=250,
        lr_start=0.025,
    )
    model_path = os.path.join(output_dir, 'arch.pickle')
    print('Saving model to disk')
    print(model_path)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print('Extracting visual attribute vectors')
    model.phase = 'test'
    train_feed, test_feed = dataset.celeba.feeds(
        img_size, batch_size=batch_size, epoch_size=epoch_size,
        with_attributes=True, split='test',
    )

    n_attr_imgs = 10000
    x = img_transform(train_feed.x[:n_attr_imgs], to_bc01=False)
    y = train_feed.y[:n_attr_imgs]
    z = model.encode(x)

    all_attributes = list(dp.dataset.CelebA().attribute_names)
    selected_attributes = [
        'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Bushy_Eyebrows',
        'Eyeglasses', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
        'Mustache', 'Pale_Skin', 'Rosy_Cheeks', 'Smiling', 'Straight_Hair',
        'Wavy_Hair', 'Wearing_Lipstick', 'Young',
    ]
    attr_idxs = [all_attributes.index(attr) for attr in selected_attributes]
    attr_vecs = []
    for attr_idx in attr_idxs:
        on_mask = y[:, attr_idx] == 1.0
        off_mask = np.logical_not(on_mask)
        vec = (np.mean(z[on_mask, :], axis=0, dtype=float) -
               np.mean(z[off_mask, :], axis=0, dtype=float))
        attr_vecs.append(vec)

    print('Outputting visual attribute vectors')
    original_x = test_feed.batches().next()[0]
    original_z = model.encode(original_x)
    attributes_dir = os.path.join(output_dir, 'attributes')
    if not os.path.exists(attributes_dir):
        os.mkdir(attributes_dir)
    for attr_idx, attr_vec in zip(attr_idxs, attr_vecs):
        attr_name = all_attributes[attr_idx].lower()
        attrs_z = original_z + attr_vec
        attrs_x = model.decode(attrs_z.astype(dp.float_))
        attrs_x = img_inverse_transform(attrs_x)
        for i, attr_x in enumerate(attrs_x):
            path = os.path.join(attributes_dir, '%.3d_%s.png' % (i, attr_name))
            sp.misc.imsave(path, attr_x)


if __name__ == '__main__':
    run()
