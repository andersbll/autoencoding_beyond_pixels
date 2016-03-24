#!/usr/bin/env python

import os
import pickle
import numpy as np
import scipy as sp
import deeppy as dp

import architectures
from model import gan
import output
from video import Video


def mnist_transform(imgs, to_flat=True):
    imgs = np.reshape(imgs, (len(imgs), -1))
    imgs = imgs.astype(dp.float_)
    imgs /= 255.0
    return imgs


def mnist_inverse_transform(imgs, to_img=True):
    imgs = (imgs*255.0).astype(np.uint8)
    imgs = np.reshape(imgs, (len(imgs), 28, 28))
    return imgs


def run():
    n_hidden = 128
    real_vs_gen_weight = 0.75
    gan_margin = 0.3

    lr_start = 0.04
    lr_stop = 0.0001
    lr_gamma = 0.75
    n_epochs = 150
    epoch_size = 250
    batch_size = 64

    experiment_name = 'mnist_gan'
    experiment_name += '_nhidden%i' % n_hidden

    out_dir = os.path.join('out', experiment_name)
    arch_path = os.path.join(out_dir, 'arch.pickle')
    start_arch_path = arch_path
    start_arch_path = None

    print('experiment_name', experiment_name)
    print('start_arch_path', start_arch_path)
    print('arch_path', arch_path)

    # Setup network
    if start_arch_path is None:
        print('Creating new model')
        _, decoder, discriminator = architectures.mnist()
    else:
        print('Starting from %s' % start_arch_path)
        with open(start_arch_path, 'rb') as f:
            decoder, discriminator = pickle.load(f)

    model = gan.GAN(
        n_hidden=n_hidden,
        generator=decoder,
        discriminator=discriminator,
        real_vs_gen_weight=real_vs_gen_weight,
    )

    # Fetch dataset
    dataset = dp.dataset.MNIST()
    x_train, y_train, x_test, y_test = dataset.arrays()
    x_train = mnist_transform(x_train)
    x_test = mnist_transform(x_test)

    # Prepare network feeds
    train_feed = dp.Feed(x_train, batch_size, epoch_size)
    test_feed = dp.Feed(x_test, batch_size)

    # Plotting
    n_examples = 64
    original_x, = test_feed.batches().next()
    original_x = np.array(original_x)[:n_examples]
    samples_z = np.random.normal(size=(n_examples, n_hidden))
    samples_z = (samples_z).astype(dp.float_)

    # Train network
    learn_rule = dp.RMSProp()
    trainer = gan.GradientDescent(model, train_feed, learn_rule,
                                  margin=gan_margin)
    annealer = dp.GammaAnnealer(lr_start, lr_stop, n_epochs, gamma=lr_gamma)
    try:
        sample_video = Video(os.path.join(out_dir, 'convergence_samples.mp4'))
        sp.misc.imsave(os.path.join(out_dir, 'examples.png'),
                       dp.misc.img_tile(mnist_inverse_transform(original_x)))
        for e in range(n_epochs):
            model.phase = 'train'
            model.setup(*train_feed.shapes)
            learn_rule.learn_rate = annealer.value(e) / batch_size
            trainer.train_epoch()

            model.phase = 'test'
            samples_x = model.decode(samples_z)
            samples_x = mnist_inverse_transform(model.decode(samples_z))
            sample_video.append(dp.misc.img_tile(samples_x))
    except KeyboardInterrupt:
        pass
    print('Saving model to disk')
    with open(arch_path, 'wb') as f:
        pickle.dump((decoder, discriminator), f)

    model.phase = 'test'
    n_examples = 100
    samples_z = np.random.normal(size=(n_examples, n_hidden)).astype(dp.float_)
    output.samples(model, samples_z, out_dir, mnist_inverse_transform)
    output.walk(model, samples_z, out_dir, mnist_inverse_transform)


if __name__ == '__main__':
    run()
