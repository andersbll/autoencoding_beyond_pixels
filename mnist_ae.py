#!/usr/bin/env python

import os
import pickle
import numpy as np
import scipy as sp
import deeppy as dp

import architectures
from model import ae
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
    n_hidden = 64
    ae_kind = 'variational'

    lr_start = 0.01
    lr_stop = 0.0001
    lr_gamma = 0.75
    n_epochs = 150
    epoch_size = 250
    batch_size = 64

    experiment_name = 'mnist_ae'
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
        encoder, decoder, _ = architectures.mnist()
        if ae_kind == 'variational':
            latent_encoder = architectures.vae_latent_encoder(n_hidden)
        elif ae_kind == 'adversarial':
            latent_encoder = architectures.aae_latent_encoder(n_hidden)
    else:
        print('Starting from %s' % start_arch_path)
        with open(start_arch_path, 'rb') as f:
            decoder, discriminator = pickle.load(f)

    model = ae.Autoencoder(
        encoder=encoder,
        latent_encoder=latent_encoder,
        decoder=decoder,
    )
    model.recon_error = ae.NLLNormal()

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
    trainer = dp.GradientDescent(model, train_feed, learn_rule)
    annealer = dp.GammaAnnealer(lr_start, lr_stop, n_epochs, gamma=lr_gamma)
    try:
        recon_video = Video(os.path.join(out_dir, 'convergence_recon.mp4'))
        sample_video = Video(os.path.join(out_dir, 'convergence_samples.mp4'))
        sp.misc.imsave(os.path.join(out_dir, 'examples.png'),
                       dp.misc.img_tile(mnist_inverse_transform(original_x)))
        for e in range(n_epochs):
            model.phase = 'train'
            model.setup(*train_feed.shapes)
            learn_rule.learn_rate = annealer.value(e) / batch_size
            loss = trainer.train_epoch()

            model.phase = 'test'
            original_z = model.encode(original_x)
            recon_x = model.decode(original_z)
            samples_x = model.decode(samples_z)
            recon_x = mnist_inverse_transform(recon_x)
            samples_x = mnist_inverse_transform(model.decode(samples_z))
            recon_video.append(dp.misc.img_tile(recon_x))
            sample_video.append(dp.misc.img_tile(samples_x))
            likelihood = model.likelihood(test_feed)
            print('epoch %i   Train loss:%.4f  Test likelihood:%.4f' %
                  (e, np.mean(loss), np.mean(likelihood)))
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
