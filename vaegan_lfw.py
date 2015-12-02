#!/usr/bin/env python

import pickle
import numpy as np
import scipy as sp
import deeppy as dp
import deeppy.expr as expr

import vaegan
import lfw
from util import img_tile, random_walk
from video import Video


def affine(n_out, gain):
    return expr.nnet.Affine(n_out=n_out, weights=dp.AutoFiller(gain))


def conv(n_filters, filter_size, stride=1, gain=1.0):
    return expr.nnet.Convolution(
        n_filters=n_filters, strides=(stride, stride),
        weights=dp.AutoFiller(gain), filter_shape=(filter_size, filter_size),
        border_mode='same',
    )

def backconv(n_filters, filter_size, stride=2, gain=1.0):
    return expr.nnet.BackwardConvolution(
        n_filters=n_filters, strides=(stride, stride),
        weights=dp.AutoFiller(gain), filter_shape=(filter_size, filter_size),
        border_mode='same',
    )


def pool(method='max'):
    return expr.nnet.Pool(win_shape=(3, 3), method=method, strides=(2, 2),
                          border_mode='same')


def upscale():
    return expr.nnet.Rescale(factor=2, method='perforated')


def model_expressions(img_shape):
    n_channels = img_shape[0]
    gain = 1.0
    sigma = 0.001
    n_encoder = 1024
    n_discriminator = 1024
    n_hidden = 512
    hidden_shape = (128, 8, 8)
    n_generator = np.prod(hidden_shape)

    encoder = expr.Sequential([
        conv(32, 5, gain=gain),
        pool(),
        expr.nnet.ReLU(),
        conv(64, 5, gain=gain),
        pool(),
        expr.nnet.ReLU(),
        conv(128, 5, gain=gain),
        expr.nnet.SpatialBatchNormalization(),
        pool(),
        expr.nnet.ReLU(),
        conv(128, 3, gain=gain),
        expr.nnet.ReLU(),
        expr.Reshape((-1, 128*8*8)),
        affine(n_encoder, gain),
        expr.nnet.ReLU(),
    ])
    sampler = vaegan.NormalSampler(
        n_hidden,
        weight_filler=dp.AutoFiller(gain),
        bias_filler=dp.NormalFiller(sigma),
    )
    generator = expr.Sequential([
        affine(n_generator, gain),
        expr.nnet.BatchNormalization(),
        expr.Reshape((-1,) + hidden_shape),
        expr.nnet.ReLU(),
        backconv(256, 5, gain=gain),
        expr.nnet.SpatialBatchNormalization(),
        expr.nnet.ReLU(),
        backconv(256, 5, gain=gain),
        expr.nnet.SpatialBatchNormalization(),
        expr.nnet.ReLU(),
        backconv(n_channels, 5, gain=gain),
        expr.Tanh(),
    ])
    discriminator = expr.Sequential([
        conv(32, 5, stride=2, gain=gain),
        expr.nnet.ReLU(),
        conv(64, 5, stride=2, gain=gain),
        expr.nnet.SpatialBatchNormalization(),
        expr.nnet.ReLU(),
        expr.nnet.SpatialDropout(0.2),
        conv(96, 5, stride=2, gain=gain),
        expr.nnet.SpatialBatchNormalization(),
        expr.nnet.ReLU(),
        expr.nnet.SpatialDropout(0.2),
        expr.Reshape((-1, 96*8*8)),
        affine(n_discriminator, gain),
        expr.nnet.BatchNormalization(),
        expr.nnet.ReLU(),
        expr.nnet.Dropout(0.25),
        affine(1, gain),
        expr.nnet.Sigmoid(),
    ])
    return encoder, sampler, generator, discriminator


def clip_range(imgs):
    return ((imgs+1)*0.5*255).astype(np.uint8)


def run():
    mode = 'vaegan'
    vae_grad_scale = 0.0001
    kld_weight = 1.0
    z_gan_prop = False

    experiment_name = mode
    experiment_name += '_scale%.1e' % vae_grad_scale
    experiment_name += '_kld%.2f' % kld_weight
    if z_gan_prop:
        experiment_name += '_zprop'

    filename = 'savestates/lfw_' + experiment_name + '.pickle'
    in_filename = None

    print('experiment_name', experiment_name)
    print('in_filename', in_filename)
    print('filename', filename)

    # Fetch dataset
    x_train = lfw.lfw_imgs(alignment='deepfunneled', size=64, crop=50,
                           shuffle=True)
    img_shape = x_train.shape[1:]

    # Normalize pixel intensities
    scaler = dp.UniformScaler(low=-1, high=1)
    x_train = scaler.fit_transform(x_train)

    # Setup network
    if in_filename is None:
        print('Creating new model')
        expressions = model_expressions(img_shape)
    else:
        print('Starting from %s' % in_filename)
        with open(in_filename, 'rb') as f:
            expressions = pickle.load(f)

    encoder, sampler, generator, discriminator = expressions
    model = vaegan.VAEGAN(
        encoder=encoder,
        sampler=sampler,
        generator=generator,
        discriminator=discriminator,
        mode=mode,
        vae_grad_scale=vae_grad_scale,
        kld_weight=kld_weight,
    )

    # Prepare network inputs
    batch_size = 64
    train_input = dp.Input(x_train, batch_size=batch_size, epoch_size=250)

    # Plotting
    n_examples = 100
    examples = x_train[:n_examples]
    samples_z = np.random.normal(size=(n_examples, model.sampler.n_hidden))
    samples_z = samples_z.astype(dp.float_)


    recon_video = Video('plots/lfw_' + experiment_name + '_reconstruction.mp4')
    sample_video = Video('plots/lfw_' + experiment_name + '_samples.mp4')
    sp.misc.imsave('lfw_examples.png', img_tile(dp.misc.to_b01c(examples)))


    def plot():
        model.phase = 'test'
        examples_z = model.embed(examples)
        reconstructed = clip_range(model.reconstruct(examples_z))
        recon_video.append(img_tile(dp.misc.to_b01c(reconstructed)))
        z = model.embed(x_train)
        z_mean = np.mean(z, axis=0)
        z_std = np.std(z, axis=0)
        model.hidden_std = z_std
        z_std = np.diagflat(z_std)
        samples_z = np.random.multivariate_normal(mean=z_mean, cov=z_std,
                                                  size=(n_examples,))
        samples_z = samples_z.astype(dp.float_)
        samples = clip_range(model.reconstruct(samples_z))
        sample_video.append(img_tile(dp.misc.to_b01c(samples)))

        model.phase = 'train'
        model.setup(**train_input.shapes)

    # Train network
    runs = [
        (150, dp.RMSProp(learn_rate=0.05)),
        (250, dp.RMSProp(learn_rate=0.03)),
        (100, dp.RMSProp(learn_rate=0.01)),
        (15, dp.RMSProp(learn_rate=0.005)),
    ]
    try:
        import timeit
        for n_epochs, learn_rule in runs:
            if mode == 'vae':
                vaegan.train(model, train_input, learn_rule, n_epochs,
                             epoch_callback=plot)
            else:
                vaegan.margin_train(model, train_input, learn_rule, n_epochs,
                                    epoch_callback=plot)
    except KeyboardInterrupt:
        pass

    raw_input('\n\nsave model to %s?\n' % filename)
    with open(filename, 'wb') as f:
        expressions = encoder, sampler, generator, discriminator
        pickle.dump(expressions, f)


    model.phase = 'test'
    batch_size = 128
    model.sampler.batch_size=128
    z = model.embed(x_train)
    z_mean = np.mean(z, axis=0)
    z_std = np.std(z, axis=0)
    z_cov = np.cov(z.T)
    print(np.mean(z_mean), np.std(z_mean))
    print(np.mean(z_std), np.std(z_std))
    print(z_mean.shape, z_std.shape, z_cov.shape)

    model.sampler.batch_size=100
    samples_z = model.embed(examples)

    print('Generating latent space video')
    walk_video = Video('plots/lfw_' + experiment_name + '_walk.mp4')
    for z in random_walk(samples_z, 500, n_dir_steps=10, mean=z_mean, std=z_cov):
        samples = clip_range(model.reconstruct(z))
        walk_video.append(img_tile(dp.misc.to_b01c(samples)))



if __name__ == '__main__':
    run()
