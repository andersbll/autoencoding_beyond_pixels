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


def conv(n_filters, filter_size, gain=1.0):
    return expr.nnet.Convolution(
        n_filters=n_filters, strides=(1, 1), weights=dp.AutoFiller(gain),
        filter_shape=(filter_size, filter_size), border_mode='same',
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
        upscale(),
        conv(256, 5, gain=gain),
        expr.nnet.SpatialBatchNormalization(),
        expr.nnet.ReLU(),
        upscale(),
        conv(256, 5, gain=gain),
        expr.nnet.SpatialBatchNormalization(),
        expr.nnet.ReLU(),
        upscale(),
        conv(128, 5, gain=gain),
        expr.nnet.SpatialBatchNormalization(),
        expr.nnet.ReLU(),
        conv(n_channels, 3, gain=gain),
    ])
    discriminator = expr.Sequential([
        conv(32, 5, gain=gain),
        pool(),
        expr.nnet.ReLU(),
        expr.nnet.SpatialDropout(0.2),
        conv(64, 5, gain=gain),
        pool(),
        expr.nnet.ReLU(),
        expr.nnet.SpatialDropout(0.2),
        conv(96, 5, gain=gain),
        pool(),
        expr.nnet.ReLU(),
        expr.nnet.SpatialDropout(0.2),
        expr.Reshape((-1, 96*8*8)),
        affine(n_discriminator, gain),
        expr.nnet.ReLU(),
        expr.nnet.Dropout(0.5),
        affine(1, gain),
        expr.nnet.Sigmoid(),
    ])
    return encoder, sampler, generator, discriminator


def clip_range(imgs):
    return np.tanh(imgs*0.5)


def run():
    mode = 'gan'
    experiment_name = mode
    filename = 'savestates/lfw_' + experiment_name + '.pickle'
    in_filename = filename
    in_filename = None
    print('experiment_name', experiment_name)
    print('in_filename', in_filename)
    print('filename', filename)

    # Fetch dataset
    x_train = lfw.lfw_imgs(alignment='deepfunneled', size=64, crop=50,
                           shuffle=True)
    img_shape = x_train.shape[1:]

    # Normalize pixel intensities
    scaler = dp.StandardScaler()
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
    )

    # Prepare network inputs
    batch_size = 64
    train_input = dp.Input(x_train, batch_size=batch_size, epoch_size=75)

    # Plotting
    n_examples = 100
    examples = x_train[:n_examples]
    samples_z = np.random.normal(size=(n_examples, model.sampler.n_hidden))
    samples_z = samples_z.astype(dp.float_)

    recon_video = Video('plots/lfw_' + experiment_name + '_reconstruction.mp4')
    sample_video = Video('plots/lfw_' + experiment_name + '_samples.mp4')
    sp.misc.imsave('lfw_examples.png', img_tile(dp.misc.to_b01c(examples)))

    def plot():
        examples_z = model.embed(examples)
        reconstructed = clip_range(model.reconstruct(examples_z))
        recon_video.append(img_tile(dp.misc.to_b01c(reconstructed)))
        samples = clip_range(model.reconstruct(samples_z))
        sample_video.append(img_tile(dp.misc.to_b01c(samples)))
        model.setup(**train_input.shapes)

    # Train network
    runs = [
        (150, dp.RMSProp(learn_rate=0.07)),
        (150, dp.RMSProp(learn_rate=0.06)),
        (100, dp.RMSProp(learn_rate=0.05)),
        (100, dp.RMSProp(learn_rate=0.03)),
        (50, dp.RMSProp(learn_rate=0.025)),
        (25, dp.RMSProp(learn_rate=0.0125)),
        (5, dp.RMSProp(learn_rate=0.005)),
    ]
    try:
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

    print('Generating latent space video')
    walk_video = Video('plots/lfw_' + experiment_name + '_walk.mp4')
    for z in random_walk(samples_z, 500, step_std=0.15):
        samples = clip_range(model.reconstruct(z))
        walk_video.append(img_tile(dp.misc.to_b01c(samples)))


if __name__ == '__main__':
    run()
