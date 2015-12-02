#!/usr/bin/env python

import pickle
import numpy as np
import scipy as sp
import deeppy as dp
import deeppy.expr as expr

import cond_vaegan
import vaegan
from util import img_tile, one_hot, random_walk
from video import Video


def affine(n_out, gain):
    return expr.nnet.Affine(n_out=n_out, weights=dp.AutoFiller(gain))


def model_expressions(img_shape):
    gain = 1.0
    sigma = 0.001
    n_in = np.prod(img_shape)
    n_encoder = 1024
    n_hidden = 32
    n_generator = 2048
    n_discriminator = 2048

    encoder = cond_vaegan.ConditionalSequential([
        expr.Concatenate(axis=1),
        affine(n_encoder, gain),
        expr.nnet.BatchNormalization(),
        expr.nnet.ReLU(),
        expr.nnet.Dropout(0.5),
        expr.Concatenate(axis=1),
        affine(n_encoder, gain),
        expr.nnet.BatchNormalization(),
        expr.nnet.ReLU(),
    ])
    sampler = vaegan.NormalSampler(
        n_hidden,
        weight_filler=dp.AutoFiller(gain),
        bias_filler=dp.NormalFiller(sigma),
    )
    generator = cond_vaegan.ConditionalSequential([
        expr.Concatenate(axis=1),
        affine(n_generator, gain),
        expr.nnet.BatchNormalization(),
        expr.nnet.ReLU(),
        expr.Concatenate(axis=1),
        affine(n_generator, gain),
        expr.nnet.BatchNormalization(),
        expr.nnet.ReLU(),
        expr.Concatenate(axis=1),
        affine(n_in, gain),
        expr.nnet.Sigmoid(),
    ])
    discriminator = cond_vaegan.ConditionalSequential([
        expr.Concatenate(axis=1),
        affine(n_discriminator, gain),
        expr.nnet.ReLU(),
        expr.nnet.Dropout(0.25),
        expr.Concatenate(axis=1),
        affine(n_discriminator, gain),
        expr.nnet.BatchNormalization(),
        expr.nnet.ReLU(),
        expr.nnet.Dropout(0.25),
        expr.Concatenate(axis=1),
        affine(1, gain),
        expr.nnet.Sigmoid(),

    ])
    return encoder, sampler, generator, discriminator


def to_b01c(imgs_flat, img_shape):
    imgs = np.reshape(imgs_flat, (-1,) + img_shape)
    return dp.misc.to_b01c(imgs)


def run():
    mode = 'vaegan'
    vae_grad_scale = 0.025
    experiment_name = mode + 'scale_%.5f' % vae_grad_scale
    filename = 'savestates/mnist_' + experiment_name + '.pickle'
    in_filename = filename
    in_filename = None
    print('experiment_name', experiment_name)
    print('in_filename', in_filename)
    print('filename', filename)

    # Fetch dataset
    dataset = dp.dataset.MNIST()
    x_train, y_train, x_test, y_test = dataset.arrays(dp_dtypes=True)
    n_classes = dataset.n_classes
    img_shape = x_train.shape[1:]

    # Normalize pixel intensities
    scaler = dp.UniformScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    y_train = one_hot(y_train, n_classes).astype(dp.float_)
    y_test = one_hot(y_test, n_classes).astype(dp.float_)
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))


    # Setup network
    if in_filename is None:
        print('Creating new model')
        expressions = model_expressions(img_shape)
    else:
        print('Starting from %s' % in_filename)
        with open(in_filename, 'rb') as f:
            expressions = pickle.load(f)

    encoder, sampler, generator, discriminator = expressions
    model = cond_vaegan.ConditionalVAEGAN(
        encoder=encoder,
        sampler=sampler,
        generator=generator,
        discriminator=discriminator,
        mode=mode,
        reconstruct_error=expr.nnet.BinaryCrossEntropy(),
        vae_grad_scale=vae_grad_scale,
    )

    # Prepare network inputs
    batch_size = 128
    train_input = dp.SupervisedInput(x_train, y_train, batch_size=batch_size,
                                     epoch_size=250)

    # Plotting
    n_examples = 100
    examples = x_test[:n_examples]
    examples_y = y_test[:n_examples]
    samples_z = np.random.normal(size=(n_examples, model.sampler.n_hidden))
    samples_z = samples_z.astype(dp.float_)
    samples_y = ((np.arange(n_examples) // 10) % n_classes)
    samples_y = one_hot(samples_y, n_classes).astype(dp.float_)

    recon_video = Video('plots/mnist_' + experiment_name +
                        '_reconstruction.mp4')
    sample_video = Video('plots/mnist_' + experiment_name + '_samples.mp4')
    sp.misc.imsave('plots/mnist_examples.png',
                   img_tile(to_b01c(examples, img_shape)))

    def plot():
        model.phase = 'test'
        model.sampler.batch_size=100
        examples_z = model.embed(examples, examples_y)
        examples_recon = model.reconstruct(examples_z, examples_y)
        recon_video.append(img_tile(to_b01c(examples_recon, img_shape)))
        samples = model.reconstruct(samples_z, samples_y)
        sample_video.append(img_tile(to_b01c(samples, img_shape)))
        model.setup(**train_input.shapes)
        model.phase = 'train'


    # Train network
    runs = [
        (75, dp.RMSProp(learn_rate=0.075)),
        (25, dp.RMSProp(learn_rate=0.05)),
        (5, dp.RMSProp(learn_rate=0.01)),
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

    model.phase = 'test'
    batch_size = 128
    model.sampler.batch_size=128
    z = []
    i = 0
    z = model.embed(x_train, y_train)
    print(z.shape)
    z_mean = np.mean(z, axis=0)
    z_std = np.std(z, axis=0)
    z_cov = np.cov(z.T)
    print(np.mean(z_mean), np.std(z_mean))
    print(np.mean(z_std), np.std(z_std))
    print(z_mean.shape, z_std.shape, z_cov.shape)


    raw_input('\n\ngenerate latent space video?\n')
    print('Generating latent space video')
    walk_video = Video('plots/mnist_' + experiment_name + '_walk.mp4')
    for z in random_walk(samples_z, 500, n_dir_steps=10, mean=z_mean, std=z_cov):
        samples = model.reconstruct(z, samples_y)
        walk_video.append(img_tile(to_b01c(samples, img_shape)))



    print('Generating AdversarialMNIST dataset')
    _, y_train, _, y_test = dataset.arrays(dp_dtypes=True)
    n = 0
    batch_size = 512
    advmnist_size = 1e6
    x_advmnist = np.empty((advmnist_size, 28*28))
    y_advmnist = np.empty((advmnist_size,))
    while n < advmnist_size:
        samples_z = np.random.multivariate_normal(mean=z_mean, cov=z_cov,
                                                  size=batch_size)
        samples_z = samples_z.astype(dp.float_)
        start_idx = n % len(y_train)
        stop_idx = (n + batch_size) % len(y_train)
        if start_idx > stop_idx:
            samples_y = np.concatenate([y_train[start_idx:], y_train[:stop_idx]])
        else:
            samples_y = y_train[start_idx:stop_idx]
        y_advmnist[n:n+batch_size] = samples_y[:advmnist_size-n]
        samples_y = one_hot(samples_y, n_classes).astype(dp.float_)
        samples = model.reconstruct(samples_z, samples_y)
        x_advmnist[n:n+batch_size] = samples[:advmnist_size-n]
        n += batch_size


    x_train = x_advmnist
    y_train = y_advmnist
    import sklearn.neighbors
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='brute', n_jobs=-1)
    clf.fit(x_train, y_train)
    print('KNN predict')
    step = 2500
    errors = []
    i = 0
    while i < len(x_test):
        print(i)
        errors.append(clf.predict(x_test[i:i+step]) != y_test[i:i+step])
        i += step
    error = np.mean(errors)
    print('Test error rate: %.4f' % error)

    print('DONE ' + experiment_name)

if __name__ == '__main__':
    run()
