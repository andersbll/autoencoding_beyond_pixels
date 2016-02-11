import numpy as np
import deeppy as dp
import deeppy.expr as expr

import model.ae


def affine(n_out, gain, wdecay=0.0, bias=0.0):
    return expr.nnet.Affine(
        n_out=n_out, bias=bias,
        weights=dp.Parameter(dp.AutoFiller(gain), weight_decay=wdecay),
    )


def conv(n_filters, filter_size, stride=1, gain=1.0, wdecay=0.0,
         bias=0.0, border_mode='same'):
    return expr.nnet.Convolution(
        n_filters=n_filters, strides=(stride, stride),
        weights=dp.Parameter(dp.AutoFiller(gain), weight_decay=wdecay),
        bias=bias, filter_shape=(filter_size, filter_size),
        border_mode=border_mode,
    )


def backconv(n_filters, filter_size, stride=2, gain=1.0, wdecay=0.0,
             bias=0.0):
    return expr.nnet.BackwardConvolution(
        n_filters=n_filters, strides=(stride, stride),
        weights=dp.Parameter(dp.AutoFiller(gain), weight_decay=wdecay),
        bias=bias, filter_shape=(filter_size, filter_size), border_mode='same',
    )


def pool(method='max', win_size=3, stride=2, border_mode='same'):
    return expr.nnet.Pool(win_shape=(win_size, win_size), method=method,
                          strides=(stride, stride), border_mode=border_mode)


def vae_latent_encoder(n_hidden):
    latent_encoder = model.ae.NormalEncoder(n_hidden, dp.AutoFiller())
    return latent_encoder


def aae_latent_encoder(n_hidden, n_discriminator=1024, recon_weight=0.025):
    wgain = 1.0
    discriminator = dp.expr.Sequential([
        affine(n_discriminator, wgain, bias=None),
        expr.nnet.BatchNormalization(),
        expr.nnet.ReLU(),
        affine(n_discriminator, wgain, bias=None),
        expr.nnet.BatchNormalization(),
        expr.nnet.ReLU(),
        affine(1, wgain),
        expr.nnet.Sigmoid(),
    ])
    latent_encoder = model.ae.AdversarialEncoder(
        n_hidden, discriminator, dp.AutoFiller(), recon_weight=recon_weight,
    )
    return latent_encoder


def mnist(wgain=1.0, wdecay=0, bn_noise_std=0.0, n_units=1024):
    img_shape = (28, 28)
    n_in = np.prod(img_shape)
    n_encoder = n_units
    n_decoder = n_units
    n_discriminator = n_units

    def block(n_out):
        return [
            affine(n_encoder, wgain, wdecay=wdecay),
            expr.nnet.BatchNormalization(noise_std=bn_noise_std),
            expr.nnet.ReLU(),
        ]
    encoder = dp.expr.Sequential(
        block(n_encoder) +
        block(n_encoder)
    )
    decoder = dp.expr.Sequential(
        block(n_decoder) +
        block(n_decoder) +
        [
            affine(n_in, wgain),
            expr.nnet.Sigmoid(),
        ]
    )
    discriminator = dp.expr.Sequential(
        block(n_discriminator) +
        block(n_discriminator) +
        [
            affine(1, wgain),
            expr.nnet.Sigmoid(),
        ]
    )
    return encoder, decoder, discriminator


def img32x32(wgain=1.0, wdecay=1e-5, bn_mom=0.9, bn_eps=1e-6,
             bn_noise_std=0.0):
    n_channels = 3
    n_encoder = 1024
    n_discriminator = 512
    decode_from_shape = (256, 4, 4)
    n_decoder = np.prod(decode_from_shape)

    def conv_block(n_filters, backward=False):
        block = []
        if backward:
            block.append(backconv(n_filters, 5, stride=2, gain=wgain,
                                  wdecay=wdecay, bias=None))
        else:
            block.append(conv(n_filters, 5, stride=2, gain=wgain,
                              wdecay=wdecay, bias=None))
        block.append(expr.nnet.SpatialBatchNormalization(
            momentum=bn_mom, eps=bn_eps, noise_std=bn_noise_std
        ))
        block.append(expr.nnet.ReLU())
        return block

    encoder = dp.expr.Sequential(
        conv_block(64) +
        conv_block(128) +
        conv_block(256) +
        [
            expr.Reshape((-1, 256*4*4)),
            affine(n_encoder, gain=wgain, wdecay=wdecay, bias=None),
            expr.nnet.BatchNormalization(noise_std=bn_noise_std),
            expr.nnet.ReLU(),
        ]
    )

    decoder = dp.expr.Sequential(
        [
            affine(n_decoder, gain=wgain, wdecay=wdecay, bias=None),
            expr.nnet.BatchNormalization(noise_std=bn_noise_std),
            expr.nnet.ReLU(),
            expr.Reshape((-1,) + decode_from_shape),
        ] +
        conv_block(192, backward=True) +
        conv_block(128, backward=True) +
        conv_block(32, backward=True) +
        [
            conv(n_channels, 5, wdecay=wdecay, gain=wgain),
            expr.Tanh(),
        ]
    )

    discriminator = dp.expr.Sequential(
        [
            conv(32, 5, wdecay=wdecay, gain=wgain),
            expr.nnet.ReLU(),
        ] +
        conv_block(128) +
        conv_block(192) +
        conv_block(256) +
        [
            expr.Reshape((-1, 256*4*4)),
            affine(n_discriminator, gain=wgain, wdecay=wdecay, bias=None),
            expr.nnet.BatchNormalization(noise_std=bn_noise_std),
            expr.nnet.ReLU(),
            affine(1, gain=wgain, wdecay=wdecay),
            expr.nnet.Sigmoid(),
        ]
    )
    return encoder, decoder, discriminator


def img64x64(wgain=1.0, wdecay=1e-5, bn_mom=0.9, bn_eps=1e-6,
             bn_noise_std=0.0):
    n_channels = 3
    n_encoder = 1024
    n_discriminator = 512
    decode_from_shape = (256, 8, 8)
    n_decoder = np.prod(decode_from_shape)

    def conv_block(n_filters, backward=False):
        block = []
        if backward:
            block.append(backconv(n_filters, 5, stride=2, gain=wgain,
                                  wdecay=wdecay, bias=None))
        else:
            block.append(conv(n_filters, 5, stride=2, gain=wgain,
                              wdecay=wdecay, bias=None))
        block.append(expr.nnet.SpatialBatchNormalization(
            momentum=bn_mom, eps=bn_eps, noise_std=bn_noise_std
        ))
        block.append(expr.nnet.ReLU())
        return block

    encoder = dp.expr.Sequential(
        conv_block(64) +
        conv_block(128) +
        conv_block(256) +
        [
            expr.Reshape((-1, 256*8*8)),
            affine(n_encoder, gain=wgain, wdecay=wdecay, bias=None),
            expr.nnet.BatchNormalization(noise_std=bn_noise_std),
            expr.nnet.ReLU(),
        ]
    )

    decoder = dp.expr.Sequential(
        [
            affine(n_decoder, gain=wgain, wdecay=wdecay, bias=None),
            expr.nnet.BatchNormalization(noise_std=bn_noise_std),
            expr.nnet.ReLU(),
            expr.Reshape((-1,) + decode_from_shape),
        ] +
        conv_block(256, backward=True) +
        conv_block(128, backward=True) +
        conv_block(32, backward=True) +
        [
            conv(n_channels, 5, wdecay=wdecay, gain=wgain),
            expr.Tanh(),
        ]
    )

    discriminator = dp.expr.Sequential(
        [
            conv(32, 5, wdecay=wdecay, gain=wgain),
            expr.nnet.ReLU(),
        ] +
        conv_block(128) +
        conv_block(256) +
        conv_block(256) +
        [
            expr.Reshape((-1, 256*8*8)),
            affine(n_discriminator, gain=wgain, wdecay=wdecay, bias=None),
            expr.nnet.BatchNormalization(noise_std=bn_noise_std),
            expr.nnet.ReLU(),
            affine(1, gain=wgain, wdecay=wdecay),
            expr.nnet.Sigmoid(),
        ]
    )
    return encoder, decoder, discriminator
