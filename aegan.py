import os
import numpy as np
import scipy as sp
import deeppy as dp

import architectures
from model import aegan
from video import Video
import output
from dataset.util import img_inverse_transform


def build_model(experiment_name, img_size, n_hidden=128, recon_depth=9,
                recon_vs_gan_weight=5e-5, real_vs_gen_weight=0.5,
                discriminate_sample_z=True, discriminate_ae_recon=True,
                wgain=1.0, wdecay=1e-5, bn_noise_std=0.0):
    if n_hidden != 128:
        experiment_name += '_nhidden%i' % n_hidden
    experiment_name += '_reconganweight%.1e' % recon_vs_gan_weight
    if recon_depth > 0:
        experiment_name += '_recondepth%i' % recon_depth
    if not np.isclose(real_vs_gen_weight, 0.5):
        experiment_name += '_realgenweight%.2f' % real_vs_gen_weight
    if not discriminate_sample_z:
        experiment_name += '_nodissamplez'
    if not discriminate_ae_recon:
        experiment_name += '_nodisaerecon'
    if not np.isclose(wgain, 1.0):
        experiment_name += '_wgain%.1e' % wgain
    if not np.isclose(wdecay, 1e-5):
        experiment_name += '_wdecay%.1e' % wdecay
    if not np.isclose(bn_noise_std, 0.0):
        experiment_name += '_bnnoise%.2f' % bn_noise_std

    # Setup network
    if img_size == 32:
        encoder, decoder, discriminator = architectures.img32x32(
            wgain=wgain, wdecay=wdecay, bn_noise_std=bn_noise_std
        )
    elif img_size == 64:
        encoder, decoder, discriminator = architectures.img64x64(
            wgain=wgain, wdecay=wdecay, bn_noise_std=bn_noise_std
        )
    else:
        raise ValueError('no architecture for img_size %i' % img_size)
    latent_encoder = architectures.vae_latent_encoder(n_hidden)
    model = aegan.AEGAN(
        encoder=encoder,
        latent_encoder=latent_encoder,
        decoder=decoder,
        discriminator=discriminator,
        recon_depth=recon_depth,
        discriminate_sample_z=discriminate_sample_z,
        discriminate_ae_recon=discriminate_ae_recon,
        recon_vs_gan_weight=recon_vs_gan_weight,
        real_vs_gen_weight=real_vs_gen_weight,
    )
    return model, experiment_name


def train(model, output_dir, train_feed, test_feed, lr_start=0.01,
          lr_stop=0.00001, lr_gamma=0.75, n_epochs=150, gan_margin=0.35):
    n_hidden = model.latent_encoder.n_out

    # For plotting
    original_x = np.array(test_feed.batches().next()[0])
    samples_z = np.random.normal(size=(len(original_x), n_hidden))
    samples_z = (samples_z).astype(dp.float_)
    recon_video = Video(os.path.join(output_dir, 'convergence_recon.mp4'))
    sample_video = Video(os.path.join(output_dir, 'convergence_samples.mp4'))
    original_x_ = original_x
    original_x_ = img_inverse_transform(original_x)
    sp.misc.imsave(os.path.join(output_dir, 'examples.png'),
                   dp.misc.img_tile(original_x_))

    # Train network
    learn_rule = dp.RMSProp()
    annealer = dp.GammaAnnealer(lr_start, lr_stop, n_epochs, gamma=lr_gamma)
    trainer = aegan.GradientDescent(model, train_feed, learn_rule,
                                    margin=gan_margin)
    try:
        for e in range(n_epochs):
            model.phase = 'train'
            model.setup(*train_feed.shapes)
            learn_rule.learn_rate = annealer.value(e) / train_feed.batch_size
            trainer.train_epoch()
            model.phase = 'test'
            original_z = model.encode(original_x)
            recon_x = model.decode(original_z)
            samples_x = model.decode(samples_z)
            recon_x = img_inverse_transform(recon_x)
            samples_x = img_inverse_transform(samples_x)
            recon_video.append(dp.misc.img_tile(recon_x))
            sample_video.append(dp.misc.img_tile(samples_x))
    except KeyboardInterrupt:
        pass

    model.phase = 'test'
    n_examples = 100
    test_feed.reset()
    original_x = np.array(test_feed.batches().next()[0])[:n_examples]
    samples_z = np.random.normal(size=(n_examples, n_hidden))
    output.samples(model, samples_z, output_dir, img_inverse_transform)
    output.reconstructions(model, original_x, output_dir,
                           img_inverse_transform)
    original_z = model.encode(original_x)
    output.walk(model, original_z, output_dir, img_inverse_transform)
    return model
