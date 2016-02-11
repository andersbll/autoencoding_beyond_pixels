#!/usr/bin/env python

import os
import pickle

import dataset.cifar
import aegan


def run():
    experiment_name = 'cifar'

    img_size = 32
    epoch_size = 250
    batch_size = 64
    train_input, test_input, _ = dataset.cifar.inputs(
        split='test', batch_size=batch_size, epoch_size=epoch_size
    )

    model, experiment_name = aegan.build_model(
        experiment_name, img_size, n_hidden=128, recon_depth=6,
        recon_vs_gan_weight=5e-6, real_vs_gen_weight=0.66,
    )
    print('experiment_name: %s' % experiment_name)
    output_dir = os.path.join('out', experiment_name)
    aegan.train(
        model, output_dir, train_input, test_input,
    )
    model_path = os.path.join(output_dir, 'arch.pickle')
    print('Saving model to disk')
    print(model_path)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    run()
