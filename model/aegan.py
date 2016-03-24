from copy import deepcopy
import numpy as np
import cudarray as ca
import deeppy as dp
import deeppy.expr as ex

from util import ScaleGradient, WeightedParameter
from ae import NLLNormal


class AEGAN(dp.base.Model, dp.base.CollectionMixin):
    def __init__(self, encoder, latent_encoder, decoder, discriminator,
                 recon_depth=0, discriminate_sample_z=True,
                 discriminate_ae_recon=True, recon_vs_gan_weight=1e-2,
                 real_vs_gen_weight=0.5, eps=1e-3):
        self.encoder = encoder
        self.latent_encoder = latent_encoder
        self.discriminator = discriminator
        self.recon_depth = recon_depth
        self.discriminate_sample_z = discriminate_sample_z
        self.discriminate_ae_recon = discriminate_ae_recon
        self.recon_vs_gan_weight = recon_vs_gan_weight
        self.real_vs_gen_weight = real_vs_gen_weight
        self.eps = eps
        self.recon_error = NLLNormal()
        self.decoder = decoder
        self.collection = [self.encoder, self.latent_encoder, self.decoder,
                           self.discriminator]
        decoder.params = [WeightedParameter(p, self.recon_vs_gan_weight,
                                            -(1.0-self.recon_vs_gan_weight))
                          for p in decoder.params]
        self.decoder_neggrad = deepcopy(decoder)
        self.decoder_neggrad.params = [p.share() for p in decoder.params]
        self.collection += [self.decoder_neggrad]
        if recon_depth > 0:
            recon_layers = discriminator.collection[:recon_depth]
            print('Reconstruction error at layer #%i: %s'
                  % (recon_depth, recon_layers[-1].__class__.__name__))
            dis_layers = discriminator.collection[recon_depth:]
            discriminator.collection = recon_layers
            discriminator.params = [WeightedParameter(p, 1.0, 0.0)
                                    for p in discriminator.params]
            self.discriminator_recon = deepcopy(discriminator)
            self.discriminator_recon.params = [p.share() for p in
                                               discriminator.params]
            discriminator.collection += dis_layers
            self.collection += [self.discriminator_recon]

    def _encode_expr(self, x, batch_size):
        enc = self.encoder(x)
        z, encoder_loss = self.latent_encoder.encode(enc, batch_size)
        return z

    def _decode_expr(self, z, batch_size):
        return self.decoder(z)

    def setup(self, x_shape):
        batch_size = x_shape[0]
        self.x_src = ex.Source(x_shape)
        loss = 0
        # Encode
        enc = self.encoder(self.x_src)
        z, self.encoder_loss = self.latent_encoder.encode(enc, batch_size)
        loss += self.encoder_loss
        # Decode
        x_tilde = self.decoder(z)
        if self.recon_depth > 0:
            # Reconstruction error in discriminator
            x = ex.Concatenate(axis=0)(x_tilde, self.x_src)
            d = self.discriminator_recon(x)
            d_x_tilde, d_x = ex.Slices([batch_size])(d)
            loss += self.recon_error(d_x_tilde, d_x)
        else:
            loss += self.recon_error(x_tilde, self.x_src)
        # Kill gradient from GAN loss to AE encoder
        z = ScaleGradient(0.0)(z)
        # Decode for GAN loss
        gen_size = 0
        if self.discriminate_ae_recon:
            gen_size += batch_size
            # Kill gradient from GAN loss to AE encoder
            z = ScaleGradient(0.0)(z)
        if self.discriminate_sample_z:
            gen_size += batch_size
            z_samples = self.latent_encoder.samples(batch_size)
            if self.discriminate_ae_recon:
                z = ex.Concatenate(axis=0)(z, z_samples)
            else:
                z = z_samples
        if gen_size == 0:
            raise ValueError('GAN does not receive any generated samples.')
        x = self.decoder_neggrad(z)
        x = ex.Concatenate(axis=0)(self.x_src, x)
        # Scale gradients to balance real vs. generated contributions to GAN
        # discriminator
        dis_batch_size = batch_size + gen_size
        real_weight = self.real_vs_gen_weight
        gen_weight = (1-self.real_vs_gen_weight) * float(batch_size)/gen_size
        weights = np.zeros((dis_batch_size,))
        weights[:batch_size] = real_weight
        weights[batch_size:] = gen_weight
        dis_weights = ca.array(weights)
        shape = np.array(x_shape)**0
        shape[0] = dis_batch_size
        dis_weights_inv = ca.array(1.0 / np.reshape(weights, shape))
        x = ScaleGradient(dis_weights_inv)(x)
        # Discriminate
        d = self.discriminator(x)
        d = ex.Reshape((-1,))(d)
        d = ScaleGradient(dis_weights)(d)
        sign = np.ones((gen_size + batch_size,), dtype=ca.float_)
        sign[batch_size:] = -1.0
        offset = np.zeros_like(sign)
        offset[batch_size:] = 1.0
        self.gan_loss = ex.log(d*sign + offset + self.eps)
        self.loss = ex.sum(loss) - ex.sum(self.gan_loss)
        self._graph = ex.graph.ExprGraph(self.loss)
        self._graph.setup()
        self.loss.grad_array = ca.array(1.0)

    @property
    def params(self):
        enc_params = self.encoder.params + self.latent_encoder.params
        dec_params = self.decoder.params
        dis_params = self.discriminator.params
        return enc_params, dec_params, dis_params

    def update(self, x):
        self.x_src.array = x
        self._graph.fprop()
        self._graph.bprop()
        encoder_loss = 0
        d_x_loss = 0
        d_z_loss = 0
        encoder_loss = np.array(self.encoder_loss.array)
        gan_loss = -np.array(self.gan_loss.array)
        batch_size = x.shape[0]
        d_x_loss = float(np.mean(gan_loss[:batch_size]))
        d_z_loss = float(np.mean(gan_loss[batch_size:]))
        return d_x_loss, d_z_loss, encoder_loss

    def _batchwise(self, feed, expr_fun):
        feed = dp.Feed.from_any(feed)
        src = ex.Source(feed.x_shape)
        sink = expr_fun(src, feed.batch_size)
        graph = ex.graph.ExprGraph(sink)
        graph.setup()
        z = []
        for x, in feed.batches():
            src.array = x
            graph.fprop()
            z.append(np.array(sink.array))
        z = np.concatenate(z)[:feed.n_samples]
        return z

    def encode(self, feed):
        return self._batchwise(feed, self._encode_expr)

    def decode(self, feed):
        return self._batchwise(feed, self._decode_expr)


class GradientDescent(dp.GradientDescent):
    def __init__(self, model, feed, learn_rule, margin=0.4, equilibrium=0.68):
        super(GradientDescent, self).__init__(model, feed, learn_rule)
        self.margin = margin
        self.equilibrium = equilibrium

    def reset(self):
        self.feed.reset()
        self.model.setup(*self.feed.shapes)
        self.params_enc, self.params_dec, self.params_dis = self.model.params

        def states(params):
            return [self.learn_rule.init_state(p) for p in params
                    if not isinstance(p, dp.parameter.SharedParameter)]
        self.lstates_enc = states(self.params_enc)
        self.lstates_dec = states(self.params_dec)
        self.lstates_dis = states(self.params_dis)

    def train_epoch(self):
        batch_costs = []
        for batch in self.feed.batches():
            real_cost, fake_cost, encoder = self.model.update(*batch)
            batch_costs.append((real_cost, fake_cost, encoder))
            dec_update = True
            dis_update = True
            if self.margin is not None:
                if real_cost < self.equilibrium - self.margin or \
                   fake_cost < self.equilibrium - self.margin:
                    dis_update = False
                if real_cost > self.equilibrium + self.margin or \
                   fake_cost > self.equilibrium + self.margin:
                    dec_update = False
                if not (dec_update or dis_update):
                    dec_update = True
                    dis_update = True
            for param, state in zip(self.params_enc, self.lstates_enc):
                self.learn_rule.step(param, state)
            if dec_update:
                for param, state in zip(self.params_dec, self.lstates_dec):
                    self.learn_rule.step(param, state)
            if dis_update:
                for param, state in zip(self.params_dis, self.lstates_dis):
                    self.learn_rule.step(param, state)
        real_cost = np.mean([cost[0] for cost in batch_costs])
        fake_cost = np.mean([cost[1] for cost in batch_costs])
        encoder = np.mean([c[2] for c in batch_costs])
        return real_cost + fake_cost + encoder
