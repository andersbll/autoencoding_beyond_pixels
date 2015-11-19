from copy import deepcopy
import numpy as np
import cudarray as ca
import deeppy as dp
import deeppy.expr as expr

from vaegan import KLDivergence, NegativeGradient, SquareError


class AppendSpatially(expr.base.Binary):
    def __call__(self, imgs, feats):
        self.imgs = imgs
        self.feats = feats
        self.inputs = [imgs, feats]
        return self

    def setup(self):
        b, c, h, w = self.imgs.out_shape
        b_, f = self.feats.out_shape
        if b != b_:
            raise ValueError('batch size mismatch')
        self.out_shape = (b, c+f, h, w)
        self.out = ca.empty(self.out_shape)
        self.out_grad = ca.empty(self.out_shape)
        self.tmp = ca.zeros((b, f, h, w))

    def fprop(self):
        self.tmp.fill(0.0)
        feats = ca.reshape(self.feats.out, self.feats.out.shape + (1, 1))
        ca.add(feats, self.tmp, out=self.tmp)
        ca.extra.concatenate(self.imgs.out, self.tmp, axis=1, out=self.out)

    def bprop(self):
        ca.extra.split(self.out_grad, a_size=self.imgs.out_shape[1], axis=1,
                       out_a=self.imgs.out_grad, out_b=self.tmp)


class ConditionalSequential(expr.Sequential):
    def __call__(self, x, y):
        for op in self.collection:
            if isinstance(op, (expr.Concatenate, AppendSpatially)):
                x = op(x, y)
            else:
                x = op(x)
        return x


class ConditionalVAEGAN(dp.base.Model):
    def __init__(self, encoder, sampler, generator, discriminator, mode,
                 reconstruct_error=None):
        self.encoder = encoder
        self.sampler = sampler
        self.generator = generator
        self.mode = mode
        self.discriminator = discriminator
        self.eps = 1e-4
        if reconstruct_error is None:
            reconstruct_error = SquareError()
        self.reconstruct_error = reconstruct_error
        if self.mode == 'vaegan':
            self.generator_neg = deepcopy(generator)
            self.generator_neg.params = [p.share() for p in generator.params]

    def _embed_expr(self, x, y):
        h_enc = self.encoder(x, y)
        z, z_mu, z_log_sigma, z_eps = self.sampler(h_enc)
        z = z_mu
        return z

    def _reconstruct_expr(self, z, y):
        return self.generator(z, y)

    def setup(self, x_shape, y_shape):
        batch_size = x_shape[0]
        self.sampler.batch_size = x_shape[0]
        self.x_src = expr.Source(x_shape)
        self.y_src = expr.Source(y_shape)

        if self.mode in ['vae', 'vaegan']:
            h_enc = self.encoder(self.x_src, self.y_src)
            z, z_mu, z_log_sigma, z_eps = self.sampler(h_enc)
            self.kld = KLDivergence()(z_mu, z_log_sigma)
            x_tilde = self.generator(z, self.y_src)
#            if self.mode == 'vaegan':
#                x_tilde = ScaleGradient()(x_tilde)
            self.logpxz = self.reconstruct_error(x_tilde, self.x_src)
            loss = self.kld + expr.sum(self.logpxz)

        if self.mode in ['gan', 'vaegan']:
            y = self.y_src
            if self.mode == 'gan':
                z = self.sampler.samples()
                x_tilde = self.generator(z, y)
                x_tilde = NegativeGradient()(x_tilde)
                gen_size = batch_size
            elif self.mode == 'vaegan':
                z = NegativeGradient()(z)
                z = expr.Concatenate(axis=0)(z, z_eps)
                y = expr.Concatenate(axis=0)(y, self.y_src)
                x_tilde = self.generator_neg(z, y)
                x_tilde = NegativeGradient()(x_tilde)
                gen_size = batch_size*2
            x = expr.Concatenate(axis=0)(self.x_src, x_tilde)
            y = expr.Concatenate(axis=0)(y, self.y_src)
            d = self.discriminator(x, y)
            d = expr.clip(d, self.eps, 1.0-self.eps)

            real_size = batch_size
            sign = np.ones((real_size + gen_size, 1), dtype=ca.float_)
            sign[real_size:] = -1.0
            offset = np.zeros_like(sign)
            offset[real_size:] = 1.0

            self.gan_loss = expr.log(d*sign + offset)
            if self.mode == 'gan':
                loss = expr.sum(-self.gan_loss)
            elif self.mode == 'vaegan':
                loss = loss + expr.sum(-self.gan_loss)

        self._graph = expr.ExprGraph(loss)
        self._graph.out_grad = ca.array(1.0)
        self._graph.setup()

    @property
    def params(self):
        enc_params = []
        gen_params = self.generator.params
        dis_params = []
        if self.mode != 'vae':
            dis_params = self.discriminator.params
        if self.mode != 'gan':
            enc_params = self.encoder.params + self.sampler.params
        return enc_params, gen_params, dis_params

    def update(self, x, y):
        self.x_src.out = x
        self.y_src.out = y
        self._graph.fprop()
        self._graph.bprop()
        kld = 0
        d_x_loss = 0
        d_z_loss = 0
        if self.mode != 'gan':
            kld = np.array(self.kld.out)
        if self.mode != 'vae':
            gan_loss = -np.array(self.gan_loss.out)
            batch_size = x.shape[0]
            d_x_loss = float(np.mean(gan_loss[:batch_size]))
            d_z_loss = float(np.mean(gan_loss[batch_size:]))
        return d_x_loss, d_z_loss, kld

    def _batchwise(self, x, y, expr_fun):
        x = dp.input.Input.from_any(x)
        y = dp.input.Input.from_any(y)
        x_src = expr.Source(x.x_shape)
        y_src = expr.Source(y.x_shape)
        graph = expr.ExprGraph(expr_fun(x_src, y_src))
        graph.setup()
        out = []
        for x_batch, y_batch in zip(x.batches(), y.batches()):
            x_src.out = x_batch['x']
            y_src.out = y_batch['x']
            graph.fprop()
            out.append(np.array(graph.out))
        out = np.concatenate(out)[:x.n_samples]
        return out

    def embed(self, x, y):
        """ Input to hidden. """
        return self._batchwise(x, y, self._embed_expr)

    def reconstruct(self, z, y):
        """ Hidden to input. """
        return self._batchwise(z, y, self._reconstruct_expr)
