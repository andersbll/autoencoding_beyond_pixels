import numpy as np
import cudarray as ca
import deeppy as dp
import deeppy.expr as ex

from util import ScaleGradient


class NLLNormal(ex.nnet.loss.Loss):
    c = -0.5*np.log(2*np.pi)

    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.multiplier = 1.0/(2.0*self.sigma**2)

    def fprop(self):
        # c - multiplier*(pred - target)**2
        tmp = self.pred.array - self.target.array
        tmp **= 2.0
        tmp *= -self.multiplier
        tmp += self.c
        ca.sum(tmp, axis=self.axis, out=self.array)

    def bprop(self):
        ca.subtract(self.pred.array, self.target.array, self.pred.grad_array)
        if self.sigma != 1.0:
            self.pred.grad_array *= 2*self.multiplier
        self.pred.grad_array *= ca.reshape(self.grad_array, self.bcast_shape)


class KLDStandardNormal(ex.nnet.loss.Loss):
    def __call__(self, mu, logvar):
        self.mu = mu
        self.logvar = logvar
        self.inputs = [mu, logvar]
        return self

    def setup(self):
        self.setup_from_shape(self.mu.shape)

    def fprop(self):
        tmp1 = self.mu.array**2
        ca.negative(tmp1, tmp1)
        tmp1 += self.logvar.array
        tmp1 += 1
        tmp1 -= ca.exp(self.logvar.array)
        ca.sum(tmp1, axis=self.axis, out=self.array)
        self.array *= -0.5

    def bprop(self):
        grad = ca.reshape(self.grad_array, self.bcast_shape)
        ca.multiply(self.mu.array, grad, self.mu.grad_array)
        ca.exp(self.logvar.array, out=self.logvar.grad_array)
        self.logvar.grad_array -= 1
        self.logvar.grad_array *= 0.5
        self.logvar.grad_array *= grad


class NormalEncoder(ex.Op, dp.base.CollectionMixin):
    def __init__(self, n_out, weight_filler, bias_filler=0.0):
        self.weight_filler = weight_filler
        self.bias_filler = bias_filler
        self.z_mu = self._affine(n_out)
        self.z_logvar = self._affine(n_out)
        self.n_out = n_out
        self.collection = [self.z_mu, self.z_logvar]
        self._batch_size = None

    def _affine(self, n_out):
        return ex.nnet.Affine(
            n_out=n_out,
            weights=self.weight_filler,
            bias=self.bias_filler,
        )

    def samples(self, batch_size):
        if self._batch_size != batch_size:
            self.z_samples = ex.random.normal(size=(batch_size, self.n_out))
        return self.z_samples

    def encode(self, h_enc, batch_size):
        z_mu = self.z_mu(h_enc)
        z_logvar = self.z_logvar(h_enc)
        z_eps = self.samples(batch_size)
        z = z_mu + ex.exp(0.5 * z_logvar) * z_eps
        kld = KLDStandardNormal()(z_mu, z_logvar)
        return z, kld


class AdversarialEncoder(ex.Op, dp.base.CollectionMixin):
    def __init__(self, n_out, discriminator, weight_filler, bias_filler=0.0,
                 recon_weight=0.01, eps=1e-4):
        self.weight_filler = weight_filler
        self.bias_filler = bias_filler
        self.n_out = n_out
        self.z_enc = self._affine(n_out)
        self.discriminator = discriminator
        self.collection = [self.z_enc, discriminator]
        self.eps = eps
        self.recon_weight = recon_weight
        self._batch_size = None

    def _affine(self, n_out):
        return ex.nnet.Affine(
            n_out=n_out,
            weights=self.weight_filler,
            bias=self.bias_filler,
        )

    def samples(self, batch_size):
        if self._batch_size != batch_size:
            self.z_samples = ex.random.normal(size=(batch_size, self.n_out))
        return self.z_samples

    def encode(self, h_enc, batch_size):
        z = self.z_enc(h_enc)
        z_ = ScaleGradient(-1.0)(z)
        z_samples = self.samples(batch_size)
        z_ = ex.Concatenate(axis=0)(z_samples, z_)
        d_z = self.discriminator(z_)
        sign = np.ones((batch_size*2, 1), dtype=ca.float_)
        sign[batch_size:] = -1.0
        offset = np.zeros_like(sign)
        offset[batch_size:] = 1.0
        loss = ex.sum(-ex.log(d_z*sign + offset + self.eps))
        z = ScaleGradient(self.recon_weight)(z)
        return z, loss


class Autoencoder(dp.base.Model, dp.base.CollectionMixin):
    def __init__(self, encoder, latent_encoder, decoder):
        self.encoder = encoder
        self.latent_encoder = latent_encoder
        self.decoder = decoder
        self.recon_error = ex.nnet.BinaryCrossEntropy()
        self.collection = [encoder, latent_encoder, decoder]

    def _encode_expr(self, x, batch_size):
        enc = self.encoder(x)
        z, encoder_loss = self.latent_encoder.encode(enc, batch_size)
        return z

    def _decode_expr(self, z, batch_size):
        return self.decoder(z)

    def _likelihood_expr(self, x, batch_size):
        enc = self.encoder(x)
        z, encoder_loss = self.latent_encoder.encode(enc, batch_size)
        x_tilde = self.decoder(z)
        log_px_given_z = self.recon_error(x_tilde, x)
        return -encoder_loss - log_px_given_z

    def setup(self, x_shape):
        batch_size = x_shape[0]
        self.x_src = ex.Source(x_shape)
        enc = self.encoder(self.x_src)
        z, encoder_loss = self.latent_encoder.encode(enc, batch_size)
        x_tilde = self.decoder(z)
        recon_loss = self.recon_error(x_tilde, self.x_src)
        self.loss = -(encoder_loss + recon_loss)
        self._graph = ex.graph.ExprGraph(self.loss)
        self._graph.setup()
        self.loss.grad_array = ca.array(-np.ones((batch_size,)))

    def update(self, x):
        self.x_src.array = x
        self._graph.fprop()
        self._graph.bprop()
        return self.loss.array

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

    def likelihood(self, feed):
        return self._batchwise(feed, self._likelihood_expr)
