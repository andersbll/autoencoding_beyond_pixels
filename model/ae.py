import numpy as np
import cudarray as ca
import deeppy as dp
import deeppy.expr as expr


from util import ScaleGradient


class GaussianNegLogLikelihood(expr.nnet.loss.Loss):
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.multiplier = 1.0/(2.0*self.sigma**2)
        self.const = None

    def setup(self):
        super(GaussianNegLogLikelihood, self).setup()
        n, k = self.out_shape
        self.const = 0.5*(np.log(n) + k*np.log(2.0*np.pi))

    def fprop(self):
        diff = self.pred.out - self.target.out
        diff **= 2.0
        ca.sum(diff, axis=1, keepdims=True, out=self.out)
        self.out *= self.multiplier
        self.out += self.const

    def bprop(self):
        ca.subtract(self.pred.out, self.target.out, self.pred.out_grad)
        self.pred.out_grad *= 2*self.multiplier
        self.pred.out_grad *= self.out_grad


class KLDivergence(expr.Expr):
    def __call__(self, mu, log_sigma):
        self.mu = mu
        self.log_sigma = log_sigma
        self.inputs = [mu, log_sigma]
        return self

    def setup(self):
        self.out_shape = (len(self.mu.out), 1)
        self.out = ca.empty(self.out_shape)
        self.out_grad = ca.ones(self.out_shape)

    def fprop(self):
        tmp1 = self.mu.out**2
        ca.negative(tmp1, tmp1)
        tmp1 += self.log_sigma.out
        tmp1 += 1
        tmp1 -= ca.exp(self.log_sigma.out)
        ca.sum(tmp1, axis=1, keepdims=True, out=self.out)
        self.out *= -0.5

    def bprop(self):
        ca.multiply(self.mu.out, self.out_grad, self.mu.out_grad)
        self.mu.out_grad *= self.out_grad
        ca.exp(self.log_sigma.out, out=self.log_sigma.out_grad)
        self.log_sigma.out_grad -= 1
        self.log_sigma.out_grad *= 0.5
        self.log_sigma.out_grad *= self.out_grad


class NormalEncoder(expr.Expr, dp.base.CollectionMixin):
    def __init__(self, n_out, weight_filler, bias_filler=0.0):
        self.weight_filler = weight_filler
        self.bias_filler = bias_filler
        self.z_mu = self._affine(n_out)
        self.z_log_sigma = self._affine(n_out)
        self.n_out = n_out
        self.collection = [self.z_mu, self.z_log_sigma]
        self._batch_size = None

    def _affine(self, n_out):
        return expr.nnet.Affine(
            n_out=n_out,
            weights=self.weight_filler,
            bias=self.bias_filler,
        )

    def samples(self, batch_size):
        if self._batch_size != batch_size:
            self.z_samples = expr.random.normal(size=(batch_size, self.n_out))
        return self.z_samples

    def encode(self, h_enc, batch_size, phase=None):
        z_mu = self.z_mu(h_enc)
        z_log_sigma = self.z_log_sigma(h_enc)
        if phase is None:
            phase = self.phase
        if phase == 'test':
            z = z_mu
        else:
            z_eps = self.samples(batch_size)
            z = z_mu + expr.exp(0.5 * z_log_sigma) * z_eps
        kld = KLDivergence()(z_mu, z_log_sigma)
        return z, kld


class AdversarialEncoder(expr.Expr, dp.base.CollectionMixin):
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
        return expr.nnet.Affine(
            n_out=n_out,
            weights=self.weight_filler,
            bias=self.bias_filler,
        )

    def samples(self, batch_size):
        if self._batch_size != batch_size:
            self.z_samples = expr.random.normal(size=(batch_size, self.n_out))
        return self.z_samples

    def encode(self, h_enc, batch_size, phase=None):
        z = self.z_enc(h_enc)
        if phase is None:
            phase = self.phase
        if phase == 'test':
            return z, 0
        else:
            z_ = ScaleGradient(-1.0)(z)
            z_samples = self.samples(batch_size)
            z_ = expr.Concatenate(axis=0)(z_samples, z_)
            d_z = self.discriminator(z_)
            sign = np.ones((batch_size*2, 1), dtype=ca.float_)
            sign[batch_size:] = -1.0
            offset = np.zeros_like(sign)
            offset[batch_size:] = 1.0
            loss = expr.sum(-expr.log(d_z*sign + offset + self.eps))
            z = ScaleGradient(self.recon_weight)(z)
            return z, loss


class Autoencoder(dp.base.Model, dp.base.CollectionMixin):
    def __init__(self, encoder, latent_encoder, decoder):
        self.encoder = encoder
        self.latent_encoder = latent_encoder
        self.decoder = decoder
        self.recon_error = expr.nnet.BinaryCrossEntropy()
        self.collection = [encoder, latent_encoder, decoder]

    def _encode_expr(self, x, batch_size):
        enc = self.encoder(x)
        z, encoder_loss = self.latent_encoder.encode(enc, batch_size)
        return z

    def _decode_expr(self, z, batch_size):
        return self.decoder(z)

    def _likelihood_expr(self, x, batch_size):
        enc = self.encoder(x)
        z, encoder_loss = self.latent_encoder.encode(enc, batch_size,
                                                     phase='train')
        x_tilde = self.decoder(z)
        log_px_given_z = self.recon_error(x_tilde, x)
        return -encoder_loss - log_px_given_z

    def setup(self, x_shape):
        batch_size = x_shape[0]
        self.x_src = expr.Source(x_shape)
        enc = self.encoder(self.x_src)
        z, encoder_loss = self.latent_encoder.encode(enc, batch_size)
        x_tilde = self.decoder(z)
        recon_loss = self.recon_error(x_tilde, self.x_src)
        self._graph = expr.ExprGraph(-encoder_loss - recon_loss)
        self._graph.out_grad = ca.array(-np.ones((batch_size, 1)))
        self._graph.setup()

    def update(self, x):
        self.x_src.out = x
        self._graph.fprop()
        self._graph.bprop()
        loss = self._graph.out
        return loss

    def _batchwise(self, input, expr_fun):
        input = dp.input.Input.from_any(input)
        src = expr.Source(input.x_shape)
        graph = expr.ExprGraph(expr_fun(src, input.batch_size))
        graph.setup()
        z = []
        for x_batch in input.batches():
            src.out = x_batch['x']
            graph.fprop()
            z.append(np.array(graph.out))
        z = np.concatenate(z)[:input.n_samples]
        return z

    def encode(self, input):
        """ Input to hidden. """
        return self._batchwise(input, self._encode_expr)

    def decode(self, input):
        """ Hidden to input. """
        return self._batchwise(input, self._decode_expr)

    def likelihood(self, input):
        """ Input to hidden. """
        return self._batchwise(input, self._likelihood_expr)
