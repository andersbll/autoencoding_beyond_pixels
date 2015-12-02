from copy import copy, deepcopy
import numpy as np
import cudarray as ca
import deeppy as dp
import deeppy.expr as expr


class NormalSampler(expr.Expr, dp.base.CollectionMixin):
    def __init__(self, n_hidden, weight_filler, bias_filler=0.0):
        self.weight_filler = weight_filler
        self.bias_filler = bias_filler
        self.z_mu = self._affine(n_hidden)
        self.z_log_sigma = self._affine(n_hidden)
        self.n_hidden = n_hidden
        self.collection = [self.z_mu, self.z_log_sigma]
        self.batch_size = None

    def _affine(self, n_out):
        return expr.nnet.Affine(
            n_out=n_out,
            weights=self.weight_filler,
            bias=self.bias_filler,
        )

    def samples(self):
        return expr.random.normal(size=(self.batch_size, self.n_hidden))

    def __call__(self, h_enc):
        self.z_eps = self.samples()
        z_mu = self.z_mu(h_enc)
        z_log_sigma = self.z_log_sigma(h_enc)
        z = z_mu + expr.exp(0.5 * z_log_sigma) * self.z_eps
        return z, z_mu, z_log_sigma, self.z_eps


class KLDivergence(expr.Expr):
    def __call__(self, mu, log_sigma):
        self.mu = mu
        self.log_sigma = log_sigma
        self.inputs = [mu, log_sigma]
        return self

    def setup(self):
        self.out_shape = (1,)
        self.out = ca.empty(self.out_shape)
        self.out_grad = ca.empty(self.out_shape)

    def fprop(self):
        tmp1 = self.mu.out**2
        ca.negative(tmp1, tmp1)
        tmp1 += self.log_sigma.out
        tmp1 += 1
        tmp1 -= ca.exp(self.log_sigma.out)
        self.out = ca.sum(tmp1)
        self.out *= -0.5

    def bprop(self):
        ca.multiply(self.mu.out, self.out_grad, self.mu.out_grad)
        ca.exp(self.log_sigma.out, out=self.log_sigma.out_grad)
        self.log_sigma.out_grad -= 1
        self.log_sigma.out_grad *= 0.5
        self.log_sigma.out_grad *= self.out_grad


class NegativeGradient(expr.base.UnaryElementWise):
    def fprop(self):
        self.out = self.x.out

    def bprop(self):
        ca.negative(self.out_grad, self.x.out_grad)


class ScaleGradient(expr.base.UnaryElementWise):
    def __init__(self, scale):
        self.scale = scale

    def fprop(self):
        self.out = self.x.out

    def bprop(self):
        ca.multiply(self.out_grad, self.scale, self.x.out_grad)


class SquareError(expr.base.Expr):
    def __call__(self, y, y_pred):
        return (y_pred - y)**2


class AbsError(expr.base.Expr):
    def __call__(self, y, y_pred):
        return expr.fabs(y_pred - y)


class WeightedParameter(dp.Parameter):
    def __init__(self, parameter, weight):
        self.__dict__ = parameter.__dict__
        self.weight = weight

    def grad(self):
        grad = self.grad_array
        grad *= self.weight
        for param in self.shares:
            grad -= param.grad_array
        grad = self._add_penalty(grad)
        return grad


class VAEGAN(dp.base.Model, dp.base.CollectionMixin):
    def __init__(self, encoder, sampler, generator, discriminator, mode,
                 vae_grad_scale=1.0, kld_weight=1.0, z_gan_prop=False):
        self.encoder = encoder
        self.sampler = sampler
        self.mode = mode
        self.discriminator = discriminator
        self.vae_grad_scale = vae_grad_scale
        self.kld_weight = kld_weight
        self.eps = 1e-4
        self.hidden_std = 1.0
        self.z_gan_prop = z_gan_prop
        self.recon_error = SquareError()
        generator.params = [p.parent if isinstance(p, WeightedParameter) else p
                            for p in generator.params]
        if self.mode == 'vaegan':
            generator.params = [WeightedParameter(p, vae_grad_scale)
                                for p in generator.params]
            self.generator_neg = deepcopy(generator)
            self.generator_neg.params = [p.share() for p in generator.params]
        if self.mode == 'gan':
            generator.params = [WeightedParameter(p, -1.0)
                                for p in generator.params]
        self.generator = generator
        self.collection = [self.encoder, self.sampler, self.generator, self.discriminator]

    def _embed_expr(self, x):
        h_enc = self.encoder(x)
        z, z_mu, z_log_sigma, z_eps = self.sampler(h_enc)
        z = z_mu
        return z

    def _reconstruct_expr(self, z):
        return self.generator(z)

    def setup(self, x_shape):
        batch_size = x_shape[0]
        self.sampler.batch_size = x_shape[0]
        self.x_src = expr.Source(x_shape)

        if self.mode in ['vae', 'vaegan']:
            h_enc = self.encoder(self.x_src)
            z, z_mu, z_log_sigma, z_eps = self.sampler(h_enc)
            if isinstance(self.hidden_std, float) and self.hidden_std != 1.0:
                z_eps = z_eps*self.hidden_std
            self.kld = KLDivergence()(z_mu, z_log_sigma)
            if self.kld_weight != 1.0:
                self.kld = self.kld_weight*self.kld
            x_tilde = self.generator(z)
            self.logpxz = self.recon_error(x_tilde, self.x_src)
            loss = self.kld + expr.sum(self.logpxz)

        if self.mode in ['gan', 'vaegan']:
            if self.mode == 'gan':
                z = self.sampler.samples()
                x_tilde = self.generator(z)
                gen_size = batch_size
            elif self.mode == 'vaegan':
                if not self.z_gan_prop:
                    z = ScaleGradient(0.0)(z)
                z = expr.Concatenate(axis=0)(z, z_eps)
                x_tilde = self.generator_neg(z)
                gen_size = batch_size*2
            x = expr.Concatenate(axis=0)(self.x_src, x_tilde)
            d = self.discriminator(x)
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

    def update(self, x):
        self.x_src.out = x
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

    def _batchwise(self, input, expr_fun):
        input = dp.input.Input.from_any(input)
        src = expr.Source(input.x_shape)
        graph = expr.ExprGraph(expr_fun(src))
        graph.setup()
        z = []
        for x_batch in input.batches():
            src.out = x_batch['x']
            graph.fprop()
            z.append(np.array(graph.out))
        z = np.concatenate(z)[:input.n_samples]
        return z

    def embed(self, input):
        """ Input to hidden. """
        return self._batchwise(input, self._embed_expr)

    def reconstruct(self, input):
        """ Hidden to input. """
        return self._batchwise(input, self._reconstruct_expr)


def train(model, input, learn_rule, n_epochs, epoch_callback=None):
    model.phase = 'train'
    learn_rule = learn_rule
    learn_rule_g = copy(learn_rule)
    learn_rule_d = copy(learn_rule)
    model.setup(**input.shapes)
    params, g_params, d_params = model.params
    learn_rule.learn_rate /= input.batch_size
    learn_rule_g.learn_rate /= input.batch_size
    learn_rule_d.learn_rate /= input.batch_size*2
    states = [learn_rule.init_state(p) for p in params]
    g_states = [learn_rule_g.init_state(p) for p in g_params]
    d_states = [learn_rule_d.init_state(p) for p in d_params]
    for epoch in range(n_epochs):
        batch_costs = []
        for batch in input.batches():
            real_cost, gen_cost, kld = model.update(**batch)
            batch_costs.append((real_cost, gen_cost, kld))
            for param, state in zip(params, states):
                learn_rule.step(param, state)
            for param, state in zip(g_params, g_states):
                learn_rule_g.step(param, state)
            for param, state in zip(d_params, d_states):
                learn_rule_d.step(param, state)
        real_cost = np.mean([cost[0] for cost in batch_costs])
        gen_cost = np.mean([cost[1] for cost in batch_costs])
        kld = np.mean([c[2] for c in batch_costs])
        print('epoch %d    kld:%.4f  real_cost:%.4f  gen_cost:%.4f'
              % (epoch, kld, real_cost, gen_cost))
        if epoch_callback is not None:
            epoch_callback()


def margin_train(model, input, learn_rule, n_epochs, margin=0.25,
                 equilibrium=0.68314718, epoch_callback=None):
    model.phase = 'train'
    learn_rule = learn_rule
    learn_rule_g = copy(learn_rule)
    learn_rule_d = copy(learn_rule)
    model.setup(**input.shapes)
    params, g_params, d_params = model.params
    learn_rule.learn_rate /= input.batch_size
    learn_rule_g.learn_rate /= input.batch_size
    learn_rule_d.learn_rate /= input.batch_size*2
    states = [learn_rule.init_state(p) for p in params]
    g_states = [learn_rule_g.init_state(p) for p in g_params]
    d_states = [learn_rule_d.init_state(p) for p in d_params]
    for epoch in range(n_epochs):
        batch_costs = []
        for batch in input.batches():
            real_cost, gen_cost, kld = model.update(**batch)
            batch_costs.append((real_cost, gen_cost, kld))
            update_g = True
            update_d = True
            if margin is not None:
                if real_cost < equilibrium - margin or \
                   gen_cost < equilibrium - margin:
                    update_d = False
                if real_cost > equilibrium + margin or \
                   gen_cost > equilibrium + margin:
                    update_g = False
                if not (update_g or update_d):
                    update_g = True
                    update_d = True
            for param, state in zip(params, states):
                learn_rule.step(param, state)
            if update_g:
                for param, state in zip(g_params, g_states):
                    learn_rule_g.step(param, state)
            if update_d:
                for param, state in zip(d_params, d_states):
                    learn_rule_d.step(param, state)
        real_cost = np.mean([cost[0] for cost in batch_costs])
        gen_cost = np.mean([cost[1] for cost in batch_costs])
        kld = np.mean([c[2] for c in batch_costs])
        print('epoch %d    kld:%.4f  real_cost:%.4f  gen_cost:%.4f'
              % (epoch, kld, real_cost, gen_cost))
        if epoch_callback is not None:
            epoch_callback()
