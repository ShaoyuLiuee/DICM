import math
import torch
from .functions import normal_kl, discretized_gaussian_loglik, flat_mean
from ddpm_torch.scheduler import get_schedule_jump
import uuid

def _warmup_beta(beta_start, beta_end, timesteps, warmup_frac, dtype):
    betas = beta_end * torch.ones(timesteps, dtype=dtype)
    warmup_time = int(timesteps * warmup_frac)
    betas[:warmup_time] = torch.linspace(beta_start, beta_end, warmup_time, dtype=dtype)
    return betas


def get_beta_schedule(beta_schedule, beta_start, beta_end, timesteps, dtype=torch.float64):
    if beta_schedule == 'quad':
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps, dtype=dtype) ** 2
    elif beta_schedule == 'linear':
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=dtype)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, timesteps, 0.1, dtype=dtype)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, timesteps, 0.5, dtype=dtype)
    elif beta_schedule == 'const':
        betas = beta_end * torch.ones(timesteps, dtype=dtype)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / torch.linspace(timesteps, 1, timesteps, dtype=dtype)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (timesteps, )
    return betas


class GaussianDiffusion:

    def __init__(
            self,
            betas,
            model_mean_type,
            model_var_type,
            loss_type,
            **kwargs
    ):
        assert isinstance(betas, torch.Tensor) and betas.dtype == torch.float64
        assert (betas > 0).all() and (betas <= 1).all()
        self.betas = betas
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        self.timesteps = len(betas)

        alphas = 1 - betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.as_tensor([1., ], dtype=torch.float64), self.alphas_bar[:-1]])

        # q(x_t | x_0)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar)

        # q(x_{t-1} | x_t, x_0)
        # refer to the formula 1-3 in README.md
        sqrt_alphas_bar_prev = torch.sqrt(alphas_bar_prev)
        self.sqrt_recip_alphas_bar = torch.sqrt(1. / self.alphas_bar)
        self.sqrt_recip_m1_alphas_bar = torch.sqrt(1. / self.alphas_bar - 1.)  # m1: minus 1
        self.posterior_var = betas * (1. - alphas_bar_prev) / (1. - self.alphas_bar)
        self.posterior_logvar_clipped = torch.log(torch.cat([self.posterior_var[[1]], self.posterior_var[1:]]))
        self.posterior_mean_coef1 = betas * sqrt_alphas_bar_prev / (1. - self.alphas_bar)
        self.posterior_mean_coef2 = torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - self.alphas_bar)

        # for fixed model_var_type's
        self.fixed_model_var, self.fixed_model_logvar = {
            "fixed-large": (self.betas, torch.log(torch.cat([self.posterior_var[[1]], self.betas[1:]]))),
            "fixed-small": (self.posterior_var, self.posterior_logvar_clipped)
        }[self.model_var_type]

    def undo(self, img_after, t):
        beta = self._extract(self.betas, t, img_after)

        return torch.sqrt(1 - beta) * img_after + \
            torch.sqrt(beta) * torch.randn_like(img_after)

    @staticmethod
    def _extract(
            arr, t, x,
            dtype=torch.float32, device=torch.device("cpu"), ndim=4):
        if x is not None:
            dtype = x.dtype
            device = x.device
            ndim = x.ndim
        out = torch.as_tensor(arr, dtype=dtype, device=device).gather(0, t)
        return out.reshape((-1, ) + (1, ) * (ndim - 1))

    def q_mean_var(self, x_0, t):
        mean = self._extract(self.sqrt_alphas_bar, t, x_0) * x_0
        var = self._extract(1. - self.alphas_bar, t, x_0)
        logvar = self._extract(torch.log(1 - self.alphas_bar), t, x_0)
        return mean, var, logvar

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        coef1 = self._extract(self.sqrt_alphas_bar, t, x_0)
        coef2 = self._extract(self.sqrt_one_minus_alphas_bar, t, x_0)
        return coef1 * x_0 + coef2 * noise

    def q_posterior_mean_var(self, x_0, x_t, t):
        posterior_mean_coef1 = self._extract(self.posterior_mean_coef1, t, x_0)
        posterior_mean_coef2 = self._extract(self.posterior_mean_coef2, t, x_0)
        posterior_mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        posterior_var = self._extract(self.posterior_var, t, x_0)
        posterior_logvar = self._extract(self.posterior_logvar_clipped, t, x_0)
        return posterior_mean, posterior_var, posterior_logvar

    def p_mean_var(self, denoise_fn, x_t, t, clip_denoised, return_pred):
        B, C, H, W = x_t.shape
        out = denoise_fn(x_t, t)

        if self.model_var_type == "learned":
            assert all(out.shape == (B, 2 * C, H, W))
            out, model_logvar = out.chunk(2, dim=1)
            model_var = torch.exp(model_logvar)
        elif self.model_var_type in ["fixed-small", "fixed-large"]:
            model_var, model_logvar = self._extract(self.fixed_model_var, t, x_t),\
                                      self._extract(self.fixed_model_logvar, t, x_t)
        else:
            raise NotImplementedError(self.model_var_type)

        # calculate the mean estimate
        _clip = (lambda x: x.clamp(-1., 1.)) if clip_denoised else (lambda x: x)
        if self.model_mean_type == "mean":
            pred_x_0 = _clip(self._pred_x_0_from_mean(x_t=x_t, mean=out, t=t))
            model_mean = out
        elif self.model_mean_type == "x_0":
            pred_x_0 = _clip(out)
            model_mean, *_ = self.q_posterior_mean_var(x_0=pred_x_0, x_t=x_t, t=t)
        elif self.model_mean_type == "eps":
            pred_x_0 = _clip(self._pred_x_0_from_eps(x_t=x_t, eps=out, t=t))
            model_mean, *_ = self.q_posterior_mean_var(x_0=pred_x_0, x_t=x_t, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        if return_pred:
            return model_mean, model_var, model_logvar, pred_x_0
        else:
            return model_mean, model_var, model_logvar

    def _pred_x_0_from_mean(self, x_t, mean, t):
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t)
        return mean / coef1 - coef2 / coef1 * x_t

    def _pred_x_0_from_eps(self, x_t, eps, t):
        coef1 = self._extract(self.sqrt_recip_alphas_bar, t, x_t)
        coef2 = self._extract(self.sqrt_recip_m1_alphas_bar, t, x_t)
        return coef1 * x_t - coef2 * eps

    # === sample ===

    def p_sample_step(self, denoise_fn, x_t, t, pred_x_0, images=None, gt_keep_mask=None, device=None, clip_denoised=True, return_pred=False, generator=None):
        if pred_x_0 is not None:
            alpha_bar = self._extract(self.alphas_bar, t, x_t)
            gt_weight = torch.sqrt(alpha_bar)
            gt_part = gt_weight.to(device) * images.to(device)
            noise_weight = torch.sqrt(1 - alpha_bar)
            noise_weight = noise_weight.to(device)
            x_t = x_t.to(device)
            noise_part = noise_weight * torch.randn_like(x_t)
            weighed_gt = gt_part + noise_part
            gt_keep_mask = gt_keep_mask.to(device)
            x_t = (
                gt_keep_mask * (
                    weighed_gt
                )
                +
                (1 - gt_keep_mask) * (
                    x_t
                )
            )                  
        model_mean, _, model_logvar, pred_x_0 = self.p_mean_var(
            denoise_fn, x_t, t, clip_denoised=clip_denoised, return_pred=True)
        noise = torch.empty_like(x_t).normal_(generator=generator)
        nonzero_mask = (t > 0).reshape((-1,) + (1,) * (x_t.ndim - 1)).to(x_t)
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_logvar) * noise
        return (sample, pred_x_0) if return_pred else sample

    @torch.inference_mode()
    # def p_sample(self, denoise_fn, shape=None, device=torch.device("cpu"), noise=None, seed=None):
    def p_sample(self, denoise_fn, shape=None, device=torch.device("cpu"), images=None, gt_keep_mask=None, pred_x_0=None, noise=None, seed=None):
        B = (shape or noise.shape)[0]
        t = torch.empty((B, ), dtype=torch.int64, device=device)
        pred_x_0 = None
        rng = None
        if seed is not None:
            rng = torch.Generator(device).manual_seed(seed)
        if noise is None:
            x_t = torch.empty(shape, device=device).normal_(generator=rng)
        else:
            x_t = noise.to(device)
        if noise is None:
            noise = torch.randn_like(images).to(device)  

        schedule1 = {'t_T': 500,
        'n_sample': images.shape[0],
        'jump_length': 25, 'jump_n_sample': 25}
        schedule=get_schedule_jump(**schedule1)
        time_pairs = list(zip(schedule[:-1], schedule[1:]))
        from tqdm.auto import tqdm

        time_pairs = tqdm(time_pairs)          
      
        # for t_last, t_cur in time_pairs:
        #     t_last_t = torch.full((B,), t_last, device=device, dtype=torch.long)
        #     if t_cur < t_last:
        #         x_t, pred_x_0 = self.p_sample_step(
        #              denoise_fn, x_t, t_last_t, pred_x_0, images, gt_keep_mask,device=device,return_pred=True, generator=rng)

        #     else:
        #         t_shift = 1
        #         x_t = self.undo(x_t, t=t_last_t+t_shift)

        # return x_t

        for ti in range(self.timesteps - 1, -1, -1):
            t.fill_(ti)
            x_t, pred_x_0 = self.p_sample_step(denoise_fn, x_t, t, pred_x_0, images, gt_keep_mask,device=device,return_pred=True, generator=rng)
        return x_t


    # === log likelihood ===
    # bpd: bits per dimension

    def _loss_term_bpd(self, denoise_fn, x_0, x_t, t, clip_denoised, return_pred):
        # calculate L_t
        # t = 0: negative log likelihood of decoder, -\log p(x_0 | x_1)
        # t > 0: variational lower bound loss term, KL term
        true_mean, _, true_logvar = self.q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)
        model_mean, _, model_logvar, pred_x_0 = self.p_mean_var(
            denoise_fn, x_t=x_t, t=t, clip_denoised=clip_denoised, return_pred=True)
        kl = normal_kl(true_mean, true_logvar, model_mean, model_logvar)
        kl = flat_mean(kl) / math.log(2.)  # natural base to base 2
        decoder_nll = discretized_gaussian_loglik(x_0, model_mean, log_scale=0.5 * model_logvar).neg()
        decoder_nll = flat_mean(decoder_nll) / math.log(2.)
        output = torch.where(t.to(kl.device) > 0, kl, decoder_nll)
        return (output, pred_x_0) if return_pred else output

    def train_losses(self, denoise_fn, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise=noise)
        # calculate the loss
        # kl: weighted
        # mse: unweighted
        if self.loss_type == "kl":
            losses = self._loss_term_bpd(
                denoise_fn, x_0=x_0, x_t=x_t, t=t, clip_denoised=False, return_pred=False)
        elif self.loss_type == "mse":
            assert self.model_var_type != "learned"
            if self.model_mean_type == "mean":
                target = self.q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)[0]
            elif self.model_mean_type == "x_0":
                target = x_0
            elif self.model_mean_type == "eps":
                target = noise
            else:
                raise NotImplementedError(self.model_mean_type)
            model_out = denoise_fn(x_t, t)
            losses = flat_mean((target - model_out).pow(2))
        else:
            raise NotImplementedError(self.loss_type)

        return losses

    def _prior_bpd(self, x_0):
        B, T = len(x_0), self.timesteps
        T_mean, _, T_logvar = self.q_mean_var(
            x_0=x_0, t=(T - 1) * torch.ones((B, ), dtype=torch.int64))
        kl_prior = normal_kl(T_mean, T_logvar, mean2=0., logvar2=0.)
        return flat_mean(kl_prior) / math.log(2.)
