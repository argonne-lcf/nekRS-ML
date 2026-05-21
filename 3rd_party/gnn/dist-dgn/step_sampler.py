import torch
from abc import abstractmethod


class StepSampler():
    def __init__(
        self,
        num_diffusion_steps: int = 100,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_diffusion_steps = num_diffusion_steps
        self.device = device
        self.dtype = dtype
        self.step = 0

    @property
    @abstractmethod
    def weights(self) -> torch.Tensor:
        pass

    def sample(
        self,
        batch_size: int = 1,
    ) -> torch.Tensor:
        assert batch_size <= self.num_diffusion_steps, "Batch size cannot be greater than the number of diffusion steps!"

        self.step += 1
        w = self.weights
        p = w / torch.sum(w)
        
        # Sampling could be faster on CPU and then transferred to the device
        # but for now we do it on the device.
        # indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        samples = p.multinomial(num_samples=batch_size, replacement=False)

        # Compute the importance weights to correct loss to be unbiased loss average
        importance_weights = 1 / (p.size()[0] * p[samples])

        return samples, importance_weights
    

class UniformSampler(StepSampler):
    """Uniform sampler for the diffusion steps"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def weights(self) -> torch.Tensor:
        return torch.ones([self.num_diffusion_steps], dtype=self.dtype).to(self.device)


class ExponentialSampler(StepSampler):
    """Exponential sampler for the diffusion steps favoring early timesteps.
       Computes: weights = exp(-decay_rate * r), r being the diffusion step.
    """
    def __init__(self, decay_rate=0.03, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay_rate = decay_rate
    
    @property
    def weights(self) -> torch.Tensor:
        timesteps = torch.arange(self.num_diffusion_steps, dtype=self.dtype)
        weights = torch.exp(-self.decay_rate * timesteps).to(self.device)
        return weights


class AdaptiveExponentialSampler(StepSampler):
    """Adaptive exponential sampler for the diffusion steps favoring early timesteps.
       Computes: weights = exp(-decay_rate * r), r being the diffusion step 
       The decay rate starts very small to approximate uniform sampling and 
       progressively increases to favor early timesteps.
    """
    def __init__(self, 
        initial_decay_rate=0.0001, 
        final_decay_rate=0.03,
        decay_rate_increment=0.01, # 1% increment per step
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.initial_decay_rate = initial_decay_rate
        self.final_decay_rate = final_decay_rate
        self.decay_rate_increment = decay_rate_increment
    
    @property
    def weights(self) -> torch.Tensor:
        # Compute the adaptive decay rate
        self.decay_rate = self.initial_decay_rate * (1 + self.decay_rate_increment) ** self.step
        self.decay_rate = min(self.decay_rate, self.final_decay_rate)

        # Compute the weights
        timesteps = torch.arange(self.num_diffusion_steps, dtype=self.dtype)
        weights = torch.exp(-self.decay_rate * timesteps).to(self.device)
        return weights