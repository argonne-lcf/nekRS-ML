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

    @property
    @abstractmethod
    def weights(self) -> torch.Tensor:
        pass

    def sample(
        self,
        batch_size: int = 1,
    ) -> torch.Tensor:
        w = self.weights
        p = w / torch.sum(w)
        # NB: Sampling could be faster on CPU and then transferred to the device
        # but for now we do it on the device.
        # indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = p.multinomial(num_samples=batch_size, replacement=False)
        weights = 1 / (p.size()[0] * p[indices])
        return indices, weights
    

class UniformStepSampler(StepSampler):
    """Uniform sampler for the diffusion steps."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def weights(self) -> torch.Tensor:
        return torch.ones([self.num_diffusion_steps], dtype=self.dtype).to(self.device)