"""
Training schedulers
"""
import numpy as np

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''
    def __init__(self, optimizer, n_phase1_steps, n_phase2_steps, n_phase3_steps, lr_phase12, lr_phase23):
        self._optimizer = optimizer
        self.n_phase1_steps = n_phase1_steps
        self.n_phase2_steps = n_phase2_steps
        self.n_phase3_steps = n_phase3_steps
        self.lr_phase12 = lr_phase12
        self.lr_phase23 = lr_phase23
        self.n_steps = 0 

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def reset_n_steps(self, step):
        "Manually set the current training step (useful when restarting)" 
        self.n_steps = step  

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr(self):
        if self.n_steps < self.n_phase1_steps:
            # Phase 1: Linear ramp from 0 to lr_phase12
            lr = (self.n_steps / self.n_phase1_steps) * self.lr_phase12

        elif self.n_steps < (self.n_phase1_steps + self.n_phase2_steps):
            # Phase 2: Half-cosine decay from lr_phase12 down to lr_phase23
            p = (self.n_steps - self.n_phase1_steps) / self.n_phase2_steps  # progress in [0, 1]
            lr = self.lr_phase23 + (self.lr_phase12 - self.lr_phase23) * ((np.cos(np.pi * p) + 1.) / 2.) 

        else:
            # Phase 3: Constant at lr_phase23
            lr = self.lr_phase23
        return lr

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self._get_lr()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
