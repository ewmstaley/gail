'''
Copyright © 2024 The Johns Hopkins University Applied Physics Laboratory LLC
 
Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import torch
import numpy as np


class GaussianPolicy(torch.nn.Module):
    '''
    A gaussian policy, taking code from both the PPO and SAC implementations from spinning up.
    If conditional_dev=True, the (log)standard deviation is a function of the input, otherwise it is a learned parameter.
    '''
    def __init__(self, state_size, action_size, hidden_size=256, conditional_dev=True, device=None):
        super().__init__()

        self.fc1 = torch.nn.Linear(state_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, action_size)

        self.conditional_dev = conditional_dev
        if conditional_dev:
            self.log_std_fc = torch.nn.Linear(hidden_size, action_size)
        else:
            log_std = -0.5 * np.ones(action_size, dtype=np.float32)
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.device = device

    def forward(self, x, existing_a=None, deterministic=False):
        x = torch.nn.functional.elu(self.fc1(x))
        x = torch.nn.functional.elu(self.fc2(x))
        x = torch.nn.functional.elu(self.fc3(x))
        mu = self.fc4(x)

        if deterministic:
            a = mu
            return a, 1.0

        if self.conditional_dev:
            log_std = self.log_std_fc(x)
        else:
            log_std = self.log_std

        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = torch.distributions.normal.Normal(mu, std)

        if existing_a is not None:
            a = existing_a
        else:
            a = dist.rsample()

        logp_pi = dist.log_prob(a).sum(axis=-1)

        return a, logp_pi

    def act(self, s, existing_a=None, deterministic=False):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s).to(torch.float32)
            if self.device is not None:
                s = s.to(self.device)
        a, unclipped_a, logp = self.forward(s, existing_a=existing_a, deterministic=deterministic)
        a = a.detach().cpu().numpy()
        return a