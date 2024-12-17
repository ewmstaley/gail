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
import scipy.signal

def totensors(dtype, device, *args):
    return [torch.tensor(x).to(dtype).to(device) for x in args]


def grad_penalty_loss(real, fake, discriminator):
    B = real.shape[0]
    interps = torch.rand(B, 1).to(fake.device)
    inbetween_samples = (real - fake)*interps + fake
    inbetween_samples = torch.autograd.Variable(inbetween_samples, requires_grad=True).to(fake.device)
    inbetween_pred_labels = discriminator(inbetween_samples)
    input_grads = torch.autograd.grad(
        outputs=inbetween_pred_labels, 
        inputs=inbetween_samples, 
        grad_outputs=torch.ones_like(inbetween_pred_labels),
        create_graph=True,
        retain_graph=True
    )[0]
    gradients_norm = torch.sqrt(torch.sum(input_grads ** 2, dim=1) + 1e-12)
    gp_loss = ((gradients_norm - 1) ** 2).mean()
    return gp_loss