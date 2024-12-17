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
import random
import numpy as np
from buffers import SubTrajectoryBuffer
from utils import totensors, grad_penalty_loss
from mpi4py import MPI
from ppo_mpi_base.mpi_utils import average_grads_across_processes, sync_weights_across_processes
from torch.utils.tensorboard import SummaryWriter
from itertools import chain


class Discriminator(torch.nn.Module):
    def __init__(self, state_size, action_size, output_size=1):
        super().__init__()
        h = 512
        self.fc1 = torch.nn.Linear(state_size+action_size, h)
        self.fc2 = torch.nn.Linear(h, h)
        self.fc3 = torch.nn.Linear(h, h)
        self.fc4 = torch.nn.Linear(h, output_size)

    def forward(self, s, a=None):
        # if actions not given, assume they are already concatenated
        if a is not None:
            x = torch.cat([s, a], dim=-1)
        else:
            x = s
        x = torch.nn.functional.elu(self.fc1(x))
        x = torch.nn.functional.elu(self.fc2(x))
        x = torch.nn.functional.elu(self.fc3(x))
        x = self.fc4(x)
        return x


# manages GAIL additions to normal PPO loop
class GAIL:

    def __init__(self, state_size, action_size, expert_buffer, log_directory, use_wgan=False, 
        update_iters=10, states_only=False, image_states=False, device=None):
        self.device = device

        self.state_preprocessor = lambda x: x # currently a placeholder
        disc_input_size = state_size

        self.expert_buffer = expert_buffer
        other_size = disc_input_size if states_only else action_size
        self.discriminator = Discriminator(disc_input_size, other_size, 1)
        if self.device is not None:
            self.discriminator.to(self.device)
        self.policy_buffer = SubTrajectoryBuffer([state_size, (action_size,)], max_size=50000)

        self.opt = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001)

        self.use_wgan = use_wgan
        self.update_iters = update_iters
        self.states_only = states_only
        self.image_states = image_states

        if MPI.COMM_WORLD.Get_rank() == 0:
            self.logstep = 0
            self.summary_writer = SummaryWriter(log_dir=log_directory)

    def receive_rollout_tuples(self, tuples):
        # from ppo, receives tuples of (s, a, r, v, logp)
        # we want to store these and replace the reward with our GAN metric

        # store
        states_and_actions = [(x[0], x[1]) for x in tuples]
        self.policy_buffer.add_samples(states_and_actions)

        if self.states_only:
            states = [x[0] for x in states_and_actions[:-1]]
            other = [x[0] for x in states_and_actions[1:]]
        else:
            states = [x[0] for x in states_and_actions]
            other = [x[1] for x in states_and_actions]
        states = torch.tensor(np.array(states)).to(torch.float32)
        other = torch.tensor(np.array(other)).to(torch.float32)

        if self.device is not None:
            states = states.to(self.device)
            other = other.to(self.device)

        with torch.no_grad():
            states = self.state_preprocessor(states)
            if self.states_only:
                other = self.state_preprocessor(other)

            expect_data = self.discriminator(states, other)

            # TODO: these should both be used (sigmoid but with grad penalty)
            if self.use_wgan:
                rewards = torch.mean(expect_data, dim=-1).detach().cpu().numpy()
            else:
                probs = torch.nn.functional.sigmoid(expect_data) + 0.00001
                rewards = torch.log(probs).detach().cpu().numpy().squeeze()

        return_tuples = []
        if self.states_only:
            tuples = tuples[:-1] # we dont know the following state for the last entry

        for i,tup in enumerate(tuples):
            s, a, r, v, p = tup
            return_tuples.append((s, a, rewards[i], v, p))

        # to make ppo happy, we can duplicate a random state if we need to. This is a not-so-elegant hack.
        if self.states_only:
            return_tuples.append(random.choice(return_tuples))

        if MPI.COMM_WORLD.Get_rank() == 0:
            avg_return_reward = np.mean(rewards)
            self.summary_writer.add_scalar("gail_reward", avg_return_reward, self.logstep)
            self.logstep += 1
            self.summary_writer.flush()

        return return_tuples


    def update(self, policy=None):
        BZ = 32 if self.image_states else 512

        sync_weights_across_processes(MPI.COMM_WORLD, self.discriminator.parameters())

        losses = []
        real_outs = []
        fake_outs = []
        for i in range(self.update_iters):
            self.opt.zero_grad()

            # get expert and fake data
            if self.states_only:
                batch1, batch2 = self.expert_buffer.sample_subtrajectories(subtraj_len=2, amount=BZ)
                expert_states = batch1[0]
                expert_other = batch2[0]
                batch1, batch2 = self.policy_buffer.sample_subtrajectories(subtraj_len=2, amount=BZ)
                policy_states = batch1[0]
                policy_other = batch2[0]
            else:
                expert_states, expert_other = self.expert_buffer.sample_buffers(BZ)
                policy_states, policy_other = self.policy_buffer.sample_buffers(BZ)

            expert_states = torch.tensor(expert_states).to(torch.float32)
            expert_other = torch.tensor(expert_other).to(torch.float32)
            policy_states = torch.tensor(policy_states).to(torch.float32)

            if policy is not None:
                with torch.no_grad():
                    policy_other = policy(policy_states)

            policy_other = torch.tensor(policy_other).to(torch.float32)

            if self.device is not None:
                expert_states = expert_states.to(self.device)
                expert_other = expert_other.to(self.device)
                policy_states = policy_states.to(self.device)
                policy_other = policy_other.to(self.device)

            expert_states = self.state_preprocessor(expert_states)
            policy_states = self.state_preprocessor(policy_states)
            if self.states_only:
                expert_other = self.state_preprocessor(expert_other)
                policy_other = self.state_preprocessor(policy_other)

            real_data = torch.cat([expert_states, expert_other], dim=-1)
            fake_data = torch.cat([policy_states, policy_other], dim=-1)

            if self.use_wgan:
                expect_real = torch.mean(self.discriminator(real_data))
                expect_fake = torch.mean(self.discriminator(fake_data))
                gp_loss = grad_penalty_loss(real_data, fake_data, self.discriminator)
                loss = expect_fake - expect_real + 10.0*gp_loss

                losses.append((expect_fake - expect_real).item())
                real_outs.append(expect_real.item())
                fake_outs.append(expect_fake.item())
            else:
                pred_real = torch.nn.functional.sigmoid(self.discriminator(real_data))
                loss_real = torch.nn.functional.binary_cross_entropy(pred_real, torch.ones_like(pred_real))

                pred_fake = torch.nn.functional.sigmoid(self.discriminator(fake_data))
                loss_fake = torch.nn.functional.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake))

                gp_loss = grad_penalty_loss(real_data, fake_data, self.discriminator)

                loss = loss_real + loss_fake + 10.0*gp_loss

                losses.append((torch.mean(pred_real) - (1-torch.mean(pred_fake))).item())
                real_outs.append(torch.mean(pred_real).item())
                fake_outs.append(torch.mean(pred_fake).item())
            loss.backward()

            average_grads_across_processes(MPI.COMM_WORLD, self.discriminator.parameters())

            self.opt.step()

        if MPI.COMM_WORLD.Get_rank() == 0:
            self.summary_writer.add_scalar("disc_real", np.mean(real_outs), self.logstep)
            self.summary_writer.add_scalar("disc_fake", np.mean(fake_outs), self.logstep)

        return np.mean(losses)
