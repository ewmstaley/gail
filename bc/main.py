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

import gymnasium as gym
import numpy as np
import torch
from ppo_mpi_base.wrappers import GymnasiumToGymWrapper
from buffers import SubTrajectoryBuffer
import pickle
import copy
from torch.utils.tensorboard import SummaryWriter
from gaussian_policy import GaussianPolicy


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.drop = torch.nn.Dropout(p=0.1)
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, **kwargs):
        x = torch.nn.functional.elu(self.fc1(x))
        x = torch.nn.functional.elu(self.fc2(x))
        x = self.drop(x)
        x = torch.nn.functional.elu(self.fc3(x))
        return self.fc4(x), 1.0




def run_experiment(data_path, output_dir, hidden=64, model_output_distribution=False,
        conditional_dev=False):

    summary_writer = SummaryWriter(log_dir=output_dir)

    device = torch.device("cuda")

    # make environment
    env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
    env = GymnasiumToGymWrapper(env)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # load expert data
    all_trajs = pickle.load(open( data_path, "rb" ) )
    expert_buffer = SubTrajectoryBuffer([(state_size,), (action_size,)], max_size=100000)
    for traj in all_trajs:
        tuples = [(x["state"], x["action"]) for x in traj]
        expert_buffer.add_samples(tuples)

    ITERS = 30000

    
    if model_output_distribution:
        policy = GaussianPolicy(state_size, action_size, hidden_size=hidden, 
            conditional_dev=conditional_dev, device=device).to(device)
    else:
        policy = MLP(state_size, hidden, action_size).to(device)

    opt_p = torch.optim.Adam(policy.parameters(), lr=0.002)
    sched = torch.optim.lr_scheduler.LinearLR(opt_p, start_factor=1.0, end_factor=0.01, total_iters=ITERS)

    policy_slow = copy.deepcopy(policy).to(device)
    policy_slow.eval()
    polyak = 0.995


    losses = []
    for i in range(ITERS):

        states, actions = expert_buffer.sample_buffers(128)
        states = torch.tensor(states).to(torch.float32).to(device)
        actions = torch.tensor(actions).to(torch.float32).to(device)

        opt_p.zero_grad()

        if model_output_distribution:
            a, logp_pi = policy(states, existing_a=actions)
            loss = -torch.mean(logp_pi)
        else:
            pred_a, _ = policy(states)
            loss = torch.nn.functional.mse_loss(pred_a, actions)

        loss.backward()
        opt_p.step()
        sched.step()

        losses.append(loss.item())

        # slow policy update, since the main policy seems noisy
        with torch.no_grad():
            for p, p_targ in zip(policy.parameters(), policy_slow.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        if (i+1)%100 == 0:
            print("loss:", np.mean(losses[-20:]))
            summary_writer.add_scalar("loss", np.mean(losses[-20:]), i)

        # test main
        TEST_POLICY = policy
        if (i+1)%100 == 0:
            TEST_POLICY.eval()

            results = []
            for k in range(50):
                s = env.reset()
                rtotal = 0
                done = False
                stepno = 0
                while not done and stepno < 100:
                    s = torch.tensor(s).to(torch.float32).to(device)
                    a, _ = TEST_POLICY(s, deterministic=True)
                    a = a.detach().cpu().numpy()
                    a = np.clip(a, -1.0, 1.0)
                    s, r, done, _ = env.step(a)
                    rtotal += r
                    stepno += 1
                results.append(rtotal)

            print(i+1, "Performance (main):", np.mean(results), np.std(results))
            summary_writer.add_scalar("rewards main", np.mean(results), i)
            TEST_POLICY.train()

        # test slow
        TEST_POLICY = policy_slow
        if (i+1)%100 == 0:
            TEST_POLICY.eval()
            results = []
            for k in range(50):
                s = env.reset()
                rtotal = 0
                done = False
                stepno = 0
                while not done and stepno < 100:
                    s = torch.tensor(s).to(torch.float32).to(device)
                    a, _ = TEST_POLICY(s, deterministic=True)
                    a = a.detach().cpu().numpy()
                    a = np.clip(a, -1.0, 1.0)
                    s, r, done, _ = env.step(a)
                    rtotal += r
                    stepno += 1
                results.append(rtotal)

            print(i+1, "Performance (slow):", np.mean(results), np.std(results))
            summary_writer.add_scalar("rewards slow", np.mean(results), i)
            TEST_POLICY.train()


if __name__ == "__main__":

    # data = "../data/traj_stochastic_one.p"
    # output_dir = "./output_policy/"

    # run_experiment(data, output_dir+"cond"+"/", hidden=64, model_output_distribution=True, conditional_dev=True)
    # run_experiment(data, output_dir+"no_cond"+"/", hidden=64, model_output_distribution=True, conditional_dev=False)
    # run_experiment(data, output_dir+"mu_only"+"/", hidden=64, model_output_distribution=False, conditional_dev=False)


    # # datas = ["../data/traj_deterministic.p", "../data/traj_stochastic_half.p", "../data/traj_stochastic_one.p", "../data/traj_stochastic_one_five.p", "../data/traj_stochastic_two.p"]
    # # names = ["determ", "0p5", "1p0", "1p5", "2p0"]

    datas = ["../data/traj_stochastic_one.p"]
    names = ["1p0"]

    # run all experiments a few times
    for trial in range(3):
        for dsx in range(len(datas)):
            data = datas[dsx]
            dataname = names[dsx]

            for hidden in [16, 64, 256, 1024, 4096]:
                output_dir = "./output_scale/" + dataname + "_" + str(hidden) + "_trial_" + str(trial) + "/"
                run_experiment(data, output_dir, hidden=hidden)
