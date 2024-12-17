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
import pickle
from ppo_mpi_base.ppo import PPO
from ppo_mpi_base.wrappers import GymnasiumToGymWrapper
from gail import GAIL
from buffers import SubTrajectoryBuffer
from cnn import CNN
from pixel_wrapper import MujocoRenderObservationWrapper


# ==================================================================

# generic network that we can use
class Network(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==================================================================


# define our environment
def env_fn():
    env = gym.make("HalfCheetah-v4", render_mode="none")

    # for simplicity, force old gym interface
    env = GymnasiumToGymWrapper(env)
    return env


# define a policy network architecture
def policy_net_fn(obs, act):
    return Network(obs, act)

# define a value network architecture
def value_net_fn(obs):
    return Network(obs, 1)


if __name__ == "__main__":

    DATA = "../data/traj_stochastic_one.p"
    PREFIX = "long"

    # set up the GAN component
    env = env_fn()
    state_size = env.observation_space.shape
    state_size = (state_size[0],)
    action_size = env.action_space.shape[0]

    # load expert data
    all_trajs = pickle.load(open( DATA, "rb" ) )
    expert_buffer = SubTrajectoryBuffer([state_size, (action_size,)], max_size=100000)
    all_rewards = []
    for traj in all_trajs:
        tuples = [(x["state"], x["action"]) for x in traj]
        all_rewards.append(np.sum([x["reward"] for x in traj]))
        expert_buffer.add_samples(tuples)
    print(np.mean(all_rewards), np.std(all_rewards))

    # gail
    gail = GAIL(state_size, action_size, expert_buffer, "./output/"+PREFIX+"/logs/cheetah/", 
        use_wgan=False, update_iters=20, states_only=False, image_states=False)

    # pre-load gail with some random actions and some updates
    print("preloading gail.")
    for ep in range(10):
        s = env.reset()
        steps = 0
        done = False
        tuples = []
        while not done and steps < 100:
            steps += 1
            a = env.action_space.sample()
            ns, r, done, info = env.step(a)
            tuples.append((s, a, 0, 0, 0))

        gail.receive_rollout_tuples(tuples)

    # run the thing
    PPO(
        total_steps=200e6,
        env_fn=env_fn, 
        network_fn=policy_net_fn,
        value_network_fn=value_net_fn,
        seed=0, 
        rollout_length_per_worker=512,
        train_batch_size=128,  
        gamma=0.99, 
        clip_ratio=0.2, 
        entropy_coef=0.0,
        pi_lr=3e-4,
        vf_lr=1e-3, 
        pi_grad_clip=1.0,
        v_grad_clip=1.0,
        train_pi_epochs=5, 
        train_v_epochs=15, 
        lam=0.97, 
        max_ep_len=100,
        target_kl=0.01, 
        log_directory="./output/"+PREFIX+"/logs/cheetah/", 
        save_location="./output/"+PREFIX+"/saved_model/cheetah/",
        rollout_interception_callback=gail.receive_rollout_tuples,
        external_update_callback=gail.update,
    )