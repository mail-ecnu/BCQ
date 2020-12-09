import utils
import argparse
import torch
import numpy as np
import DQN
from copy import deepcopy

# Atari Specific
USE_PBRS = True
atari_preprocessing = {
    "frame_skip": 4,
    "frame_size": 84,
    "state_history": 4,
    "done_on_life_loss": False,
    "reward_clipping": True,
    "max_episode_timesteps": 27e3
}

atari_parameters = {
    # Exploration
    "start_timesteps": 2e4,
    "initial_eps": 1,
    "end_eps": 1e-2,
    "eps_decay_period": 25e4,
    # Evaluation
    "eval_freq": 5e4,
    "eval_eps": 1e-3,
    # Learning
    "discount": 0.99,
    "buffer_size": 1e6,
    "batch_size": 32,
    "optimizer": "Adam",
    "optimizer_parameters": {
        "lr": 0.0000625,
        "eps": 0.00015
    },
    "train_freq": 4,
    "polyak_target_update": False,
    "target_update_freq": 8e3,
    "tau": 1
}

regular_parameters = {
    # Exploration
    "start_timesteps": 1e3,
    "initial_eps": 0.1,
    "end_eps": 0.1,
    "eps_decay_period": 1,
    # Evaluation
    "eval_freq": 5e3,
    "eval_eps": 0,
    # Learning
    "discount": 0.99,
    "buffer_size": 1e6,
    "batch_size": 64,
    "optimizer": "Adam",
    "optimizer_parameters": {
        "lr": 3e-4
    },
    "train_freq": 1,
    "polyak_target_update": True,
    "target_update_freq": 1,
    "tau": 0.005
}

# Load parameters
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="MountainCar-v0")  # OpenAI gym environment name
parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--buffer_name", default="Default")  # Prepends name to filename
parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment or train for
parser.add_argument("--BCQ_threshold", default=0.3, type=float)  # Threshold hyper-parameter for BCQ
parser.add_argument("--low_noise_p", default=0.2,
                    type=float)  # Probability of a low noise episode when generating buffer
parser.add_argument("--rand_action_p", default=0.2,
                    type=float)  # Probability of taking a random action when generating buffer, during non-low noise episode
parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral policy
parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
args = parser.parse_args()
env, is_atari, state_dim, num_actions = utils.make_env(args.env, atari_preprocessing)
parameters = atari_parameters if is_atari else regular_parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

replay_buffer = utils.ReplayBuffer(state_dim, is_atari, atari_preprocessing, parameters["batch_size"],
                                   parameters["buffer_size"], device)
setting = f"{args.env}_{args.seed}"
buffer_name = f"{args.buffer_name}_{setting}"
new_buffer_name = f"{args.buffer_name}_{setting}" + "_PBRS"
replay_buffer.load(f"./buffers/{buffer_name}")
print("Load Replay OK")
new_replay_buffer = deepcopy(replay_buffer)
if USE_PBRS:
    policy = DQN.DQN(
        is_atari,
        num_actions,
        state_dim,
        device,
        parameters["discount"],
        parameters["optimizer"],
        parameters["optimizer_parameters"],
        parameters["polyak_target_update"],
        parameters["target_update_freq"],
        parameters["tau"],
        parameters["initial_eps"],
        parameters["end_eps"],
        parameters["eps_decay_period"],
        parameters["eval_eps"],
    )
    policy.load(f"./models/behavioral_{setting}")
    for i in range(len(new_replay_buffer.state) - 4):
        s_0 = new_replay_buffer.state[i: i + new_replay_buffer.state_history]
        s_1 = new_replay_buffer.state[i + 1: i + new_replay_buffer.state_history + 1]
        s_0 = torch.FloatTensor(torch.from_numpy(s_0.astype(np.float32))).reshape(policy.state_shape).to(policy.device)
        s_1 = torch.FloatTensor(torch.from_numpy(s_1.astype(np.float32))).reshape(policy.state_shape).to(policy.device)
        phi_0 = policy.Q_target(s_0).max(1, keepdim=True)[0]
        phi_1 = policy.Q_target(s_1).max(1, keepdim=True)[0]
        shaped_reward = policy.discount * phi_1 - phi_0
        new_replay_buffer.reward[i] = new_replay_buffer.reward[i] + shaped_reward.data.to("cpu").numpy()
        print(i, new_replay_buffer.reward[i])
    new_replay_buffer.save(f"./new_buffers/{new_buffer_name}")

