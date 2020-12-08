import argparse
import time
import numpy
import torch
import DQN
from array2gif import write_gif

import utils


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default='demo',
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=10,
                    help="number of episodes to visualize")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, args.seed)
for _ in range(args.shift):
    env.seed(args.seed)
    env.reset()
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n
print("Environment loaded\n")

# Load agent
is_atari = False
parameters = {
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
agent = DQN.DQN(
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

setting = f"{args.env}_{args.seed}"
agent.load(f"./models/behavioral_{setting}")

print("Agent loaded\n")

# Run the agent and create gif
frames = []

for episode in range(args.episodes):
    env.seed(args.seed)
    obs = env.reset()

    while True:
        frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))
        action = agent.select_action(obs, eval=True)
        obs, reward, done, _ = env.step(action)

        if done:
            break

print("Saving gif... ", end="")
write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
print("Done.")