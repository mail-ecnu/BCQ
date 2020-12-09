#!/bin/sh

python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0 --rand_action_p 0
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0 --rand_action_p 0.2
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0 --rand_action_p 0.4
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0 --rand_action_p 0.6
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0 --rand_action_p 0.8

python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.2 --rand_action_p 0
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.2 --rand_action_p 0.2
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.2 --rand_action_p 0.4
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.2 --rand_action_p 0.6
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.2 --rand_action_p 0.8

python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.4 --rand_action_p 0
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.4 --rand_action_p 0.2
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.4 --rand_action_p 0.4
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.4 --rand_action_p 0.6
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.4 --rand_action_p 0.8

python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.6 --rand_action_p 0
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.6 --rand_action_p 0.2
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.6 --rand_action_p 0.4
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.6 --rand_action_p 0.6
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.6 --rand_action_p 0.8

python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.8 --rand_action_p 0
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.8 --rand_action_p 0.2
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.8 --rand_action_p 0.4
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.8 --rand_action_p 0.6
python main.py --generate_buffer --env MiniGrid-DoorKey-8x8-v0 --low_noise_p 0.8 --rand_action_p 0.8