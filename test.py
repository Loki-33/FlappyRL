import torch
import pygame
import numpy as np 
import gymnasium as gym 
from flappy import Flappy
from main import AC
import time 

env = Flappy(render_mode='human')

state, _ = env.reset()
obs_shape = env.observation_space.shape 
n_actions = env.action_space.n 


model = AC(obs_shape, n_actions)
model.load_state_dict(torch.load('model.pth'))

model.eval()

done=False 
total_reward = 0

while not done:
	env.render()
	with torch.no_grad():
		state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) / 255.0
		logits, _ = model(state_tensor)
		probs = torch.softmax(logits, dim=-1)
		action = torch.multinomial(probs, 1).item()
	next_state, reward, terminated, truncated, _ = env.step(action)
	done = truncated or terminated
	total_reward += reward
	time.sleep(0.003)
env.close()
print(f"REWARD: {total_reward}")