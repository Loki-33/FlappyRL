import torch
import torch.optim as optim
import torch.nn as nn
import pygame
import time
from flappy import Flappy 
import numpy as np

from collections import deque

class AC(nn.Module):
	def __init__(self, input_size, n_actions):
		super(AC, self).__init__()
		c,h,w = input_size
		self.net = nn.Sequential(
			nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
			nn.ReLU(),
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
			nn.ReLU(),
			nn.Flatten(),
			)

		with torch.no_grad():
			dummy = torch.zeros(1, c, h, w)
			n_flatten = self.net(dummy).shape[1]

		self.actor = nn.Linear(n_flatten, n_actions)

		self.critic = nn.Linear(n_flatten, 1)

	def forward(self, x):
		x = self.net(x)
		return self.actor(x), self.critic(x).squeeze(-1)



def compute_gae(rewards, dones, values, gamma=0.99, lmbda=0.95):
	gae = 0
	advantages = torch.zeros(len(rewards), dtype=torch.float32)

	for i in reversed(range(len(rewards))):
		if i == len(rewards) - 1:
			next_value = 0
		else:
			next_value =values[i+1]

		delta = rewards[i]+gamma*next_value*(1-dones[i]) - values[i]
		gae = delta + gamma * lmbda * (1-dones[i]) * gae 
		advantages[i] = gae 
	return advantages

env = Flappy(render_mode='rgb_array')
n_actions = env.action_space.n
obs_shape = env.observation_space.shape

gamma = 0.99
entropy_coef = 0.03
value_coef = 0.5
clip_ratio=0.2
lr = 1e-3
batch_size = 8
model = AC(obs_shape, n_actions)
optimizer = optim.Adam(model.parameters(), lr=lr)
mini_batch_size = 64


#train 
def train(episodes):

	batch_data = []
	episode_reward = []
	performance = float('-inf')

	for ep in range(episodes):
		state, _ = env.reset()
		done = False 
		rewards, dones, values = [],[],[]
		log_probs, entropies, actions, states = [],[],[],[]

		while not done:
			state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) / 255.0
			
			with torch.no_grad():	
				logits, value = model(state_tensor)
				probs = torch.softmax(logits, dim=-1)
				dist = torch.distributions.Categorical(probs)
				action = dist.sample()
				entropy = dist.entropy()
				log_prob = dist.log_prob(action)
			
			next_state, reward, done, truncated, _ = env.step(action.item())

			states.append(state)
			actions.append(action.cpu().item())
			rewards.append(float(reward))
			dones.append(float(done or truncated))
			values.append(value.cpu().item())
			log_probs.append(log_prob.cpu().item())
			entropies.append(entropy.cpu().item())

			state=next_state
			done=done or truncated

		batch_data.append({
			'states':np.array(np.array(states), dtype=np.uint8),
			'actions': np.array(actions, dtype=np.int64),
	        'rewards': np.array(rewards, dtype=np.float32),
	        'dones': np.array(dones, dtype=np.float32),
	        'values': np.array(values, dtype=np.float32),
	        'log_probs': np.array(log_probs, dtype=np.float32),
	        'entropies': np.array(entropies, dtype=np.float32),
   			})
		
		episode_reward.append(sum(rewards))
		
		if len(batch_data) >= batch_size:
			all_states = np.concatenate([data['states'] for data in batch_data], axis=0)
			all_actions = np.concatenate([data['actions'] for data in batch_data], axis=0)
			all_rewards = np.concatenate([data['rewards'] for data in batch_data], axis=0)
			all_dones = np.concatenate([data['dones'] for data in batch_data], axis=0)
			all_values = np.concatenate([data['values'] for data in batch_data], axis=0)
			all_log_probs = np.concatenate([data['log_probs'] for data in batch_data], axis=0)
			all_entropies = np.concatenate([data['entropies'] for data in batch_data], axis=0)

			all_states = torch.tensor(all_states, dtype=torch.float32)/255.0
			all_actions = torch.tensor(all_actions, dtype=torch.long)
			all_rewards = torch.tensor(all_rewards, dtype=torch.float32)
			all_dones = torch.tensor(all_dones, dtype=torch.float32)
			all_values = torch.tensor(all_values, dtype=torch.float32)
			all_log_probs = torch.tensor(all_log_probs, dtype=torch.float32)
			all_entropies = torch.tensor(all_entropies, dtype=torch.float32)

			advantages = compute_gae(all_rewards, all_dones, all_values)
			advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)
			
			dataset_size = all_states.shape[0]
			returns = advantages + all_values 

			for _ in range(4):
				ids = np.arange(dataset_size)
				np.random.shuffle(ids)

				for start in range(0, dataset_size, mini_batch_size):
					mb_ids = ids[start:start+mini_batch_size]
					mb_states = all_states[mb_ids]
					mb_actions = all_actions[mb_ids]
					mb_old_logp = all_log_probs[mb_ids]
					mb_adv = advantages[mb_ids]
					mb_ret = returns[mb_ids]
					

					logits, new_values = model(mb_states)
					probs = torch.softmax(logits, dim=-1)
					dist = torch.distributions.Categorical(probs)

					mb_new_log_probs = dist.log_prob(mb_actions)
					mb_entropy = dist.entropy().mean()

					ratio = torch.exp(mb_new_log_probs - mb_old_logp)
					
					clipped_ratio= torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio)
					
					policy_loss = -torch.min(ratio*mb_adv, clipped_ratio*mb_adv).mean()
					
					value_loss = ((mb_ret - new_values) ** 2).mean()
					
					entropy_loss = -entropy_coef * mb_entropy
					
					loss = policy_loss + 0.5 * value_loss + entropy_loss
					
					optimizer.zero_grad()
					loss.backward()
					nn.utils.clip_grad_norm_(model.parameters(), 0.5)
					optimizer.step()
			
			avg_reward = np.mean(episode_reward[-batch_size:])
			print(f"Episode {ep+1}, Loss: {loss.item():.4f}, Avg Return: {avg_reward:.2f}")
			
			if avg_reward > performance:
				performance = avg_reward
				torch.save(model.state_dict(), 'model.pth')
				print(f"New best average reward: {performance:.2f} - Model saved!")
			batch_data = []
	env.close()



if __name__ == '__main__':
    train(episodes=1000)