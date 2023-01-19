from env import Env
import numpy as np
import gym
from ppo import PPO, PPOConfig
import random

class CartPoleEnv(Env): 
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        super().__init__(observation_shape=np.array([4]), action_shape=np.array([2]))
        
    @property
    def is_multi_discrete(self):
        return self.action_shape.size > 1
    
    def _run_reset(self):
        return self.env.reset()

    def _run_step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            observation = self.env.reset()
        return observation, reward, done, info
    
    def randomise_seed(self):
        pass

    def close(self):
        self.env.close()

# Training
config = {
    'n_episodes': 100000,
    'hidden_size_1': 4,
    'hidden_size_2': 4,
    'lr': 3e-5,
    'lr_lambda': 0.9999,
    'gamma': 0.99,
    'lamda': 0.98,
    'ent_coef': 0.01,
    'vf_coef': 0.5,             
    'k_epochs': 4,       
    'kl_max': 0.02,              # early stop to prevent big kl         
    'eps_clip': 0.1,  
    'clip_vf': False,                            # Clip range
    'max_grad_norm': 0.5,
    'batch_size': 32,                   # Update every batch_size samples
    'minibatch_size': 8,               # Divide batch into smaller chunks
    'network_arch': 'separate',
    'reward_scaling': False,
    'show_training_progress': True,
    'device': 'cpu',
    'seed': 0
}

env = CartPoleEnv()
observation = env.reset()
rewards = 0
for _ in range(475):
    observation, reward, done, info = env.step(random.randint(0, 1))
    rewards += reward
    if done:
        break
print('Random policy rewards: %d' % (rewards))

ppo_config = PPOConfig(**config)
ppo = PPO(env, ppo_config)
ppo.set_log('training')
ppo.learn()
observation = env.reset()
rewards = 0
for _ in range(475):
    observation, reward, done, info = env.step(ppo.act(observation))
    rewards += reward
    if done:
        break
print('PPO rewards: %d' % (rewards))


env.close()