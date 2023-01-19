import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

def ortho_init(layer, scale=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain=scale)
    nn.init.constant_(layer.bias, 0)
    return layer

class SDModel(nn.Module): 
    def __init__(self, obs_dim, n_action, hidden_size_1, hidden_size_2):
        super(SDModel, self).__init__()
        self.critic = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden_size_1)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, hidden_size_2)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, 1), scale=1)
        )
        self.actor = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden_size_1)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, hidden_size_2)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, n_action), scale=0.01)
        )

    def get_value(self, obs):
        return self.critic(obs)

    def get_action(self, obs, action=None):
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        if action is None: 
            action = probs.sample()
        logprob = probs.log_prob(action)
        entropy = probs.entropy()
        return action, logprob, entropy

class MDModel(nn.Module): 
    def __init__(self, obs_dim, action_space, hidden_size_1, hidden_size_2):
        super(MDModel, self).__init__()
        self.action_nvec = action_space
        self.shared_network = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden_size_1)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, hidden_size_2)),
            nn.Tanh(),
        )
        self.actor = ortho_init(nn.Linear(hidden_size_1, self.action_nvec.sum()), scale=0.01)
        self.critic = ortho_init(nn.Linear(hidden_size_1, 1), scale=1)

    def get_value(self, obs):
        return self.critic(self.shared_network(obs))

    def get_action(self, obs, action=None):
        logits = self.actor(self.shared_network(obs))
        split_logits = torch.split(logits, self.action_nvec.tolist(), dim=1)
        multi_dists = [Categorical(logits=logits) for logits in split_logits]
        if action is None: 
            action = torch.stack([dist.sample() for dist in multi_dists])
        else: 
            action = action.T
        logprob = torch.stack([dist.log_prob(a) for a, dist in zip(action, multi_dists)])
        entropy = torch.stack([dist.entropy() for dist in multi_dists])
        return action.T, logprob.sum(dim=0, dtype=torch.float64), entropy.sum(dim=0, dtype=torch.float64)

class MDSModel(nn.Module): 
    def __init__(self, obs_dim, observation_space, hidden_size_1, hidden_size_2):
        super(MDSModel, self).__init__()
        self.action_nvec = observation_space.nvec
        self.critic = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden_size_1)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, hidden_size_2)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, 1), scale=1)
        )
        self.actor = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden_size_1)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, hidden_size_2)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, self.action_nvec.sum()), scale=0.01)
        )

    def get_value(self, obs):
        return self.critic(obs)

    def get_action(self, obs, action=None):
        logits = self.actor(obs)
        split_logits = torch.split(logits, self.action_nvec.tolist(), dim=1)
        multi_dists = [Categorical(logits=logits) for logits in split_logits]
        if action is None: 
            action = torch.stack([dist.sample() for dist in multi_dists])
        else: 
            action = action.T
        logprob = torch.stack([dist.log_prob(a) for a, dist in zip(action, multi_dists)])
        entropy = torch.stack([dist.entropy() for dist in multi_dists])
        return action.T, logprob.sum(dim=0, dtype=torch.float64), entropy.sum(dim=0, dtype=torch.float64)
    
    def get_det_action(self, obs, action=None):
        logits = self.actor(obs)
        split_logits = torch.reshape(logits, (self.action_nvec.size, self.action_nvec[0]))
        return torch.argmax(split_logits, dim=1)

class MDSCModel(nn.Module): 
    def __init__(self, obs_dim, action_space, hidden_size_1, hidden_size_2):
        super(MDSCModel, self).__init__()
        self.action_nvec = action_space.nvec
        self.critic = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden_size_1)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, hidden_size_2)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, 1), scale=1)
        )
        self.actor_means = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden_size_1)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, hidden_size_2)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, self.action_nvec.shape[0]), scale=0.01)
        )
        self.actor_logstds = nn.Parameter(torch.zeros(1, np.prod(self.action_nvec.shape[0])))

    def get_value(self, obs):
        return self.critic(obs)

    def get_action(self, obs, actions=None):
        action_means = self.actor_means(obs)
        action_logstds = self.actor_logstds.expand_as(action_means)
        action_stds = torch.exp(action_logstds)
        probs = Normal(action_means, action_stds)
        if actions is None:
            actions = probs.sample()

        actions[actions < 0] = 0
        actions[actions > 0] = self.action_nvec[0] - 1
        return actions.int(), probs.log_prob(actions).sum(dim=1, dtype=torch.float64), probs.entropy().sum(dim=1, dtype=torch.float64)

class RunningMeanStd:
    # https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/69019cf9b1624db3871d4ed46e29389aadfdcb02/4.PPO-discrete/normalization.py
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

class RewardScaler:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def scale(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)
