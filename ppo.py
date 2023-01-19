import torch
from tqdm import tqdm
import numpy as np
from torch.optim import lr_scheduler
from env import Env
from models import SDModel, MDModel, MDSModel, MDSCModel, RewardScaler
from dataclasses import dataclass
from torch.utils.data import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from time import gmtime, strftime

@dataclass
class PPOConfig:
    n_episodes: int
    hidden_size_1: int
    hidden_size_2: int
    lr: float
    lr_lambda: float
    gamma: float # GAE parameter
    lamda: float # GAE parameter
    ent_coef: float # Entropy coefficient
    vf_coef: float # Value function coefficient
    k_epochs: int
    kl_max: float
    eps_clip: float
    clip_vf: bool
    max_grad_norm: float # Clip gradient
    batch_size: int
    minibatch_size: int
    network_arch: str
    reward_scaling: bool
    show_training_progress: bool
    device: str
    seed: int

class PPO():
    def __init__(self, env: Env, config: PPOConfig):
        self.env = env
        self.config = config
        self.obs_dim = self.env.observation_shape[0]
        self.device =  torch.device(self.config.device)
        self.total_steps = 0
        self.writer = None

        torch.manual_seed(self.config.seed)

        if self.env.is_multi_discrete: 
            if self.config.network_arch == 'shared':
                self.model = MDModel(self.obs_dim, self.env.action_shape, self.config.hidden_size_1, self.config.hidden_size_2).to(self.device) 
            elif self.config.network_arch == 'separate':
                self.model = MDSModel(self.obs_dim, self.env.action_shape, self.config.hidden_size_1, self.config.hidden_size_2).to(self.device) 
            elif self.config.network_arch == 'continuous':
                self.model = MDSCModel(self.obs_dim, self.env.action_shape, self.config.hidden_size_1, self.config.hidden_size_2).to(self.device)
            else:
                assert self.config.network_arch not in ['shared', 'separate', 'continuous']
        else: 
            self.model = SDModel(self.obs_dim, self.env.action_shape[0], self.config.hidden_size_1, self.config.hidden_size_2).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, eps=1e-5)
    
    def set_log(self, logdir):
        if logdir: 
            run_name = f"{strftime('%Y%m%d', gmtime())}"
            self.writer = SummaryWriter(f"{logdir}/{run_name}")

            self.writer.add_text(
                "Environment hyperparameters",
                "|param|value|\n|---|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.config).items()])),
            )
            self.writer.add_text(
                "Agent hyperparameters",
                "|param|value|\n|---|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.config).items()])),
            )

    def act(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(obs).unsqueeze(dim=0).float()
        action, _, _ = self.model.get_action(obs.to(self.device))
        return action

    def learn(self):
        ep_returns = np.zeros(self.config.n_episodes)
        pbar = tqdm(range(int(self.config.n_episodes)), disable=not bool(self.config.show_training_progress))
        return_factor = int(self.config.n_episodes*0.01 if self.config.n_episodes >= 100 else 1)
        if self.env.is_multi_discrete:
            action_batch = torch.zeros((self.config.batch_size,self.env.action_shape.size)).to(self.device)
        else: 
            action_batch = torch.zeros(self.config.batch_size).to(self.device)
        obs_batch = torch.zeros(self.config.batch_size, self.obs_dim).to(self.device)
        next_obs_batch = torch.zeros(self.config.batch_size, self.obs_dim).to(self.device)
        logprob_batch = torch.zeros(self.config.batch_size).to(self.device)
        rewards_batch = torch.zeros(self.config.batch_size).to(self.device)
        done_batch = torch.zeros(self.config.batch_size).to(self.device)
        batch_head = 0
    
        if self.config.reward_scaling:
            reward_scaler = RewardScaler(shape=1, gamma=self.config.gamma) 

        scheduler = lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lambda epoch: self.config.lr_lambda) 

        for i_episode in pbar:
            current_ep_reward = 0
            self.env.randomise_seed() # get different sequence
            obs = self.env.reset()
            done = False
            while not done:
                action, logprob, _ = self.model.get_action(obs.to(self.device))
                next_obs, reward, done, _ = self.env.step(action)
                self.total_steps += 1
                
                if self.config.reward_scaling:
                    reward_t = reward_scaler.scale(reward)[0]
                else: 
                    reward_t = reward

                action_batch[batch_head] = torch.flatten(action)
                obs_batch[batch_head] = torch.flatten(obs)
                next_obs_batch[batch_head] = torch.flatten(next_obs)
                logprob_batch[batch_head] = logprob.item()
                rewards_batch[batch_head] = reward_t
                done_batch[batch_head] = done
                batch_head += 1

                if batch_head >= self.config.batch_size:
                    self.update(action_batch, obs_batch, next_obs_batch, logprob_batch, rewards_batch, done_batch)
                    batch_head = 0
                    scheduler.step()
                
                obs = next_obs

                current_ep_reward += reward  # For logging

            ep_returns[i_episode] = current_ep_reward
            if self.writer: 
                self.writer.add_scalar('Training/ep_return', current_ep_reward, i_episode)
                self.writer.add_scalar('Training/lr', scheduler.get_last_lr()[0], i_episode)

            if i_episode > return_factor: 
                pbar.set_description("Return %.2f" % np.median(ep_returns[i_episode-return_factor:i_episode]))

    def update(self, action_batch, obs_batch, next_obs_batch, logprob_batch, rewards_batch, done_batch):

        done_batch = done_batch.int()

        # GAE advantages         
        
        with torch.no_grad():      
            gae = 0     
            advantages = torch.zeros_like(rewards_batch).to(self.device)
            values_batch = torch.flatten(self.model.get_value(obs_batch))
            next_values = torch.flatten(self.model.get_value(next_obs_batch))
            deltas = rewards_batch + (1 - done_batch) * self.config.gamma * next_values - values_batch
            for i in reversed(range(len(deltas))):
                gae = deltas[i] + (1 - done_batch[i]) * self.config.gamma * self.config.lamda * gae 
                advantages[i] = gae
            
            returns = advantages + values_batch

        clipfracs = []


        for epoch in range(self.config.k_epochs):
            minibatches = BatchSampler(
                SubsetRandomSampler(range(self.config.batch_size)), 
                batch_size=self.config.minibatch_size, 
                drop_last=False)

            for bi, minibatch in enumerate(minibatches):
                adv_minibatch = advantages[minibatch]
                adv_minibatch = (adv_minibatch - adv_minibatch.mean()) / (adv_minibatch.std() + 1e-8) # Adv normalisation
                _, newlogprob, entropy = self.model.get_action(obs_batch[minibatch], action_batch[minibatch])
                log_ratios = newlogprob - logprob_batch[minibatch] # KL divergencey
                ratios = torch.exp(log_ratios)
                assert bi != 0 or epoch != 0 or torch.all(torch.abs(ratios - 1.0) < 2e-4), 'epoch: %d mb: %d log ratios should be zeros:\n %s' % (epoch, bi, str(log_ratios)) # newlogprob == logprob_batch in epoch 1 minibatch 1
                if -log_ratios.mean() > self.config.kl_max:
                    break
                clipfracs.append(((ratios - 1.0).abs() > self.config.eps_clip).float().mean().item())
                
                surr = -ratios * adv_minibatch
                surr_clipped = -torch.clamp(ratios, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * adv_minibatch
                loss_clipped = torch.max(surr, surr_clipped).mean()

                newvalues = self.model.get_value(obs_batch[minibatch])
                loss_vf = torch.square(newvalues - returns[minibatch])
                v_clipped = values_batch[minibatch] + torch.clamp(newvalues - values_batch[minibatch], -self.config.eps_clip, self.config.eps_clip)
                loss_vf_clipped = torch.square(v_clipped - returns[minibatch])
                if self.config.clip_vf:
                    loss_vf = torch.max(loss_vf, loss_vf_clipped)
                loss_vf = 0.5 * loss_vf.mean() # unclipped should be better

                loss = loss_clipped - self.config.ent_coef * entropy.mean() + self.config.vf_coef * loss_vf # maximise equation (9) from the original PPO paper

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

        if self.writer: 
            self.writer.add_scalar('training/loss_clipped', loss_clipped.item(), self.total_steps)
            self.writer.add_scalar('training/loss_vf', loss_vf.item(), self.total_steps)
            self.writer.add_scalar('training/entropy', entropy.mean().item(), self.total_steps)
            self.writer.add_scalar('training/loss', loss.item(), self.total_steps)
            self.writer.add_scalar('training/kl', -log_ratios.mean().item(), self.total_steps)
            self.writer.add_scalar('training/clipfracs', np.mean(clipfracs), self.total_steps)
