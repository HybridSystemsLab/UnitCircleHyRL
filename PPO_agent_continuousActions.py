# Based on the code by Phil Tabor 
# https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO/torch
import os 
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        
        self.batch_size = batch_size
    
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.logprobs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches
                
    def store_memory(self, state, action, logprobs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprobs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, n_actions, 
                 fc1_dims=64, fc2_dims=64, chkpt_dir='checkpoints'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.reparam_noise = 1e-6
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.max_action = max_action
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, *self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, *self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        self.initialize_weights()
        
    def forward(self, state):
        policy = self.fc1(state)
        policy = F.relu(policy)
        policy = self.fc2(policy)
        policy = F.relu(policy)
        
        mu = self.mu(policy)
        sigma = self.sigma(policy)
    
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)
        
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)
        
        if reparameterize:
            actions = probabilities.rsample()
            # adding some noise to achieve additional exploration
        else:
            actions = probabilities.sample()
        
        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise) # comes from the appendix SAC paper
        log_probs = log_probs.sum(1, keepdim=True)
        
        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                y = 1.0 / np.sqrt(m.in_features)
                #nn.init.normal_(m.weight, 0.0, np.sqrt(y))
                nn.init.uniform_(m.weight, -y, y)

    
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=64,
                 fc2_dims=64, chkpt_dir='checkpoints'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        self.initialize_weights()
        
    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        
        v = self.v(state_value)
        
        return v
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                y = 1.0 / np.sqrt(m.in_features)
                nn.init.zeros_(m.weight)
                #nn.init.normal_(m.weight, 0, np.sqrt(y))
                #nn.init.uniform_(m.weight, -y, y)
                
class Agent:
    def __init__(self, n_actions, max_actions, input_dims, gamma=0.99, alpha=3e-4, 
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, N=2048, 
                 fc1_dims=64, fc2_dims=64, n_epochs=10, beta=1e-5):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        
        self.actor = ActorNetwork(alpha, input_dims, max_actions, n_actions, 
                                  fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.critic = CriticNetwork(beta, input_dims, fc1_dims=fc1_dims, 
                                    fc2_dims=fc2_dims)
        self.memory = PPOMemory(batch_size)
        
    def remember(self, state, action, logprobs, vals, reward, done):
        self.memory.store_memory(state, action, logprobs, vals, reward, done)
        
    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        
    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        
    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float32).to(self.actor.device)
        actions, log_probs = self.actor.sample_normal(state, reparameterize=(False))
        value = self.critic(state)
        
        #action = T.squeeze(action).item()
        #action = action.cpu().detach().numpy()[0]
        #log_probs = log_probs.cpu().detach().numpy()[0]
        value = T.squeeze(value).item()

        return actions.cpu().detach().numpy()[0], \
            log_probs.cpu().detach().numpy()[0], \
                value
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_log_probs_arr, vals_arr,\
            reward_arr, done_arr, batches = \
                    self.memory.generate_batches()
        
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount * (reward_arr[k] + self.gamma*values[k+1]*\
                                       (1-int(done_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)
            
            
            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_log_probs = T.tensor(old_log_probs_arr[batch]).to(self.actor.device)
                #actions = T.tensor(action_arr[batch]).to(self.actor.device)
                
                _, new_log_probs = self.actor.sample_normal(states)
                #dist = self.actor(states)
                critic_value = self.critic(states)
                
                critic_value = T.squeeze(critic_value)
                
                #new_log_probs = dist.log_prob(actions)
                prob_ratio = new_log_probs.exp() / old_log_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                                            1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                
                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
        self.memory.clear_memory()