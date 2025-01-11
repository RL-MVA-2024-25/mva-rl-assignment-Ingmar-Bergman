from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os


import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


#Ce que j'ai écrit 

def normalize_reward(reward, max_reward=10000000.0):
    return reward / max_reward

import random
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device

        
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (
            torch.tensor(s, dtype=torch.float32).to(self.device),
            torch.tensor(a, dtype=torch.long).to(self.device),
            torch.tensor(r, dtype=torch.float32).to(self.device),
            torch.tensor(s_, dtype=torch.float32).to(self.device),
            torch.tensor(d, dtype=torch.float32).to(self.device),
        )
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(torch.stack, zip(*batch)))
    def __len__(self):
        return len(self.data)



import torch

def greedy_action(network, state_tensor, device='cuda'):
    """
    Selects the action with the highest Q-value for the given state.

    Args:
        network (torch.nn.Module): The neural network estimating Q-values.
        state (np.ndarray): The current state.
        device (torch.device): The device to perform computations on.

    Returns:
        int: The index of the action with the highest Q-value.
    """
    network.eval()  # Set network to evaluation mode
    with torch.no_grad():
        Q_values = network(state_tensor.unsqueeze(0))  # Add batch dimension
        action = torch.argmax(Q_values, dim=1).item()
    network.train()  # Set back to training mode if necessary #A SUPPRIEMR 
    return action 


class ProjectAgent:
    def __init__(self):
        """
        Initializes the DQN agent with the given configuration and model.
        """
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")


        env = TimeLimit(
        env=HIVPatient(domain_randomization=False), max_episode_steps=200)
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n 

        self.config = {'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'randomize_prob': 0,
          'gamma': 0.98,
          'buffer_size': 100000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 10000,
          'epsilon_delay_decay': 200,
          'batch_size': 640,
          'gradient_steps': 4,
          'update_target_strategy': 'replace', # or 'ema'
          'update_target_freq': 600, #PRENDRE UN TRUC PLUS FAIBLE
          'update_target_tau': 0.005,
          'criterion': torch.nn.SmoothL1Loss()}



        # Initialize hyperparameters
        self.nb_actions = self.config['nb_actions']
        self.gamma = self.config.get('gamma', 0.95)
        self.batch_size = self.config.get('batch_size', 100)
        buffer_size = self.config.get('buffer_size', int(1e5))
        self.epsilon_max = self.config.get('epsilon_max', 1.0)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.epsilon_stop = self.config.get('epsilon_decay_period', 1000)
        self.epsilon_delay = self.config.get('epsilon_delay_decay', 20)
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop

        # Initialize Replay Buffer (Assuming ReplayBuffer is defined elsewhere)
        self.memory = ReplayBuffer(buffer_size, self.device)

        # Initialize networks

        nb_neurons=256
        model = torch.nn.Sequential(
    nn.Linear(env.observation_space.shape[0], nb_neurons),
    nn.PReLU(),
    nn.Linear(nb_neurons, nb_neurons),
    nn.PReLU(),
    nn.Linear(nb_neurons, nb_neurons),
    nn.PReLU(), 
    nn.Linear(nb_neurons, env.action_space.n)).to(self.device)


        self.model = model.to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)

        # Initialize loss criterion and optimizer
        self.criterion = self.config.get('criterion', nn.MSELoss())
        lr = self.config.get('learning_rate', 0.001)
        self.optimizer = self.config.get('optimizer', optim.Adam(self.model.parameters(), lr=lr))

        # Initialize target network update parameters
        self.nb_gradient_steps = self.config.get('gradient_steps', 1)
        self.update_target_strategy = self.config.get('update_target_strategy', 'replace')
        self.update_target_freq = self.config.get('update_target_freq', 20)
        self.update_target_tau = self.config.get('update_target_tau', 0.005)
        
    def act(self, observation, use_random=False):
        """
        Select an action based on the given observation.

        Args:
            observation (np.array): The current state observation.
            use_random (bool): Whether to force random action selection (e.g., for exploration).

        Returns:
            int: The action selected.
        """
        # if use_random or np.random.rand() < self.epsilon:
        #     # Random action
        #     return np.random.randint(self.model[-1].out_features)  # Number of actions
        # else:
            # Greedy action
        state_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            action = torch.argmax(self.model(state_tensor)).item()
        self.model.train()
        return action

    def print_device_info(self):
        """
        Prints the device assignments of the model and target model.
        """
        print(f"Model is on device: {next(self.model.parameters()).device}")
        print(f"Target Model is on device: {next(self.target_model.parameters()).device}")

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 


    def save_model(self, path):
        """
        Saves the model's state dictionary to the specified path.

        Args:
            path (str): The file path to save the model (e.g., 'models/dqn_model.pth').
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self):
        """
        Loads the model's state dictionary from the specified path.

        Args:
            path (str): The file path from which to load the model (e.g., 'models/dqn_model.pth').
        """
        self.model.load_state_dict(torch.load('best_model_bis (10).pth', map_location=self.device))
        self.model.to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)

    def train(self, env, max_episode):
        """
        Trains the DQN agent on the given environment for a specified number of episodes.

        Args:
            env (gym.Env): The Gym environment to train on.
            max_episode (int): The number of episodes to train.

        Returns:
            list: A list of cumulative rewards per episode.
        """
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # Update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)
            
            # Convert state to tensor on device
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            
            # Select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state_tensor)
            
            # Step in the environment
            next_state, reward, done, trunc, _ = env.step(action)
            reward = normalize_reward(reward)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            
            # Train the agent
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            
            # Update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            elif self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau * model_state_dict[key] + (1 - tau) * target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            
            # Next transition
            step += 1
            if done or trunc:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return
