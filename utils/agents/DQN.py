from collections import Counter

import torch
import numpy as np
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from .Base_Agent import Base_Agent
from .Q_predictor import Q_predictor
from utils.exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utils.trainer.Replay_Buffer import Replay_Buffer

class DQN(Base_Agent):
    """A deep Q learning agent"""
    agent_name = "DQN"
    def __init__(self, config, cfg, env, action_mask):
        Base_Agent.__init__(self, config, cfg, env, action_mask)
        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"], config.seed, self.device)
        self.q_network_local = Q_predictor(cfg)
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyperparameters["learning_rate"], eps=1e-4)
        self.exploration_strategy = Epsilon_Greedy_Exploration(config)
        self.mask_init = action_mask.reshape(-1,1)
        self.mask = self.mask_init.copy()
        self.action_idx_ls = []

    def reset_game(self):
        super(DQN, self).reset_game()
        self.update_learning_rate(self.hyperparameters["learning_rate"], self.q_network_optimizer)
        self.mask = self.mask_init.copy()
        self.action_idx_ls = []

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.action = self.pick_action()
            self.conduct_action(self.action)
            if self.time_for_q_network_to_learn():
                for _ in range(self.hyperparameters["learning_iterations"]):
                    self.learn()
            self.save_experience()
            self.state = self.next_state #this is to set the state for the next iteration
            self.global_step_number += 1
        self.episode_number += 1

    def pick_action(self, state=None):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        if state is None: state = self.state
        if isinstance(state, np.int64) or isinstance(state, int): state = np.array([state])
        #state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # if len(state.shape) < 2: state = state.unsqueeze(0)
        self.q_network_local.eval() #puts network in evaluation mode
        with torch.no_grad():
            action_values = self.q_network_local(state)
            action_values = action_values * (1-self.mask)
            print('max action = {}'.format(np.argmax(action_values)))
        self.q_network_local.train() #puts network back in training mode
#         action = self.exploration_strategy.perturb_action_for_exploration_purposes(
#             {"action_values": action_values,
#              "turn_off_exploration": self.turn_off_exploration,
#              "episode_number": self.episode_number}
#         )
        if(random.random() < 0.05):
            action = np.random.randint(0, 270)
        else:
            action = torch.argmax(action_values).item()
        self.logger.info("Q values {} -- Action chosen {}".format(action_values, action))
        return action

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        if experiences is None: states, actions, rewards, next_states, dones = self.sample_experiences() #Sample experiences
        else: states, actions, rewards, next_states, dones = experiences
        loss = self.compute_loss(states, next_states, rewards, actions, dones)

        actions_list = [action_X.item() for action_X in actions ]

        self.logger.info("Action counts {}".format(Counter(actions_list)))
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss, self.hyperparameters["gradient_clipping_norm"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        """Computes the q_targets we will compare to predicted q values to create the loss to train the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        #Q_targets_next = self.q_network_local(next_states).detach().max(1)[0].unsqueeze(1) #This is for BATCH x ACTIONS
        Q_targets_next = self.q_network_local(next_states).detach().max(0)[0].unsqueeze(1) #ACTIONS x 1
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        Q_targets_current = rewards + (self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        #Q_output = (ACTIONS, 1)
        #Q_expected = self.q_network_local(states).gather(1, actions.long()) #must convert actions to long so can be used as index
        Q_expected = self.q_network_local(states).gather(0, actions.long())
        return Q_expected

    def locally_save_policy(self):
        """Saves the policy"""
        ckpt_path = "{}/agent{}.ckpt".format(self.SAVE_PATH, 0)
        torch.save(
            {
            "model": self.q_network_local.state_dict(),
            "optimizer": self.q_network_optimizer.state_dict(), 
            "episode": self.episode_number,
            "config": self.config
            },
            ckpt_path
        )

    def time_for_q_network_to_learn(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin and there are
        enough experiences in the replay buffer to learn from"""
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()

    def right_amount_of_steps_taken(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin"""
        return self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def sample_experiences(self):
        """Draws a random sample of experience from the memory buffer"""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones