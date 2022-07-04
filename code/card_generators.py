import numpy as np
import torch
import random


from scipy.stats import pareto



def generate_game(num_iterations, card_generator, agents):
    agent_rewards = {key:[] for key in agents.keys()}

    def get_reward(hidden_number_is_bigger, actual):
        if hidden_number_is_bigger == actual:
            return 1
        else:
            return 0
     
    for i in range(num_iterations):
        selected_numbers = card_generator.draw_cards()
        shown_number = selected_numbers[0]
        hidden_number = selected_numbers[1]

        actual = hidden_number > shown_number

        for agent in agents.keys():
            hidden_number_is_bigger = agents[agent].decision(shown_number)
            reward = get_reward(hidden_number_is_bigger, actual) 
            agent_rewards[agent].append(reward)
            if hasattr(agents[agent], 'update_rule'):
                agents[agent].update_rule(actual.type(torch.float32))     

    return agent_rewards



class NormalCards:
    def __init__(self, mu, sigma, num_cards = 2):
        self.num_cards = 2
        self.mu = mu
        self.sigma = sigma


    def draw_cards(self, flip_coin = False):
        selected_numbers = self.sigma*torch.randn(self.num_cards)+self.mu
        if flip_coin:
            shuffled = torch.randperm(self.num_cards)     
            return selected_numbers[shuffled]
        else:
            return selected_numbers


class ParetoCards:
    def __init__(self, a, b, num_cards = 2):
        self.num_cards = 2
        self.a = a
        self.b = b

    def draw_cards(self, flip_coin = False):
        numpy_pareto = pareto.rvs(self.a, self.b, size = self.num_cards)
        selected_numbers = torch.from_numpy(numpy_pareto.astype(np.float32))
        if flip_coin:
            shuffled = torch.randperm(self.num_cards)     
            return selected_numbers[shuffled]
        else:
            return selected_numbers        


class NumberCards:
    def __init__(self, numbers, num_cards = 2):
        self.num_cards = num_cards
        self.numbers = numbers
        self.iter = 0

    def draw_cards(self, flip_coin = False):
        start_indx, end_indx = self.iter, self.iter+self.num_cards
        if end_indx > len(self.numbers):
            print("out of numbers")
            return None
        
        selected_numbers = torch.from_numpy(self.numbers[start_indx:end_indx])
        self.iter = end_indx
        if flip_coin:
            shuffled = torch.randperm(self.num_cards)     
            return selected_numbers[shuffled]
        else:
            return selected_numbers  