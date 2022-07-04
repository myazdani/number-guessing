import torch
from torch import nn
import numpy as np
from torch import optim
import random

class RandomDecision:

    def decision(self, shown_number=0):
        self.prediction = bool(random.randint(0,1))
        return self.prediction

    def update_rule(self, actual):
        pass


class AlwaysTrue:
    def decision(self, shown_number=0):
        self.prediction = True
        return self.prediction

    def update_rule(self, actual):
        pass  


class AlwaysFalse:
    def decision(self, shown_number=0):
        self.prediction = False
        return self.prediction

    def update_rule(self, actual):
        pass          


class OptimalDecision:

    def sigmoid(self, x):
        """
        A numerically stable version of the logistic sigmoid function.
        """
        pos_mask = (x >= 0)
        neg_mask = (x < 0)
        z = np.zeros_like(x)
        z[pos_mask] = np.exp(-x[pos_mask])
        z[neg_mask] = np.exp(x[neg_mask])
        top = np.ones_like(x)
        top[neg_mask] = z[neg_mask]
        return top / (1 + z)

    def decision(self, shown_number=0):
        z = np.random.rand()
        shown_number_squashed = self.sigmoid(shown_number)
        if z <= shown_number_squashed:
            return 0
        else:
            return 1



class LinearDecision:
    # Reference for online logistic regerssion learning rule: 
    # https://courses.cs.washington.edu/courses/cse547/16sp/slides/logistic-SGD.pdf
    def __init__(self, lr = 0.1):
        self.w0 = torch.randn(1)
        self.w1 = torch.randn(1)
        self.sigmoid = nn.Sigmoid()
        self.lr = lr
        self.t = 1

    def z_scale(self, shown_number):
        if self.t == 1:
            self.mu = 0
            self.sigma = 0
        self.mu = ((self.t-1)*self.mu + shown_number)/self.t
        self.sigma = ((self.t-1)*self.sigma + (shown_number-self.mu)**2)/self.t
        if self.t > 1:
            z = (shown_number-self.mu)/self.sigma**.5
        else:
            z = shown_number 
        return z            
    
    def decision(self, shown_number):
        self.shown_number = self.z_scale(shown_number)
        self.prediction = self.sigmoid(self.shown_number*self.w1 + self.w0)
        decision = bool(self.prediction > 0.5)
        return decision

    def update_rule(self, actual):
        delta = actual - self.prediction
        eta = self.lr/(1+np.log(self.t))
        self.w0 = self.w0 + eta*delta
        self.w1 = self.w1 + eta*self.shown_number*delta
        self.t+=1
                    


class LinearMLPDecision:
    def __init__(self, lr = 0.1):
        self.model = nn.Sequential(
            nn.Linear(1, 1), nn.Sigmoid()
        )
        self.lr = lr
        self.t = 1
        self.loss = nn.BCELoss()
        self.sgd = optim.SGD(self.model.parameters(), lr=self.lr)


    def z_scale(self, shown_number):
        if self.t == 1:
            self.mu = 0
            self.sigma = 0
        self.mu = ((self.t-1)*self.mu + shown_number)/self.t
        self.sigma = ((self.t-1)*self.sigma + (shown_number-self.mu)**2)/self.t
        if self.t > 1:
            z = (shown_number-self.mu)/self.sigma**.5
        else:
            z = shown_number 
        return z          


    def decision(self, shown_number):
        self.shown_number = self.z_scale(shown_number)
        self.prediction = self.model(self.shown_number.view(1,1))
        decision = bool(self.prediction.item() > 0.5)
        return decision

    def update_rule(self, actual):
        eta = self.lr/(1+np.log(self.t))
        for g in self.sgd.param_groups:
            g['lr'] = eta
        loss = self.loss(self.prediction, actual.view(1,1))
        self.sgd.zero_grad()
        loss.backward()
        self.sgd.step()
        self.t+=1
        

class MLPDecision:
    def __init__(self, lr = 0.1):
        self.model = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(), 
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.lr = lr
        self.t = 1
        self.loss = nn.BCELoss()
        #self.sgd = optim.Adam(self.model.parameters(), lr=self.lr)
        self.sgd = optim.SGD(self.model.parameters(), lr=self.lr)

    def z_scale(self, shown_number):
        if self.t == 1:
            self.mu = 0
            self.sigma = 0
        self.mu = ((self.t-1)*self.mu + shown_number)/self.t
        self.sigma = ((self.t-1)*self.sigma + (shown_number-self.mu)**2)/self.t
        if self.t > 1:
            z = (shown_number-self.mu)/self.sigma**.5
        else:
            z = shown_number 
        return z  

    def decision(self, shown_number):
        self.shown_number = self.z_scale(shown_number)
        self.prediction = self.model(self.shown_number.view(1,1))
        decision = bool(self.prediction.item() > 0.5)
        return decision

    def update_rule(self, actual):
        eta = self.lr/(1+np.log(self.t))
        for g in self.sgd.param_groups:
            g['lr'] = eta
        loss = self.loss(self.prediction, actual.view(1,1))
        self.sgd.zero_grad()
        loss.backward()
        self.sgd.step()
        self.t+=1
                
class StatefulMLPDecision:
    def __init__(self, lr = 0.1):
        self.model = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(), 
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.lr = lr
        self.t = 1
        self.loss = nn.BCELoss()
        self.sgd = optim.Adam(self.model.parameters(), lr=self.lr)
        #self.sgd = optim.SGD(self.model.parameters(), lr=self.lr, momentum=.9)
        self.all_shown_numbers = []
        self.all_actuals = []

    def z_scale(self, shown_number):
        if self.t == 1:
            self.mu = 0
            self.sigma = 0
        self.mu = ((self.t-1)*self.mu + shown_number)/self.t
        self.sigma = ((self.t-1)*self.sigma + (shown_number-self.mu)**2)/self.t
        if self.t > 1:
            z = (shown_number-self.mu)/self.sigma**.5
        else:
            z = shown_number 
        return z  

    def decision(self, shown_number):
        self.shown_number = self.z_scale(shown_number)
        self.all_shown_numbers.append(self.shown_number)
        self.prediction = self.model(self.shown_number.view(1,1))
        decision = bool(self.prediction.item() > 0.5)
        return decision

    def update_rule(self, actual):
        # eta = self.lr/(1+np.log(self.t))
        # for g in self.sgd.param_groups:
        #     g['lr'] = eta
        self.all_actuals.append(actual)
        predictions = self.model(torch.Tensor(self.all_shown_numbers).view(-1,1))
        loss = self.loss(predictions, torch.Tensor(self.all_actuals).view(-1,1))
        self.sgd.zero_grad()
        loss.backward()
        self.sgd.step()
        self.t+=1                


class MomentumMLPDecision:
    def __init__(self, lr = 0.1):
        self.model = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(), 
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.lr = lr
        self.t = 1
        self.loss = nn.BCELoss()
        #self.sgd = optim.Adam(self.model.parameters(), lr=self.lr)
        self.sgd = optim.SGD(self.model.parameters(), lr=self.lr, momentum=.9)
        self.all_shown_numbers = []
        self.all_actuals = []

    def z_scale(self, shown_number):
        if self.t == 1:
            self.mu = 0
            self.sigma = 0
        self.mu = ((self.t-1)*self.mu + shown_number)/self.t
        self.sigma = ((self.t-1)*self.sigma + (shown_number-self.mu)**2)/self.t
        if self.t > 1:
            z = (shown_number-self.mu)/self.sigma**.5
        else:
            z = shown_number 
        return z  

    def decision(self, shown_number):
        self.shown_number = self.z_scale(shown_number)
        self.all_shown_numbers.append(self.shown_number)
        self.prediction = self.model(self.shown_number.view(1,1))
        decision = bool(self.prediction.item() > 0.5)
        return decision

    def update_rule(self, actual):
        # eta = self.lr/(1+np.log(self.t))
        # for g in self.sgd.param_groups:
        #     g['lr'] = eta
        self.all_actuals.append(actual)
        predictions = self.model(torch.Tensor(self.all_shown_numbers).view(-1,1))
        loss = self.loss(predictions, torch.Tensor(self.all_actuals).view(-1,1))
        self.sgd.zero_grad()
        loss.backward()
        self.sgd.step()
        self.t+=1        