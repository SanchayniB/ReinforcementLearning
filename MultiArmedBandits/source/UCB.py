import numpy as np
import os
import matplotlib.pyplot as plt
import colored

from numpy.random import normal as GaussianDistribution

class UCBBandit:
    def __init__(self,c, arm_vals , estimates , arm_vals_var = 1):
        # If arm values not provided generate random values between [0,1] of length 10
        if arm_vals == []:
            self.arm_values = np.random.normal(0,1,10) # actual mean value of actions # mean reward
        else:
            self.arm_values = arm_vals
        
        self.k = len(self.arm_values)
        self.K = np.zeros(self.k) # number of actions
        self.arm_values_var = arm_vals_var
        self.c = c

        if estimates == []:
            self.est_values = np.zeros(self.k) # estimated value of value of actions
        elif len(estimates)!= self.k:
            print(colored('Size of estimates should be  equal to size of arm values provided','red'))
        

    def get_reward(self,action):
        reward = GaussianDistribution(loc=self.arm_values[action], scale= self.arm_values_var, size=1)[0]
        return reward
    
    def choose_action(self,c,t):
        UCB_est = []
        for i in self.est_values:
            est = self.est_values[i]
            UCB_est.append(est + ( c * np.sqrt(np.log(t/self.K[i])) ))

        return np.argmax(UCB_est)

    
    def update_est(self,action,reward,alpha = None):
        self.K[action] += 1
        if alpha == None:
            alpha = 1./self.K[action]
        self.est_values[action] += alpha * (reward - self.est_values[action]) 
        # Updating the estimate value with the mean value of rewards
        # keeps running average of rewards

    def is_optimal(self,action):
        if action == np.argmax(self.arm_values):
            return True
        else:
            return False

    def update_arm_vals(self,pull):
        # Adding non stationarity
        # changing the value of arm values with time
        random_walk = GaussianDistribution(loc=0, scale=0.01, size=self.k)
        self.arm_values += random_walk

        # for i in range(len(self.arm_values)):
        #     self.arm_values[i] += np.random.normal(0,0.1)
