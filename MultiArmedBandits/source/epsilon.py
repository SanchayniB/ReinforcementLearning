import numpy as np
import os
import matplotlib.pyplot as plt
import colored

from numpy.random import normal as GaussianDistribution
'reference git: https://github.com/brianfarris/RLtalk/blob/master/RLtalk.ipynb'

class Bandit:
    def __init__(self, arm_vals , estimates , arm_vals_var = 1):
        # If arm values not provided generate random values between [0,1] of length 10
        if arm_vals == []:
            self.arm_values = np.random.normal(0,1,10) # actual mean value of actions # mean reward
        else:
            self.arm_values = arm_vals
        
        self.k = len(self.arm_values)
        self.K = np.zeros(self.k) # number of actions
        self.arm_values_var = arm_vals_var

        if estimates == []:
            self.est_values = np.zeros(self.k) # estimated value of value of actions
        elif len(estimates)!= self.k:
            print(colored('Size of estimates should be  equal to size of arm values provided','red'))
        

    def get_reward(self,action):
        reward = GaussianDistribution(loc=self.arm_values[action], scale= self.arm_values_var, size=1)[0]
        return reward
    
    def choose_eps_greedy(self,epsilon):
        rand_num = np.random.random()
        if epsilon>rand_num:
            return np.random.randint(self.k) # Exploring # selecting random k value
        else:
            return np.argmax(self.est_values) # Exploiting # select action k with highest reward


    def choose_UCBaction(self,c,t):
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

def run_experiment(bandit, Npulls=1000, epsilon=0.01, alpha=None):
    history = []
    correct_pct = []

    for i in range(Npulls):
        action = bandit.choose_eps_greedy(epsilon) 

        # % selection of correct action
        correct_pct.append(float(bandit.is_optimal(action)))
        R = bandit.get_reward(action)
        bandit.update_est(action, R, alpha)
        history.append(R)
    return {'Qa': np.array(history) , 'correctness': correct_pct}

def run_experiment_ns(bandit, Npulls=1000, epsilon=0.01, alpha= []):
    history = []
    correct_pct = []
    ct = 0
   
    # print(optimal_action)

    for i in range(Npulls):
        action = bandit.choose_eps_greedy(epsilon) 
        # % selection of correct action
        correct_pct.append(float(bandit.is_optimal(action)))

        R = bandit.get_reward(action)
        bandit.update_est(action, R, alpha) 
        bandit.update_arm_vals(i)
        history.append(R)

    return {'Qa' : np.array(history), 'correctness' : np.array(correct_pct)}

def run_experiment_ucb(bandit, Npulls=1000, epsilon=0.01, alpha= [], c = 2):
    history = []
    correct_pct = []
    ct = 0
   
    # print(optimal_action)

    for i in range(Npulls):
        action = bandit.choose_UCBaction(epsilon,c,i+1) 
        # % selection of correct action
        correct_pct.append(float(bandit.is_optimal(action)))

        R = bandit.get_reward(action)
        bandit.update_est(action, R, alpha) 
        bandit.update_arm_vals(i)
        history.append(R)

    return {'Qa' : np.array(history), 'correctness' : np.array(correct_pct)}

def run_simulation(Nexp=20, Npulls=1000, epsilons=[0.05], 
                    bandit_args={'arm_vals' : [], 'alpha' : None , 'estimates' : [] , 'c': 2},
                    experiment=run_experiment):
    
    avg_reward = {}
    avg_correctness = {}

    for epsilon in epsilons:
        avg_reward[str(epsilon)]  = np.zeros(Npulls)
        avg_correctness[str(epsilon)] = np.zeros(Npulls)

        for i in range(Nexp):
            bandit = Bandit(bandit_args['arm_vals'], bandit_args['estimates'])
            if experiment == run_experiment_ucb:
                result = experiment(bandit, Npulls, epsilon, bandit_args['alpha'], c = bandit_args['c'])
            else:
                result = experiment(bandit, Npulls, epsilon, bandit_args['alpha'])
            
            avg_reward[str(epsilon)]  += result['Qa']
            avg_correctness[str(epsilon)]  += result['correctness']
        
        avg_reward[str(epsilon)] /= np.float(Nexp)
        avg_correctness[str(epsilon)] /= np.float(Nexp)
    
    os.system('say "Done compiling"')

    return (avg_reward,avg_correctness)



def plotting(dictn, xlabel = 'Npulls', ylabel = 'Reward', ylim = (None,None)):

    for eps in dictn:
        plt.plot(dictn[eps],label="eps = {}".format(eps))
    
    if ylim[0] != None or ylim[1] != None:    
        plt.ylim(ylim[0],ylim[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()