#!/usr/bin/env python3
import os
import itertools
import json

import matplotlib
from collections import defaultdict

import numpy as np
import pymc3 as pm
import theano
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
from scipy.stats import entropy as H
import scipy as sp
import scipy.stats as st


class Agent():

    def __init__(self, config):
        self.n_iteration_for_parameter_evaluation= config["n_iteration_for_parameter_evaluation"]
        self.name = config["name"]
        self.k = config["k"]  # k sample
        self.iterations = config["iterations"]  # number of iterations per sample
        self.N = config["N"]  # normalization denominator
        self.n_action_state = config["n_action_state"]  # dimension of action and state space
        self.time_cost_sample = config["time_cost_sample"]  # step cost parameter
        self.beta_values = config["beta"]  # param for softmax
        self.a = config["a"]  # param for beta distribution
        self.b = config["b"]  # param for beta distribution

        self.data_store = self.__class__.make_nested_dict()
        self.dir_setup()
        self.init_dict(self.data_store)
        self.path_to_plot = os.path.join('./plots/', self.name)
        self.path_to_output = os.path.join('./outputs/', self.name)

        #######Utility Matrices#######
        self.squared_loss = self.utility_matrix_squared(self.n_action_state)  # list of matrices
        self.linear_loss = self.utility_matrix_linear(self.n_action_state)
        self.paper_utility = self.utility_matrix_mode(self.n_action_state)
        self.sparse_matrix = [sp.sparse.random(state, state, density=0.6).A for state in self.n_action_state]


    def dir_setup(self):
        if not os.path.exists('./plots'):
            os.mkdir('./plots')
        if not os.path.exists('./plots/'+self.name):
            os.mkdir('./plots/'+self.name)
        if not os.path.exists('./outputs'):
            os.mkdir("./outputs")
        if not os.path.exists('./outputs/'+self.name):
            os.mkdir("./outputs/"+self.name)


    @staticmethod
    def make_nested_dict():
        return defaultdict(__class__.make_nested_dict)


    def init_dict(self, nested_dict):
        loss = ["square", "onehot", "linear", "sparse"]
        y = ["sample_size", "loss"]
        data = ["mean", "low", "high"]
        for level_1 in loss:
            for level_2 in y:
                for level_3 in data:
                    nested_dict[level_1][level_2][level_3] = []
        self.data_store = nested_dict


    def plot(self, y, n_as):
        name = 'plot_for_{}_action_{}.png'.format(str(n_as), y)
        path = os.path.join(self.path_to_plot, name)

        if y == "sample_size":
            title = "Optimal Sample Size Comparison on Different Utility Function"
            ylabel = "Optimal Sample Size"
        elif y == "loss":
            title = "Loss Comparison on Different Utility Function"
            ylabel = "Loss"
        else:
            raise ImportError("plotting only support \"sample_size\" and \"loss\"")

        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.set_title(title)
        axe.set_xlabel("Beta ")
        axe.set_ylabel(ylabel)
        x_axis = self.beta_values
        plt.plot(x_axis, self.data_store["square"][y]["mean"], 'r')
        plt.fill_between(x_axis, self.data_store["square"][y]["low"], self.data_store["square"][y]["high"],
                         color='r', alpha=0.2)

        plt.plot(x_axis, self.data_store["onehot"][y]["mean"], 'g')
        plt.fill_between(x_axis, self.data_store["onehot"][y]["low"], self.data_store["onehot"][y]["high"],
                         color='green', alpha=0.2)

        plt.plot(x_axis, self.data_store["linear"][y]["mean"], 'y')
        plt.fill_between(x_axis, self.data_store["linear"][y]["low"], self.data_store["linear"][y]["high"],
                         color='yellow', alpha=0.2)

        plt.plot(x_axis, self.data_store["sparse"][y]["mean"], 'b')
        plt.fill_between(x_axis, self.data_store["sparse"][y]["low"], self.data_store["sparse"][y]["high"],
                         color='blue', alpha=0.2)

        plt.gca().legend(('Squared Loss', 'One Hot Loss', 'Linear Loss', 'Sparse Reward Matrix'))
        plt.savefig(path)
        plt.clf()
        plt.close('all')

    def run(self):
        """
        run agent with Maximum Likelihood Policy
        :return:
        """
        for i, n_as in enumerate(self.n_action_state):
            ###Sampling from a betabinomial distrubtion with given parameters(n,alpha,beta)
            states = list(range(n_as))
            p_true_betabino = self.sample_Beta_Binomial(n_as - 1, self.N)
            true_p = self.normalization(p_true_betabino, n_as)
            print("The generative distribution is is:\n", true_p)

            ###Defining a Transition matrix where an action is defined as a map
            # form a state into another state
            transitions = np.random.choice(states, 100000, p=true_p)
            t_m = self.transition_matrix(transitions)

            for beta in self.beta_values:
                self.actions("square", n_as, true_p, self.squared_loss[i], t_m, beta)
                self.actions("onehot", n_as, true_p, self.paper_utility[i], t_m, beta)
                self.actions("linear", n_as, true_p, self.linear_loss[i], t_m, beta)
                self.actions("sparse", n_as, true_p, self.sparse_matrix[i], t_m, beta)

            # save the data locally
            name = 'data_ml_' + str(n_as) + '.json'
            path = os.path.join(self.path_to_output, name)
            with open(path, 'w') as f:
                json.dump(self.data_store, f)
            # plot
            self.plot('sample_size', n_as)
            self.plot('loss', n_as)
            # init data_store
            self.init_dict(self.data_store)

    def create_model(self,observations, m, n_as):

        if self.name == "max_likelihood":
            states = list(range(n_as))
            # belief about the probability of an action given a state formed
            # by proprtion of samples
            counts = [observations.tolist().count(x) for x in set(states)]
            total_counts = sum(counts)
            agent_belief = [x / total_counts for x in counts]
            return agent_belief

        elif self.name == "bayesian" or "info_bayesian":
            with pm.Model() as model:
                sparsity = 3  # not zero
                alpha = np.full(n_as, 1)  # input for dirichlet

                # Weakly informative priors for unknown model parameters
                theta = pm.Dirichlet('theta', alpha / sparsity)

                # Likelihood (sampling distribution) of observations
                likelihood = pm.Multinomial('likelihood', m, theta, observed=observations)

                # a starting point can be given passing a dictionary where each of the keys containes a probaility cell-
                # This distribution can be placed certain standard deviations away from the mean of the underling distribution
                # for experimenting with distance
                # Uncomment to use Nuts
                #trace = pm.sample(1000,tune=500, cores=4,target_accept=0.95)
                trace= pm.sample(1000,step=pm.Metropolis(), tune=500)
            basic = pm.summary(trace).round(2)
            print(basic)

            # extract the inferred probability distribution of belief from the paramater space
            theta_infe = trace['theta'].mean(0).flatten()

            return theta_infe


    def sequential(self, orizon, optimal_agent, agent_belief, action_s,
                         states, utility_matrix, utilities_action,
                         t_m, o, utilities_optimal, p_true, beta):
        # draw a state from the transition matrix given the action take
        c_state = np.random.choice(states, 1, p=t_m[action_s[0]])
        o_state = np.random.choice(states, 1, p=t_m[optimal_agent])

        if o <= orizon:
            o += 1
            # next state is obtain through from the current state given by the action
            # (for example s1->s2)
            # via a transiton matrix probability
            s_next = np.random.choice(states, 1, p=t_m[action_s[0]])
            s_t = np.random.choice(states, 1, p=t_m[optimal_agent])

            # select next action based from the belief generated from the observation
            softmax_p = self.softmax(np.average(utility_matrix, axis=1, weights=agent_belief), agent_belief, beta)

            next_action_bounded = np.random.choice(states, 1, p=softmax_p)
            next_action_optimal = np.argmax(np.average(utility_matrix, axis=1, weights=t_m[s_t[0]]))

            utilities_action.append(
                utility_matrix[action_s[0]][c_state] + (0.6 ** o) * (utility_matrix[next_action_bounded[0]][s_next[0]]))
            utilities_optimal.append(
                utility_matrix[optimal_agent][o_state] + (0.6 ** o) * (utility_matrix[next_action_optimal][s_t[0]]))
            # call the function for o times, with a new state and new action#list to collect future expected utilities
            self.sequential(orizon, next_action_optimal, agent_belief, next_action_bounded,
                         states, utility_matrix, utilities_action,
                         t_m, o, utilities_optimal, p_true, beta)

        else:
            return utilities_action, utilities_optimal


    def actions(self, utility_type, n_as, p_true, utility_matrix, t_m, beta):

        # assumption same number of actions that for states

        states = list(range(n_as))
        list_optimal_stopping = []
        error_list = []
        orizon = 6
        # kl_costs= []
        for i in range(self.n_iteration_for_parameter_evaluation):
             # list where the elements are optimal stopping point of utility curves under the given sample cost
            action_list = []

            utilities_action = []
            utilities_optimal = []
            utilities_list = []
            low_confidence_interval = []
            high_confidence_interval = []
            utilities_list_optimal = []
            costs = []
            error_with_cost_list = []

            for m in range(1,self.k+1):
                for _ in range(self.iterations):
                    # draw k samples from p_true j times the bottom procedure is repeated
                    samples = np.random.choice(states, m, p=p_true)
                    agent_belief = self.create_model(samples,m,n_as)

                    for _ in range(self.iterations):
                        eu = np.average(utility_matrix, axis=1, weights=agent_belief)
                        softmax_p = self.softmax(eu, beta,agent_belief)
                        action_s = np.random.choice(states, 1, p=softmax_p)

                        # optimal agent action
                        optimal_agent = np.argmax(np.average(utility_matrix, axis=1, weights=p_true))
                        # adding selected choices to a list
                        action_list.append(action_s)
                        # orizon
                        o = 1

                        self.sequential(orizon, optimal_agent, agent_belief, action_s,
                                         states, utility_matrix, utilities_action,
                                         t_m, o, utilities_optimal, p_true, beta)

                    # average utility given k observations for bounded approximate agent
                    mean_utility_sample_agent = np.mean(utilities_action)
                    # average utility given k observations for optimal agent
                    mean_utility_optimal = np.mean(utilities_optimal)

                costs.append(m * self.time_cost_sample)
                # collect average utilities for k observation
                utilities_list.append(mean_utility_sample_agent)
                utilities_list_optimal.append(mean_utility_optimal)

                #use the 1000 utilities obtained by picking an action formed with k
                #observation, to be bootstraped and the 5th up to 95th percentile are extracted
                boot = self.bootstrap(list(itertools.chain(*utilities_action)), n=1000)
                low_conf_curr, high_conf_curr = boot(.95)
                low_confidence_interval.append(low_conf_curr)
                high_confidence_interval.append(high_conf_curr)

            # calculate element wise difference for the two agents utilities list
            utilities_difference = [x - y for x, y in zip(utilities_list_optimal, utilities_list)]
            # normalize the difference
            norm_difference = [x / np.max(utilities_difference) for x in utilities_difference]
            # total utilities differences given linear costs
            total_utilities_difference = ([x + i for x, i in zip(norm_difference, costs)])
            #add them to a list
            error_list.append(norm_difference)
            error_with_cost_list.append(total_utilities_difference)

            # optimal stopping point on the curve
            optimal_number_x = np.argmin(total_utilities_difference)
            list_optimal_stopping.append(optimal_number_x)


        # save data for optimal stopping
        optimal_stopping_mean = np.mean(list_optimal_stopping)
        boot = self.bootstrap(list_optimal_stopping, n=1000)
        low_sample_95, high_sample_95 = boot(.95)

        self.data_store[utility_type]["sample_size"]["mean"].append(optimal_stopping_mean)
        self.data_store[utility_type]["sample_size"]["low"].append(low_sample_95)
        self.data_store[utility_type]["sample_size"]["high"].append(high_sample_95)

        # save data for loss
        error_mean= np.mean(error_with_cost_list, axis=1)  # TODO: check if the input is correct
        boot = self.bootstrap(error_with_cost_list[0], n=1000)  # TODO: check if the input is correct
        low_loss_95, high_loss_95 = list(boot(.95))

        self.data_store[utility_type]["loss"]["mean"].append(list(error_mean)[0])  # TODO: check if the append target is correct
        self.data_store[utility_type]["loss"]["low"].append(low_loss_95)
        self.data_store[utility_type]["loss"]["high"].append(high_loss_95)


    def bootstrap(self,data, n=1000, func=np.mean):
        """
        Generate `n` bootstrap samples, evaluating `func`
        at each resampling. `bootstrap` returns a function,
        which can be called to obtain confidence intervals
        of interest.
        """
        simulations = list()
        sample_size = len(data)
        for c in range(n):
            itersample = np.random.choice(data, size=sample_size, replace=True)
            simulations.append(func(itersample))
        simulations.sort()

        def ci(p):
            """
            Return 2-sided symmetric confidence interval specified
            by p.
            """
            u_pval = (1 + p) / 2.
            l_pval = (1 - u_pval)
            l_indx = int(np.floor(n * l_pval))
            u_indx = int(np.floor(n * u_pval))
            return (simulations[l_indx], simulations[u_indx])

        return (ci)


    def transition_matrix(self, transitions):
        """
        Creates a conditional distribution of moving to any other state given the current one
        via samples obtained from the underling distribution
        """
        n = 1+max(transitions)  # number of states and actios
        M = [[0 for x in range(n)] for y in range(n)]

        # each entry in the matrix represent a probability distribution of going
        # from a given state to another
        for (i, j) in zip(transitions, transitions[1:]):
            M[i][j] += 1

        # now convert to probabilities:
        for row in M:
            s = sum(row)
            row[:] = [f / s for f in row if s > 0]
        return M

    def sample_Beta_Binomial(self, n, size=None):
        """
        Generate n sample from a beta-binomial with parameters
        a,b
        """
        p = np.random.beta(self.a, self.b, size=size)
        r = np.random.binomial(n, p)
        return r

    def normalization(self, p_betabino, n):
        """
        Function that normalize the sample obtained from the beta-binomail to form
        a sound probability distribution
        """
        p = np.zeros(n, dtype=np.float64)  # histogram
        for v in p_betabino:  # fill it
            p[v] += 1.0

        p /= np.float64(self.N)  # normalization

        return p

    def softmax(self, eu, beta, agent_belief):
        """Compute softmax values for each sets of scores in x."""
        if self.name in ["max_likelihood","bayesian"]:
            e_x = np.exp(eu * beta)
            return e_x / e_x.sum(axis=0)  # only difference

        elif self.name == "info_bayesian":
            e_x = agent_belief * np.exp(eu * beta)
            return e_x / e_x.sum(axis=0)  # only difference

    def utility_matrix_squared(self, n_action_state):
        # Creates a Squared Loss Utility Matrix
        return [np.fromfunction(lambda i, j: (1 - abs(i - j) ** 2) / np.max(abs(i - j) ** 2),
                               (state, state), dtype=float) for state in n_action_state]

    def utility_matrix_mode(self, n_action_state):
        # Creates a Utility matrix with 0 penalty for every entry except for the diagonal

        return [np.eye(state, state, dtype=float) for state in n_action_state]

    def utility_matrix_linear(self, n_action_state):
        # Creates a Linear Loss Utility Matrix

        return [np.fromfunction(lambda i, j: 1 - abs(i - j) / np.max(abs(i - j)), (state, state),
                               dtype=float) for state in n_action_state]
