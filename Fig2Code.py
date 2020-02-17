# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Description of the Code:
# This code generates the Figure 2 of the paper titled "Estimating Power-law Degree Distributions via Friendship
# Paradox Sampling". It compares the estimate proposed in the paper (friendship paradox based maximum likelihood
# estimate) with the widely used vanilla method (maximum likelihood estimation with uniform sampling) in terms of:
#   1: Variance
#   2: Mean-squared error (sum of squared bias and variance)
#   3: Cramer-Rao Lower Bound (best achievable variance for any unbiased estimate),
# for different values of the power-law parameter alpha.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import bernoulli
from scipy.stats import pareto
import networkx as nx
import matplotlib.pyplot as plt
import random
import scipy.io
import collections
import pickle


random.seed(123)  # Initializing the random number generators

No_of_Nodes = 5000  # Number of nodes in each of the power-law graphs
n = 100  # Number of samples drawn from the degree distribution to compute the MLEs
No_iterations = 3000  # Number of independent iterations in the Monte-Carlo simulation

# Lists for storing the bias and variance for each value of alpha
Bias2_Var_MSE_X_vec = []
Bias2_Var_MSE_Y_vec = []

for alpha in np.linspace(2.1, 3.6, num=30):
    i = 1  # i is the current iteration number

    # Lists for storing the estimates produced in each iteration
    Estimate_X_vec_alpha = []
    Estimate_Y_vec_alpha = []

    # The following while loop generates vanilla and friendship paradox based MLEs in each iteration
    while i < No_iterations:

        # Generating degree sequence by sampling from continuous power-distribution and,
        #  then rounding to the nearest integer.
        deg_sequence = pareto.rvs(alpha - 1, loc=0, scale=1, size=No_of_Nodes)
        deg_sequence = np.round(deg_sequence, decimals=0)

        # Ensuring the sum of the sampled degrees even
        # (to be valid sequence of degrees the sum should be even since it is equal to twice the number of edges)
        if (np.sum(deg_sequence) % 2) != 0:
            deg_sequence[0] = deg_sequence[0] + 1
        deg_sequence = deg_sequence.astype(int)

        # Generating a network G with the degree sequence generated previously by using the configuration model
        G = nx.configuration_model(list(deg_sequence))

        # Sampling n Random Nodes (X_1, X_2,....X_n) independently from the set of nodes of the network G
        X_nodes = list(np.random.choice(G.nodes, size=n))

        # Sampling n Random Friends (Y_1, Y_2,....Y_n) independently
        # by listing all friends and then sampling the resulting list uniformly
        Y_nodes = list(np.random.choice(np.array(list(G.edges(data=False))).flatten(), size=n))

        # The degrees (d(X_1), d(X_2),....d(X_n)) of the n random nodes
        #  and the degrees (d(Y_1), d(Y_2),....d(Y_n)) of n random friends are stored as lists X and Y respectively
        X = [G.degree(x) for x in X_nodes]
        Y = [G.degree(y) for y in Y_nodes]

        # Computing the vanilla MLE and storing it
        alpha_MLE_X = (n/np.sum(np.log(X))) + 1
        Estimate_X_vec_alpha.append(alpha_MLE_X)

        # Computing the friendship paradox based MLE and storing it
        alpha_MLE_Y = (n/np.sum(np.log(Y))) + 2
        Estimate_Y_vec_alpha.append(alpha_MLE_Y)
        i = i + 1

        print(alpha)
        print(i)

    # Computing the squared bias, variance and MSE of the vanilla and friendship paradox based MLEs
    # that were generated from the while loop  (for the considered value of alpha)
    Bias2_Var_MSE_X_vec.append((alpha,
                                (np.mean(Estimate_X_vec_alpha) - alpha)**2,
                                np.mean((Estimate_X_vec_alpha-np.mean(Estimate_X_vec_alpha))**2),
                                np.mean((Estimate_X_vec_alpha-alpha)**2)))
    Bias2_Var_MSE_Y_vec.append((alpha,
                                (np.mean(Estimate_Y_vec_alpha) - alpha) ** 2,
                                np.mean((Estimate_Y_vec_alpha - np.mean(Estimate_Y_vec_alpha)) ** 2),
                                np.mean((Estimate_Y_vec_alpha - alpha) ** 2)))

# Storing the squared bias, variance and MSE in lists for considered values of alpha
Alpha_vec, Bias_X, Var_X, MSE_X = zip(*Bias2_Var_MSE_X_vec)
Alpha_vec, Bias_Y, Var_Y, MSE_Y = zip(*Bias2_Var_MSE_Y_vec)


# Setting up the parameters of the plot
plt.rc('text', usetex=True)
MarkerSize = 2.5
LineWidth = 1.0
alpha = 0.7
plt.figure(figsize=(3.43, 1.7))

# Plotting the values of variance and MSE values of the vanilla MLE
# against the considered values of alpha
plot_Var_X = plt.plot(Alpha_vec, Var_X,
                      linestyle='--',
                      dashes=(1, 1),
                      marker='^',
                      alpha=alpha+0.2,
                      markerfacecolor='none',
                      markeredgecolor='r',
                      c='r',
                      linewidth=LineWidth - 0.1,
                      markersize=MarkerSize - 1,
                      label = r'$\mathrm{Var}\{\hat{\alpha}_{\mathrm{vanilla}}\}$')

plot_MSE_X = plt.plot(Alpha_vec, MSE_X,
                      linestyle='--',
                      dashes=(1, 1),
                      marker='<',
                      alpha=alpha,
                      markerfacecolor='none',
                      markeredgecolor='g',
                      c='g',
                      linewidth=LineWidth - 0.2,
                      markersize=MarkerSize - 1,
                      label=r'$\mathrm{MSE}\{\hat{\alpha}_{\mathrm{vanilla}}\}$')

# Computing and plotting the theoretical Cramer-Rao Lower Bound values of the vanilla MLE
# against the considered values of alpha
plot_CRLB_X = plt.plot(Alpha_vec, (np.array(Alpha_vec)-1)**2/n,
                       linestyle='--',
                       dashes=(1, 1),
                       marker='v',
                       alpha=alpha-0.2,
                       markerfacecolor='none',
                       markeredgecolor='b',
                       c='b',
                       linewidth=LineWidth-0.4,
                       markersize=MarkerSize - 0.5,
                       label=r'${\mathrm{CRLB}}_{\mathrm{vanilla}}(\alpha)$')


# Plotting the values of variance and MSE values of the friendship paradox based MLE
# against the considered values of alpha
plot_Var_Y = plt.plot(Alpha_vec, Var_Y,
                      linestyle=':',
                      dashes=(1, 1, 3, 1),
                      marker='1',
                      alpha=alpha+0.1,
                      markerfacecolor='none',
                      markeredgecolor='r',
                      c='r',
                      linewidth=LineWidth + 0.2,
                      markersize=MarkerSize + 1.5 + 0.5,
                      markeredgewidth=1,
                      label=r'$\mathrm{Var}\{\hat{\alpha}_{\mathrm{FP}}\}$')

plot_MSE_Y = plt.plot(Alpha_vec, MSE_Y,
                      linestyle=':',
                      dashes=(3, 1, 1, 1),
                      marker='2',
                      alpha=alpha,
                      markerfacecolor='none',
                      markeredgecolor='g',
                      c='g',
                      linewidth=LineWidth - 0.4,
                      markersize=MarkerSize + 1.5 - 0.75,
                      markeredgewidth=1,
                      label=r'$\mathrm{MSE}\{\hat{\alpha}_{\mathrm{FP}}\}$')

# Computing and plotting the theoretical Cramer-Rao Lower Bound values of the friendship paradox based MLE
# against the considered values of alpha
plot_CRLB_Y = plt.plot(Alpha_vec, (np.array(Alpha_vec)-2)**2/n,
                       linestyle=':',
                       dashes=(3, 1, 1, 1),
                       marker='3',
                       alpha=alpha,
                       markerfacecolor='none',
                       markeredgecolor='b',
                       c='b',
                       linewidth=LineWidth - 0.5,
                       markersize=MarkerSize + 1.5 - 1.25,
                       markeredgewidth=1,
                       label=r'${\mathrm{CRLB}}_{\mathrm{FP}}(\alpha)$')

# Setting up the parameters of the plot and saving
plt.xlabel(r'True power-law coefficient $\alpha$', fontsize=7)
plt.ylabel(r'Variance, MSE and CRLB', fontsize=7)
plt.xlim((2.1, 3.65))
plt.ylim((-0.02, 1))
plt.xticks(np.arange(2.1, 3.71, step=0.2), fontsize=7)
plt.yticks(np.arange(0, 1.01, step=0.2), fontsize=7)
plt.legend(ncol=1, loc='upper left', fontsize=7)
plt.savefig('Var_MSE_CRLB.pdf', bbox_inches='tight')



