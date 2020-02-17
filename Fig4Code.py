# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Description of the Code:
# This code generates the Figure 4 of the paper titled "Estimating Power-law Degree Distributions via Friendship
# Paradox Sampling". It compares the estimate proposed in the paper (friendship paradox based maximum likelihood
# estimate) with another method that takes the discrete nature of the power-law degree distribution into account
#  in terms of the mean-squared error.
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


random.seed(123)  # Initializing the random number generators

No_of_Nodes = 5000  # Number of nodes in each of the power-law graphs
n = 100  # Number of samples drawn from the degree distribution to compute the MLEs
No_iterations = 3000  # Number of independent iterations in the Monte-Carlo simulation

# Lists for storing the bias and variance for each value of alpha
Bias2_Var_MSE_X_vec = []
Bias2_Var_MSE_X_discrete_vec = []
Bias2_Var_MSE_Y_vec = []

for alpha in np.linspace(2.1, 3.6, num=30):
    i = 1  # i is the current iteration number

    # Lists for storing the estimates produced in each iteration
    Estimate_X_discrete_vec_alpha = []
    Estimate_Y_vec_alpha = []

    # The following while loop generates vanilla, discrete-specific and friendship paradox based MLEs in each iteration
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

        # Computing the discrete-specific MLE and storing it
        alpha_MLE_X_discrete = (n/np.sum(np.log(np.array(X)/(1-0.5)))) + 1
        Estimate_X_discrete_vec_alpha.append(alpha_MLE_X_discrete)

        # Computing the friendship paradox based MLE and storing it
        alpha_MLE_Y = (n/np.sum(np.log(Y))) + 2
        Estimate_Y_vec_alpha.append(alpha_MLE_Y)
        i = i + 1

    # Computing the squared bias, variance and MSE of the discrete-specific and friendship paradox based MLEs
    # that were generated from the while loop  (for the considered value of alpha)
    Bias2_Var_MSE_X_discrete_vec.append((alpha,
                                        (np.mean(Estimate_X_discrete_vec_alpha) - alpha) ** 2,
                                        np.mean((Estimate_X_discrete_vec_alpha - np.mean(Estimate_X_discrete_vec_alpha)) ** 2),
                                        np.mean((Estimate_X_discrete_vec_alpha - alpha) ** 2)))

    Bias2_Var_MSE_Y_vec.append((alpha,
                                (np.mean(Estimate_Y_vec_alpha) - alpha) ** 2,
                                np.mean((Estimate_Y_vec_alpha - np.mean(Estimate_Y_vec_alpha)) ** 2),
                                np.mean((Estimate_Y_vec_alpha - alpha) ** 2)))

# Storing the squared bias, variance and MSE in lists for considered values of alpha
Alpha_vec, Bias_X_discrete, Var_X_discrete, MSE_X_discrete = zip(*Bias2_Var_MSE_X_discrete_vec)
Alpha_vec, Bias_Y, Var_Y, MSE_Y = zip(*Bias2_Var_MSE_Y_vec)

# Setting up the parameters of the plot
plt.rc('text', usetex=True)
MarkerSize = 2.5
LineWidth = 0.75
alpha = 1
plt.figure(figsize=(3.43, 0.9))

# Plotting the values of variance, MSE and theoretical CRLB values of the
# discrete-specific and friendship paradox based MLEs against the considered values of alpha
plot_MSE_X_discrete = plt.plot(Alpha_vec, MSE_X_discrete,
                               linestyle=':',
                               dashes=(1, 1),
                               marker='o',
                               alpha=alpha,
                               markerfacecolor='none',
                               markeredgecolor='r',
                               c='r',
                               linewidth=LineWidth,
                               markersize=MarkerSize,
                               label=r'$\mathrm{MSE}\{\hat{\alpha}_{\mathrm{vanilla{\_}discrete}}\}$')

plot_MSE_Y = plt.plot(Alpha_vec, MSE_Y,
                      linestyle='-.',
                      dashes=(1, 1, 3, 1),
                      marker='2',
                      alpha=alpha,
                      markerfacecolor='none',
                      markeredgecolor='g',
                      c='g',
                      linewidth=LineWidth,
                      markersize=MarkerSize + 1.5,
                      markeredgewidth=1,
                      label=r'$\mathrm{MSE}\{\hat{\alpha}_{\mathrm{FP}}\}$')

# Setting up the parameters of the plot and saving
plt.xlabel(r'True power-law coefficient $\alpha$', fontsize=7)
plt.ylabel(r'MSE', fontsize=7)
plt.xlim((2.1, 3.65))
plt.ylim((-0.1, 3))
plt.xticks(np.arange(2.1, 3.71, step=0.2), fontsize=7)
plt.yticks(np.arange(0, 3.01, step=1), fontsize=7)
plt.yticks(fontsize=7)
plt.legend(ncol=1, loc='upper left', fontsize=7)
plt.savefig('MSE_discrete_vs_continuous.pdf', bbox_inches='tight')
