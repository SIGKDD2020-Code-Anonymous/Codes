# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Description of the Code:
# This code generates the Figure 3 of the paper titled "Estimating Power-law Degree Distributions via Friendship
# Paradox Sampling". It compares the estimate proposed in the paper (friendship paradox based maximum likelihood
# estimate) with the widely used vanilla method (maximum likelihood estimation with uniform sampling) in terms of
# the mean-squared error under four different sample sizes (n = 100, 200, 300, 400).
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
No_iterations = 3000  # Number of independent iterations in the Monte-Carlo simulation

# colors and markers for different values of the sample size (j indicates which element from the list is being taken)
line_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
markers_X = ['^', '<', '>', 'v']
markers_Y = ['1', '2', '3', '4']
j = 0

plt.figure(figsize=(3.43, 2.4))  # Width, Height of the figure

# The following for loop considers four different values of the sample size
for n in np.arange(100, 500, 100):
    # Lists for storing the bias and variance for each value of alpha for a given value of sample size n
    Bias2_Var_MSE_X_vec = []
    Bias2_Var_MSE_Y_vec = []

    for alpha in np.linspace(2.1, 3.6, num=30):
        i = 1  # i is the current iteration number

        # Lists for storing the estimates produced in each iteration
        Estimate_X_vec_alpha = []
        Estimate_Y_vec_alpha = []

        # The following while loop generates vanilla and friendship paradox based MLEs in each iteration
        while i < No_iterations:
            # print('n = ' + str(n))  # Number of samples drawn from the degree distribution
            # print('alpha = ' + str(alpha))  # value of alpha
            # print('i = ' + str(i))  # Iteration number

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

    # Setting the parameters of the plots
    plt.rc('text', usetex=True)
    MarkerSize = 2.5
    MarkerEdgeWidth = 0.75
    LineWidth = 1.2
    alpha = 0.7

    # Plotting the values of MSE of the vanilla and friendship paradox based MLEs against the
    # considered values of alpha and sample size
    plot_MSE_X = plt.plot(Alpha_vec, MSE_X,
                          linestyle='--',
                          dashes=(6 - j, 1+j),
                          marker=markers_X[j],
                          alpha=alpha - (j * 0.05),
                          markerfacecolor='none',
                          markeredgecolor=line_colors[j],
                          c=line_colors[j],
                          linewidth=LineWidth - (j * 0.2),
                          markersize=MarkerSize,
                          label=r'${\mathrm{MSE}\{\hat{\alpha}_{\mathrm{vanilla}}\},\, n = }$' + str(n))

    plot_MSE_Y = plt.plot(Alpha_vec, MSE_Y,
                          linestyle='-.',
                          dashes=(6 - j, 1+j),
                          marker=markers_Y[j],
                          alpha=alpha - (j * 0.075),
                          markerfacecolor='none',
                          markeredgecolor=line_colors[j],
                          c=line_colors[j],
                          linewidth=LineWidth - (j * 0.2),
                          markersize=MarkerSize + 1,
                          markeredgewidth=MarkerEdgeWidth,
                          label=r'${\mathrm{MSE}\{\hat{\alpha}_{\mathrm{FP}}\},\, n = }$' + str(n))

    j = j+1

# Setting the parameters of the plots and saving
plt.xlabel(r'True power-law coefficient $\alpha$', fontsize=7)
plt.ylabel(r'MSE', fontsize=7)
plt.xlim((2.1, 3.65))
plt.ylim((-0.02, 1))
plt.xticks(np.arange(2.1, 3.71, step=0.2), fontsize=7)
plt.yticks(np.arange(0, 1.01, step=0.2), fontsize=7)
plt.legend(ncol=1, loc='upper left', fontsize=7)
plt.savefig('SampleSize_vs_MSE.pdf', bbox_inches='tight')
