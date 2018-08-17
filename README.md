# Reversible-Jump-Markov-Chain-Monte-Carlo(RJMCMC) with Simulated Annealing

## Introduction

This repo is about using Reversible Jump MCMC(RJMCMC) and Simulated Annealing algorithm(SA) to train Radial Basis Function(RBF) network, so that we can obtain a model with uncertain parameter dimensions. Besides, different model choosing approaches including AIC, BIC, MDL, MAP, HQC, and their performance are compared.

## Code

1. Metropolis-Hastings&Gibbs 
Use Metropolis Hastings algorithm and Gibbs Sampling to estimate parameters in 2D Gaussian distribution.

2. RJMCMC
A simple example of Reversible Jump MCMC.

3. RJMCMC+SA
Use RJMCMC and SA to train RBF network.

4. Model Choosing
A comparison between different model choosing criteria, including AIC, BIC, MDL, MAP, HQC.
