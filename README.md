# Two-step-python

A python implement of the Two-step algorithm

More details about the Two-step algorithm can be found in the original [repo](https://github.com/cabal-cmu/Two-Step).

A dataset X of non-Gaussian variables is required as input, together with a positive value for the penalization parameter, lambda.

Two-Step outputs the causal coefficients matrix B, from X = BX + E.

In B, the causal direction goes from column to row, such that a matrix entry Bij, implies Xj --> Xi

A toy example is given in 'two-step/two_step_CD.py'

