# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:39:15 2021

@author: Hans
"""
import numpy as np
from numpy.linalg import inv
from scipy.stats import unitary_group
import matplotlib.pyplot as plt
import time

def sequence_with_noise(m, Lmbda, seed=1):
    oper = np.eye(2)
    inver_op = np.eye(2)
    for i in range(m):
        u_map = unitary_group.rvs(2)
        # add "Lamda u_(i+1)" to "Lmbda u_(i)..."
        # add noise and random unitary each iteration
        oper = np.matmul(Lmbda, np.matmul(u_map, oper))
        # add "u_(i+1)^(-1)" to "u_(i)^(-1)..."
        inver_op = np.matmul(inv(u_map), inver_op)
    return inver_op, oper

seed = 1
np.random.seed(seed)
time_mark = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
ket_0 = np.array([1,0]).reshape(2,1)
ket_1 = np.array([0,1]).reshape(2,1)
rho = np.array([[1,0],[0,0]])
proj_O = np.kron(ket_0, ket_0.T)        # np.kron(ket_0, ket_0.T) - np.kron(ket_1, ket_1.T)
pd = 1.333
# 0.01 ~ 0.05 current best; 0.01 no better than 0.1; 0.0001 mess # 0.1 close, but not start from ~1
D = 2^1     # Eq.2.9 in RB_Clifford 1811.10040
# Lmbda = (1-pd)*np.eye(2) + (pd/D)*(np.kron(ket_0, ket_0.T) - np.kron(ket_1, ket_1.T))
Lmbda = (1-pd)*np.eye(2) + (pd/D)*(np.kron(ket_0, ket_1.T) + np.kron(ket_1, ket_0.T))
M = 40
Fm = np.zeros(M)
sample_size = int(1e2)
fm = np.zeros(sample_size)

for m in range(1, M+1):
    for i in range(sample_size):
        # inver_op = u_(m)^(-1) u_(m-1)^(-1)...
        # oper = Lamda u_(m) Lmbda u_(m-1)...
        inver_op, oper = sequence_with_noise(m, Lmbda, seed)
        # Sm here only consider the no-dagger part.
        # Sm = Lmbda inver_op Lmbda ... u_2 Lmbda u_1
        Sm = np.matmul(Lmbda, np.matmul(inver_op, oper))
        # Sm_rho = Sm rho Sm_dagger
        Sm_rho = np.matmul(Sm, np.matmul(rho, np.conj(Sm).T))
        fm[i] = np.trace(np.matmul(proj_O, Sm_rho).real)
        
    Fm[m-1] = np.average(fm)
    # if m % 10 == 0:
    #     np.savetxt("../data/Fm_" + str(M) + "_avg_" + str(sample_size) + "_seed_" + str(seed) + "_" + time_mark + ".txt", Fm)
        
plt.plot(range(1, M+1), Fm, 'bo')


    

        

        