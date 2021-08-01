#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:14:26 2021

@author: sxyang
"""

import numpy as np
from numpy.linalg import inv
from scipy.stats import unitary_group
from scipy import linalg
import matplotlib.pyplot as plt
import time
# from fit_RB import compare_Fm

#----------------------------------------------------
# Reproducing Fig.4 unitary non-Markovian noise, 2 qubit, environment is the first qubit.
#----------------------------------------------------

def sequence_with_unitary_noise(m, rho, noise_u):
    # apply unitary noise in the sequence
    tmp_rho = rho
    inver_op = np.eye(2)
    for i in range(m):
        u_map = unitary_group.rvs(2)
        tmp_rho = noise_u @ np.kron(I, u_map) @ tmp_rho @ np.conj(np.kron(I, u_map)).T @ np.conj(noise_u).T
        
        inver_op = inver_op @ inv(u_map)
    return tmp_rho, inver_op

def Markovianised_map(rho_e, rho_s, noise_u):
    # Input the state of environment, system and non-MArkovian noise, outputs the corresponding Markovinaised noise map in Eq.16.
    # Calculate Eq.16 to obtain A, B (Eq.14)
    M_noise_u = noise_u * np.kron(rho_e, rho_s) * np.conj(noise_u).T
    M_noise_u = np.trace(M_noise_u.reshape(2,2,2,2), axis1=0, axis2=2)
    return M_noise_u
    
def trace_Markovianised(noise_u):
    I = np.eye(2, dtype=complex)
    ket_0 = np.array([1,0], dtype=complex).reshape(2,1)
    ket_1 = np.array([0,1], dtype=complex).reshape(2,1)
    k0 = np.kron(ket_0.T, I) @ noise_u @ np.kron(ket_0, I)
    k1 = np.kron(ket_1.T, I) @ noise_u @ np.kron(ket_1, I)
    tr = abs(np.trace(k0)) + abs(np.trace(k1))
    return tr

if __name__ == "__main__":
    
    # Set parameters for unitary
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    Y = np.array([[0, -1j],[1j, 0]], dtype=complex)
    Z = np.array([[1, 0],[0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    ds = 2   # dim(rho_s)
    de = 2
    J = 1.7
    hx = 1.47
    hy = -1.05
    delta = 0.03

    H = J * np.kron(X,X) + hx * (np.kron(X, I) + np.kron(I, X)) + hy * (np.kron(Y, I) + np.kron(I, Y))
    
    # Non-Markovian noise, assume unitary.
    noise_u = linalg.expm(-1j*delta*H)

    
    seed = 1
    np.random.seed(seed)
    time_mark = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    
    ket_0 = np.array([1,0], dtype=complex).reshape(2,1)
    ket_1 = np.array([0,1], dtype=complex).reshape(2,1)
    
    rho = np.kron(np.kron(ket_0, ket_0.T), np.kron(ket_0, ket_0.T))
    proj_O = np.kron(ket_0, ket_0.T)        # np.kron(ket_0, ket_0.T) - np.kron(ket_1, ket_1.T)
    rho_s = np.trace(rho.reshape(2,2,2,2), axis1=0, axis2=2)
    rho_e = np.trace(rho.reshape(2,2,2,2), axis1=1, axis2=3)
    #---------------------------------------
    # Set Markovianised F_m
    p = ((trace_Markovianised(noise_u) - 1)/(ds**2 - 1))
    A = np.trace(proj_O * Markovianised_map(rho_e, rho_s - I/ds, noise_u)).real
    B = np.trace(proj_O * Markovianised_map(rho_e, I/ds, noise_u)).real
    
    
    M = 100
    sample_size = int(50)
    fm = np.zeros(sample_size)
    Fm = np.zeros(M)
    print('Setup state, projector, and some parameters.')
    
    for m in range(1, M+1):
        for i in range(sample_size):
            tmp_rho, inver_op = sequence_with_unitary_noise(m, rho, noise_u)
            tmp_rho = np.kron(I, inver_op) @ tmp_rho @ np.conj(np.kron(I, inver_op)).T
            final_state = noise_u @ tmp_rho @ noise_u.T
            f_sys_state = np.trace(final_state.reshape(2,2,2,2), axis1=0, axis2=2)
            fm[i] = np.trace(proj_O @ f_sys_state).real
            
        Fm[m-1] = np.average(fm)
        print("m = ", str(m), " finished.")
        # if m % 20 == 0:
        #     np.savetxt("../data/Fm_" + str(M) + "_avg_" + str(sample_size) + '_' + noise_mode +'_'+ str(noise_para )+ "_seed_" + str(seed) + "_" + time_mark + ".txt", Fm)
            # np.savetxt("../data/Fm_" + str(M) + "_avg_" + str(sample_size) + "_Amp_damp_001_K1_seed_" + str(seed) + "_" + time_mark + ".txt", Fm)
    
    # pic_path = "../data/Fm_" + str(M) + "_avg_" + str(sample_size) + '_' + noise_mode +'_'+ str(noise_para) + "_seed_" + str(seed) + "_" + time_mark + ".png"
    
    
    m = np.array(range(1, 1 + M))
    plt.plot(range(1, M+1), Fm, 'o', label='data')
    plt.plot(range(1, M+1), A*p**m + B, '-', label=r'$F_m^{(M)}$')
    # plt.title()
    plt.xlabel('Sequence length')
    plt.ylabel('Average sequence fidelity')
        