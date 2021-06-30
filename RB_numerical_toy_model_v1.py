# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:20:36 2021

@author: Hans
"""
import numpy as np
from numpy.linalg import inv
from scipy.stats import unitary_group

ket_0 = np.array([1,0]).reshape(2,1)
ket_1 = np.array([0,1]).reshape(2,1)
rho = np.array([[1,0],[0,0]])
proj_O = np.kron(ket_0, ket_0.T)        # np.kron(ket_0, ket_0.T) - np.kron(ket_1, ket_1.T)
m = 3
lmbda = 0.2
K = np.zeros((4,2,2))
K[0] = np.sqrt(1-3*lmbda/4)*np.eye(2)
K[1] = np.sqrt(lmbda/4)*(np.kron(ket_0, ket_1.T) + np.kron(ket_1, ket_0.T))
K[2] = np.sqrt(lmbda/4)*(-1j*np.kron(ket_0, ket_1.T) + 1j*np.kron(ket_1, ket_0.T))
K[3] = np.sqrt(lmbda/4)*(np.kron(ket_0, ket_0.T) - np.kron(ket_1, ket_1.T))
noise_on = np.zeros((2,2))
inver_op = np.eye(2)

# Kraus operators have 4 terms to sum, CP map (unitary, CPTP) may not be hermitian,
# can not multiply rho after sum over.
# Calculate A rho A+ each iteration.
for i in range(m):
    cp_map = unitary_group.rvs(2)
    # CP on the state
    cp_on = cp_map @ rho @ np.conj(cp_map).T
    # noise Lambda affect the state
    for k in range(4):
        noise_on += K[k] @ cp_on @ np.conj(K[k]).T
    inver_cp = np.matmul(inv(cp_map), inver_op)
    
Sm = np.matmul(inver_op, noise_oper)
Sm_rho = np.matmul(Sm, np.matmul(rho, np.conj(Sm).T))
Fm = np.trace(np.matmul(proj_O, Sm_rho).real)
# F = np.trace(np.matmul(np.conj(Sm_rho).T, np.matmul(proj_O, Sm_rho)))

print('Fm = ', Fm)
# print('F =', F)
    
    
    