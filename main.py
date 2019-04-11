import os
import sys
import numpy as np
from scipy import optimize

from rayleigh import cerate_A_sym_mat
from rayleigh import RayleighQuotient

# Set Seed Value
# np.random.seed(3)

# First Eigen Vector
dim = 4
A,V,L = cerate_A_sym_mat(dim)

# To handle Rank defeciency
c = 10
A = A + c*np.eye(dim)

rq_obj = RayleighQuotient(A)

x0 = np.random.randn(dim)
x0 = x0/np.linalg.norm(x0)
result = optimize.minimize(rq_obj.eval, x0, method='Newton-CG', jac=rq_obj.grad, hess=rq_obj.hessian, options={'xtol': 1e-3, 'disp': False})  
V0 = result.x/np.linalg.norm(result.x)
l0 = result.fun


rq_obj.update_func(V0.reshape((dim,1)))
x0 = np.random.randn(dim)
x0 = x0/np.linalg.norm(x0)
result1 = optimize.minimize(rq_obj.eval, x0, method='Newton-CG', jac=rq_obj.grad, hess=rq_obj.hessian, options={'xtol': 1e-3, 'disp': False})  
V1 = result1.x/np.linalg.norm(result1.x)
V1 = V1 - np.dot(V1,V0)*V0 # Making sure that V1 is perpendicular to V0
V1 = V1/np.linalg.norm(V1)
l1 = result1.fun

print('lambda_0 = {}, lambda_0 = {}'.format(np.sign(l0)*l0 - c,np.sign(l1)*l1 - c))

