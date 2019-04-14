import os
import sys
import numpy as np

from rayleigh import create_A_sym_mat
from rayleigh import RayleighQuotient

# Set Seed Value
# np.random.seed(3)


dim = 100
A,V,L = create_A_sym_mat(dim)

# To handle Rank defeciency
c = 10
A = A + c*np.eye(dim)

print('Marix Created')
rq_obj = RayleighQuotient(A)


# First Eigen Vector
result = rq_obj.optimize()
V0 = result.x/np.linalg.norm(result.x)
l0 = rq_obj.eval_rq(V0)
print('First Eigen Vector...')

# Second Eigen Vector
rq_obj.update_func(V0.reshape((dim,1))) # modify the objective function
print('Updated Objective Function...')
result = rq_obj.optimize()
V1 = result.x/np.linalg.norm(result.x)
V1 = V1 - np.dot(V1,V0)*V0 # Making sure that V1 is perpendicular to V0
V1 = V1/np.linalg.norm(V1)
l1 = rq_obj.eval_rq(V1)
print('Second Eigen Vector...')

print('lambda_0 = {}, lambda_0 = {}'.format(np.sign(l0)*l0 - c,np.sign(l1)*l1 - c))

