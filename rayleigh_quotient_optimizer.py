import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy import optimize


def cerate_A_sym_mat(dim=10):
	Q,_ = np.linalg.qr(np.random.randn(dim,dim) *np.random.randn(dim,dim))
	s = np.random.choice(dim) + 1
	S = np.diag(list(reversed(range(s,s+dim))))**2
	return Q.dot(S).dot(Q.T), Q, S

def create_X(dim=10):
	X = ['x'+str(i) for i in range(dim)]
	return np.array(sp.symbols(X)).T

# First Eigen Vector
dim = 3
A,V,L = cerate_A_sym_mat(dim)

# To handle Rank defeciency
c = 10
A = A + c*np.eye(dim)

X = create_X(dim)
f0 = -1*X.T.dot(A).dot(X)/X.T.dot(X)
df0 = f0.diff(X)
H0 = df0.diff(X)

def eval_0(x):
	value = {}
	for var,val in zip(range(len(x)),x):
		value['x'+str(var)] = val
	return f0.subs(value)

def grad_0(x):
	value = {}
	for var,val in zip(range(len(x)),x):
		value['x'+str(var)] = val
	return np.array(df0.subs(value)).astype('float')

def hessian_0(x):
	d = len(x)
	value = {}
	for var,val in zip(range(len(x)),x):
		value['x'+str(var)] = val
	return np.array(H0.subs(value)).astype('float').reshape(d,d)

X0 = np.random.randn(dim)
X0 = X0/np.linalg.norm(X0)
result = optimize.minimize(eval_0, X0, method='Newton-CG', jac=grad_0, hess=hessian_0, options={'xtol': 1e-3, 'disp': False})  
V0 = result.x/np.linalg.norm(result.x)
l0 = result.fun

# Second Eigen Vector
Y = X - np.dot(X,V0)*V0
f1 = -1*Y.T.dot(A).dot(Y)/Y.T.dot(Y)
df1 = f1.diff(X)
H1 = df1.diff(X)

def eval_1(x):
	value = {}
	for var,val in zip(range(len(x)),x):
		value['x'+str(var)] = val
	return f1.subs(value)

def grad_1(x):
	value = {}
	for var,val in zip(range(len(x)),x):
		value['x'+str(var)] = val
	return np.array(df1.subs(value)).astype('float')

def hessian_1(x):
	d = len(x)
	value = {}
	for var,val in zip(range(len(x)),x):
		value['x'+str(var)] = val
	return np.array(H1.subs(value)).astype('float').reshape(d,d)


X0 = np.random.randn(dim)
X0 = X0/np.linalg.norm(X0)
result1 = optimize.minimize(eval_1, X0, method='Newton-CG', jac=grad_1, hess=hessian_1, options={'xtol': 1e-3, 'disp': False})  

V1 = result1.x
V1 = V1 - np.dot(V1,V0)*V0
V1 = V1/np.linalg.norm(V1)
l1 = result1.fun

print('lambda_0 = {}, lambda_0 = {}'.format(np.sign(l0)*l0 - c,np.sign(l1)*l1 - c))
