import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy import optimize
import pdb

def cerate_A_sym_mat(dim=3):
	Q,_ = np.linalg.qr(np.random.randn(dim,dim) *np.random.randn(dim,dim))
	s = np.random.choice(dim) + 1
	S = np.diag(list(reversed(range(s,s+dim))))**2
	return Q.dot(S).dot(Q.T), Q, S

class RayleighQuotient():
	def __init__(self, A):
		if not self.check_symmetric(A):
			print('Matrix must be symmetric!!')
			sys.exit()

		self.dim = A.shape[0]
		self.A = A
		self.X = self.create_var_X(self.dim)
		self.rq = self.func()
		self.f = self.func()
		self.df = self.f.diff(self.X)
		self.H = self.df.diff(self.X)

	def check_symmetric(self, A, tol=1e-8):
		    return np.allclose(A, A.T, atol=tol)

	def create_var_X(self, dim=3):
		X = ['x'+str(i) for i in range(dim)]
		return np.array(sp.symbols(X)).T

	def func(self, V=None):
		if V is None:
			return -1*self.X.T.dot(self.A).dot(self.X)/self.X.T.dot(self.X)

		Y = self.X
		for i in range(V.shape[1]):
			v = V[:,i].reshape(self.dim)
			Y = Y - np.dot(self.X,v)*v
		
		return -1*Y.T.dot(self.A).dot(Y)/Y.T.dot(Y)	

	def update_func(self, V):
		self.f = self.func(V)
		self.df = self.f.diff(self.X)
		self.H = self.df.diff(self.X)
		return

	def eval_rq(self, x):
		value = {}
		for var,val in zip(range(len(x)),x):
			value['x'+str(var)] = val
		return self.rq.subs(value)

	def eval(self, x):
		value = {}
		for var,val in zip(range(len(x)),x):
			value['x'+str(var)] = val
		return self.f.subs(value)

	def grad(self, x):
		value = {}
		for var,val in zip(range(len(x)),x):
			value['x'+str(var)] = val
		return np.array(self.df.subs(value)).astype('float')

	def hessian(self, x):
		d = len(x)
		value = {}
		for var,val in zip(range(len(x)),x):
			value['x'+str(var)] = val
		return np.array(self.H.subs(value)).astype('float').reshape(d,d)

	def optimize(self, x0=None):
		if x0 is None:
			x0 = np.random.randn(self.dim)
			x0 = x0/np.linalg.norm(x0)
		result = optimize.minimize(	self.eval, x0, method='Newton-CG', \
									jac=self.grad, hess=self.hessian, \
									options={'xtol': 1e-3, 'disp': False}) 
		return result 

