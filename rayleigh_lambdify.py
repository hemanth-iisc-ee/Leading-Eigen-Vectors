import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy import optimize
import pdb

def create_A_sym_mat(dim=3):
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
		self.rq = self.func()[0]
		self.f = self.func()[0]
		self.df = self.f.diff(self.X)
		self.H = self.df.diff(self.X).reshape(self.dim,self.dim)

		self.rq_func = sp.lambdify(np.array([self.X]),self.rq) 
		self.f_func = sp.lambdify(np.array([self.X]),self.f)
		self.df_func = sp.lambdify(np.array([self.X]),self.df)
		self.H_func = sp.lambdify(np.array([self.X]),self.H)




	def check_symmetric(self, A, tol=1e-8):
		    return np.allclose(A, A.T, atol=tol)

	def create_var_X(self, dim=3):
		X = ['x'+str(i) for i in range(dim)]
		return sp.Matrix(sp.symbols(X))

	def func(self, V=None):
		if V is None:
			return -1*self.X.T*self.A*self.X/(self.X.T*self.X)

		Y = self.X
		for i in range(V.shape[1]):
			Y = Y - (sum((self.X.T*V[:,i])[0])*V[:,i]).reshape(self.dim,1)
		
		return -1*Y.T*self.A*Y/(Y.T*Y)	

	def update_func(self, V):
		self.f = self.func(V)[0]
		self.df = self.f.diff(self.X)
		self.H = self.df.diff(self.X).reshape(self.dim,self.dim)

		self.f_func = sp.lambdify(np.array([self.X]),self.f)
		self.df_func = sp.lambdify(np.array([self.X]),self.df)
		self.H_func = sp.lambdify(np.array([self.X]),self.H)

		return

	def eval_rq(self, x):
		return self.rq_func(x)

	def eval(self, x):
		return self.f_func(x)

	def grad(self, x):
		return np.array(self.df_func(x)).reshape(self.dim)

	def hessian(self, x):
		return np.array(self.H_func(x))

	def optimize(self, x0=None):
		if x0 is None:
			x0 = np.random.randn(self.dim)
			x0 = x0/np.linalg.norm(x0)
		result = optimize.minimize(	self.eval, x0, method='Newton-CG', \
									jac=self.grad, hess=self.hessian, \
									options={'xtol': 1e-3, 'disp': False}) 
		return result 

