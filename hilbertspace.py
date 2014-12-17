import numpy as np
import sys
sys.setrecursionlimit(1500)


class HilbertSpace:
	def __init__(self, shape, components = []):
		self.shape = shape
		self.components = components
		
	def append_component(self, C):
		self.components.append(C)
		
	def add(self, U,V):
		if U.shape is V.shape and U.shape <= self.shape:
			W = U + V
			return W
		else:
			raise ShapeException
			
	def sub(self, U,V):
		if U.shape is V.shape and U.shape <= self.shape:
			W = U - V
			return W
		else:
			raise ShapeException

	def inner_product(self, U,V):
		if U.shape is V.shape and U.shape <= self.shape:
			p = complex(0,0)
			for i in range(0,U.shape):
				z = complex(U.real[i], U.imag[i])
				w = complex(V.real[i], -V.imag[i])
				p = p+(z*w)
			return np.array([p.real, p.imag])
		else:
			 raise ShapeException
			 	
	def norm(self, U):
		return np.sqrt(self.inner_product(U,U))[0]
		
	def distance (self, U,V):
		return self.inner_product(U-V, U-V)[0]		
	
			
class ShapeException(Exception):
	pass
	
class HilbertVector:
	
	def __init__(self, H, real, imag):
		a = real.shape[0] - imag.shape[0]
		if a > 0:
			imag = np.concatenate((imag, np.zeros(abs(a))))
		elif a<0:	
			real = np.concatenate((real, np.zeros(abs(a))))
			
		H.append_component(self)
		self.real, self.imag = real, imag
		self.shape = real.shape[0]

	def get_conjugate(self):
		return HilbertVector(H, self.real, -1*self.imag)
	
	def __add__(self, V):
		return HilbertVector(H, self.real+V.real, self.imag+V.imag)
		
	def __sub__(self, V):
		return HilbertVector(H, self.real-V.real, self.imag-V.imag)	
		
	def __mul__(self, scalar):
		return HilbertVector(H, self.real*scalar, self.imag*scalar)
		
	def __str__(self):
		s = '''('''
		for i in range(0,self.real.shape[0]):
			s = s + '{0}+{1}j \n'.format(self.real[i], self.imag[i])
		return s + ')'
		
class HilbertSingleVariableFunction(HilbertVector):
	
	def __init__(self, H, real, imag):
		self.v = HilbertVector(H, real,imag)
		self.shape = self.v.shape
		
	def __call__(self, x):
		U = HilbertVector(H, np.zeros(self.shape), np.zeros(self.shape))
		for i in range(0,self.shape):
			U.real[i] = self.v.real[i](x)
			U.imag[i] = self.v.imag[i](x)
		return U	
		
			
class HilbertMultiVariableFunction(HilbertVector):

	def __init__(self, H, real, imag,n):
		self.v = HilbertVector(H, real,imag)
		self.shape, self.n = self.v.shape, n
		
	def __call__(self, x):
		U = HilbertVector(H, np.zeros(self.shape), np.zeros(self.shape))
		for i in range(0,self.shape):
			mr = len(inspect.getargspec(U.real[i])[0]) 
			mi = len(inspect.getargspec(U.real[i])[0]) 
			yr = np.concatenate((x, np.zeros(abs(mr-len(x)))))
			yi = np.concatenate((x, np.zeros(abs(mi-len(x)))))
			U.real[i] = self.v.real[i](*yr)
			U.imag[i] = self.v.imag[i](*yi)	
		return U	
				
if __name__ == '__main__':
	H = HilbertSpace(3)
	U = HilbertVector(H, np.array([2,3,1]), np.array([3,4,5]))
	V = HilbertVector(H, np.array([4,3,1]), np.array([5,4,5]))
	W = H.add(U,V)
	print W
	A = H.sub(U,V)
	print A
	z = H.inner_product(U,V)
	l = H.norm(U)
	print l
	Z = HilbertSingleVariableFunction(H, np.array([lambda x: x**2, lambda x: 2*x, lambda x: 1]), np.array([lambda x: 0, lambda x: np.sqrt(x), lambda x: x**x]))
	print Z(1)
