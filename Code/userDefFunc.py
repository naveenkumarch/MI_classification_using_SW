import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
from scipy.linalg import sqrtm
from scipy import linalg


def my_function(X1,X2):

  R1 = (np.matmul(X1,X1.transpose()))/np.trace(np.matmul(X1,X1.transpose()))
  R2 = (np.matmul(X2,X2.transpose()))/np.trace(np.matmul(X2,X2.transpose()))

  R = R1+R2

  EValsum, EVecsum = LA.eig(R)

  ind = np.argsort(EValsum)
  ind = ind[::-1]
  EVecsum = EVecsum[:,ind]
  EValsum = EValsum[ind]
  EValsum = np.diag(EValsum)

  temp=np.transpose(EVecsum)
  P=np.matmul(sqrtm(inv(EValsum)),temp)

  S1=np.matmul(np.matmul(P,R1),np.transpose(P))
  S2=np.matmul(np.matmul(P,R2),np.transpose(P))

  w, vr=linalg.eig(S1, S2, False, True)
  ind = np.argsort(w.real)
  U = vr[:,ind]



  result = np.matmul(np.transpose(U),P)

  return result


def f_logVar(X):

  X_var = np.var(X,1,ddof=0)
  X_var_sum = np.sum(X_var)
  XLog = np.log(np.divide(X_var, X_var_sum))

  return XLog
