from random import*
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def point_fixe(emax,f,x0, nmax):
  n=0 #compte des itérations
  x=f(x0)
  condition= np.linalg.norm(f(x0)-x)>emax
  #condition =(np.abs(f(x)-x)<emax).all() #condition pour norme max
  #condition = not condition

  #on cherche le point fixe en acceptant un certain niveau d'erreur
  while(True):
    y=f(x)
    n+=1
    #condition = np.linalg.norm(y-x)>emax #option pour norme usuelle
    #condition =(np.abs(f(x)-x)<emax).all() #condition pour norme max
    #condition = not condition
    if (np.abs(y - x) < emax).all():
      break

    x=y


    if(n>=nmax):
      print("nombre d'itérations dépassé pour la recherche de point fixe")
      break

  return x

#Cette fonction sera utilisée plusieurs fois dans la suite
def b(x,p):
  #pour pallier à des erreurs d'arrondi. On le commente pour gagner en tps de calcul.
  # epsilon = 1e-10
  # if(p<epsilon):
  #   p=epsilon
  # if(p>1):
  #   p=1
  return (p**x)*(1-p)**(1-x)

def update_tau(pi, mu, t, X, nmax_ptf, opti = False):
  # a priori, tourne plus vite ?
  t_n = t.copy()
  count = 0
  n = len(X[0])
  k = len(pi)
        # Fix point iteration

  while True:
    old_t = t_n.copy()

    #proposition de calcul optimisé:
    if opti:
      adjmat_1= X[1]
      adjmat_0= X[0]
      old_t = t_n.copy()
      #calcul de A_0:
      for i in adjmat_1:
        for q in range(k):
            t_n[i,q] = pi[q]*np.prod([(mu[q][l])**old_t[j][l] for j in adjmat_1[i] for l in range(k) if (j != i) and mu[q][l]!=0])
            t_n[i,q] = t_n[i,q]*np.prod([(1-mu[q][l])**old_t[j][l] for j in adjmat_0[i] for l in range(k) if (j != i) and mu[q][l]!=0])
    else:
      for i in range(n):
        #norm_const = np.exp(np.sum([tau[i,r] for r in range(self.q)]))
        for q in range(k):
            t_n[i,q] = pi[q] * np.prod([
                b(X[i][j], mu[q][l]) ** old_t[j][l]
                for j in range(n)
                for l in range(k) if (j != i) and mu[q][l]!=0
            ])

    for i in range(n):
        c = np.sum(t_n[i,:])
        for q in range(k):

            t_n[i,q] = t_n[i,q]/c


    count += 1
    if count == nmax_ptf:
      break
    if (np.abs(t_n - old_t) < 1e-3).all():
        break        


  return t_n      

from sklearn.cluster import KMeans
import scipy

def K_clust(k, X):
  mat = []
  #calcul k cluster à partir d'une matrice d'adjacence

  kmeans = KMeans(n_clusters=k, n_init = 100).fit(X)
  for i in range(len(kmeans.labels_)):
    mat.append(list(np.eye(k)[kmeans.labels_[i], :]))

  return np.array(mat)

def borne_inf(t, mu, pi, n, K,adjmat):
  "Calcule la vraissemblance en un point de l'estimation des blocs"
  a_1 = np.sum([t[i][q]*np.log(pi[q]) for i in range(n) for q in range(K)])
  b_1 = (1/2)*np.sum([t[i][q]*t[j][l]*np.log(b(adjmat[i][j],mu[q][l])) for j in range (n) for i in range(n) for q in range(K) for l in range(K) if j!=i])
  c_1 = np.sum([t[i][q]*np.log(t[i][q]) for i in range(n) for q in range(K)])
  return a_1+b_1-c_1


def ICL(adjmat,K, pi_N, mu_N, t_N, Z_N):
  #à améliorer
  n=len(adjmat)
  m = -K*(K+1)/4*np.log(n*(n-1)/2)-(K-1)/2*np.log(n) - n*np.log(n)
  for k in range (1, K+1):
    l = np.sum(Z_N[q, k] for q in range(K))
    m += l*np.log(l)
  m += 1/2*np.sum([Z_N[i,q]*Z_N[j,l]**np.log(b(adjmat[i][j], mu_N[q][l])) for q in range(K) for l in range(K) for i in range(n) for j in range(n) if j!= i])
  return m


