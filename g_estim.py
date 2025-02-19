from random import*
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from math import *
#Cette fonction sera utilisée plusieurs fois dans la suite
def b(x,p):
  #pour pallier à des erreurs d'arrondi. On le commente pour gagner en tps de calcul.
  # epsilon = 1e-10
  # if(p<epsilon):
  #   p=epsilon
  # if(p>1):
  #   p=1
  return (p**x)*(1-p)**(1-x)

def update_tau(pi, mu, t, X, nmax_ptf):
    "met tau à jour en cherchant le point fixe de la relation ci-dessus"
    t_n = t.copy()
    count = 0
    n = len(X[0])
    k = len(pi)

    while True:
        old_t = t_n.copy()

        adjmat_1= X[1]
        adjmat_0= X[0]
        old_t = t_n.copy()
        #on applique la formule de l'article avec une nuance, on ne calcule plus directement b parce qu'on utilise une liste d'adjacence
        #avec la position des 1
        for i in adjmat_1:
            for q in range(k):
                t_n[i,q] = pi[q]*np.prod([(mu[q][l])**old_t[j][l] for j in adjmat_1[i] for l in range(k) if (j != i) and mu[q][l]!=0])
                t_n[i,q] = t_n[i,q]*np.prod([(1-mu[q][l])**old_t[j][l] for j in adjmat_0[i] for l in range(k) if (j != i) and mu[q][l]!=0])
        
        #on normalise
        for i in range(n):
            c = np.sum(t_n[i,:])
            for q in range(k):
                t_n[i,q] = t_n[i,q]/c


        count += 1
        if count >= nmax_ptf:
          break
        if (np.abs(t_n - old_t) < 1e-3).all():
            break


    return t_n
  
from sklearn.cluster import KMeans
import scipy



def K_clust(k, X):
  mat = []
  "calcule k cluster à partir d'une matrice d'adjacence selon l'algorithme des K-moyennes"
  #on lance 50 fois l'algorithme pour réduire la variabilité et prendre le meilleur essai
  #en effet comme l'algorithme K-mean dépend lui aussi du hasard pour son point de départ, il est important de lelancer plusieurs fois
  kmeans = KMeans(n_clusters=k, n_init = 50).fit(X)
  for i in range(len(kmeans.labels_)):
    mat.append(list(np.eye(k)[kmeans.labels_[i], :]))

  return np.array(mat)

def borne_inf(t, mu, pi, n, K,adjmat):
  "Calcule la vraissemblance en un point"
  a_1 = np.sum([t[i][q]*np.log(pi[q]) for i in range(n) for q in range(K)])
  a_2 = (1/2)*np.sum([t[i][q]*t[j][l]*np.log(b(adjmat[i][j],mu[q][l])) for j in range (n) for i in range(n) for q in range(K) for l in range(K) if j!=i])
  a_3 = np.sum([t[i][q]*np.log(t[i][q]) for i in range(n) for q in range(K)])
  return a_1+a_2-a_3


def ICL(K, pi_N, mu_N, t_N, Z_N, adjmat):
  "calcul de l'ICL qui nous permet d'estimer la qualité d'une approximation"

  n = len(adjmat[0])
  m = -K*(K+1)/4*log(n*(n-1)/2)-(K-1)/2*np.log(n) - n*np.log(n)
  l = np.sum(Z_N, axis = 0)
  for j in range(len(l)):
      m += l[j]*log(l[j])
  m += 1/2*np.sum([Z_N[i,q]*Z_N[j,l]*np.log(b(adjmat[i][j], mu_N[q][l])) for q in range(K) for l in range(K) for i in range(n) for j in range(n) if j!= i])
  return m


