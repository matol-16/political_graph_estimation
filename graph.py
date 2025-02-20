from random import*
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

from g_estim import *
class Graph:
    def __init__(self, adjacency_matrix, blocs=None):
        self.adjacency_matrix = adjacency_matrix
        self.num_vertices = len(adjacency_matrix)
        if blocs is None:
            self.blocs = np.ones((self.num_vertices, 1)) #un seul blocs = graphe standard
        else:
            self.blocs = blocs
        self.blocs = blocs
        self.graph = self._create_graph()
        self.estimated_blocs=None

    def _create_graph(self):
        graph = {}
        for i in range(self.num_vertices):
            graph[i] = []
            for j in range(self.num_vertices):
                if self.adjacency_matrix[i][j] != 0:
                    graph[i].append(j)
        return graph

    def __repr__(self):
        return str(self.graph)

    def simu_Wgraph(n,W):
        #simule un W-graphe pour le graphon W et n sommets (question 6)
        U = np.random.random_sample(n)
        X = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i>j:
                    p = W(U[i], U[j])
                    X[i, j] = np.random.binomial(1, p)
                    X[j, i] = X[i, j]
        return Graph(X)
    
    def simu_blocsgraph(n,pi,mu):
        #simule un graphe à blocs stochastiques 

        K = len(pi) #nombre de blocs 
        #on construir le vecteur décrivant le bloc d'appartenance de chaque sommet
        labels = np.random.choice(K, size=n, p=pi)
        Z= [[0 if j!=labels[i] else 1  for j in range(K)] for i in range(n)]

        #la matrice d'adjacence
        X = np.zeros((n, n))

        #Pour chaque couple de sommet possible (on ne boucle que sur une moitié de X car le graphe n'est pas orienté)
        for i in range(n-1):
            for j in range(i+1, n):
                #on récupère le paramètre de la loi binomiale du sommet Xij
                p = mu[labels[i], labels[j]]
                #On procède à un tirage pour ce sommet
                X[i][j] = np.random.binomial(1, p)
                #On incrémente symétriquement la matrice d'adjacence
                X[j][i] = X[i][j]

        return Graph(X, Z)

    def graphon_croiss(pi,mu,u,v):
        #Cette fonction commence par partitionner [0,1] tel que g soit croissante en u
        #Ce changement de parition ne change pas les probabilités associées à chaque blocs
        #Elle retourne ensuite la probabilité mu_(i,j) de lien entre 2 blocs associée à (u,v) selon cette nouvelle partition.

        #On commence par construire la liste des valeurs à trier :
        #la parittion de [0,1] est pour le moment celle de la question 2
        K=len(pi)
        l_tri = []
        mu=np.array(mu)
        pi=np.array(pi)
        for j in range(K):
            l_tri.append(mu[j]@pi)
        #on récupère les indices des élélemnts de la liste dans l'ordre croissant
        ind_triés = sorted(range(len(l_tri)), key=l_tri.__getitem__)

        #on créé ensuite la liste des i_j partitionnant [0,1] en K intervalles
        l_cr=[pi[ind_triés[0]]]
        for j in range(1,10):
            l_cr.append(l_cr[j-1]+pi[ind_triés[j]])
        #On fixe bien le dernier à 1 pour s'assurer de partitionner [0,1] entièrement
        l_cr[K-1]=1
        #On cherche enfin l'intervalle de [0,1]x[0,1] auquel appartient (u,v):
        x=0
        i=-1
        while(u>=x and i<K-1):
            i+=1
            x= l_cr[i]
        y=0
        j=-1
        while(v>=y and j<K-1):
            j+=1
            y= l_cr[j]
        #on en déduit la probabilité de lien entre u et v

        return mu[ind_triés[i],ind_triés[j]], l_cr, ind_triés


    def estim_kk(self, K, Nmax_glob, nmax_ptf, emax):
    
      "Implémentation de l'algorithme de recherche de max de vraissemblance"
      #adjmat: matrice d'adjacence
      #Nmax le max d'itérations
      #Nmax_glob le max d'itérations de la fonction de recherche de point fixe "maison" si elle est utilisée
      #emax l'erreur acceptée pour le point fixe"
    
      n = self.num_vertices #le nombre de noeuds
      adjmat = self.adjacency_matrix
      N = 0 #le compte des itérations
    
      #On initialise nos variables à itérer. On prend des 10^-16 à la place des 0 pour éviter les problèmes de logarithmes
      t_N = K_clust(K, adjmat)
      t_N[t_N == 0] = 10**(-16)
    
      #on initialise les mu et pi qu'on voudra estimer
      mu_N=np.zeros((K,K))
      pi_N=np.zeros(K)
    
      condition = True
    
      #on initialise la vraissemblance
      V_old = -1e11
    
      #on implémente une  matrice creuse pour aller plus vite
      adjmat_1= {i: [j for j, val in enumerate(row) if val != 0] for i, row in enumerate(adjmat)}
      adjmat_0= {i: [j for j, val in enumerate(row) if val == 0] for i, row in enumerate(adjmat)}
      adjmat_opti = [adjmat_0, adjmat_1]
    
      while(condition):
        ancien_t = t_N
    
        #on met à jour les paramètres du modèle à blocs stochastiques suivant les FOC
        pi_N = np.mean(t_N, axis=0)
    
        for q in range(K):
          for l in range(K):
              numerat = np.sum([t_N[i,q]*t_N[j,l]*adjmat[i][j] for i in range(n) for j in range(n) if j!= i])
              denom = np.sum([t_N[i,q] * t_N[j,l] for i in range(n) for j in range(n) if j != i])
              if(denom!=0):
                mu_N[q,l] = numerat/denom
              else:
                 mu_N[q,l]=0
    
        #on fait attention à enlever tous les 0 et 1 pour ne pas avoir de soucis dans nos logs, on utilise l'epsilon machine
        mu_N[mu_N == 0] = 10**(-16)
        mu_N[mu_N == 1] = 1 - 10**(-16)
        pi_N[pi_N == 0] = 10**(-16)
    
        #on met à jour tau selon la méthode du point fixe décrit dans l'article
        t_N = update_tau(pi_N, mu_N, t_N, adjmat_opti, nmax_ptf)
    
        #on remplace les 0 par 10^-16 pour éviter les problèmes de logarithmes dans les calcules
        t_N[t_N == 0] = 10**(-16)
    
        #calcul de la vraissemblance
        V = borne_inf(t_N,mu_N,pi_N,n,K,adjmat) 
          
        diff=V-V_old
        if(diff<0):
          print(f"vraissemblance non croissante pour l'itération {N}")
        V_old=V
    
        #On incrémente le nombre d'itérations
        N+=1
    
        #on affiche les itérations multiples de 5
        if(N%5==0):
          print(f"itération {N} et vraissemblance = {V}")
    
        #on arrête d'itérer après un nombre prédéfini d'itérations
        if(N>=Nmax_glob):
          print(f"nombre d'itérations maximal {Nmax_glob} dépassé pour l'algorithme général")
          print(f"la diférence entre 2 itérations est : {np.linalg.norm(t_N-ancien_t)} ")
    
          condition = False
    
        #on arrête d'itérer si les valeurs de la vraissemblance restent très proches les unes des autres
        if abs(diff)<emax:
          print(f"Convergence de la vraissemblance atteinte en {N} itération à un niveau de {emax}")
          condition = False
    
      #Enfin, on infère le bloc d'appartenance pour chaque noeud
      Z_N = np.eye(n)[np.argmax(t_N, axis=1)][:,0:K]
      print(f"Vraissemblance atteinte: {V}")
      self.estimated_blocs=Z_N
    
      return pi_N, mu_N, t_N, Z_N, V
    
    def estim_kk_MC(self, N_mt, K, Nmax_glob, nmax_ptf, emax):
        "on calcul le clustering en K partie politique de notre jeu de donnée en lançant 10 fois l'algorithme"
        "et en prenant la meilleure valeure d'ICL sur ces 10 performances"
        Z_tot = []
        likl = []
        adjmat = self.adjacency_matrix
        for l in range(N_mt):
            print(f"Essai type Monte Carlo numéro:{l}")
            pi_N, mu_N, t_N, Z_N, V = self.estim_kk(K, Nmax_glob , nmax_ptf , emax )
            Z_tot.append(Z_N)
            likl.append(ICL(K, pi_N, mu_N, t_N, Z_N, adjmat))
        
        #on récupère le meilleur essai
        m, Z = max(likl), Z_tot[likl.index(max(likl))]
        self.estimated_blocs=Z
        print("La Vraissemblance atteinte est de ", m)
        return m, Z

    def estim_findK(self, Nmax_glob, nmax_pft, emax,N_mt=5):
        adjmat=self.adjacency_matrix
        icl = []
        Z_tot = []
        for K in range(1, 14):
            print(f"------------Estimation pour {K} blocs----------")
            ICL, Z = self.estim_kk_MC(N_mt,K, Nmax_glob, nmax_pft, emax)
            icl.append(ICL)
            Z_tot.append(Z)
        self.blocs = Z_tot[icl.index(max(icl))]
        return icl, Z_tot[icl.index(max(icl))], (icl.index(max(icl)) + 1)
    
    def display_comp_table(self, colonne = None):
        assert self.estimated_blocs is not None, "You must estimate the blocs first"
        Z_vrai = self.blocs
        best_Z = self.estimated_blocs


        n = len(Z_vrai[0])
        m = len(best_Z[0])

        # Créer un tableau croisé pour compter les correspondances
        tableau_croise_best = np.zeros((m,n), dtype = int)


        # Remplir le tableau croisé
        for j in range(m):
            for i in range(len(best_Z)):  
                if best_Z[i][j] == 1:  
                    for k in range(n):  
                        if Z_vrai[i][k] == 1:  
                            tableau_croise_best[j][k] += 1


        noms_colonnes = colonne if colonne is not None else [f"Vrai bloc{i+1}" for i in range(n)]

        noms_lignes = [f"bloc attribué{j+1}" for j in range(m)]

        # Conversion en DataFrame
        df_tableau = pd.DataFrame(tableau_croise_best, columns=noms_colonnes, index=noms_lignes)

        print("Tableau croisé avec noms des lignes et colonnes :")
        print(df_tableau)
        return df_tableau

            
