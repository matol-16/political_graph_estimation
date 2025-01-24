from random import*
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import g_estim as ge

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
        Z = np.random.choice(K, size=n, p=pi)

        #la matrice d'adjacence
        X = np.zeros((n, n))

        #Pour chaque couple de sommet possible (on ne boucle que sur une moitié de X car le graphe n'est pas orienté)
        for i in range(n-1):
            for j in range(i+1, n):
                #on récupère le paramètre de la loi binomiale du sommet Xij
                p = mu[Z[i], Z[j]]
                #On procède à un tirage pour ce sommet
                X[i, j] = np.random.binomial(1, p)
                #On incrémente symétriquement la matrice d'adjacence
                X[j, i] = X[i, j]

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
    

    def estim_blocs(self, K, t_0=None, debut_clusters = True, vraissemblance = True, pf_opti= True, Nmax_glob=50, nmax_ptf=50, emax_pf = 0.0001, emax_it=0.001, debug= False, debug_detail = False):
        #estime les blocs du graphe self
        
        #Implémentation de l'algorithme de recherche de max de vraissemblance
        #X: matrice d'adjacence
        #t0 est une matrice de taille (K, n)
        #Nmax le max d'itérations
        #Nmax_glob le max d'itérations de la fonction de recherche de point fixe "maison" si elle est utilisée
        #emax l'erreur acceptée pour le point fixe

        adjmat=self.adjacency_matrix


        n=self.num_vertices #le nombre de noeuds
        N=0 #le compte des itérations
        #On initialise nos variables à itérer
        if(debut_clusters):
            t_0 = ge.K_clust(K, adjmat)
        t_N = t_0
        t_N[t_N == 0] = 10**(-16)

        #on initialise les mu et pi qu'on voudra estimer
        mu_N=np.zeros((K,K))
        pi_N=np.zeros(K)

        condition = True

        V_old = -100000000000

        if pf_opti:
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

            mu_N[mu_N == 0] = 10**(-16)
            mu_N[mu_N == 1] = 1 - 10**(-16)
            pi_N[pi_N == 0] = 10**(-16)

        
            #On met ensuite à jour le paramètre définissant l'estimation de la fonction graphon:

            #Pour cela, on doit implémenter un algo de recherche de point fixe:

            #On recherche ensuite son point fixe (qui est le point fixe recherché à une constante près)

            #Pour cela, on définit d'abord une fonction auxiliaire ne prenant que t en argument:

            def tau_opti_aux(t):
                return ge.tau_opti(pi_N,mu_N,t, adjmat)


            #Puis on utilise notre fonction de recherche de point fixe:
            #Ici on a le choix entre la méthode "maison" (ci dessous), et une fonction de la bibliothèque scipy.optimize :
            #t_N=point_fixe(emax, tau_opti_aux, t_N, nmax_ptf)
            #t_N = fixed_point(tau_opti_aux, t_N) #la fonction de la bibliothèque scipy.optimize
            if pf_opti:
                t_N = ge.update_tau(pi_N, mu_N, t_N, adjmat_opti, nmax_ptf, opti=True)
            else:
                t_N = ge.update_tau(pi_N, mu_N, t_N, adjmat, nmax_ptf, opti = False)

            #on teste la croissance de la vraissemblance
            t_N[t_N == 0] = 10**(-16)
            if vraissemblance:
                V=ge.borne_inf(t_N,mu_N,pi_N,n,K,adjmat) #calcul de la vraissemblance
                diff=V-V_old

                if(diff<0):
                    print(f"vraissemblance non croissante pour l'itération {N}")
                # break à mettre pour garder seulement les essais avec une vraisemblance croissante
                if debug_detail:
                    print(f"Vraissemblance: {V}")
                V_old=V


            #On incrémente le nombre d'itérations
            N+=1

            #on affiche que les itérations en dessous de 10 ou le smultipes de 10
            if((not debug) and N%10==0):
                print(f"itération {N}")


            if(debug):
                print(f"itération {N}")


            if(debug_detail):
                print(f"pi_N: {pi_N}")

            #on arrête d'itérer après un nombre prédéfini d'itérations
            if(N>=Nmax_glob):
                print(f"nombre d'itérations maximal {Nmax_glob} dépassé pour l'algorithme général")
                print(f"la diférence entre 2 itérations est : {np.linalg.norm(t_N-ancien_t)} en norme usuelle et {np.abs(t_N-ancien_t).max()} en norme max")

                condition = False

            #on arrête d'itérer si les valeurs de tau restent très proches les unes des autres
            #on a plusieurs options pour la norme: celle usuelle de numpy, ou la norme max:
            #if(np.linalg.norm(t_N-ancien_t)<emax_it): # norme usuelle de numpy
            if vraissemblance:
                if abs(diff)<emax_it:
                    print(f"Convergence de la vraissemblance atteinte en {N} itération à un niveau de {emax_it}")
                    condition=False

            else:
                if (np.abs(t_N-ancien_t)<emax_it).all(): #norme max
                    print(f"Convergence atteinte en {N} itération à un niveau de {emax_it}")
                    condition=False

        #Enfin, on infère le bloc d'appartenance pour chaque noeud

        Z_N = np.eye(n)[np.argmax(t_N, axis=1)][:,0:K]
        print(f"Vraissemblance atteinte: {V}")

        return pi_N, mu_N, t_N, Z_N, V
    
    def find_blocs(self, N=15,random = True,Nmax_glob=30, nmax_ptf=50, emax_pf = 0.0001, emax_it=0.001, debug= False, debug_detail = False):
        adjmat=self.adjacency_matrix
        maxvrais = 0
        indicemaxvrais = 0
        n=self.num_vertices
        resultats={0:None}
        for K in range(4, N+1) :
            (pi_N, mu_N, t_N, Z_N) = self.estim_blocs(K, Nmax_glob=30, nmax_ptf=50, emax_pf = 0.001, emax_it=0.0001, debug= False, debug_detail = False)
            m = ge.ICL(K, pi_N, mu_N, t_N, Z_N)
            resultats[K]=(pi_N, mu_N, t_N, Z_N)
            if m>maxvrais :
                m = maxvrais
                indicemaxvrais = K
        return maxvrais, indicemaxvrais, resultats