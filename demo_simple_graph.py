
import graph
import numpy as np
import g_display

vanilla = True


if __name__ == "__main__":
    #Paramètres modèle
    n = 100
    k=5
    pi = (1/5, 1/5, 1/5, 1/5, 1/5)
    beta=0.4
    nmax_glob=30
    nmax_ptf=20
    emax=1e-4

    #Matrice des probabilités d'interconnexion
    mu = np.ones((n,n))/100 + (beta - 0.01)*np.eye(n)

    #Estimation et affichage des résultats

    g=graph.Graph.simu_blocsgraph(n, pi, mu)

    #true_label=g.blocs

    #true_label = [[0 if j!=true_label[i] else 1  for j in range(K)] for i in range(n)]

if vanilla: 
    print("----------------Estimation des 5 blocs:------------------------------------")
    _,_,_,Z_pred, V = g.estim_kk(5, nmax_glob, nmax_ptf, emax)
    print("----------------Graphe simulé à partir des paramètres choisis:-----------")
    g_display.display_graph(g)
    print("----------------Graphe à blocs estimé:----------------------------------")
    g_display.display_graph(g, take_blocs_estim=True)

    print("-------Tableau de comparaison des blocs réels/estimés:------------------")

    graph.Graph.display_comp_table(g)

else:
        print("----------------Estimation des K blocs:------------------------------------")
        _,Z_pred_betas,_ = g.estim_findK(Nmax_glob, nmax_ptf, emax, Kmax=8)
        np.save('output/Z_pred_betas.npy', Z_pred_betas)
        print("----------------Graphe simulé à partir des paramètres choisis:-----------")
        g_display.display_graph(g)
        np.save('output/Z_betas.npy', g.blocs)

        print("----------------Graphe estimé K inconnu:----------------------------------")
        g_display.display_graph(g, take_blocs_estim=True)