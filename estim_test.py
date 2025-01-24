from sklearn import metrics
import numpy as np
import graph as g
import matplotlib.pyplot as plt

def test_betas():
    Nmax_glob = 70
    nmax_ptf = 30
    emax = 1e-5
    B = np.linspace(0.45, 0.5, 30)
    #B=[0.6]
    K = 5
    n = 100
    pi = (1/5, 1/5, 1/5, 1/5, 1/5)
    score = []
    for beta in B :
        print(f"--------Test pour beta = {beta}---------")

        mu = np.ones((n,n))/100 + (beta - 0.01)*np.eye(n)
        G=g.Graph.simu_blocsgraph(n, pi, mu)
        X, true_label = G.adjacency_matrix, G.blocs
        _,_,_,Z_pred, V = G.estim_blocs(5, Nmax_glob = Nmax_glob, nmax_ptf=nmax_ptf, emax_it=emax)

        pred_label = []
        for i in range(n):
            pred_label.append(list(Z_pred[i]).index(1))


        score.append(metrics.adjusted_rand_score(true_label, pred_label))

    print(f"scores: {score}")
    plt.plot(B, score)
    plt.show()
    return score