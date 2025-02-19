from sklearn import metrics
import numpy as np
import graph as g
import matplotlib.pyplot as plt


def test_beta():
    #ensemble des valeures de beta possible
    B = np.linspace(0.05, 0.5, 20) 
    Nmax_glob = 25
    nmax_ptf = 15
    emax = 0.001
    K = 5
    n = 100
    pi = (1/5, 1/5, 1/5, 1/5, 1/5)
    score = []
    for beta in B :
        temp = []
        #on crée 20 graphes différents
        for j in range(20):
            print(f"--------Test pour beta = {beta}---------")
            mu = np.ones((n,n))/100 + (beta - 0.01)*np.eye(n)
            beta_graph= g.simu_blocsgraph(n, pi, mu)
            _,_,_,Z_pred, V = beta_graph.estim_kk(5, Nmax_glob, nmax_ptf, emax)

            pred_label = []
            for i in range(n):
                pred_label.append(list(Z_pred[i]).index(1))

            true_label = beta_graph.true_label
            temp.append(metrics.adjusted_rand_score(true_label, pred_label))
        #on ajoute la moyenne des scores
        score.append(np.mean(temp))
    return score, B


if __name__ == "__main__":
    score, B = test_beta()
    plt.plot(B, score, marker='o', color='orange', label='Score en fonction de Beta')
    plt.title("Évolution du score en fonction de Beta")
    plt.xlabel("Valeurs de Beta")
    plt.ylabel("Score obtenu")
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()