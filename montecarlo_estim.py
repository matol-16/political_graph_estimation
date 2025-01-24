def test_Montecarlo_random_V2(N_mt,n1,K1,pi1,mu1, cluster=False,Nmax_glob_test=50, nmax_ptf_test=20,emax=1e-4,emax_pf_test=0.0001):
  #teste une méthode de carlo pour un graphe aléatoire à K2 blocs

  X2, Z2 =simuBlocs(n1, pi1, mu1)
  Z2=np.eye(K1)[Z2]

  print(f"Test pour n = {n1}")
  print(f"Probabilité modèle: {pi1}")
  print(f"Fréquence de chaque bloc: {np.mean(Z2, axis=0)}")

  l_rez =[]
  l_v=[]
  for i in  range(1,N_mt+1):
    print(f"----------Essai{i}----------")
    dep= np.random.uniform(0.000001, 1, (n1, K1))
    for i in range(n1):
     if (dep[i].sum())!=0:
       dep[i]=dep[i]/(dep[i].sum())

    result= estimp_Qfixe(X2,K1,t_0=dep, debut_clusters=False, vraissemblance = True,Nmax_glob=Nmax_glob_test, nmax_ptf=nmax_ptf_test,emax_it=emax, emax_pf=emax_pf_test, debug=False, debug_detail=False)
    print(f"pi estimé: {result[0]}")
    l_rez.append(result)
    l_v.append(result[4])
  maxvrais=np.argmax(l_v)
  print("----------RESULTATS GLOBAUX---------")
  print(f"La meilleure itération est la numéro {maxvrais+1}, à un niveau {l_v[maxvrais]}")

  return l_rez[maxvrais], Z2, X2


def main():
    n1=150
    K1=4
    pi1=np.array([0.1,0.2,0.3,0.4])
    mu1=np.array([[0.8,0.2,0.3,0.3],[0.1,0.6,0.1,0.1],[0.2,0.3,0.9,0.05],[0.3,0.2,0.1,0.7]])
    res_V2=test_Montecarlo_random_V2(3,n1,K1,pi1,mu1, emax_pf_test=0.001)
    return res_V2
