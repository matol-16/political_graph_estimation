import g_display
import data_retrieve
import graph
import g_estim
import g_display
import matplotlib.pyplot as plt

import os
import pickle

display_graph = False

estim_blocs = True #estimer les blocs pour un nombre de blocs fixé à 10


output_dir = os.path.join('C:', 'output')
output_path_data = os.path.join(output_dir, 'polgraph.pkl')

load_data = not os.path.exists(output_path_data)


if load_data:
    polgraph,_,_=data_retrieve.retrieve_political_data()

    os.makedirs(output_dir, exist_ok=True)

    with open(output_path_data, 'wb') as f:
        pickle.dump(polgraph, f)

    print(f'Political graph saved to {output_path_data}')

else:
    output_path = os.path.join(output_dir, 'polgraph.pkl')
    with open(output_path, 'rb') as f:
        polgraph = pickle.load(f)

    print('Political graph loaded successfully')

if display_graph:
    g_display.display_graph(polgraph)

if estim_blocs:
    #paramètres de l'estimation
    # Nmax_glob = 25
    # nmax_ptf = 15
    # emax = 0.001
    # Z_tot = []
    # l_v=[] #liste des vraissemblances
    # for i in range(5):
    #     print(f"---------Estimation {i+1}/5-------------")
    #     result = polgraph.estim_blocs(10, debut_clusters = True, Nmax_glob=50, nmax_ptf=25, emax_pf = 0.001, emax_it=0.001, debug= True, debug_detail = True)
    #     Z_estim=result[3]
    #     Z_tot.append(Z_estim)
    #     l_v.append(result[4])

    # max = max(l_v)
    # Z_best_estim=Z_tot[l_v.index(max)]
    # print("Meilleure vraisemblance : ", max)

    Nmax_glob = 1
    nmax_ptf = 2
    emax = 0.001

    icl, best_Z, best_K = polgraph.estim_findK(Nmax_glob, nmax_ptf, emax)

    print("le nombre de cluster optimal est ", best_K)

    #plot l'évolution de l'ICL
    plt.plot(range(1, 14), icl, marker='o', color='blue', label="ICL (Critère d'information)")
    plt.axvline(x=best_K, color='red', linestyle='--', label=f"Meilleur K ({best_K})")
    plt.title("Évolution du critère ICL en fonction du nombre de clusters")
    plt.xlabel("Nombre de clusters (K)")
    plt.ylabel("Valeur du critère ICL")
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()

    
    g_display(polgraph)


    output_path_estimblocs10V3 = os.path.join(output_dir, 'estimatedblocs10V3.pkl')

    with open(output_path_estimblocs10V3, 'wb') as f:
        pickle.dump(best_Z, f)

    print(f'Estimated blocs saved to {output_path_estimblocs10V2}')

    print('Graph estimated')
    g_display.display_graph(graph.Graph(polgraph.adjacency_matrix,best_Z))

else:
    output_path = os.path.join(output_dir, 'estimatedblocs10V2.pkl')
    with open(output_path, 'rb') as f:
        Z = pickle.load(f)

    print('Political graph loaded successfully')
    g_display.display_graph(graph.Graph(polgraph.adjacency_matrix,Z))
    print(Z)

