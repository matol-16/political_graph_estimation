
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import graph
import g_display
import matplotlib.pyplot as plt

import data_processing.data_retrieve as data_retrieve
import pickle
import numpy as np


display_graph = False

estim_blocs = True #estimer les blocs pour un nombre de blocs fixé à 10

searchblocs = False

if not os.path.exists('C:/output'):
    os.makedirs('C:/output')
depart = np.load(os.path.join('C:','output','Z_10.npy'))

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


    Nmax_glob = 1
    nmax_ptf = 2
    emax = 0.001

    if searchblocs:
        icl, best_Z_V2, best_K = polgraph.estim_findK(Nmax_glob, nmax_ptf, emax, depart = depart)   
        print("le nombre de cluster optimal est ", best_K)
        estimated_graph = best_Z_V2

        #plot l'évolution de l'ICL
        plt.plot(range(1, 14), icl, marker='o', color='blue', label="ICL (Critère d'information)")
        plt.axvline(x=best_K, color='red', linestyle='--', label=f"Meilleur K ({best_K})")
        plt.title("Évolution du critère ICL en fonction du nombre de clusters")
        plt.xlabel("Nombre de clusters (K)")
        plt.ylabel("Valeur du critère ICL")
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.show()

    else:
        _,_,_, Z_10_V2, m= polgraph.estim_kk(10, Nmax_glob, nmax_ptf, emax, depart=depart)
        best_K=10
        estimated_graph = Z_10_V2


    
    g_display.display_graph(polgraph)


    output_path_estimblocs10V3 = os.path.join(output_dir, 'estimatedblocs10V3.pkl')

    with open(output_path_estimblocs10V3, 'wb') as f:
        #pickle.dump(best_Z_V2, f)
        pickle.dump(Z_10_V2, f)

    print(f'Estimated blocs saved to {output_path_estimblocs10V3}')

    print('Graph estimated')

    g_display.display_graph(graph.Graph(polgraph.adjacency_matrix,estimated_graph))


else:
    output_path = os.path.join(output_dir, 'estimatedblocs10V2.pkl')
    with open(output_path, 'rb') as f:
        Z = pickle.load(f)

    print('Political graph loaded successfully')
    g_display.display_graph(graph.Graph(polgraph.adjacency_matrix,Z))
    print(Z)

