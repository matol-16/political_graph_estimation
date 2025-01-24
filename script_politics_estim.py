import g_display
import data_retrieve
import graph
import g_estim

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
    Z_tot = []
    l_v=[] #liste des vraissemblances
    for i in range(5):
        print(f"---------Estimation {i+1}/5-------------")
        result = polgraph.estim_blocs(10, debut_clusters = True, Nmax_glob=50, nmax_ptf=25, emax_pf = 0.001, emax_it=0.001, debug= True, debug_detail = True)
        Z_estim=result[3]
        Z_tot.append(Z_estim)
        l_v.append(result[4])

    max, Z_best_estim = max(l_v), Z_tot[max.index(max(l_v))]
    print("Meilleure vraisemblance : ", max)
    
    output_path_estimblocs10 = os.path.join(output_dir, 'estimatedblocs10.pkl')

    with open(output_path_estimblocs10, 'wb') as f:
        pickle.dump(Z_best_estim, f)

    print(f'Estimated blocs saved to {output_path_estimblocs10}')

    print('Graph estimated')
    g_display.display_graph(graph.Graph(polgraph.adjacency_matrix,Z_estim))
