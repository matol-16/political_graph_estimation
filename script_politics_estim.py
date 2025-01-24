import g_display
import data_retrieve
import graph

import os
import pickle

display_graph = False

estim_blocs = False


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
    result = graph.estimate_graph(polgraph)
    Z_estim=result[3]

    output_path_estimblocs = os.path.join(output_dir, 'estimatedblocs.pkl')

    with open(output_path_estimblocs, 'wb') as f:
        pickle.dump(Z_estim, f)

    print(f'Estimated blocs saved to {output_path_estimblocs}')

    print('Graph estimated')
    g_display.display_graph(graph.Graph(polgraph.adjacency_matrix,Z_estim))
