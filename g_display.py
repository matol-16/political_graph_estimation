
import networkx as nx
import numpy as np

import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from scipy.integrate import quad
import graph
import plotly.graph_objects as go

def display_graph(G, take_blocs_estim=False, seed=20):
    madj = G.adjacency_matrix
    if take_blocs_estim:
        mblocs= G.estimated_blocs
    else:
        mblocs = G.blocs
    A = [list(mblocs[i]).index(1)+1 for i in range(len(mblocs))]
    A = np.array(A).T

    G = nx.from_numpy_array(madj)
    pos = nx.spring_layout(G, k=0.2)
    x_nodes = [pos[node][0] for node in G.nodes()]
    y_nodes = [pos[node][1] for node in G.nodes()]
    node_labels = [str(node) for node in G.nodes()]

    # 4. Catégoriser les nœuds par bloc selon la matrice Z
    block_colors = {1: 'green', 2: 'blue', 3: 'lightblue', 4: 'red', 5: 'pink', 6: 'orange', 7: 'white', 8: 'yellow', 9: 'grey', 10: 'cyan'}  # Choisir les couleurs pour chaque bloc
    node_colors = [block_colors[A[i]] for i in range(len(A))]  # Assigner les couleurs aux nœuds
    block_colors = {i:block_colors[i] for i in block_colors if i<=len(mblocs)}

    # 5. Visualiser le graphe avec NetworkX et Matplotlib
    edges_x = []
    edges_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edges_x.append(x0)
        edges_x.append(x1)
        edges_y.append(y0)
        edges_y.append(y1)

    # Créer un objet plotly pour les nœuds
    node_trace = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',  # Choisir une palette de couleurs
            size=10,
            color=node_colors,  # Affecter les couleurs aux nœuds
            line_width=2
        )
    )

    # Créer un objet plotly pour les arêtes
    edge_trace = go.Scatter(
        x=edges_x,
        y=edges_y,
        line=dict(width=0.5, color='gray'),
        hoverinfo='none',
        mode='lines'
    )


    if take_blocs_estim:
        tit="Graphe des blocs estimés avec Légende"
    else:
        tit ="Graphe des blocs réels"
    # Créer le graphique
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        title =tit,
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False),
                        plot_bgcolor='white',
                        margin=dict(l=0, r=0, b=0, t=0),

                    ))

    # Afficher le graphe
    fig.show()


def display_graphon(mu,pi):
    #j'ai crée des copies des tableaux comme ça quand je faisais mes tests je modofiais pas les vrai mu et pi à chaque fois
    mu_2 = np.array([[mu[i][j] for j in range(n)] for i in range(n)])
    pi_2 = np.array([pi[i] for i in range(n)])

    block_colors = {1: 'green', 2: 'blue', 3: 'lightblue', 4: 'red', 5: 'pink', 6: 'darkred', 7: 'brown', 8: 'yellow', 9: 'grey', 10: 'cyan'}  # Choisir les couleurs pour chaque bloc

    sorted = np.argsort(mu_2@pi_2)  # Assurer une somme croissante pour les intégrales

    pi_2 = [pi_2[i] for i in sorted]

    # Calcul des bornes cumulées
    cumu_pi = np.cumsum(pi_2)
    cumu_pi = np.insert(cumu_pi, 0, 0)  # Ajouter 0 au début

    # Définir les paramètres de la grille
    u = np.linspace(0, 1, 100)
    v = np.linspace(0, 1, 100)
    u, v = np.meshgrid(u, v)
    face_colors = np.zeros_like(u, dtype=object)

    z = np.zeros_like(u)
    for l1 in range(1, n + 1):
        for l2 in range(1, n + 1):
            mask = (
                (u >= cumu_pi[l1 - 1]) & (u < cumu_pi[l1]) &
                (v >= cumu_pi[l2 - 1]) & (v < cumu_pi[l2])
            )
            z[mask] = mu_2[sorted[l1 - 1]][sorted[ l2 - 1]]
            mask = (
                (u >= cumu_pi[l1 - 1]) & (u < cumu_pi[l1]) &
                (v >= cumu_pi[l2 - 1]-0.02) & (v < cumu_pi[l2])
            )
            if l1 == l2:
                face_colors[mask] = block_colors[sorted[l1-1]+1]
            else:
                face_colors[mask] = "white"


    # Créer une figure et une projection 3D
    fig = plt.pyplot.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')


    # Appliquer les couleurs personnalisées
    cmap = ListedColormap([block_colors[i] for i in range(1, len(block_colors) + 1)])
    surf = ax.plot_surface(u, v, z, facecolors=face_colors, edgecolor='none')
    patches = [
        mpatches.Patch(color=block_colors[i], label=f"{parti_politique[i - 1]}")
        for i in range(1, len(parti_politique) + 1)
    ]

    plt.pyplot.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.2, 1), title="Partis politiques")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


    plt.pyplot.show()


def display_graphon_integral(mu,pi):
    def integrand(v, u, pi, mu):
        return graph.Graph.graphon_croiss(pi, mu, u, v)[0]
    #on cacule l'intégrande avec un pas de 0,002 car la fonction W est très sensible pour les petites valeurs
    rez = []
    for u in np.linspace(0, 1, 500):
        integral_result, error =quad(integrand, 0, 1, args=(u, pi, mu))
        rez.append(integral_result)
    plt.plot(np.linspace(0, 1, 500), rez, label="$\int_0^1 W_{croissante}(u, v) \, dv$")
    plt.title("Graphe de $\int_0^1 W_{croissante}(u, v) \, dv$ pour $u \in [0,1]$")
    plt.xlabel('u')
    plt.legend()
    plt.show()
    return rez