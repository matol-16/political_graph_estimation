import numpy as np
import graph

def retrieve_political_data():
    #On commence par extraire des listes des blocs (liste_blog), des partis politiques (parti_politique), la matrice d'adjacence (X) ainsi que les estimateurs des paramètres de la loi multinomiale (pi)
    #et des lois binomiales (mu) qui interviennent dans le modèle. Cela nous permet également de construire progresivement Z.

    with open("blog.txt", 'r', encoding='utf-8') as fichier: #j'ai modifié légèrement le fichier txt, un 'PCF LCR" était ecrit sans le "-" dans la deuxième partie le 165
            contenu = fichier.readlines()[:1]
            contenu = " ".join(contenu)
            # Séparer les éléments par des virgules ou des lignes
            elements = contenu.replace('\n', ' ').split(' ')
            # Supprimer les espaces et les guillemets autour des éléments
            liste_blog = [elm.strip().strip('"') for elm in elements if elm.strip().strip('"') != ""]

    parti_politique = ['Les Verts', 'UMP', 'UDF', 'PS', 'Parti Radical de Gauche', 'PCF - LCR', 'FN - MNR - MPF', 'liberaux', 'Commentateurs Analystes', 'Cap21']

    #On initialise la matrice Z, qui prend les blogs en ligne et les partis politiques en colonne. Elle indique pour chaque blog son parti d'appartenance.


    Z_vrai = [[0]*len(parti_politique) for i in range(len(liste_blog))]

    with open("blog.txt", 'r', encoding='utf-8') as fichier:
            contenu = fichier.readlines()[5 + len(liste_blog):]
            contenu = " ".join(contenu).split("\n")


    for j in range(len(liste_blog)):
        parti = contenu[j].split('" "')
        parti = [i.strip('"').strip() for i in parti]
        Z_vrai[j][parti_politique.index(parti[1])] = 1

    pi = np.array(Z_vrai).sum(axis = 0)/len(Z_vrai)

    with open("blog.txt", "r") as fichier:
        X = []
        lignes = fichier.readlines()[1: len(liste_blog)+1]  # On ignore la première ligne

        for ligne in lignes:
            # Supposer que les valeurs sont séparées par des espaces ou des tabulations
            X.append(ligne.strip().split())


    X = np.array(X)
    #On exclut les 2 premières colonnes, qui sont le nom d'une blog et un espace
    X = X[: , 2:].astype(int)

    n = len(parti_politique)

    #On peut maintenant construire l'estimateur des paramètres mu_(k,l) à partir de la matrice d'adjacence X et de l'appartenance aux blocs
    #décrite dans Z.
    mu = np.zeros((n, n))


    for k in range(n):
        for l in range(n):
                for i in range(len(liste_blog)):
                    mu[k][l] = mu[k][l] + Z_vrai[i][k]*(X@Z_vrai)[i][l]
                if k == l:
                    mu[k][k] = mu[k][k]/(pi[k]*len(Z_vrai)*(pi[k]*len(Z_vrai)-1))
                else :
                    mu[k][l] = mu[k][l]/(pi[k]*len(Z_vrai)*(pi[l]*len(Z_vrai)))
    return graph.Graph(X,Z_vrai), pi, mu