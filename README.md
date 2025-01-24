# Political graph estimaiton
This repository is dedicated to an academic project at école Polytechnique deamed to analyzing political interactions on internet through a Erdös-Renyi random graph model

The code is organized as follows:

- data_retrieve permet d'extraire les données du fichier blog.txt sous forme de matrice d'adjacence et d'appartenance de blocs

- le fichier graph contient la classe Graph qui code les fonctions permettant de simuler et d'estimer des graphes

- estim_test contient les tests des estimations pour les beta-graphes (cf readme)

- estim_polgraph contient la fonction qui implémente l'estimation des blocs du graphe des partis politiques sur le web

- g_display permet d'afficher un graphe à partir de sa matrice d'adjacence

- script_betas_tests permet d'exécuter les tests de la fonction estim_test
