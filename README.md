# Political graph estimaiton
Ce repository est dédié à un projet académique à l'école Polytechnique. Il permet d'analyser les interactions politiques sur internet via un graphe aléatoire d'Erdös-Renyi.

Le code est organisé de la façon suivante: 

- data_retrieve permet d'extraire les données du fichier blog.txt sous forme de matrice d'adjacence et d'appartenance de blocs

- le fichier graph contient la classe Graph qui code les fonctions permettant de simuler et d'estimer des graphes

- estim_test contient les tests des estimations pour les beta-graphes (cf readme)

- estim_polgraph contient la fonction qui implémente l'estimation des blocs du graphe des partis politiques sur le web

- g_display permet d'afficher un graphe à partir de sa matrice d'adjacence

- script_betas_tests permet d'exécuter les tests de la fonction estim_test
