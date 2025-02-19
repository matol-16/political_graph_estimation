# Political graph estimaiton (encore en production)
Ce dépôt est dédié à un projet académique à l'école Polytechnique. Il permet d'analyser les interactions politiques sur internet via un graphe aléatoire d'Erdös-Renyi.

Le code est organisé de la façon suivante: 

- demo_simple_graph permet d'utiliser l'algorithme d'estimation sur un graphe simple et d'afficher les résultats

- data_retrieve permet d'extraire les données du fichier blog.txt sous forme de matrice d'adjacence et d'appartenance de blocs

- le fichier graph contient la classe Graph qui code les fonctions permettant de simuler et d'estimer des graphes

- estim_test contient les tests des estimations pour les beta-graphes (cf le rapport). C'est un premier test simple de fonctionnement de l'algorithme. Le fichier permet également d'effectuer les tests directement

- estim_polgraph contient la fonction qui implémente l'estimation des blocs du graphe des partis politiques sur le web

- g_display permet d'afficher un graphe à partir de sa matrice d'adjacence

Ce dépôt contient également, dans le dossier **rapports*, des documents décrivant le projet:

- Le rapport final de ce projet, qui détaille les mathématiques sous-jacentes, les algorithmes employés et nos résultas

- Un notebook avec le code exécuté qui permet de visualiser directement nos résultats avec le code associé.

- Le support de la soutenance de ce projet



