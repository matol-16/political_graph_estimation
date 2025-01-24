import graph
import estim_test
import g_display
import data_retrieve

import os
import pickle


#Permet de tester pour les betas
score = estim_test.test_betas()

estim_betas = False


output_dir = os.path.join('C:', 'output')
output_path_betas = os.path.join(output_dir, 'graphique_betas.pkl')


if estim_betas:
    image = estim_test.test_betas()

    os.makedirs(output_dir, exist_ok=True)

    with open(output_path_betas, 'wb') as f:
        pickle.dump(image, f)

    print(f'Political graph saved to {output_path_betas}')




