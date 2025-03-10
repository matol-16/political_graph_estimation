import numpy as np
import pandas as pd

def extract_matrix_from_excel(file_path, output_path):
    df = pd.read_csv(file_path, header=None)
    matrix = df.to_numpy()
    print(np.shape(matrix))
    np.save(output_path, matrix)
    return 

extract_matrix_from_excel('Z_10.csv', 'Z_10.npy')
extract_matrix_from_excel('best_Z.csv', 'best_Z.npy')
