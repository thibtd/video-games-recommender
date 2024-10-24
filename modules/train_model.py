from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd


def create_model(neighbors:int=11, metric:str="cosine", algo:str="auto")->NearestNeighbors:
    """
    This function creates a NearestNeighbors model
    inputs: neighbours:int - the number of neighbours to consider
            metric:str - the distance metric to use
            algo:str - the algorithm to use
    output: model:NearestNeighbors - the model object
    """
    model = NearestNeighbors(n_neighbors=neighbors, metric=metric, algorithm=algo)
    return model

def train_model(model:NearestNeighbors, df:pd.DataFrame)->tuple:
    """
    This function trains the model on the data
    inputs: model:NearestNeighbors - the model object
            df:pd.DataFrame - the data to train on
    output: vg_distances:np.ndarray - the distances of the neighbours
            vg_indices:np.ndarray - the indices of the neighbours
    """

    model.fit(df)
    vg_distances, vg_indices = model.kneighbors(df)
    return vg_distances, vg_indices

def save_model(distances:np.ndarray, indices:np.ndarray)->None:
    """

    """
    np.save("models/vg_distances.npy", distances)
    np.save("models/vg_indices.npy", indices)

def load_model()->tuple:
    distances = np.load("models/vg_distances.npy")
    indices = np.load("models/vg_indices.npy")
    return distances, indices
