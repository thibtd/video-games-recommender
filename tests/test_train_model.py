from modules.train_model import create_model, train_model, save_model, load_model
from modules.data_pipeline import get_data, clean_data, feature_engineering
import numpy as np
from modules.utilities import get_vectorizer


def make_data():
    data = get_data()
    _, data = clean_data(data)
    vector_plot = get_vectorizer(1000)
    return feature_engineering(data, vector_plot)


def test_create_model():
    algo = "auto"
    neighbours = 3
    metric = "cosine"
    model_full_param = create_model(neighbors=neighbours, metric=metric, algo=algo)
    model_default = create_model()
    assert model_full_param.metric == metric
    assert model_full_param.n_neighbors == neighbours
    assert model_full_param.algorithm == algo

    assert model_default.metric == "cosine"
    assert model_default.n_neighbors == 11
    assert model_default.algorithm == "auto"


def test_train_model():
    data = make_data()
    neigh = 3
    model = create_model(neighbors=neigh)
    vg_distances, vg_indices = train_model(model, data)
    assert vg_distances.shape == (data.shape[0], neigh)
    assert vg_indices.shape == (data.shape[0], neigh)


def test_save_load_model():
    data = make_data()
    model = create_model()
    vg_distances, vg_indices = train_model(model, data)
    save_model(vg_distances, vg_indices)
    dist_loaded, inds_loaded = load_model()
    assert np.array_equal(
        vg_distances, dist_loaded
    ), "Saved and loaded distance data differ"
    assert np.array_equal(
        vg_indices, inds_loaded
    ), "Saved and loaded indices data differ"
