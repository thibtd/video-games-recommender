from modules.data_pipeline import get_data, clean_data, feature_engineering, tf_idf_names
import pandas as pd


def test_get_data():
    data = get_data()
    assert data.shape == (20803, 16),"the original data has changed"

def test_clean_data():
    data = get_data()
    url, data_clean = clean_data(data)
    assert data_clean.shape == (10944, 14), "data clean shape is not correct"
    assert url.shape == (10944,), "url shape is not correct"

def test_feature_engineering():
    data = get_data()
    _, data_clean = clean_data(data)
    data_engineered = feature_engineering(data_clean)
    assert data_engineered.shape == (10944, 1017),"data engineered shape is not correct"
    assert all(isinstance(col, str) for col in data_engineered.columns), "Not all column names are strings"

def test_tf_idf_names():
    data = get_data()
    _, data_clean = clean_data(data)
    names_tfidf = tf_idf_names(data_clean['name'])
    assert names_tfidf.shape == (10944, 1000),"names tfidf shape is not correct"