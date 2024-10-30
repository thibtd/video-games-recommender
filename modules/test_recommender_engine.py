import pytest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from modules.recommender_engine import recommend_video_games


@pytest.fixture(scope="module", autouse=True)
def sample_data():
    return pd.DataFrame(
        {
            "name": [
                "Max Payne",
                "GTA V",
                "Red Dead Redemption",
                "The Witcher 3",
                "Cyberpunk 2077",
            ],
            "rating": [9.0, 9.5, 9.7, 9.8, 8.5],
            "genres": [
                "Action",
                "Action, Adventure",
                "Action, Adventure",
                "RPG",
                "RPG",
            ],
            "plot": ["Plot 1", "Plot 2", "Plot 3", "Plot 4", "Plot 5"],
            "url": [
                "http://example.com/game1",
                "http://example.com/game2",
                "http://example.com/game3",
                "http://example.com/game4",
                "http://example.com/game5",
            ],
        }
    )


@pytest.fixture(scope="module", autouse=True)
def sample_distances():
    return np.array(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [0.1, 0.0, 0.2, 0.3, 0.4],
            [0.2, 0.1, 0.0, 0.3, 0.4],
            [0.3, 0.2, 0.1, 0.0, 0.4],
            [0.4, 0.3, 0.2, 0.1, 0.0],
        ]
    )


@pytest.fixture(scope="module", autouse=True)
def sample_indices():
    return np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 2, 3, 4],
            [2, 1, 0, 3, 4],
            [3, 2, 1, 0, 4],
            [4, 3, 2, 1, 0],
        ]
    )


@pytest.fixture(scope="module", autouse=True)
def sample_vectorizer():
    vectorizer = TfidfVectorizer()
    vectorizer.fit(
        ["Max Payne", "GTA V", "Red Dead Redemption", "The Witcher 3", "Cyberpunk 2077"]
    )
    return vectorizer


def test_recommend_video_games_existing_game():
    vg_name_input = "Max Payne"
    result = recommend_video_games(
        vg_name_input, sample_data, sample_distances, sample_indices, sample_vectorizer
    )
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "distance" in result.columns


def test_recommend_video_games_non_existing_game():
    vg_name_input = "Unknown Game"
    result = recommend_video_games(
        vg_name_input, sample_data, sample_distances, sample_indices, sample_vectorizer
    )
    assert isinstance(result, str)
