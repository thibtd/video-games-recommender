# Pipeline for data retrieval, preprocessing, and feature engineering.

import pandas as pd
import numpy as np
import kagglehub
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def get_data() -> pd.DataFrame:
    """
    Retrieve the dataset from Kaggle and load it as a pandas DataFrame.
    input: None
    output: pd.DataFrame - dataset
    """
    # Download latest version
    path: str = kagglehub.dataset_download("muhammadadiltalay/imdb-video-games")
    # Load the dataset
    data: pd.DataFrame = pd.read_csv(
        path + "/imdb-videogames.csv", index_col="Unnamed: 0"
    )
    return data


def clean_data(df: pd.DataFrame) -> tuple:
    """
    Clean the dataset by removing missing values, duplicates, and filling missing values.
    input: df: pd.DataFrame - dataset to be cleaned
    output: tuple - urls: pd.Series - urls of the games, data: pd.DataFrame - cleaned dataset
    """
    # remove rows with missing years.
    data = df.dropna(subset=["year"])
    # remove duplicated names
    data = data.drop_duplicates(subset=["name"])
    # removes rows with 'Add a plot' in the plot column
    data = data[data["plot"] != "Add a Plot"]
    # fill missing values in certificate column with 'Not Rated'
    data["certificate"] = data.certificate.fillna("Not Rated")
    data["certificate"] = data.certificate.replace(
        {
            "K-A": "E",
            "EC": "E",
            "GA": "E",
            "CE": "T",
            "12": "T",
            "PG-13": "T",
            "MA-13": "T",
            "G": "E",
            "MA-17": "M",
            "PG": "E",
            "TV-MA": "M",
            "TV-14": "T",
            "Unrated": "Not Rated",
            "Passed": "Not Rated",
            "Approved": "Not Rated",
        }
    )
    # change year to integer
    data["year"] = data["year"].astype(int)
    # keep only released games (i.e. year <= 2024)
    data = data[data["year"] <= 2024].reset_index(drop=True)
    # fill missing rating with median value.
    median = data["rating"].median()
    data["rating"] = data["rating"].fillna(median)
    # drop votes and urls columns
    urls = data["url"]  # save urls for later
    data = data.drop(columns=["votes", "url"], axis=1)

    return urls, data


def feature_engineering(
    df: pd.DataFrame, vectorizer_plot: TfidfVectorizer
) -> pd.DataFrame:
    """
    Perform feature engineering by creating dummy variables for categorical columns and standardizing the data.
    input: df: pd.DataFrame - cleaned dataset
    output: pd.DataFrame - standardized dataset with dummy variables for categorical columns
    """
    # create a tf-idf matrix for the plot column
    games_plots = df["plot"]
    vectorizer_plot = vectorizer_plot.fit(games_plots)
    game_plots_vectors = vectorizer_plot.transform(games_plots)
    data = df.drop("plot", axis=1)
    tfidf_matrix_dense = game_plots_vectors.toarray()

    # make categorical columns into dummy variables
    categorical_columns = [name for name in data.columns if data[name].dtype == "O"]
    categorical_columns = categorical_columns[1:]
    video_games_df_dummy = pd.get_dummies(data=data, columns=categorical_columns)
    # standardize the data, excluding tf-idf matrix
    features = video_games_df_dummy.drop("name", axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_features = pd.DataFrame(scaled_features, columns=features.columns)
    # combine the scaled features with the tf-idf matrix
    df_combined = pd.concat([scaled_features, pd.DataFrame(tfidf_matrix_dense)], axis=1)
    df_combined.columns = df_combined.columns.astype(str)
    return df_combined


def tf_idf_names(names: pd.Series, game_names_vectors: TfidfVectorizer) -> np.ndarray:
    """
    Create a tf-idf matrix for the game names.
    input: names: pd.Series - names of the games
    output: np.ndarray - tf-idf matrix for the game names
    """
    vectorizer_name = game_names_vectors.fit(names)
    game_names_vectors = vectorizer_name.transform(names)
    return game_names_vectors


def main():
    """print("Data Pipeline")
    print("----------------")
    print("Get Data")
    print("----------------")
    data = get_data()
    print(data.shape)
    print("----------------")
    print('clean data')
    print("----------------")
    url, data_clean = clean_data(data)
    print(data_clean.shape)
    data_display = pd.concat([url,data_clean],axis=1)
    print(data_display.shape)
    print("----------------")
    print("Feature Engineering")
    print("----------------")
    vector_plot = get_vectorizer(1000)
    data_engineered = feature_engineering(data_clean,vector_plot)
    print(data_engineered.shape)
    print("----------------")
    print("TF-IDF Names")
    print("----------------")
    vector_names = get_vectorizer(500)
    names_tfidf = tf_idf_names(data_clean['name'],vector_names)
    print(names_tfidf.shape)"""
    


if __name__ == "__main__":
    main()
