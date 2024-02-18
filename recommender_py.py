import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


def get_data():
    # Load the data
    df = pd.read_csv('imdb-videogames.csv',index_col='Unnamed: 0')
    return df

def preprocess_data(df):
  df.drop_duplicates(subset=['name'], inplace=True)
  df.dropna(subset=["year"], inplace=True) 
  df.certificate.fillna('Not Rated',inplace=True)
  df.certificate.replace({'K-A':'E', 'EC':'E','GA':'E', 'CE':'T', '12':'T','PG-13':'T','MA-13':'T','G':'E', 'MA-17':'M','PG':'E','TV-MA':'M','TV-14':'T','Unrated':'Not Rated','Passed':'Not Rated','Approved':'Not Rated'}, inplace=True)
  df_2024 = df[df['year']<=2024].reset_index(drop=True)
  df_2024.drop(['url','votes'],axis=1, inplace=True)
  median = df_2024['rating'].median()
  df_2024['rating'].fillna(median, inplace=True)
  return df_2024

def features_engineering(df_2024):
  game_plots = df_2024['plot']
  vectorizer_plot = TfidfVectorizer( max_features=10000, stop_words='english').fit(game_plots)
  game_plots_vectors = vectorizer_plot.transform(game_plots)
  df_2024.drop('plot',axis=1, inplace=True)
  tfidf_matrix_dense = game_plots_vectors.toarray()
  categorical_columns = [name for name in df_2024.columns if df_2024[name].dtype=='O']
  categorical_columns = categorical_columns[1:]
  video_games_df_dummy = pd.get_dummies(data=df_2024, columns=categorical_columns)
  features = video_games_df_dummy.drop('name', axis=1)
  scale = StandardScaler()
  scaled_features = scale.fit_transform(features)
  scaled_features = pd.DataFrame(scaled_features, columns=features.columns)
  df_combined =  pd.concat([scaled_features, pd.DataFrame(tfidf_matrix_dense)], axis=1)
  df_combined.columns = df_combined.columns.astype(str)
  return game_plots, df_combined

def tf_idf_names(df):
   game_names = df.name
   vectorizer_name = TfidfVectorizer( max_features=10000, stop_words='english').fit(game_names)
   game_name_vectors = vectorizer_name.transform(game_names)
   return game_names, vectorizer_name,game_name_vectors

def model(df):
   model = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='auto').fit(df)
   vg_distances, vg_indices = model.kneighbors(df)
   return vg_distances, vg_indices

def save_model(distances, indices):
    np.save('distances.npy', distances)
    np.save('indices.npy', indices)

def load_model():
    distances = np.load('vg_distances.npy')
    indices = np.load('vg_indices.npy')
    return distances, indices

def VideoGameTitleRecommender(video_game_name, game_names, game_name_vectors, vectorizer_name):
    '''
    This function will recommend a game title that has the closest match to the input
    '''
    query_vector = vectorizer_name.transform([video_game_name])
    similarity_scores = cosine_similarity(query_vector, game_name_vectors)

    closest_match_index = similarity_scores.argmax()
    closest_match_game_name = game_names[closest_match_index]

    return closest_match_game_name

  
def VideoGameRecommender(video_game_name, df_2024, vg_distances, vg_indices, game_plots, game_names, game_name_vectors, vectorizer_name):

    video_game_idx = df_2024.query("name == @video_game_name").index

    
    if video_game_idx.empty:
        # If the game entered by the user doesn't exist in the records, the program will recommend a new game similar to the input
        closest_match_game_name = VideoGameTitleRecommender(video_game_name, game_names, game_name_vectors, vectorizer_name)

        print(f"'{video_game_name}' doesn't exist in the records.\n")
        print(f"You may want to try '{closest_match_game_name}', which is the closest match to the input.")
    
    else:
        # Place in a separate dataframe the indices and distances, then sort the record by distance in ascending order       
        vg_combined_dist_idx_df = pd.DataFrame()
        for idx in video_game_idx:
            # Remove from the list any game that shares the same name as the input
            vg_dist_idx_df = pd.concat([pd.DataFrame(vg_indices[idx][1:]), pd.DataFrame(vg_distances[idx][1:])], axis=1)
            vg_combined_dist_idx_df = pd.concat([vg_combined_dist_idx_df, vg_dist_idx_df])

        vg_combined_dist_idx_df = vg_combined_dist_idx_df.set_axis(['Index', 'Distance'], axis=1)
        vg_combined_dist_idx_df = vg_combined_dist_idx_df.reset_index(drop=True)
        vg_combined_dist_idx_df = vg_combined_dist_idx_df.sort_values(by='Distance', ascending=True)

        video_game_list = df_2024.iloc[vg_combined_dist_idx_df['Index']]
        plot_list = game_plots.iloc[vg_combined_dist_idx_df['Index']]

        # Remove any duplicate game names to provide the user with a diverse selection of recommended games
        video_game_list = video_game_list.drop_duplicates(subset=['name'], keep='first')
        
        # Get the first 10 games in the list
        video_game_list = video_game_list.head(10)

        # Get the distance of the games similar to the input
        recommended_distances = np.array(vg_combined_dist_idx_df['Distance'].head(10))


        print(f"Top 10 Recommended Video Games for '{video_game_name}'")
        plot_list = plot_list.reset_index(drop=True)
        video_game_list = video_game_list.reset_index(drop=True)
        recommended_video_game_list = pd.concat([video_game_list,plot_list, 
                                                 pd.DataFrame(recommended_distances, columns=['Similarity_Distance'])], axis=1)

    return recommended_video_game_list

