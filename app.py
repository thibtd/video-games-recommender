import pandas as pd
import numpy as np
import streamlit as st
from modules.utilities import get_vectorizer
from modules.recommender_engine import recommend_video_games
from modules.data_pipeline import get_data, clean_data, tf_idf_names
from modules.train_model import load_model
import time
import os



def stream_data(data: pd.DataFrame):
    """
    This function will stream the data to the user
    """
    i = 1
    for index, row in data.iterrows():
        string_final = f"{i}. **{row['name']}** \n\r Rating: *{row['rating']}*, Genres: *{row['genres']}* \n\n {row['plot']} \n\n  Check it out on IMDb: *{row['url']}* "
        for word in string_final.split(" "):
            yield word + " " 
            time.sleep(0.02)
        i +=1
        yield " \n\r"

if __name__ == "__main__":
    st.set_page_config(
         page_title="Home",
         page_icon="💰")
    
    # Load the data
    data = get_data()
    url,data = clean_data(data)
    data_display = pd.concat([url,data],axis=1)
    vector_names= get_vectorizer(100)
    names_vectorized = tf_idf_names(data_display['name'],vector_names)
    vg_distance, vg_indices = load_model()

    st.title("New games finder")
    game_input = st.text_input("Enter the name of a game you like and we'll give you similar ones!", "Call of Duty: World at War")
    recommandation = ""
    if st.button(" recommend!"):
        recommandation = recommend_video_games(game_input, data_display, vg_distance, vg_indices, vector_names)
        if isinstance(recommandation, str):
            st.write(recommandation)
        else:
            columns_genre = ['Action','Adventure','Comedy','Crime','Family','Fantasy','Mystery','Sci-Fi','Thriller']
            recommandation['genres']= recommandation.apply(lambda row: ', '.join([col for col in columns_genre if row[col]]), axis=1)
            st.write_stream(stream_data(recommandation))
