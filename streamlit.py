import pandas as pd
import numpy as np
import streamlit as st
import recommender_py as rec


if __name__ == "__main__":
    # st.set_page_config(
    #     page_title="Home",
    #     page_icon="💰",
    # )
    df = rec.get_data()
    df_preproc = rec.preprocess_data(df)
    plots, df_combined = rec.features_engineering(df_preproc)
    game_names, vectorizer_name, game_name_vectors = rec.tf_idf_names(df_preproc)
    vg_distances, vg_indices = rec.load_model()
    
    st.title('Recommendation de jeux videos')
    game = st.text_input('entrer nom du jeu','Call of Duty: World at War')
    recommandation = ''
    if st.button(' recommend!'):
        recommandation = rec.VideoGameRecommender(game,df_preproc,vg_distances,vg_indices,plots,game_names,game_name_vectors,vectorizer_name)
    st.write(recommandation)
