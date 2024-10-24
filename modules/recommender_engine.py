import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def recommend_video_game_name(
    vg_name_input: str, game_names: pd.Series, vectorizer_name: TfidfVectorizer
) -> str:
    """
    This function will recommend a game title that has the closest match to the input
    """
    query_vector = vectorizer_name.transform([vg_name_input])
    game_names_vectors = vectorizer_name.transform(game_names)
    similarity_scores = cosine_similarity(query_vector, game_names_vectors)

    closest_match_index = similarity_scores.argmin()
    closest_match_game_name = game_names[closest_match_index]

    return closest_match_game_name


def recommend_video_games(
    vg_name_input: str,
    data_display: pd.DataFrame,
    vg_distances: np.ndarray,
    vg_indices: np.ndarray,
    vectorizer_name: TfidfVectorizer,
) -> pd.DataFrame:

    vg_inpt_idx = data_display.query("name.str.lower() == @vg_name_input.lower()").index

    if vg_inpt_idx.empty:
        # If the game entered by the user doesn't exist in the records, the program will recommend a new game similar to the input
        game_names = data_display["name"]
        closest_match_game_name = recommend_video_game_name(
            vg_name_input, game_names, vectorizer_name
        )
        out = f"'{vg_name_input}' doesn't exist in the records.\n You may want to try '{closest_match_game_name}', which is the closest match to the input."
    else:
        vg_combined_dist_idx_df = pd.DataFrame()
        for idx in vg_inpt_idx:
            # remove any game that shares the same name as the input
            vg_dist_idx_df = pd.concat(
                [
                    pd.DataFrame(vg_indices[idx][1:]),
                    pd.DataFrame(vg_distances[idx][1:]),
                ],
                axis=1,
            )
            vg_combined_dist_idx_df = pd.concat(
                [vg_combined_dist_idx_df, vg_dist_idx_df]
            )
        vg_combined_dist_idx_df = vg_combined_dist_idx_df.set_axis(
            ["index", "distance"], axis=1
        )
        vg_combined_dist_idx_df = vg_combined_dist_idx_df.sort_values(
            by="distance", ascending=True
        )
        games_recommended = data_display.loc[vg_combined_dist_idx_df["index"]][:10]
        games_recommended["distance"] = vg_combined_dist_idx_df["distance"][:10].values
        out = games_recommended
    return out


def main():
    """print("---------")
    print("Setup")
    print("---------")
    data = get_data()
    url,data_cleaned = clean_data(data)
    data_display = pd.concat([url,data_cleaned],axis=1)
    vector_names= get_vectorizer(100)
    vg_distance, vg_indices = load_model()
    print("---------")
    print("Recommendation")
    print("---------")
    vg_name_input = 'Max Payne'
    recommended_games = recommend_video_games(vg_name_input,data_display,vg_distance,vg_indices,vector_names)
    print(recommended_games.head(10))"""
    

if __name__ == "__main__":
    main()
