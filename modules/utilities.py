from sklearn.feature_extraction.text import TfidfVectorizer 


def get_vectorizer(max_feat:int=100,stop_words:str='english')->TfidfVectorizer:
    return TfidfVectorizer( max_features=max_feat, stop_words=stop_words)
