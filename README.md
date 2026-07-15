"""
preprocess.py
-------------
Loading + cleaning helpers shared by the content-based and
collaborative-filtering recommenders.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLUMNS = [
    "danceability", "energy", "valence",
    "acousticness", "tempo", "loudness",
]


def load_songs(path: str = "data/songs.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="song_id").reset_index(drop=True)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].mean())
    return df


def load_ratings(path: str = "data/ratings.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.dropna(subset=["user_id", "song_id", "rating"])


def scale_features(df: pd.DataFrame, columns=FEATURE_COLUMNS) -> pd.DataFrame:
    """Min-max scale audio features to [0, 1] so no single feature
    (e.g. tempo, which ranges 60-200) dominates the similarity score."""
    scaled = df.copy()
    scaler = MinMaxScaler()
    scaled[columns] = scaler.fit_transform(scaled[columns])
    return scaled
