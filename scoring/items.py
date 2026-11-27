"""Item lists in the exact row order ThalamusDB uses for each multimodal column.

Index i of each list is the item whose id is i in the ThalamusDB scores table.
Keep in sync with tdb/nldbs.py (both read through these functions).
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def craigslist_furniture():
    return pd.read_csv(DATA / "craigslist/furnitures.csv")


def craigslist_imgs():
    return pd.read_csv(DATA / "craigslist/imgs.csv")


def youtube_table():
    """YouTube rows restricted to clips whose AudioCaps audio is available."""
    df = pd.read_csv(DATA / "youtube/youtube.csv")
    audio_dir = DATA / "youtube/audio"
    have = {p.stem for p in audio_dir.glob("*.flac")}
    df = df[df["youtube_id"].isin(have)].drop_duplicates("youtube_id").reset_index(drop=True)
    df["audio_path"] = [str(audio_dir / f"{y}.flac") for y in df["youtube_id"]]
    return df


def netflix_movies():
    return pd.read_csv(DATA / "netflix/movies_with_reviews.csv")


def craigslist_images():
    return [str(DATA / p) for p in craigslist_imgs()["img"]]


def craigslist_titles():
    return craigslist_furniture()["title"].fillna("").tolist()


def youtube_descriptions():
    return youtube_table()["description"].fillna("").tolist()


def youtube_audio():
    return youtube_table()["audio_path"].tolist()


def netflix_reviews():
    return netflix_movies()["featured_review"].fillna("").tolist()
