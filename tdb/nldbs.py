"""Port of the authors' nldbs.py: same tables, columns, and foreign keys, but NL columns are
backed by cached Gemma scores (data/scores/*.parquet) instead of live CLIP/BERT/audio models.
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scoring"))
import items  # noqa: E402

from datatype import DataType  # noqa: E402
from nlfilter import CachedProcessor  # noqa: E402
from schema import NLDatabase, NLTable, NLColumn  # noqa: E402

SCORES = ROOT / "data/scores"
dfs = {}

# Column -> {predicate: score-cache task name}. Predicates follow the authors' benchmark.py.
CACHE_TASKS = {
    ("craigslist", "img"): {
        p: f"craigslist_img_{p.replace(' ', '_')}"
        for p in ["blue chair", "leather sofa", "wooden chair", "dining table", "sofa", "table", "wooden"]
    },
    ("craigslist", "title_u"): {"wood": "craigslist_title_wood"},
    ("youtubeaudios", "audio"): {"voices": "youtube_audio_voices"},
    ("youtubeaudios", "description_u"): {"cooking": "youtube_description_cooking"},
    ("netflix", "featured_review_u"): {"positive": "netflix_review_positive"},
}


def processor(db, col, nr_total):
    caches = {}
    for pred, task in CACHE_TASKS[(db, col)].items():
        path = SCORES / f"{task}.parquet"
        if not path.exists():
            continue  # Predicate not scored (e.g. optional Fig. 11 predicates).
        df = pd.read_parquet(path).sort_values("idx")
        assert len(df) == nr_total and (df["idx"].to_numpy() == np.arange(nr_total)).all(), f"incomplete cache {task}"
        caches[pred] = (df["score"].to_numpy(), df["latency"].to_numpy())
    return CachedProcessor(caches, nr_total)


def get_df_by_name(table_name):
    if table_name not in dfs:
        if table_name == "furniture":
            df = items.craigslist_furniture()
            df["title_u"] = np.arange(len(df))
        elif table_name == "furniture_imgs":
            df = items.craigslist_imgs()
        elif table_name == "youtube":
            df = items.youtube_table().drop(columns=["audio_path"])
            df["description"] = df["description"].fillna("")
            df["likes"] = df["likes"].fillna(0)
            df["description_u"] = np.arange(len(df))
            df["audio"] = np.arange(len(df))
        elif table_name == "movies":
            df = items.netflix_movies().drop("review_label", axis=1, errors="ignore")
            df["featured_review_u"] = np.arange(len(df))
        elif table_name == "ratings":
            df = duckdb.sql(f"SELECT * FROM read_csv_auto('{ROOT / 'data/netflix/ratings.csv'}')").df()
            df.columns = [c.lower() for c in df.columns]
        else:
            raise ValueError(f"Wrong table name: {table_name}")
        dfs[table_name] = df
    return dfs[table_name]


def get_nldb_by_name(dbname):
    return {"craigslist": craigslist, "youtubeaudios": youtubeaudios, "netflix": netflix}[dbname]()


def add_count_columns(relationships, name2df):
    for table_dim, col_dim, table_fact, col_fact in relationships:
        df_counts = name2df[table_fact][col_fact].value_counts().reset_index()
        col_count = col_dim + "_c"
        df_counts.columns = [col_dim, col_count]
        merged = name2df[table_dim].merge(df_counts, on=col_dim, how="left")
        merged[col_count] = merged[col_count].fillna(0)
        name2df[table_dim] = merged
    return name2df


def _connect(name2df):
    con = duckdb.connect(database=":memory:")
    for name, df in name2df.items():
        con.register("tmp_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
    return con


def craigslist():
    df_furniture = get_df_by_name("furniture")
    df_images = get_df_by_name("furniture_imgs").copy()
    img_processor = processor("craigslist", "img", len(df_images))
    df_images["img"] = np.arange(len(df_images))
    name2df = {"furniture": df_furniture, "images": df_images}
    relationships = [("furniture", "aid", "images", "aid")]
    name2df = add_count_columns(relationships, name2df)
    con = _connect(name2df)
    con.execute("CREATE UNIQUE INDEX furniture_aid_idx ON furniture (aid)")
    title_processor = processor("craigslist", "title_u", len(df_furniture))

    furniture = NLTable("furniture")
    furniture.add(
        NLColumn("aid", DataType.NUM),
        NLColumn("time", DataType.NUM),
        NLColumn("neighborhood", DataType.TEXT),
        NLColumn("title", DataType.TEXT),
        NLColumn("title_u", DataType.NUM, title_processor),
        NLColumn("url", DataType.TEXT),
        NLColumn("price", DataType.NUM),
    )
    images = NLTable("images")
    images.add(NLColumn("img", DataType.IMG, img_processor), NLColumn("aid", DataType.NUM))
    nldb = NLDatabase("craigslist", con)
    nldb.add(furniture, images)
    nldb.add_relationships(*relationships)
    for table_dim, col_dim, _, _ in relationships:
        nldb.tables[table_dim].add(NLColumn(f"{col_dim}_c", DataType.NUM))
    nldb.init_info()
    return nldb


def youtubeaudios():
    df = get_df_by_name("youtube")
    con = _connect({"youtube": df})
    youtube = NLTable("youtube")
    youtube.add(
        NLColumn("youtube_id", DataType.TEXT),
        NLColumn("audio", DataType.AUDIO, processor("youtubeaudios", "audio", len(df))),
        NLColumn("title", DataType.TEXT),
        NLColumn("category", DataType.TEXT),
        NLColumn("viewcount", DataType.NUM),
        NLColumn("author", DataType.TEXT),
        NLColumn("length", DataType.NUM),
        NLColumn("duration", DataType.TEXT),
        NLColumn("likes", DataType.NUM),
        NLColumn("description", DataType.TEXT),
        NLColumn("description_u", DataType.NUM, processor("youtubeaudios", "description_u", len(df))),
    )
    nldb = NLDatabase("youtubeaudios", con)
    nldb.add(youtube)
    nldb.init_info()
    return nldb


def netflix():
    df_movie = get_df_by_name("movies")
    df_rating = get_df_by_name("ratings")
    name2df = {"movies": df_movie, "ratings": df_rating}
    relationships = [("movies", "movieid", "ratings", "movieid")]
    name2df = add_count_columns(relationships, name2df)
    con = _connect(name2df)
    movies = NLTable("movies")
    movies.add(
        NLColumn("movieid", DataType.NUM),
        NLColumn("releaseyear", DataType.NUM),
        NLColumn("movietitle", DataType.TEXT),
        NLColumn("featured_review", DataType.TEXT),
        NLColumn("featured_review_u", DataType.NUM, processor("netflix", "featured_review_u", len(df_movie))),
    )
    ratings = NLTable("ratings")
    ratings.add(
        NLColumn("custid", DataType.NUM),
        NLColumn("rating", DataType.NUM),
        NLColumn("date", DataType.TEXT),
        NLColumn("movieid", DataType.NUM),
    )
    nldb = NLDatabase("netflix", con)
    nldb.add(movies, ratings)
    nldb.add_relationships(*relationships)
    for table_dim, col_dim, _, _ in relationships:
        nldb.tables[table_dim].add(NLColumn(f"{col_dim}_c", DataType.NUM))
    nldb.init_info()
    return nldb
