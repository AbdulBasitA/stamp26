"""Phase 0 smoke: toy end-to-end — synthetic captions -> embeddings -> UMAP -> ToponymyClusterer
-> Toponymy naming against the live local namer -> DataMapPlot offline HTML."""
import warnings

warnings.filterwarnings("ignore")
import numpy as np

themes = {
    "highway-night": "Ego vehicle cruises on a {adj} highway at night, {detail}.",
    "urban-pedestrian": "At a {adj} urban intersection, a pedestrian {detail} crosses ahead of the ego vehicle.",
    "rainy-rearend": "In {adj} rain on a wet road, the lead vehicle brakes hard and the ego vehicle {detail}.",
}
adjs = ["busy", "quiet", "dark", "well-lit", "crowded", "empty", "foggy", "wide", "narrow", "congested",
        "two-lane", "four-lane", "downtown", "suburban", "industrial", "residential", "rural", "elevated",
        "curved", "straight"]
details = ["with light traffic", "suddenly", "while signaling", "at speed", "cautiously", "abruptly",
           "without warning", "near a truck", "beside a bus", "behind a sedan", "in heavy traffic",
           "during rush hour", "near an exit ramp", "by a crosswalk", "under a bridge", "past parked cars",
           "alongside cyclists", "approaching a light", "after merging", "before an off-ramp"]
captions, true_theme = [], []
for theme, tpl in themes.items():
    for i in range(20):
        captions.append(tpl.format(adj=adjs[i], detail=details[i]))
        true_theme.append(theme)

from sentence_transformers import SentenceTransformer

st = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")  # namer owns all 4 GPUs during this test
embs = np.asarray(st.encode(captions, normalize_embeddings=True))
print(f"embeddings: {embs.shape}")

import umap

clusterable = umap.UMAP(n_components=5, n_neighbors=10, min_dist=0.0, metric="cosine",
                        random_state=42).fit_transform(embs)
map2d = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1, metric="cosine",
                  random_state=42).fit_transform(embs)

from toponymy import Toponymy, ToponymyClusterer
from toponymy.llm_wrappers import OpenAINamer

llm = OpenAINamer(api_key="EMPTY", model="namer", base_url="http://localhost:8000/v1")
topic_model = Toponymy(
    llm_wrapper=llm,
    text_embedding_model=st,
    clusterer=ToponymyClusterer(min_clusters=2, min_samples=2, base_min_cluster_size=5),
    object_description="ego-centric dashcam driving clip captions",
    corpus_description="a collection of dashcam driving scenario clips",
)
topic_model.fit(captions, embs, clusterable)

layers = topic_model.cluster_layers_
print(f"layers: {len(layers)}")
for li, layer in enumerate(layers):
    names = sorted(set(layer.topic_name_vector) - {"Unlabelled"})
    print(f"  layer {li}: {len(names)} topics: {names}")

import datamapplot

plot = datamapplot.create_interactive_plot(
    map2d, layers[0].topic_name_vector,
    hover_text=captions,
    title="STAMP26 Phase-0 toy map",
    offline_mode=True, inline_data=True,
)
import os

os.makedirs("artifacts/smoke", exist_ok=True)
plot.save("artifacts/smoke/toy_map.html")
size = os.path.getsize("artifacts/smoke/toy_map.html")
print(f"toy_map.html written: {size/1e6:.1f} MB")
assert size > 1e5
print("TOY_E2E_PASS")
