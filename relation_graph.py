import argparse
import json

import networkx as nx
import numpy as np
from pyvis.network import Network
from sklearn.neighbors import NearestNeighbors


def make_graph(G: nx.Graph, output_file: str) -> list[dict]:

    if G.number_of_nodes() == 0:
        print("No nodes to plot.")
        return []

    nt = Network(height="700px", width="1000px", notebook=False)

    for n, attr in G.nodes(data=True):
        nt.add_node(n, label=attr.get("title", ""), title=attr.get("title", ""))

    for u, v, d in G.edges(data=True):
        weight = d.get("weight", 0.5)
        # adjust edge width based on weight
        width = 0.5 + 4.5 * weight
        nt.add_edge(u, v, value=width, title=f"Similarity: {weight:.4f}")

    nt.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=150)

    nt.save_graph(output_file)


def calc_knn(vectors: list[dict], k: int, threshold: float) -> list[dict]:
    # vectors: list or array shape (n, d)
    X = np.vstack([v["vector"] for v in vectors])  # shape (n,d)

    # L2 normalization (cosine = dot)
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)

    k = 5  # Number of neighbors for each node
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X_norm)
    distances, indices = nn.kneighbors(X_norm, return_distance=True)
    # distances = 0..2 (cosine distance); similarity = 1 - distance
    sims = 1.0 - distances

    G = nx.Graph()
    for i in range(X_norm.shape[0]):
        G.add_node(i, title=vectors[i].get("title", ""))

        for j_idx, sim in zip(indices[i, 1:], sims[i, 1:]):  # skip self
            sim = float(sim)
            j = int(j_idx)
            if sim >= threshold:
                # add undirected edge with weight = similarity
                if G.has_edge(i, j):
                    # keep max similarity if duplicated
                    if G[i][j]["weight"] < sim:
                        G[i][j]["weight"] = sim
                else:
                    G.add_edge(i, j, weight=sim)

    return G


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relation graph from embedding vectors."
    )
    parser.add_argument(
        "input",
        type=str,
        help="The target JSONL file",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="The number of neighbors",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="The similarity threshold",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="graph.html",
        help="The output HTML file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_file = args.input

    """
    Input vector dict structure is:
    ```
    [
        {
            "filepath": "absolute path of the file",
            "title": "title of the file",
            "vector": [numpy array of the embedding vector]
        }, ...
    ]
    ```
    """

    vectors = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            vec = json.loads(line)
            vectors.append(vec)

    G = calc_knn(vectors, args.k, threshold=args.threshold)
    make_graph(G, args.output)
