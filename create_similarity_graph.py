import argparse
import json
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from pyvis.network import Network  # type: ignore
from sklearn.neighbors import NearestNeighbors  # type: ignore


def save_graph(G: nx.Graph, output_file: str) -> None:
    """Create a network graph visualization and return node information.

    Args:
        G: Input graph
        output_file: HTML output file path

    Returns:
        List of node dictionaries with position and other attributes
    """
    if G.number_of_nodes() == 0:
        print("No nodes to plot.")
        return

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


def make_graph(
    vectors: List[Dict[str, Any]],
    X_norm: np.ndarray,
    indices: np.ndarray,
    sims: np.ndarray,
    threshold: float,
    output_file: str,
) -> None:
    G: nx.Graph = nx.Graph()
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

    save_graph(G, output_file)


def save_similarity(
    vectors: List[Dict[str, Any]],
    X_norm: np.ndarray,
    indices: np.ndarray,
    sims: np.ndarray,
    threshold: float,
    output_file: str,
) -> None:
    # Create similarity list, own filepath, target filepath, similarity
    similarity_list = []
    for i in range(X_norm.shape[0]):
        for j_idx, sim in zip(indices[i, 1:], sims[i, 1:]):  # skip self
            sim = float(sim)
            j = int(j_idx)
            if sim >= threshold:
                similarity_list.append(
                    {
                        "source": vectors[i]["filepath"],
                        "target": vectors[j]["filepath"],
                        "similarity": sim,
                    }
                )

    # Save similarity list to JSON, similarity.json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(similarity_list, f, ensure_ascii=False, indent=2)


def calc_knn(
    vectors: List[Dict[str, Any]], k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # vectors: list or array shape (n, d)
    X = np.vstack([v["vector"] for v in vectors])  # shape (n,d)

    # L2 normalization (cosine = dot)
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)

    # Use the k parameter instead of hardcoded 5
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X_norm)
    distances, indices = nn.kneighbors(X_norm, return_distance=True)
    # distances = 0..2 (cosine distance); similarity = 1 - distance
    sims = 1.0 - distances

    return X_norm, indices, sims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relation graph from embedding vectors."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="vectors.jsonl",
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
    # choice from 'save_graph', 'save_similarity', 'both'
    parser.add_argument(
        "--mode",
        type=str,
        choices=["save_graph", "save_similarity", "both"],
        default="save_similarity",
        help="The mode of operation",
    )
    parser.add_argument(
        "--output",
        type=str,
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

    vectors: List[Dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            vec = json.loads(line)
            vectors.append(vec)

    X_norm, indices, sims = calc_knn(vectors, args.k)
    if args.mode in ("save_similarity", "both"):
        output_file = f"{args.output}.json" if args.output else "similarity.json"
        save_similarity(vectors, X_norm, indices, sims, args.threshold, output_file)
    if args.mode in ("save_graph", "both"):
        output_file = f"{args.output}.html" if args.output else "graph.html"
        make_graph(vectors, X_norm, indices, sims, args.threshold, output_file)
