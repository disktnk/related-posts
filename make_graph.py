import argparse
import json

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors


def calc_knn(vectors: list[dict], k: int = 5, threshold: float = 0.5) -> list[dict]:
    # vectors: list or array shape (n, d)
    X = np.vstack([v["vector"] for v in vectors])  # shape (n,d)

    # L2 正規化（cosine = dot）
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)

    k = 5  # 各ノードの近傍数
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

    if G.number_of_nodes() == 0:
        print("No nodes to plot.")
        return

    # layout: spring_layout (can be slow for large graphs)
    pos = nx.spring_layout(G, k=0.3, seed=42)

    # edge traces
    edge_x, edge_y = [], []
    edge_widths = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_widths.append(d.get("weight", 0.5))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="rgba(100,100,200,0.6)", width=1),
        hoverinfo="none",
    )

    # node traces
    node_x, node_y, node_text = [], [], []
    node_size = []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        title = G.nodes[n].get("title", "")
        deg = G.degree[n]
        node_text.append(f"{n}: {title} (deg={deg})")
        node_size.append(6 + deg * 2)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            size=node_size, color="skyblue", line=dict(width=1, color="DarkSlateGrey")
        ),
        text=node_text,
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Embedding similarity graph (k={k}, threshold={threshold})",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
        width=900,
    )
    fig.write_html("graph.html", include_plotlyjs="cdn", full_html=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relation graph from embedding vectors."
    )
    parser.add_argument(
        "input",
        type=str,
        help="The target JSONL file",
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

    calc_knn(vectors, k=5, threshold=0.5)
