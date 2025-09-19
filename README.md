## Run

First, make embedding list

```sh
$ uv run embed_markdown.py path/to/markdowns
```

`vectors.jsonl` will be saved.

Next, make relation graph from `vectors.jsonl`

```sh
$ uv run create_similarity_graph.py --mode save_graph
```

then, `graph.html` will be saved.

## Development

```sh
$ uv sync
$ uv sync --group dev
```

format check

```sh
$ uv run block .
$ uv run ruff check . --fix
$ uv run mypy .
```
