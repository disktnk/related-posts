## Run

First, make embedding list

```sh
$ uv run markdown_vector.py path/to/markdowns
```

vectors.jsonl will be saved.

Next, make relation graph

```sh
$ uv run relation_graph.py vectors.jsonl
```

then, graph.html will be saved.

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
