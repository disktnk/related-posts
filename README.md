## Run

```sh
$ uv run markdown_vector.py example.md
```

vectors.json will be saved.

## Development

```sh
$ uv sync
$ uv sync --group dev
```

format check

```sh
$ uv run block .
$ uv run ruff check . --fix
```
