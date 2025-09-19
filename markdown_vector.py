import argparse
import hashlib
import json
import re
from pathlib import Path

import tqdm
from sentence_transformers import SentenceTransformer


def preprocess_markdown(md_content: str) -> str:
    # if title:  in frontmatter, convert to # ... and insert right after frontmatter
    frontmatter_match = re.match(r"---\n(.*?)\n---\n", md_content, flags=re.DOTALL)
    if frontmatter_match:
        # if include "draft: true", skip this file
        if re.search(r"draft:\s*true", frontmatter_match.group(1)):
            return ""
        frontmatter = frontmatter_match.group(1)
        title_match = re.search(r"title:\s*(.+)", frontmatter)
        if title_match:
            title = title_match.group(1).strip().strip("'").strip('"')
            title_tag = f"# {title}\n"
            md_content = re.sub(
                r"---\n.*?\n---\n",
                f"---\n{frontmatter}\n---\n{title_tag}",
                md_content,
                count=1,
                flags=re.DOTALL,
            )
    # convert {{< highlight ... >}} ... {{< /highlight >}} to ```` ... ```
    md_content = re.sub(r"{{<\s*highlight\s+\w+\s*>}}", "```", md_content)
    md_content = re.sub(r"{{</\s*highlight\s*>}}", "```", md_content)
    # convert {{< relref link >}} to link
    md_content = re.sub(r"{{<\s*relref\s+(.+?)\s*>}}", r"\1", md_content)
    # remove {{% ... %}} and {{< ... >}}
    md_content = re.sub(r"{{[%<].*?[%>]}}", "", md_content, flags=re.DOTALL)
    # remove frontmatter
    md_content = re.sub(r"---\n.*?\n---\n", "", md_content, count=1, flags=re.DOTALL)
    return md_content


def convert_clarified_markdown(target: Path) -> list[dict]:
    """Convert markdown file or all markdown files in a directory to clarified text.

    Return dict structure is:
    ```
    [
        {
            "filepath": "absolute path of the file",
            "title": "title of the file",
            "text": "structured text"
        }, ...
    ]
    ```
    """

    def extract_title(structured_text: str) -> str:
        # get title from # ...
        title_match = re.search(r"#\s*(.*)", structured_text)
        return title_match.group(1).strip() if title_match else ""

    def get_hash(text: str) -> str:
        encoded = text.encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    result_values = []
    print("Convert markdown files to structured text...")
    if target.is_dir():
        md_files = list(target.rglob("*.md"))
        with tqdm.tqdm(md_files) as pbar:
            for md_file in pbar:
                with open(md_file, "r", encoding="utf-8") as f:
                    md_content = f.read()
                md_content = preprocess_markdown(md_content)
                if not md_content:
                    continue

                title = extract_title(md_content)
                content_hash = get_hash(md_content)

                pbar.set_postfix({"title": title})

                result_values.append(
                    {
                        "filepath": str(md_file.resolve()),
                        "title": title,
                        "hash": content_hash,
                        "text": md_content,
                    }
                )
    else:
        with open(target, "r", encoding="utf-8") as f:
            md_content = f.read()
        md_content = preprocess_markdown(md_content)
        if not md_content:
            return result_values

        title = extract_title(md_content)
        content_hash = get_hash(md_content)

        result_values.append(
            {
                "filepath": str(target.resolve()),
                "title": title,
                "hash": content_hash,
                "text": md_content,
            }
        )

    return result_values


def get_embedding(texts: list[dict], model_name: str, prev: Path) -> list[dict]:
    """Get embeddings for a list of texts using SentenceTransformers.

    Return dict structure is:
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
    print(f"Make embeddings using {model_name}...")

    vector_list = []
    prev_vector_map = {}
    if prev and prev.is_file():
        with open(prev, "r", encoding="utf-8") as f:
            for line in f:
                vec = json.loads(line)
                prev_vector_map[vec["filepath"]] = vec

    text_contents = []
    filepaths = []
    titles = []
    hashes = []
    for text_dict in texts:
        if text_dict["filepath"] in prev_vector_map:
            if prev_vector_map[text_dict["filepath"]]["hash"] == text_dict["hash"]:
                vector_list.append(prev_vector_map[text_dict["filepath"]])
                continue
        text_contents.append(text_dict["text"])
        filepaths.append(text_dict["filepath"])
        titles.append(text_dict["title"])
        hashes.append(text_dict["hash"])

    if not text_contents:
        print("No new or updated files to embed.")
        return vector_list

    model = SentenceTransformer(model_name, trust_remote_code=True)
    embeddings = model.encode(
        text_contents, show_progress_bar=True, batch_size=8, convert_to_numpy=True
    )

    for i, embedding in enumerate(embeddings):
        vector_list.append(
            {
                "filepath": filepaths[i],
                "title": titles[i],
                "hash": hashes[i],
                "vector": embedding.tolist(),
            }
        )

    return vector_list


def save_jsonl(vectors: list[dict], output_file: str) -> None:
    print(f"Saving embedded vectors to JSONL, {output_file} ...")
    with open(output_file, "w", encoding="utf-8") as f:
        for vec in vectors:
            f.write(json.dumps(vec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vectorize markdown files.")
    parser.add_argument(
        "target", type=str, help="The target file or directory to vectorize."
    )
    parser.add_argument(
        "--prev",
        type=str,
        default="vectors.jsonl",
        help="The previous vectors file to compare.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="pfnet/plamo-embedding-1b",
        help="The embedding model to use.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vectors.jsonl",
        help="The output file to save the embeddings.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plain_texts = convert_clarified_markdown(Path(args.target))

    vectors = get_embedding(plain_texts, args.embedding_model, Path(args.prev))

    save_jsonl(vectors, args.output)
