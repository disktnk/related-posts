import argparse
import json
import re
from pathlib import Path

import mistune
import tqdm
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


def preprocess_markdown(md_content: str) -> str:
    # convert {{< highlight ... >}} ... {{< /highlight >}} to ```` ... ```
    md_content = re.sub(r"{{<\s*highlight\s+\w+\s*>}}", "```", md_content)
    md_content = re.sub(r"{{</\s*highlight\s*>}}", "```", md_content)
    # remove {{% ... %}} and {{< ... >}}
    md_content = re.sub(r"{{[%<].*?[%>]}}", "", md_content, flags=re.DOTALL)
    # if title:  in frontmatter, convert to <title> ... </title> and insert right after frontmatter
    frontmatter_match = re.match(r"---\n(.*?)\n---\n", md_content, flags=re.DOTALL)
    if frontmatter_match:
        frontmatter = frontmatter_match.group(1)
        title_match = re.search(r"title:\s*(.+)", frontmatter)
        if title_match:
            title = title_match.group(1).strip().strip("'").strip('"')
            title_tag = f"<title>{title}</title>\n"
            md_content = re.sub(
                r"---\n.*?\n---\n",
                f"---\n{frontmatter}\n---\n{title_tag}",
                md_content,
                count=1,
                flags=re.DOTALL,
            )
    # remove frontmatter
    md_content = re.sub(r"---\n.*?\n---\n", "", md_content, count=1, flags=re.DOTALL)
    return md_content


def structured_text_from_markdown(md_content: str) -> str:
    md_content = preprocess_markdown(md_content)
    html_content = mistune.html(md_content)

    structured = []
    # extract <title> ... </title>
    title_match = re.search(r"<title>(.*?)</title>", str(html_content), flags=re.DOTALL)
    if title_match:
        title = title_match.group(1).strip()
        structured.append(f"[TITLE] {title}")
    # extract headings
    for level in range(1, 7):
        headings = re.findall(
            rf"<h{level}>(.*?)</h{level}>", str(html_content), flags=re.DOTALL
        )
        structured += [f"[HEADING{level}] {h}" for h in headings]

    # convert to plain text using BeautifulSoup
    soup = BeautifulSoup(str(html_content), "html.parser")
    plain_text = soup.get_text()
    structured += [f"[BODY] {plain_text}"]

    return "\n".join(structured)


def convert_structured_text(target: Path) -> list[dict]:
    """Convert markdown file or all markdown files in a directory to structured text.

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

    def get_structured_text(target: Path) -> str:
        with open(target, "r", encoding="utf-8") as f:
            md_content = f.read()
        # if set "draft: true" in frontmatter, ignore the file
        frontmatter_match = re.match(r"---\n(.*?)\n---\n", md_content, flags=re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            draft_match = re.search(r"draft:\s*true", frontmatter)
            if draft_match:
                return ""
        return structured_text_from_markdown(md_content)

    def extract_title(structured_text: str) -> str:
        # get title from [TITLE]
        title_match = re.search(r"\[TITLE\]\s*(.*)", structured_text)
        return title_match.group(1).strip() if title_match else ""

    result_values = []
    print("Convert markdown files to structured text...")
    if target.is_dir():
        md_files = list(target.rglob("*.md"))
        with tqdm.tqdm(md_files) as pbar:
            for md_file in pbar:
                structured_text = get_structured_text(md_file)
                if not structured_text:
                    continue

                title = extract_title(structured_text)

                pbar.set_postfix({"title": title})

                result_values.append(
                    {
                        "filepath": str(md_file.resolve()),
                        "title": title,
                        "text": structured_text,
                    }
                )
    else:
        structured_text = get_structured_text(target)
        if not structured_text:
            return result_values

        title = extract_title(structured_text)

        result_values.append(
            {
                "filepath": str(target.resolve()),
                "title": title,
                "text": structured_text,
            }
        )

    return result_values


def get_embedding(texts: list[dict], model_name: str) -> list[dict]:
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
    vector_list = []
    print(f"Make embeddings using {model_name}...")

    model = SentenceTransformer(model_name, trust_remote_code=True)

    text_contents = [text_dict["text"] for text_dict in texts]
    filepaths = [text_dict["filepath"] for text_dict in texts]
    titles = [text_dict["title"] for text_dict in texts]

    embeddings = model.encode(
        text_contents, show_progress_bar=True, batch_size=8, convert_to_numpy=True
    )

    for i, embedding in enumerate(embeddings):
        vector_list.append(
            {
                "filepath": filepaths[i],
                "title": titles[i],
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
    target = Path(args.target)
    plain_texts = convert_structured_text(target)

    vectors = get_embedding(plain_texts, args.embedding_model)

    save_jsonl(vectors, args.output)
