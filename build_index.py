"""Build the BM25 search index over HotpotQA's 2017 Wikipedia. Run once, on Kaggle. (T03)

    python build_index.py        (or press Run / Debug on this file)

The index is already built and saved as the private Kaggle dataset hotpotqa-bm25-index, so this script is only
needed if that copy is lost. It takes about 12 minutes on Kaggle and needs more memory than a laptop has.

It works in four steps, and skips any step whose result is already on disk:
  1. Download HotpotQA's processed Wikipedia (1.55 GB) into data/wiki/dump/.
  2. Unpack it: many small .bz2 files, each holding one article per line.
  3. Read every article's first paragraph (about 5.2 million of them) as {title, text}.
  4. Index each paragraph's title and text with BM25, and save the index to data/wiki/bm25_index/ (about 2.8 GB).
Each run is logged in runs/build_index/<date>/.
"""

import bz2
import json
import tarfile
import time
import urllib.request
from pathlib import Path

import bm25s

from retriever import INDEX_DIR, split_into_search_words
from setup_project import PROJECT_ROOT, note, warn, write_run_log

# Settings for this script.
WIKIPEDIA_DUMP_URL = ("https://nlp.stanford.edu/projects/hotpotqa/"
                      "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2")
DUMP_DIR = PROJECT_ROOT / "data" / "wiki" / "dump"
LOGS_DIR = PROJECT_ROOT / "runs" / "build_index"


# --- Step 1: download ---

def download_wikipedia_dump(dump_url: str, archive_path: Path) -> None:
    """Download the Wikipedia dump to archive_path, unless it's already there."""
    if archive_path.exists():
        return
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    # We download into a ".part" file and rename it only once the download has finished.
    # That way, a file with the real name is always a complete download, even if an earlier run was cut off.
    partial_path = archive_path.with_name(archive_path.name + ".part")
    urllib.request.urlretrieve(dump_url, partial_path)
    partial_path.rename(archive_path)


# --- Step 2: unpack ---

def unpack_wikipedia_dump(archive_path: Path, unpacked_dir: Path) -> None:
    """Unpack the downloaded archive into unpacked_dir, unless that's already done."""
    if unpacked_dir.exists():
        return

    # Same idea as the download: unpack into a ".part" folder, and rename it only when every file is out.
    unpacking_dir = unpacked_dir.with_name(unpacked_dir.name + ".part")
    with tarfile.open(archive_path) as archive:
        archive.extractall(unpacking_dir, filter="data")   # "data" refuses files that would land outside the folder
    unpacking_dir.rename(unpacked_dir)


# --- Step 3: read ---

def read_wikipedia_paragraphs(unpacked_dir: Path) -> list[dict]:
    """Read every article in the unpacked dump as a paragraph {title, text}."""
    # We sort the files, so the paragraphs are always read, and indexed, in the same order.
    dump_files = sorted(unpacked_dir.rglob("*.bz2"))

    paragraphs = []
    for dump_file in dump_files:
        with bz2.open(dump_file, "rt") as article_lines:
            for article_line in article_lines:
                article = json.loads(article_line)
                # The dump stores a paragraph as a list of sentences, so we join them back into one text.
                paragraph_text = "".join(article["text"])
                paragraphs.append({"title": article["title"], "text": paragraph_text})
    return paragraphs


# --- Step 4: index ---

def build_index(paragraphs: list[dict], index_dir: Path) -> None:
    """Index every paragraph's title and text with BM25, and save the index together with the paragraphs."""
    # We index the title and the text together, so a search can match words in either.
    texts_to_index = [paragraph["title"] + " " + paragraph["text"] for paragraph in paragraphs]
    words_of_each_paragraph = split_into_search_words(texts_to_index, as_word_numbers=True)

    index = bm25s.BM25()
    index.index(words_of_each_paragraph, show_progress=False)

    # Saving the paragraphs with the index lets a search hand back each paragraph's title and text.
    index.save(index_dir, corpus=paragraphs)


# --- All steps, in order ---

def build_wikipedia_index(log_lines: list[str]) -> None:
    """Download, unpack, read and index Wikipedia, skipping whatever is already done, and log each step."""
    started_at = time.time()
    if (INDEX_DIR / "params.index.json").exists():
        note(log_lines, f"The index already exists in {INDEX_DIR.relative_to(PROJECT_ROOT)}, so there is nothing to build.")
        return

    archive_path = DUMP_DIR / Path(WIKIPEDIA_DUMP_URL).name
    note(log_lines, f"1/4 Downloading {WIKIPEDIA_DUMP_URL} (1.55 GB), unless it's already there ...")
    download_wikipedia_dump(WIKIPEDIA_DUMP_URL, archive_path)
    note(log_lines, f"    {archive_path.relative_to(PROJECT_ROOT)}: {archive_path.stat().st_size:,} bytes")

    unpacked_dir = DUMP_DIR / "unpacked"
    note(log_lines, "2/4 Unpacking, unless that's already done ...")
    unpack_wikipedia_dump(archive_path, unpacked_dir)

    note(log_lines, "3/4 Reading the paragraphs ...")
    paragraphs = read_wikipedia_paragraphs(unpacked_dir)
    note(log_lines, f"    {len(paragraphs):,} paragraphs")

    note(log_lines, "4/4 Building and saving the index (about 12 minutes on Kaggle) ...")
    build_index(paragraphs, INDEX_DIR)
    note(log_lines, f"    saved to {INDEX_DIR.relative_to(PROJECT_ROOT)}")

    elapsed_minutes = (time.time() - started_at) / 60
    note(log_lines, f"Done in {elapsed_minutes:.1f} minutes. Next: `python retriever.py` checks search quality.")


if __name__ == "__main__":
    build_log_lines = []
    try:
        build_wikipedia_index(build_log_lines)
    except BaseException as build_error:
        # Here we record a failed run in the log too, then let the error show as usual.
        warn(build_log_lines, f"building the index failed: {build_error!r}")
        raise
    finally:
        print("Log saved in", write_run_log(build_log_lines, LOGS_DIR, "build_index"))
