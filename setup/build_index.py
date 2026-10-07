"""Build the BM25 search index over HotpotQA's 2017 Wikipedia. Run once, on Kaggle (needs ~12 GB RAM). (T03)

    python setup/build_index.py        (or press Debug on this file)

Downloads the Wikipedia dump (1.55 GB) into data/wiki/dump/, unpacks it, and saves the index to
data/wiki/bm25_index/ (~2.8 GB). Safe to re-run after an interruption: the download resumes where it stopped,
and a half-finished unpack starts over. The index is already built and saved as the Kaggle dataset
hotpotqa-bm25-index, so this is only needed if that copy is lost.
"""

import bz2
import http.client
import json
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path

import bm25s

from pipeline.config import INDEX_DIR, WIKI_DUMP_DIR
from pipeline.retriever import tokenize

WIKI_DUMP_URL = ("https://nlp.stanford.edu/projects/hotpotqa/"
                 "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2")
DOWNLOAD_ATTEMPTS = 20
SECONDS_BETWEEN_ATTEMPTS = 10
ONE_MEGABYTE = 1024 * 1024


# --- Download, resuming after interruptions ---

def download_remaining_bytes(url: str, partial_path: Path) -> int | None:
    """One download attempt: fetch the bytes partial_path doesn't have yet and add them to it.

    Returns the full file size the server promised (None if it didn't say).
    """
    bytes_already_downloaded = partial_path.stat().st_size if partial_path.exists() else 0
    # Here we ask the server for only the missing bytes ("Range: bytes=<first missing byte>-").
    range_request = urllib.request.Request(url, headers={"Range": f"bytes={bytes_already_downloaded}-"})
    try:
        with urllib.request.urlopen(range_request, timeout=60) as response:
            # Status 206 = the server sent only the rest, so we add it on. Status 200 = it sent the whole file,
            # so we start the .part file over.
            resuming = response.status == 206
            size_being_sent = response.headers.get("Content-Length")
            with open(partial_path, "ab" if resuming else "wb") as partial_file:
                shutil.copyfileobj(response, partial_file, length=ONE_MEGABYTE)
    except urllib.error.HTTPError as http_error:
        # Status 416 = "nothing left to send": the .part file already holds every byte.
        if http_error.code == 416:
            return bytes_already_downloaded
        raise
    if size_being_sent is None:
        return None
    return int(size_being_sent) + (bytes_already_downloaded if resuming else 0)


def download_with_resume(url: str, destination: Path, max_attempts: int = DOWNLOAD_ATTEMPTS,
                         seconds_between_attempts: int = SECONDS_BETWEEN_ATTEMPTS) -> None:
    """Download url to destination, retrying and resuming until the file is complete.

    The bytes go into "<destination>.part" and the file only gets its real name once complete,
    so a file named `destination` is always a whole download.
    """
    partial_path = destination.with_name(destination.name + ".part")
    for attempt_number in range(1, max_attempts + 1):
        try:
            expected_size = download_remaining_bytes(url, partial_path)
        except (OSError, http.client.HTTPException) as download_error:
            # Here we wait and try again after a dropped connection; the next attempt resumes the .part file.
            print(f"Download interrupted ({download_error}); attempt {attempt_number} of {max_attempts}.")
            time.sleep(seconds_between_attempts)
            continue

        downloaded_size = partial_path.stat().st_size
        # Here we accept the file only if it has every byte the server promised.
        if expected_size is None or downloaded_size == expected_size:
            partial_path.rename(destination)
            return
        print(f"Download incomplete ({downloaded_size:,} of {expected_size:,} bytes); "
              f"attempt {attempt_number} of {max_attempts}.")
    raise RuntimeError(f"Could not finish downloading {url}. Run this script again to keep resuming {partial_path}.")


# --- Unpack ---

def unpack_archive(archive_path: Path, unpacked_dir: Path) -> None:
    """Unpack the archive into unpacked_dir, all or nothing.

    Files go into "<unpacked_dir>.part" first, which is renamed only when unpacking finishes,
    so a folder named unpacked_dir always holds every file.
    """
    unpacking_dir = unpacked_dir.with_name(unpacked_dir.name + ".part")
    shutil.rmtree(unpacking_dir, ignore_errors=True)   # here we throw away a half-finished earlier unpack
    with tarfile.open(archive_path) as archive:
        archive.extractall(unpacking_dir, filter="data")
    unpacking_dir.rename(unpacked_dir)


def download_and_unpack_dump(dump_url: str, dump_dir: Path) -> list[Path]:
    """Make sure the dump is downloaded and unpacked in dump_dir (skipping finished steps); return its files in order."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dump_dir / Path(dump_url).name
    if not archive_path.exists():
        print(f"Downloading {dump_url} (1.55 GB) ...")
        download_with_resume(dump_url, archive_path)

    unpacked_dir = dump_dir / "unpacked"
    if not unpacked_dir.exists():
        print("Unpacking ...")
        unpack_archive(archive_path, unpacked_dir)
    # Here we sort the file list, so the index is built in the same order every time.
    return sorted(unpacked_dir.rglob("*.bz2"))


# --- Read and index ---

def read_wiki_paragraphs(dump_files: Iterable[Path]) -> Iterator[dict]:
    """Read the dump files (bz2, one JSON article per line) and hand out one {title, text} paragraph at a time."""
    for dump_file in dump_files:
        with bz2.open(dump_file, "rt") as dump_lines:
            for line in dump_lines:
                article = json.loads(line)
                # Here we join the article's list of sentences back into one paragraph.
                yield {"title": article["title"], "text": "".join(article["text"])}


def build_index(paragraphs: Iterable[dict], index_dir: Path, show_progress: bool = False) -> None:
    """Index every paragraph's title + text with BM25, and save the index together with the paragraphs."""
    corpus = list(paragraphs)   # the index is saved with the paragraphs, so they all stay in memory
    # Here we index title and text together, so a search can match words in either.
    paragraph_words = tokenize([f"{paragraph['title']} {paragraph['text']}" for paragraph in corpus],
                               return_word_ids=True, show_progress=show_progress)
    bm25_index = bm25s.BM25()
    bm25_index.index(paragraph_words, show_progress=show_progress)
    bm25_index.save(index_dir, corpus=corpus)


if __name__ == "__main__":
    dump_files = download_and_unpack_dump(WIKI_DUMP_URL, WIKI_DUMP_DIR)
    print(f"Reading {len(dump_files)} dump files and building the index ...")
    build_index(read_wiki_paragraphs(dump_files), INDEX_DIR, show_progress=True)
    print(f"Saved the BM25 index to {INDEX_DIR}")
