#!/usr/bin/env python
from __future__ import annotations

import argparse

from minillm.data import write_clean_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Bersihkan dan deduplikasi dataset lokal")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", default="data/clean/documents.jsonl")
    args = parser.parse_args()
    written, duplicates = write_clean_jsonl(args.input, args.output)
    print(f"Selesai: {written} dokumen ditulis, {duplicates} duplikat dibuang")


if __name__ == "__main__":
    main()
