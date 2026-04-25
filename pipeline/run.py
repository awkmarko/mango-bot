import argparse
import asyncio

from pipeline.collect import collect_all
from pipeline.clean import clean_records
from pipeline.export import export_jsonl, to_training_pairs


async def _run(data_dir: str, output: str) -> None:
    print(f"[1/4] Collecting from DB and '{data_dir}/'...")
    raw = await collect_all(data_dir)
    print(f"      {len(raw)} raw records")

    print("[2/4] Cleaning and deduplicating...")
    cleaned = clean_records(raw)
    dropped = len(raw) - len(cleaned)
    print(f"      {len(cleaned)} records kept, {dropped} dropped")

    print("[3/4] Generating training pairs...")
    pairs = to_training_pairs(cleaned)
    print(f"      {len(pairs)} pairs generated")

    print(f"[4/4] Exporting to '{output}'...")
    count = export_jsonl(pairs, output)
    print(f"      Done. {count} examples written.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mango Bot data pipeline — exports fine-tuning JSONL from DB and local files."
    )
    parser.add_argument(
        "--output",
        default="training_data.jsonl",
        help="Output JSONL file path (default: training_data.jsonl)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory with .txt and .json source files (default: data/)",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.data_dir, args.output))


if __name__ == "__main__":
    main()
