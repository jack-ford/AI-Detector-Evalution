from __future__ import annotations
import ast
import csv
from pathlib import Path
from typing import List, Dict, Tuple

RESULTS_DIR = Path("detection_results")
OUTPUT_DIR = Path("csv_results")
RAW_CSV = OUTPUT_DIR / "export_raw.csv"
STATISTICS_TXT = OUTPUT_DIR / "export_statistics.csv"
DETECTION_CSV = OUTPUT_DIR / "export_detections.csv"

BINARY_THRESHOLD = 0.50

def parse_chunk_id(chunk_id: str) -> Tuple[int, int, int]:
    parts = chunk_id.split("-")
    try:
        a = int(parts[0])
    except Exception:
        a = 10**9
    try:
        b = int(parts[1])
    except Exception:
        b = 10**9
    try:
        c = int(parts[2])
    except Exception:
        c = 10**9
    return a, b, c


def load_records(results_dir: Path) -> List[Dict]:
    """Read all .txt files in results_dir and return a sorted list of records.

    Each record: {
      'chunk_id', 'text_id', 'ai_amount', 'chunk', 'word_count', 'ai_score', 'text'
    }
    """
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    records: List[Dict] = []
    for p in sorted(results_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".txt":
            continue
        try:
            raw = p.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            data = ast.literal_eval(raw)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        chunk_id = data.get("chunk_id") or p.stem
        ai_score = data.get("ai_score")
        text = data.get("text", "")
        word_count = data.get("word_count")
        if word_count is None:
            word_count = len(text.split()) if text else 0
        try:
            ai_score = float(ai_score)
        except Exception:
            continue

        t_num, ai_pct, chunk_num = parse_chunk_id(chunk_id)

        records.append({
            "chunk_id": chunk_id,
            "text_id": t_num,
            "ai_amount": ai_pct,
            "chunk": chunk_num,
            "word_count": int(word_count),
            "ai_score": ai_score,
            "text": text,
        })

    records.sort(key=lambda r: (r["text_id"], r["ai_amount"], r["chunk"]))
    return records


def write_raw_csv(records: List[Dict], outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["chunk_id", "word_count", "text"]
    with outpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "chunk_id": r["chunk_id"],
                "word_count": r["word_count"],
                "text": r["text"],
            })


def write_detection_csv(records: List[Dict], outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "chunk_id",
        "ai_amount",
        "ai_pct",
        "predicted_label",
        "expected_label",
        "correct",
    ]
    with outpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            expected = "Human" if r["ai_amount"] < 50 else "AI"
            predicted = "AI" if r["ai_score"] >= BINARY_THRESHOLD else "Human"
            correct = "Yes" if (predicted == expected) else "No"
            writer.writerow({
                "chunk_id": r["chunk_id"],
                "ai_amount": r["ai_amount"],
                "ai_pct": f"{r['ai_score'] * 100:.2f}",
                "predicted_label": predicted,
                "expected_label": expected,
                "correct": correct,
            })


def write_statistics_csv(outpath: Path):
    """Write the summary as a CSV suitable for Google Sheets import.

    The CSV contains two tables separated by a blank row:
      - One-Proportion Z-Test: Version, p_hat, z, p_value, Reject H0?
      - One-Sample T-Test:    Version, x_bar (%), mu0 (%), t, p_value, Reject H0?
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        import analysis

        grouped = analysis.load_results(analysis.RESULTS_DIR)

        with outpath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # One-Proportion Z-Test table
            writer.writerow(["One-Proportion Z-Test (H₀: p = 0.50, Hₐ: p > 0.50)"])
            writer.writerow(["Version", "p̂", "z", "p-value", "Reject H0?"])
            for pct in sorted(analysis.EXPECTED_MEANS):
                if pct not in grouped:
                    continue
                chunks = grouped[pct]
                z_res = analysis.one_proportion_z_test(chunks, analysis.VERSION_LABELS[pct])
                version = analysis.VERSION_LABELS[pct]
                p_hat = z_res.get("p_hat", 0)
                z = z_res.get("z", 0)
                pval = z_res.get("p_value", 1)
                reject = "Yes" if z_res.get("reject_H0") else "No"
                writer.writerow([
                    version,
                    f"{p_hat:.3f}",
                    f"{z:.3f}",
                    f"{pval:.4f}",
                    reject,
                ])

            writer.writerow([])

            # One-Sample T-Test table
            writer.writerow(["One-Sample T-Test (H₀: μ = expected %, Hₐ: μ ≠ expected %)"])
            writer.writerow(["Version", "x̄ (%)", "μ₀ (%)", "t", "p-value", "Reject H0?"])
            for pct in sorted(analysis.EXPECTED_MEANS):
                if pct not in grouped:
                    continue
                chunks = grouped[pct]
                t_res = analysis.one_sample_t_test(chunks, analysis.VERSION_LABELS[pct], analysis.EXPECTED_MEANS[pct])
                version = analysis.VERSION_LABELS[pct]
                mean = t_res.get("mean", 0)
                mu0 = t_res.get("mu0", analysis.EXPECTED_MEANS.get(pct, 0))
                tstat = t_res.get("t", 0)
                pval = t_res.get("p_value", 1)
                reject = "Yes" if t_res.get("reject_H0") else "No"
                writer.writerow([
                    version,
                    f"{mean:.2f}",
                    f"{mu0:.1f}",
                    f"{tstat:.3f}",
                    f"{pval:.4f}",
                    reject,
                ])

    except Exception:
        with outpath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["STATISTICS could not be generated: ensure analysis.py and SciPy are available."])


def main():
    records = load_records(RESULTS_DIR)
    write_raw_csv(records, RAW_CSV)
    write_detection_csv(records, DETECTION_CSV)
    write_statistics_csv(STATISTICS_TXT)
    print(f"Wrote exports to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
