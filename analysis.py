import ast
import os
import math
from collections import defaultdict
from scipy import stats as scipy_stats

RESULTS_DIR = "./stat_results"
BINARY_THRESHOLD = 0.50
ALPHA = 0.05
EXPECTED_MEANS = {
    0:   0.0,    
    25:  25.0,  
    75:  75.0, 
    100: 100.0,
}
PERCENT_LABELS = {
    0:   "Human original (0%)",
    25:  "Lightly AI-edited (25%)",
    75:  "AI-gen, human-edited (75%)",
    100: "Fully AI-generated (100%)",
}


def load_results(results_dir: str) -> dict[int, list[dict]]:
    files = [os.path.join(RESULTS_DIR, file) for file in os.listdir(RESULTS_DIR) if file.endswith(".txt")]

    if not files:
        raise FileNotFoundError(
            f"No files found in '{results_dir}'"
        )

    data = defaultdict(list)

    for filepath in files:
        filename = os.path.splitext(os.path.basename(filepath))[0]  # e.g. "1-0-1"
        parts = filename.split("-")

        if len(parts) != 3:
            print(f"  Skipping '{filename}' — unexpected filename format.")
            continue

        try:
            text_num   = int(parts[0])
            percent_ai = int(parts[1])
            chunk_num  = int(parts[2])
        except ValueError:
            print(f"  Skipping '{filename}' — could not parse integers.")
            continue

        if percent_ai not in EXPECTED_MEANS:
            print(f"  Skipping '{filename}' — percent_ai={percent_ai} not in {list(EXPECTED_MEANS)}.")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            record = ast.literal_eval(f.read())

        raw_score = record.get("ai_score")
        if raw_score is None:
            print(f"  Skipping '{filename}' — no 'ai_score' field found.")
            continue

        score_pct = float(raw_score) * 100
        binary    = 1 if raw_score >= BINARY_THRESHOLD else 0

        # Correct = called human when truly human, or called AI when truly AI
        correct = (binary == 0) if percent_ai < 50 else (binary == 1)

        data[percent_ai].append({
            "chunk_id":  record.get("chunk_id", filename),
            "score_pct": score_pct,
            "binary":    binary,
            "correct":   correct,
        })

    print(f"Loaded {sum(len(v) for v in data.values())} chunks across "
          f"{len(data)} versions.\n")
    
    return dict(data)

def one_proportion_z_test(chunks: list[dict], version_label: str, p0: float = 0.50):
    """
    H₀: p = p0  (detector performs no better than chance)
    Hₐ: p > p0  (detector correctly classifies more than half)
    """
    n       = len(chunks)
    correct = sum(1 for c in chunks if c["correct"])
    p_hat   = correct / n

    # Condition check
    if not (n * p0 >= 10) and (n * (1 - p0) >= 10):
        return None

    # Test statistic
    se = math.sqrt(p0 * (1 - p0) / n)
    z  = (p_hat - p0) / se

    # One-tailed p-value from standard normal
    p_value = 1 - scipy_stats.norm.cdf(z)

    reject = p_value < ALPHA

    return {
        "version": version_label, "n": n, "correct": correct,
        "p_hat": p_hat, "z": z, "p_value": p_value, "reject_H0": reject,
    }

def mean_std(values: list[float]) -> tuple[float, float]:
    n    = len(values)
    xbar = sum(values) / n
    var  = sum((x - xbar) ** 2 for x in values) / (n - 1)
    return xbar, math.sqrt(var)

def one_sample_t_test(chunks: list[dict], version_label: str, mu0: float):
    """
    H₀: μ = mu0  (detector mean matches expected percentage)
    Hₐ: μ ≠ mu0  (two-tailed)
    """
    scores = [c["score_pct"] for c in chunks]
    n      = len(scores)
    xbar, s = mean_std(scores)
    se     = s / math.sqrt(n)
    t      = (xbar - mu0) / se
    df     = n - 1

    # Condition check
    if not n >= 30:
        return None

    p_value = 2 * (1 - scipy_stats.t.cdf(abs(t), df=df))
    reject = p_value < ALPHA

    return {
        "version": version_label, "n": n, "mean": xbar, "std": s,
        "mu0": mu0, "t": t, "df": df, "p_value": p_value, "reject_H0": reject,
    }


def print_summary(z_results: list[dict], t_results: list[dict]):
    print("═" * 65)
    print("SUMMARY TABLE")
    print("═" * 65)

    print("\nOne-Proportion Z-Test (H₀: p = 0.50, Hₐ: p > 0.50)")
    print(f"  {'Version':<25} {'p̂':>6} {'z':>7} {'p-value':>9} {'Reject H₀?':>12}")
    print("  " + "-" * 60)
    for r in z_results:
        print(f"  {r['version']:<25} {r['p_hat']:>6.3f} {r['z']:>7.3f} "
              f"{r['p_value']:>9.4f} {'Yes' if r['reject_H0'] else 'No':>12}")

    print("\nOne-Sample T-Test (H₀: μ = expected %, Hₐ: μ ≠ expected %)")
    print(f"  {'Version':<25} {'x̄ (%)':>7} {'μ₀ (%)':>7} {'t':>7} "
          f"{'p-value':>9} {'Reject H₀?':>12}")
    print("  " + "-" * 70)
    for r in t_results:
        print(f"  {r['version']:<25} {r['mean']:>7.2f} {r['mu0']:>7.1f} "
              f"{r['t']:>7.3f} {r['p_value']:>9.4f} "
              f"{'Yes' if r['reject_H0'] else 'No':>12}")

    print("\n" + "═" * 65)

def main():
    print("=" * 65)
    print("  AI DETECTOR STATISTICAL ANALYSIS")
    print("=" * 65 + "\n")

    data = load_results(RESULTS_DIR)

    z_results = []
    t_results = []

    for percent_ai in sorted(EXPECTED_MEANS):
        if percent_ai not in data:
            print(f"  No data found for percent_ai={percent_ai}, skipping.\n")
            continue

        chunks = data[percent_ai]
        label  = PERCENT_LABELS[percent_ai]
        mu0    = EXPECTED_MEANS[percent_ai]

        z_results.append(one_proportion_z_test(chunks, label))
        t_results.append(one_sample_t_test(chunks, label, mu0))

    print_summary(z_results, t_results)


if __name__ == "__main__":
    main()