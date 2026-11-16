import numpy as np
import pytest

torch = pytest.importorskip("torch")

from statproject.detectors import sampling_discrepancy


def test_sampling_discrepancy_by_hand():
    # One position, two equally likely tokens: every outcome has the same
    # log-probability, so the actual token sits exactly at the expectation and
    # the variance is zero. A second, lopsided position supplies the spread.
    probs = torch.tensor([[0.5, 0.5], [0.9, 0.1]])
    tokens = torch.tensor([0, 1])
    curvature, mean_log_likelihood = sampling_discrepancy(probs.log(), tokens)

    log_p = np.log([0.9, 0.1])
    expected = 0.9 * log_p[0] + 0.1 * log_p[1]
    variance = 0.9 * log_p[0] ** 2 + 0.1 * log_p[1] ** 2 - expected**2
    assert curvature == pytest.approx((log_p[1] - expected) / np.sqrt(variance), rel=1e-5)
    assert mean_log_likelihood == pytest.approx((np.log(0.5) + log_p[1]) / 2, rel=1e-5)


def test_text_sampled_from_the_model_scores_like_a_standard_normal():
    # If tokens really are drawn from the model's own distributions, the
    # curvature is a standardised sum and should average 0 with unit spread.
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(400, 50, generator=generator)
    probs = logits.softmax(-1)
    curvatures = [
        sampling_discrepancy(logits, torch.multinomial(probs, 1, generator=generator).squeeze(-1))[0]
        for _ in range(300)
    ]
    assert np.mean(curvatures) == pytest.approx(0, abs=0.2)
    assert np.std(curvatures) == pytest.approx(1, abs=0.15)
