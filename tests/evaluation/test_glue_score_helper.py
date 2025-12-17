from neobert.glue.train import compute_glue_score


def test_compute_glue_score_averages_pairwise_metrics():
    metrics = {"eval_accuracy": 0.5, "eval_f1": 0.9}
    assert compute_glue_score("mrpc", metrics) == (0.5 + 0.9) / 2


def test_compute_glue_score_handles_combined_score_fallback():
    metrics = {"combined_score": 0.77}
    assert compute_glue_score("stsb", metrics) == 0.77


def test_compute_glue_score_mnli_aliases():
    metrics = {"eval_accuracy": 0.8, "eval_accuracy_mm": 0.6}
    assert compute_glue_score("mnli", metrics) == 0.7
