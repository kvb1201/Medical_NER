#!/usr/bin/env python3
"""Report text in src/models/train_and_evaluate.py must be derived from the results."""

from src.models.train_and_evaluate import (
    best_models_by_track,
    describe_empirical_findings,
    format_train_time,
)


def _results(scores):
    """Build a minimal results dict: {model: {track: micro_f1}}."""
    return {
        model: {
            "train_time_sec": 1.0,
            "tracks": {
                track: {"entity_metrics": {"micro": {"f1": f1}}}
                for track, f1 in tracks.items()
            },
        }
        for model, tracks in scores.items()
    }


def test_findings_name_each_track_winner():
    results = _results({
        "memm": {"track_a_bc5cdr_original": 0.77, "track_b_synthetic": 0.95},
        "crf": {"track_a_bc5cdr_original": 0.79, "track_b_synthetic": 0.89},
    })
    assert best_models_by_track(results) == {
        "track_a_bc5cdr_original": ["CRF"],
        "track_b_synthetic": ["MEMM"],
    }
    text = describe_empirical_findings(results)
    assert "No single model" in text
    assert "Track B (Synthetic): **MEMM**" in text


def test_findings_report_a_clean_sweep():
    results = _results({
        "hmm": {"track_a_bc5cdr_original": 0.6, "track_b_synthetic": 0.7},
        "crf": {"track_a_bc5cdr_original": 0.8, "track_b_synthetic": 0.9},
    })
    assert describe_empirical_findings(results).startswith("CRF achieves the highest")


def test_unmeasured_training_time_is_not_invented():
    assert format_train_time(None) == "—"
    assert format_train_time(12.5) == "12.5s"
