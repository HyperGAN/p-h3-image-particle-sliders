"""Analytical controls for the base protocol, independent of native H3."""

import copy
import math

import pytest
import torch

from ph3_sliders.evaluation import aggregate_rows, response_statistics

KEY = ("cup", 17, "neutral")


def rows(student, teacher=(1., 1e-6), neutral=(0., 0.), key=KEY):
    result = []
    for position, (s, t, n) in enumerate(zip(student, teacher, neutral)):
        result.append(dict(zip(("scene", "seed", "path"), key), position=position,
            **response_statistics(*(torch.tensor([v], dtype=torch.float64) for v in (s, n, t)))))
    return result


def score(values, examples=(KEY,), positions=(0, 1)):
    return aggregate_rows(values, expected_examples=examples, expected_positions=positions)


@pytest.mark.parametrize("student,expected", [((0., 0.), 1.), ((1., 1e-6), 0.),
    ((-1., -1e-6), 4.), ((3., 3e-6), 4.)])
def test_off_teacher_reversal_and_overshoot(student, expected):
    assert score(rows(student))["mean_teacher_student_relative_error"] == pytest.approx(expected)


def test_weak_late_edit_does_not_dominate_but_late_errors_count():
    # The early field improves a lot; tiny late target must not invert the rank.
    assert score(rows((.8, .01)))["mean_teacher_student_relative_error"] < .041
    assert score(rows((.8, 2.)))["mean_teacher_student_relative_error"] > 4.


def test_orthogonal_corruption_and_coordinate_partition():
    n = torch.zeros(2, dtype=torch.float64)
    t = torch.tensor([1., 0.], dtype=torch.float64)
    s = torch.tensor([1., 2.], dtype=torch.float64)
    whole = dict(zip(("scene", "seed", "path"), KEY), position=0,
        **response_statistics(s, n, t))
    combined = aggregate_rows([whole], expected_examples=[KEY], expected_positions=[0])
    partitioned = score(rows((1., 2.), teacher=(1., 0.)))
    assert combined["mean_teacher_student_relative_error"] == partitioned["mean_teacher_student_relative_error"] == 4.
    assert combined["examples"][0]["orthogonal_relative_energy"] == 4.


def test_equal_example_weighting_not_global_energy_weighting():
    other = ("book", 18, "positive")
    result = score(rows((1., 1e-6)) + rows((0., 0.), teacher=(100., 100.), key=other),
                   examples=[KEY, other])
    assert result["mean_teacher_student_relative_error"] == .5
    assert result["path_relative_error"] == {"neutral": 0., "positive": 1.}


def test_units_and_order_are_invariant():
    original = score(rows((.8, .01)))
    scaled = rows((800., 10.), teacher=(1000., .001))
    assert score(scaled[::-1])["mean_teacher_student_relative_error"] == pytest.approx(original["mean_teacher_student_relative_error"])


@pytest.mark.parametrize("case", ["missing_position", "missing_example", "duplicate", "unexpected", "negative", "nonfinite", "corrupt", "zero_example"])
def test_invalid_cohorts_and_statistics_are_unscoreable(case):
    values = rows((.5, .5))
    examples = [KEY]
    if case == "missing_position":
        values.pop()
    elif case == "missing_example":
        examples.append(("book", 18, "positive"))
    elif case == "duplicate":
        values.append(copy.deepcopy(values[0]))
    elif case == "unexpected":
        values[0]["path"] = "unmatched"
    elif case == "negative":
        values[0]["student_teacher_sse"] = -1.
    elif case == "nonfinite":
        values[0]["student_teacher_sse"] = math.inf
    elif case == "corrupt":
        values[0]["student_teacher_edit_dot"] = 100.
    else:
        values = rows((.5, .5), teacher=(0., 0.))
    with pytest.raises(ValueError):
        score(values, examples=examples)


def test_statistics_are_detached_and_do_not_mutate_inputs_or_rng():
    state = torch.get_rng_state().clone()
    s = torch.tensor([.5, .2], requires_grad=True)
    before = s.detach().clone()
    response_statistics(s, torch.zeros(2), torch.ones(2))
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(s, before)
    assert s.grad is None


@pytest.mark.parametrize("student,neutral,teacher", [
    (torch.zeros(0), torch.zeros(0), torch.zeros(0)),
    (torch.zeros(2), torch.zeros(1), torch.zeros(2)),
    (torch.tensor([math.nan]), torch.zeros(1), torch.ones(1))])
def test_bad_raw_responses(student, neutral, teacher):
    with pytest.raises(ValueError):
        response_statistics(student, neutral, teacher)
