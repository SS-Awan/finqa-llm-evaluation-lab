from decimal import Decimal

from finqa_eval.scoring import answers_match, is_numeric_answer, parse_numeric_answer


def test_parse_numeric_answer_handles_currency_and_percentages() -> None:
    currency = parse_numeric_answer("$ 1,234.50")
    percentage = parse_numeric_answer("(4.1%)")

    assert currency is not None
    assert currency.value == Decimal("1234.50")
    assert currency.is_percent is False

    assert percentage is not None
    assert percentage.value == Decimal("-4.1")
    assert percentage.is_percent is True


def test_is_numeric_answer_rejects_non_numerical_ground_truth() -> None:
    assert is_numeric_answer("64.9%") is True
    assert is_numeric_answer("yes") is False
    assert is_numeric_answer("170 grants") is False
    assert is_numeric_answer("$ 27.8 per share") is False


def test_answers_match_respects_percentage_units() -> None:
    expected = parse_numeric_answer("64.9%")
    same_value_with_percent = parse_numeric_answer("64.90%")
    same_value_without_percent = parse_numeric_answer("64.9")

    assert expected is not None
    assert same_value_with_percent is not None
    assert same_value_without_percent is not None

    assert answers_match(expected, same_value_with_percent) is True
    assert answers_match(expected, same_value_without_percent) is False


def test_answers_match_allows_small_numeric_rounding_difference() -> None:
    expected = parse_numeric_answer("100")
    close_prediction = parse_numeric_answer("100.05")
    incorrect_prediction = parse_numeric_answer("100.2")

    assert expected is not None
    assert close_prediction is not None
    assert incorrect_prediction is not None

    assert answers_match(expected, close_prediction) is True
    assert answers_match(expected, incorrect_prediction) is False