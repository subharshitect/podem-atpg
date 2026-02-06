from atpg.logic5 import Logic5, evaluate_gate


def test_and_with_d():
    result = evaluate_gate("AND", [Logic5.D, Logic5.ONE]).to_logic5()
    assert result is Logic5.D


def test_and_with_zero_masks_d():
    result = evaluate_gate("AND", [Logic5.D, Logic5.ZERO]).to_logic5()
    assert result is Logic5.ZERO


def test_or_with_dbar():
    result = evaluate_gate("OR", [Logic5.DBAR, Logic5.ZERO]).to_logic5()
    assert result is Logic5.DBAR


def test_xor_with_d():
    result = evaluate_gate("XOR", [Logic5.D, Logic5.ZERO]).to_logic5()
    assert result is Logic5.D


def test_not_inverts_d():
    result = evaluate_gate("NOT", [Logic5.D]).to_logic5()
    assert result is Logic5.DBAR
