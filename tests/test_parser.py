from atpg.parser_bench import parse_bench


def test_parse_basic_bench():
    text = """
    INPUT(a)
    INPUT(b)
    n1 = AND(a, b)
    out = NOT(n1)
    OUTPUT(out)
    """
    circuit = parse_bench(text)
    assert circuit.primary_inputs == ["a", "b"]
    assert circuit.primary_outputs == ["out"]
    assert len(circuit.gates) == 2
    assert circuit.gates[0].gate_type == "AND"
    assert circuit.gates[1].gate_type == "NOT"
