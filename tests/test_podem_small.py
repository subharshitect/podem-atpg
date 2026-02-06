from atpg.fault import Fault
from atpg.parser_bench import parse_bench
from atpg.podem import podem


def test_podem_detects_fault():
    text = """
    INPUT(a)
    INPUT(b)
    n1 = AND(a, b)
    out = NOT(n1)
    OUTPUT(out)
    """
    circuit = parse_bench(text)
    fault = Fault.parse("n1/SA0")
    result = podem(circuit, fault)
    assert result.status == "DETECTED"
    assert result.test_vector["a"].to_char() == "1"
    assert result.test_vector["b"].to_char() == "1"


def test_podem_detects_fault_inversion():
    text = """
    INPUT(a)
    INPUT(b)
    n1 = OR(a, b)
    out = BUF(n1)
    OUTPUT(out)
    """
    circuit = parse_bench(text)
    fault = Fault.parse("n1/SA1")
    result = podem(circuit, fault)
    assert result.status == "DETECTED"


def test_podem_untestable_fault():
    text = """
    INPUT(a)
    INPUT(b)
    n1 = AND(a, b)
    out = OR(n1, a)
    OUTPUT(out)
    """
    circuit = parse_bench(text)
    fault = Fault.parse("n1/SA0")
    result = podem(circuit, fault)
    assert result.status == "UNTESTABLE"
