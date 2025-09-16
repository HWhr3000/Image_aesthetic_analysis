import yaml, pathlib

def test_thresholds_values():
    p = pathlib.Path(__file__).resolve().parents[1] / "thresholds.yml"
    t = yaml.safe_load(p.read_text())
    assert 88 == t["lab"]["L_min"]
    assert 22 == t["rms_contrast"]["target"]
