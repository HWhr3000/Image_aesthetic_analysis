import json, pathlib

def test_schema_loads():
    p = pathlib.Path(__file__).resolve().parents[1] / "schema.json"
    data = json.loads(p.read_text())
    assert data["title"] == "Aesthetics Evaluation Result"
