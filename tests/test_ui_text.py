"""Guards for visitor-facing wording: no function names or raw variable names in the UI."""
import re
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"


def test_every_cache_decorator_has_a_friendly_spinner_message():
    bad = []
    for path in APP.glob("*.py"):
        for m in re.finditer(r"@st\.cache_(?:data|resource)(\(.*?\))?\s*\ndef (\w+)", path.read_text(), re.S):
            args, name = m.group(1) or "", m.group(2)
            msg = re.search(r'show_spinner=("([^"]+)"|False)', args)
            if not msg or (msg.group(2) and name in msg.group(2)) or (msg.group(2) and "_" in msg.group(2)):
                bad.append(f"{path.name}:{name}")
    assert not bad, f"cache functions without a plain-language spinner: {bad}"


def test_feature_labels_are_readable():
    import sys
    sys.path.insert(0, str(APP.parent))
    sys.path.insert(0, str(APP))
    from labels import feature_label
    from profiles import get_profile

    p = get_profile("drilling")
    assert feature_label("pit_volume_m3__dev180", p) == "Pit volume: change vs its 3-hour average"
    assert feature_label("spp_bar__d5", p) == "Standpipe pressure: change over the last 5 min"
    assert feature_label("gas_units", p) == "Total gas: current value"
    assert feature_label("flow_delta_lpm__std15", p) == "Flow out minus flow in: variability over the last 15 min"


def test_every_feature_in_the_results_file_gets_a_label():
    import json
    import sys
    sys.path.insert(0, str(APP.parent))
    sys.path.insert(0, str(APP))
    from labels import feature_label
    from profiles import get_profile

    res = json.loads((APP.parent / "docs" / "eval_results.json").read_text())
    p = get_profile("drilling")
    for v in res["shap"]["multiclass"].values():
        for item in v["top"]:
            label = feature_label(item["feature"], p)
            assert "_" not in label and "__" not in label, label


def test_startup_status_is_short_and_get_detector_has_no_second_spinner():
    src = (APP / "main.py").read_text()
    m = re.search(r'st\.status\("([^"]+)"', src)
    assert m and len(m.group(1)) <= 60, m and m.group(1)
    assert re.search(r"@st\.cache_resource\(show_spinner=False\)[^\n]*\ndef get_detector", src)
    assert re.search(r"@st\.cache_data\(show_spinner=False\)[^\n]*\ndef load_drilling_scored", src)
