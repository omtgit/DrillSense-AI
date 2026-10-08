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
