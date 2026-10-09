"""Drilling-flavoured waiting indicator for slow calls (Gemini).

The call runs in a worker thread that never touches Streamlit; the main script rotates a status word
in a placeholder every 1.5 s. Nothing here logs or displays the arguments, so an API key passed
through stays out of every message. If the animation cannot start, a plain spinner is used instead.
"""
import threading

import streamlit as st

STATUS_WORDS = (
    "Circulating", "Logging the signals", "Fishing for facts", "Pressure-testing the numbers",
    "Correlating the channels", "Tripping through the report", "Sweeping the hole",
)
INTERVAL_SECONDS = 1.5

_STYLE = """<style>
.ds-wait{font-weight:600;background:linear-gradient(90deg,#8a8f98 25%,#e6e8eb 50%,#8a8f98 75%);
background-size:200% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;
animation:ds-shimmer 1.8s linear infinite}
@keyframes ds-shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
@media (prefers-reduced-motion:reduce){.ds-wait{animation:none;color:#8a8f98;background:none}}
</style>"""


def _word_html(word):
    return f'{_STYLE}<div class="ds-wait">{word}...</div>'


def run_with_status(fn, *args, fallback_text="Working...", interval=INTERVAL_SECONDS, words=STATUS_WORDS):
    """Call fn(*args) once in a worker thread and return its result, animating a status word meanwhile."""
    box = {}

    def worker():
        try:
            box["value"] = fn(*args)
        except BaseException as exc:  # handed back to the caller below, never printed
            box["error"] = exc

    try:
        placeholder = st.empty()
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    except Exception:
        with st.spinner(fallback_text):
            return fn(*args)  # nothing was started, so the call has not run yet

    shown = 0
    try:
        while thread.is_alive():
            placeholder.markdown(_word_html(words[shown % len(words)]), unsafe_allow_html=True)
            shown += 1
            thread.join(interval)
    except Exception:
        thread.join()  # the animation failed; the call still finishes, then the result is used
    finally:
        try:
            placeholder.empty()
        except Exception:
            pass
    if "error" in box:
        raise box["error"]
    return box["value"]
