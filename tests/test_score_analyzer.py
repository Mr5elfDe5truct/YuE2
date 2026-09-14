"""
Unit tests for Score Analyzer and Linter
"""

from yue2.score_analyzer import analyze_score, format_score_report_markdown, diff_two_scores


SAMPLE_ABC = """X:1
T:Test Track
M:4/4
L:1/32
Q:1/4=128
K:Am
%%stretchstaff 1
V:Vocal clef=treble
V:Chords clef=treble
[V:Vocal] A32 | c32 | e32 | a32 |]
[V:Chords] "Am" [A4c4e4] z28 | "F" [F4A4c4] z28 | "C" [C4E4G4] z28 | "G" [G4B4d4] z28 |]
"""


def test_analyze_score():
    res = analyze_score(SAMPLE_ABC)
    assert res["key"] == "Am"
    assert res["meter"] == "4/4"
    assert res["bpm"] == 128
    assert res["bars_count"] == 4
    assert res["status"] in ("valid", "warning")
    assert "Am" in res["chords"]
    assert "F" in res["chords"]
    assert "C" in res["chords"]
    assert "G" in res["chords"]


def test_format_score_report():
    res = analyze_score(SAMPLE_ABC)
    md = format_score_report_markdown(res)
    assert "Am" in md
    assert "128 BPM" in md
    assert "4 Bars" in md


def test_diff_two_scores():
    score_b = SAMPLE_ABC.replace('"G"', '"Em7"').replace("128", "140")
    diff = diff_two_scores(SAMPLE_ABC, score_b)
    assert diff["tempo_changed"] is True
    assert diff["tempo_a"] == 128
    assert diff["tempo_b"] == 140
    assert "Em7" in diff["added_chords"]
    assert "G" in diff["removed_chords"]
