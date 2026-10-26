import pytest

from localcrickai import llm, render, timeline, vision
from localcrickai.schema import AudioInstructions, EditIntent, Segment


def test_schema_rejects_bad_event():
    with pytest.raises(ValueError):
        EditIntent(events=["boundary"])


def test_heuristic_parse():
    i = llm.heuristic_parse("Show all sixes and fours by Babar Azam in overs 10 to 20", False, True)
    assert i.events == ["six", "four"] and i.over_range == (10, 20)
    assert i.target_player == "Babar Azam" and i.audio_instructions.use_background_music


def test_scorecard_and_events():
    assert vision.parse_scorecard_text("PAK 134-3 (15.2)") == {"runs": 134, "wickets": 3, "over": 15}
    st = [{"t": 1, "runs": 10, "wickets": 0, "over": 1}, {"t": 2, "runs": 16, "wickets": 0, "over": 1},
          {"t": 3, "runs": 16, "wickets": 1, "over": 1}]
    assert [e["kind"] for e in vision.detect_events(st)] == ["six", "wicket"]


def test_timeline_filters_and_merges():
    ev = [{"t": 100, "kind": "six", "over": 12}, {"t": 105, "kind": "four", "over": 12},
          {"t": 500, "kind": "six", "over": 30}]
    intent = EditIntent(events=["six", "four"], over_range=(10, 20))
    segs = timeline.build_timeline(intent, ev, [], 1000)
    assert len(segs) == 1 and segs[0].start == 92 and segs[0].end == 111


def test_render_ducking_graph():
    a = AudioInstructions(use_external_commentary=True, use_background_music=True)
    cmd = render.build_command("v.mp4", [Segment(start=1, end=5)], "o.mp4", a, "c.wav", "b.mp3")
    g = cmd[cmd.index("-filter_complex") + 1]
    assert "sidechaincompress" in g and "[1:a]atrim" in g and "[2:a]volume" in g


def test_render_default_keeps_embedded():
    cmd = render.build_command("v.mp4", [Segment(start=0, end=2)], "o.mp4", AudioInstructions())
    assert "[0:a]atrim" in cmd[cmd.index("-filter_complex") + 1]
