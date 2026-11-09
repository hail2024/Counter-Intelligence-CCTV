"""Proglint Counter Intelligence - Streamlit dashboard.

Run:  streamlit run app.py

Flow: choose or upload a video -> Analyse. The counter line and the customer/staff sides are placed
automatically from the people in the video.
All numbers shown come from the run's event ledger (the same data as the exported CSV/JSON):
the headline count is the number of INTERACTIONS (one per customer-staff exchange); each interaction
lists the observation rows (crossings, contact assessment) it is made of.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src.calibration import CalibrationError
from src.config import PROJECT_ROOT, ConfigError, load_config
from src.pipeline import effective_config, run_video
from src.pose_estimator import ModelLoadError
from src.video_io import VideoError, probe_video, read_frame

INPUT_DIR = PROJECT_ROOT / "data" / "input"
UPLOAD_DIR = INPUT_DIR / "uploads"
VIDEO_TYPES = (".mp4", ".avi", ".mov", ".mkv")
OUTCOME_STYLE = {  # colour, title
    "LIKELY_HANDSHAKE": ("#5fd47f", "Likely handshake"),
    "LIKELY_HAND_CONTACT": ("#3fb8a8", "Likely hand contact"),
    "UNCERTAIN": ("#ffbf47", "Uncertain interaction"),
    "BOUNDARY_CROSSING": ("#62a8ff", "Reach across the counter (no contact)"),
}
# The assignment's eight states (section 4), in lifecycle order, with their colour on the video and the meaning given
# in the assignment.
STATES = [
    ("NO_ACTIVE_INTERACTION", "#9a93a8", "No current hand-crossing or customer–staff hand-contact episode is being evaluated."),
    ("HAND_APPROACHING", "#b98cff", "A tracked hand is moving toward the counter boundary or the other person's hand."),
    ("BOUNDARY_CROSSED", "#62a8ff", "A wrist trajectory has crossed the counter boundary (hysteresis + persistence rule)."),
    ("POSSIBLE_HAND_CONTACT", "#ffbf47", "Customer and staff hands are sufficiently close; temporal validation is in progress."),
    ("LIKELY_HAND_CONTACT", "#5fd47f", "Proximity and persistence criteria are met across multiple frames."),
    ("LIKELY_HANDSHAKE", "#2fa84f", "The contact episode also meets the motion-pattern criteria for a handshake."),
    ("UNCERTAIN", "#ff9f43", "Occlusion, low keypoint confidence or identity ambiguity prevents reliable classification."),
    ("COOLDOWN", "#8aa0b8", "The interaction has ended; repeated events are suppressed until a new episode begins."),
]
STATE_COLOR = {n: c for n, c, _ in STATES}
ROW_NAME = {"LIKELY_HANDSHAKE": "handshake assessment", "LIKELY_HAND_CONTACT": "contact assessment",
            "UNCERTAIN": "uncertain contact assessment", "BOUNDARY_CROSSING": "crossing"}

st.set_page_config(page_title="Proglint Counter Intelligence", page_icon="🟣", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.1rem;}
.pci-title {font-size: 1.9rem; font-weight: 700; color: #ece8f4; margin-bottom: 0;}
.pci-sub {color: #a99fc0; margin-top: 0.1rem;}
.chip {display:inline-block; padding: 3px 10px; border-radius: 12px; margin: 4px 6px 0 0; font-size: 0.8rem;
       background:#231d2e; border:1px solid #3a3050;}
.chip.ok {border-color:#3fbf6a; color:#7fe0a0;} .chip.warn {border-color:#e0a030; color:#ffc860;}
.chip.bad {border-color:#e05050; color:#ff8080;}
.metric {background:#1a1622; border:1px solid #2e2640; border-radius:10px; padding:12px 14px; height:100%;}
.metric .v {font-size:1.7rem; font-weight:700;} .metric .l {color:#a99fc0; font-size:0.8rem;}
.metric.hero {border-color:#6b4fa0;} .metric.hero .v {font-size:2.2rem;}
.metric.purple .v {color:#b98cff;} .metric.green .v {color:#5fd47f;} .metric.teal .v {color:#3fb8a8;}
.metric.amber .v {color:#ffbf47;} .metric.blue .v {color:#62a8ff;} .metric.grey .v {color:#ece8f4;}
.ev {background:#1a1622; border-left:5px solid var(--c); border-radius:8px; padding:10px 14px; margin-bottom:4px;}
.ev .t {font-weight:700; color:var(--c);} .ev .m {color:#a99fc0; font-size:0.82rem;}
.ev .i {margin-top:4px;} .ev .r {margin-top:6px; font-size:0.78rem; color:#c9c0da;}
.row {display:inline-block; padding:1px 7px; border-radius:8px; background:#231d2e; margin:2px 4px 0 0;}
.path {margin-top:6px; font-size:0.76rem; line-height:2;}
.arrow {color:#6f6680; margin:0 2px;}
</style>""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("result", None)
ss.setdefault("status", "ready")
ss.setdefault("seek", 0.0)


def chip(text, kind=""):
    return f'<span class="chip {kind}">{text}</span>'


def metric(col, value, label, kind="purple"):
    col.markdown(f'<div class="metric {kind}"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


def model_status() -> tuple[str, str]:
    try:
        cfg = load_config()
    except ConfigError as exc:
        return f"config error: {exc}", "bad"
    p = cfg.resolve_path(cfg.model.path)
    if not p.exists():
        return f"model {p.name} missing: run `python main.py setup-model`", "bad"
    return f"pose model {p.stem} ({cfg.model.backend}) ready", "ok"


def state_chip(name: str, on: bool = True, small: bool = False) -> str:
    """One state chip with INLINE colours (Streamlit strips CSS custom properties from inline HTML)."""
    c = STATE_COLOR.get(name, "#999999")
    pad = "1px 7px" if small else "4px 10px"
    style = (f"display:inline-block;padding:{pad};border-radius:6px;margin:3px 5px 0 0;font-weight:600;"
             f"font-size:{'0.72rem' if small else '0.78rem'};")
    if on:
        style += f"background:{c};color:#15121b;border:1px solid {c};"
    else:
        style += f"background:transparent;color:{c};border:1px dashed {c};opacity:0.45;"
    return f'<span style="{style}">{name}</span>'


def fmt_t(v) -> str:
    return "…" if v is None else f"{v:.1f}s"


# ----------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1 · Video")
    up = st.file_uploader("Upload a new video", type=[t[1:] for t in VIDEO_TYPES])
    if up is not None:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOAD_DIR / Path(up.name).name
        if not dest.exists() or dest.stat().st_size != up.size:
            dest.write_bytes(up.getbuffer())
        if ss.get("video") != str(dest):
            ss.video, ss.result, ss.status = str(dest), None, "ready"
    videos = sorted(p for p in INPUT_DIR.glob("**/*") if p.suffix.lower() in VIDEO_TYPES)
    names = [p.relative_to(INPUT_DIR).as_posix() for p in videos]
    if names:
        cur = Path(ss.video).relative_to(INPUT_DIR).as_posix() if ss.get("video") else None
        idx = next((i for i, n in enumerate(names) if n == cur), 0)
        choice = st.selectbox("…or pick a video", names, index=idx)
        new = str(INPUT_DIR / choice)
        if ss.get("video") != new:
            ss.video, ss.result, ss.status = new, None, "ready"

    st.header("2 · Analyse")
    speed = st.radio("Speed", ["Every frame", "Fast (every 2nd frame)"], horizontal=False,
                     help="Fast analyses every 2nd frame: about 40% quicker, but brief exchanges can be missed.")
    go = st.button("▶ Analyse video", type="primary", use_container_width=True, disabled=not ss.get("video"))
    st.caption("The counter line, the customer and the staff member are found automatically: staff = the person "
               "working at the desk, customer = the person who reaches. Other people (background, passers-by) are "
               "ignored. A re-run of the same video reuses its cached poses and takes seconds.")

    swap = st.toggle("Swap customer and staff", False,
                     help="Turn this on if the customer and staff roles were assigned the wrong way round.")

ui_over = {"video": {"stride": 2 if speed.startswith("Fast") else 1}}

# ----------------------------------------------------------------------------------- header
ms, mk = model_status()
vid_name = Path(ss.video).name if ss.get("video") else "no video"
stat_kind = {"done": "ok", "failed": "bad", "running": "warn"}.get(ss.status, "")
st.markdown('<p class="pci-title">Proglint Counter Intelligence</p>'
            '<p class="pci-sub">Counter-Crossing &amp; Customer–Staff Hand-Contact Analytics</p>'
            + chip(ms, mk) + chip(f"video: {vid_name}") + chip(f"status: {ss.status}", stat_kind),
            unsafe_allow_html=True)

if not ss.get("video"):
    st.info("Upload a video in the sidebar to begin.")
    st.stop()
try:
    meta = probe_video(ss.video)
except VideoError as exc:
    st.error(f"This video cannot be read: {exc}")
    st.stop()

# ----------------------------------------------------------------------------------- processing
if go:
    ss.status = "running"
    bar = st.progress(0.0, "starting")
    try:
        cfg = effective_config(None, None, ui_over)
        r = run_video(ss.video, None, cfg, swap_roles=swap,
                      progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), m))
        ss.result = {"counts": r.ledger.counts(), "events": r.ledger.records(),
                     "interactions": r.ledger.interactions(), "icounts": r.ledger.interaction_counts(),
                     "stats": r.stats, "annotated": str(r.annotated_path) if r.annotated_path else None,
                     "csv": str(r.events_csv), "json": str(r.events_json), "icsv": str(r.interactions_csv),
                     "warnings": r.warnings, "notes": r.ledger.notes, "config": cfg.to_dict(),
                     "suppressed": len(r.ledger.suppressed),
                     "calibration": r.calibration.to_dict(), "duration": meta.duration_s,
                     "roles_uncertain": bool(r.stats.get("setup_info", {}).get("roles_uncertain")),
                     "state_timeline": r.ledger.state_timeline}
        ss.status, ss.seek = ("done" if r.annotated_path else "failed"), 0.0
        st.rerun()
    except CalibrationError as exc:
        ss.status = "failed"
        st.error(f"Automatic setup failed: {exc}")
    except (ConfigError, VideoError, ModelLoadError, RuntimeError) as exc:
        ss.status = "failed"
        st.error(f"Processing failed: {exc}")

res = ss.result
if res is None:
    st.image(cv2.cvtColor(read_frame(ss.video, min(30, max(meta.frame_count - 1, 0))), cv2.COLOR_BGR2RGB),
             caption=f"{vid_name} - {meta.width}x{meta.height}, {meta.fps:.1f} FPS, {meta.duration_s:.1f} s. "
                     "Press ▶ Analyse video.", use_container_width=True)
    st.stop()

# ----------------------------------------------------------------------------------- results
ic, inter, events = res["icounts"], res["interactions"], res["events"]
st_ = res["stats"]
cols = st.columns([1.3, 1, 1, 1, 1, 1.2])
metric(cols[0], ic["total"], "customer–staff interactions", "purple hero")
metric(cols[1], ic["LIKELY_HANDSHAKE"], "likely handshakes", "green")
metric(cols[2], ic["LIKELY_HAND_CONTACT"], "likely hand contact (not handshake)", "teal")
metric(cols[3], ic["UNCERTAIN"], "uncertain", "amber")
metric(cols[4], ic["BOUNDARY_CROSSING"], "reach across, no contact", "blue")
speed_txt = (f"{st_['elapsed_s']:.0f}s" if st_.get("elapsed_s") is not None else "–")
metric(cols[5], speed_txt, f"processing time for {res['duration']:.0f}s of video"
       + (" (cached poses)" if st_["detections_from_cache"] else f" · {st_.get('inference_backend', '')}"), "grey")
for w in res["warnings"]:
    st.warning(w)
seen_states = {x["state"] for x in res.get("state_timeline", [])}
st.markdown("<div style='margin-top:10px'><b>Interaction states reached in this video</b> "
            "<span style='color:#a99fc0;font-size:0.8rem'>(all eight states of the assignment; lit = occurred)</span><br>"
            + "".join(state_chip(n, n in seen_states) for n, _, _ in STATES) + "</div>", unsafe_allow_html=True)
if res.get("roles_uncertain"):
    st.warning("Could not tell from the hands who is the staff member, so the left (or top) person was taken as the "
               "customer. Use Swap customer and staff in the sidebar if needed.")

tab_res, tab_states, tab_table = st.tabs(["Results", "Interaction states", "Tables & timeline"])

with tab_res:
    vcol, ecol = st.columns([3, 2])
    with vcol:
        if res["annotated"] and Path(res["annotated"]).exists():
            st.video(res["annotated"], start_time=int(ss.seek))
            st.caption("Annotated video: white line = counter boundary, blue = customer side, pink = staff side; "
                       "boxes show the two tracked participants, dots their arm keypoints. "
                       "Event IDs on the video match the cards and the CSV.")
        else:
            st.error("The annotated video was not created (see warnings).")
        d = st.columns(5)
        if res["annotated"] and Path(res["annotated"]).exists():
            d[0].download_button("Video (MP4)", Path(res["annotated"]).read_bytes(), Path(res["annotated"]).name,
                                 "video/mp4", use_container_width=True)
        d[1].download_button("Interactions CSV", Path(res["icsv"]).read_bytes(), Path(res["icsv"]).name, "text/csv",
                             use_container_width=True)
        d[2].download_button("Event rows CSV", Path(res["csv"]).read_bytes(), Path(res["csv"]).name, "text/csv",
                             use_container_width=True)
        d[3].download_button("Full log JSON", Path(res["json"]).read_bytes(), Path(res["json"]).name,
                             "application/json", use_container_width=True)
        d[4].download_button("Settings", json.dumps({"config": res["config"], "scene": res["calibration"]}, indent=2),
                             "run_settings.json", "application/json", use_container_width=True)
    with ecol:
        st.markdown(f"#### What happened · {ic['total']} interaction{'s' if ic['total'] != 1 else ''}")
        if not inter:
            st.success("No hand crossing or hand contact between the customer and the staff member was detected.")
        by_id = {e["event_id"]: e for e in events}
        for k, it in enumerate(inter):
            colr, title = OUTCOME_STYLE[it["outcome"]]
            scores = []
            if it["contact_score"] is not None:
                scores.append(f"contact score {it['contact_score']}%")
            if it["handshake_score"] is not None:
                scores.append(f"handshake score {it['handshake_score']}%")
            who = " · ".join(x for x in (
                "customer #" + ", #".join(map(str, it["customer_track_ids"])) if it["customer_track_ids"] else "",
                "staff #" + ", #".join(map(str, it["staff_track_ids"])) if it["staff_track_ids"] else "") if x)
            rows = "".join(
                f'<span class="row">{eid} {ROW_NAME[by_id[eid]["event_type"]]}'
                f'{" " + by_id[eid]["crossing_direction"].replace("_", " ") if by_id[eid]["crossing_direction"] else ""}'
                f'</span>' for eid in it["event_ids"])
            unc = (f' · <span style="color:#ffbf47">{", ".join(it["uncertainty_reason"])}</span>'
                   if it["outcome"] == "UNCERTAIN" and it["uncertainty_reason"] else "")
            st.markdown(
                f'<div class="ev" style="--c:{colr}"><div class="t">{it["interaction_id"]} · {title}</div>'
                f'<div class="m">{fmt_t(it["start_timestamp"])} – {fmt_t(it["end_timestamp"])} · {who}'
                f'{" · " + " · ".join(scores) if scores else ""}{unc}</div>'
                f'<div class="i">{it["interpretation"]}</div>'
                f'<div class="path">states: ' + '<span class="arrow">→</span>'.join(
                    state_chip(x, True, small=True) for x in it.get("state_path", [])) + '</div>'
                f'<div class="r">made of: {rows}</div></div>',
                unsafe_allow_html=True)
            if st.button(f"▶ play {it['interaction_id']}", key=f"seek{k}"):
                ss.seek = max(0.0, it["start_timestamp"] - 0.5)
                st.rerun()
        st.caption("One interaction = one customer–staff exchange. A reach and the handshake that follows are one "
                   "interaction, not two. Scores are evidence scores from a small classifier trained on the labelled "
                   "project videos plus the assignment's rules; they are not calibrated probabilities. 2-D video "
                   "cannot prove physical contact, hence 'likely'.")

with tab_states:
    tl = res.get("state_timeline", [])
    table = []
    for name, colr, meaning in STATES:
        segs = [x for x in tl if x["state"] == name]
        total = sum(x["end"] - x["start"] for x in segs) + len(segs) / 30.0   # + one frame per run
        table.append({"state": name, "meaning (assignment)": meaning,
                      "occurred": "yes" if segs else "no",
                      "first at (s)": round(segs[0]["start"], 2) if segs else None,
                      "episodes": len(segs), "total time (s)": round(total, 2) if segs else 0.0})
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    if tl:
        try:
            import altair as alt
            tdf = pd.DataFrame(tl)
            tdf["end_plot"] = tdf["end"] + 1.0 / 30
            st.altair_chart(alt.Chart(tdf).mark_bar(height=16).encode(
                x=alt.X("start:Q", title="time (s)", scale=alt.Scale(domain=[0, res["duration"]])), x2="end_plot:Q",
                y=alt.Y("state:N", title=None, sort=[n for n, _, _ in STATES],
                        scale=alt.Scale(domain=[n for n, _, _ in STATES])),
                color=alt.Color("state:N", scale=alt.Scale(domain=[n for n, _, _ in STATES],
                                                           range=[c for _, c, _ in STATES]), legend=None),
                tooltip=["state", "start", "end"]).properties(height=8 * 30), use_container_width=True)
        except ImportError:
            pass
    st.caption("The state shown at each moment is the most advanced state of any customer–staff pair (or person, for "
               "approaching/crossing). States that did not occur stay empty: a video without contact never shows "
               "LIKELY_HAND_CONTACT, and that is the correct result. The same eight states are drawn as a strip at the "
               "bottom of the annotated video, with the current one highlighted.")

with tab_table:
    if not inter:
        st.info("No interactions.")
    else:
        st.markdown("**Interactions** (counting unit)")
        idf = pd.DataFrame(inter)
        for c in ("customer_track_ids", "staff_track_ids", "crossing_directions", "event_ids", "uncertainty_reason"):
            idf[c] = idf[c].apply(lambda v: ", ".join(map(str, v)))
        sel = st.dataframe(idf[["interaction_id", "outcome", "start_timestamp", "end_timestamp", "customer_track_ids",
                                "staff_track_ids", "crossing_count", "crossing_directions", "handshake_score",
                                "contact_score", "event_ids", "interpretation", "uncertainty_reason"]],
                           hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row")
        rows = sel.selection.rows if sel and sel.selection else []
        if rows:
            ss.seek = max(0.0, float(idf.iloc[rows[0]]["start_timestamp"]) - 0.5)
            st.info(f"The video in the Results tab now starts at {ss.seek:.1f} s.")
        st.markdown("**Observation rows** (event log: crossings and the contact assessment of each episode)")
        df = pd.DataFrame(events)
        show = df[["event_id", "interaction_id", "event_type", "start_timestamp", "end_timestamp", "duration_seconds",
                   "customer_track_id", "staff_track_id", "crossing_direction", "handshake_score", "contact_score",
                   "minimum_hand_distance_px", "keypoint_conf_mean", "uncertainty_reason", "end_reason"]].copy()
        show["uncertainty_reason"] = show["uncertainty_reason"].apply(lambda v: "; ".join(v))
        st.dataframe(show, hide_index=True, use_container_width=True)
        try:
            import altair as alt
            tl = df.assign(end=df["end_timestamp"].fillna(df["start_timestamp"]),
                           lane=df["interaction_id"] + " · " + df["event_type"].map(ROW_NAME))
            colors = {k: v[0] for k, v in OUTCOME_STYLE.items()}
            st.altair_chart(alt.Chart(tl).mark_bar(height=14, cornerRadius=3).encode(
                x=alt.X("start_timestamp:Q", title="time (s)", scale=alt.Scale(domain=[0, res["duration"]])),
                x2="end:Q", y=alt.Y("lane:N", title=None, sort=None),
                color=alt.Color("event_type:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["event_id", "interaction_id", "event_type", "start_timestamp", "end_timestamp",
                         "interpretation"]
            ).properties(height=max(120, 26 * len(tl))), use_container_width=True)
        except ImportError:
            pass
    st.caption(f"{res['suppressed']} close approaches were ignored as too brief, too fast or one-sided (listed with "
               "reasons in the JSON log). No row is ever created per frame.")
