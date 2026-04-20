"""
Pulsa UI Components — Reusable HTML builders for Streamlit.

All components use design_tokens for consistent styling.
Usage:
    from components import card, section_label, page_title, phase_badge, ...
    st.markdown(card(inner_html), unsafe_allow_html=True)
"""
import html as _html

from design_tokens import (
    phase_color, phase_icon, workout_color, training_phase_color,
    BRAND, GOLD, NEUTRAL, SPACE, RADIUS, SHADOW, TEXT_STYLES,
    text_style, FONT_BODY, FONT_DISPLAY,
)


# ─── Layout ─────────────────────────────────────────────────────────────────

def card(content: str, *, padding: str = "18px", margin_bottom: str = SPACE["lg"],
         extra_style: str = "") -> str:
    """Standard white card container."""
    return (
        f'<div style="background:{NEUTRAL["bg_card"]};border-radius:{RADIUS["md"]};'
        f'padding:{padding};border:1px solid {NEUTRAL["border"]};'
        f'box-shadow:{SHADOW["card"]};margin-bottom:{margin_bottom};{extra_style}">'
        f'{content}</div>'
    )


def section_label(text: str) -> str:
    """Uppercase section label (YOUR METRICS, TRAINING PACES, etc.)."""
    return (
        f'<div style="{text_style("h3")}color:{NEUTRAL["text_secondary"]};'
        f'margin-bottom:{SPACE["sm"]};">{_html.escape(text)}</div>'
    )


def page_title(title: str, subtitle: str = "") -> str:
    """Page title (TODAY, PLAN, INSIGHTS, etc.)."""
    sub = ""
    if subtitle:
        sub = (
            f'<div style="{text_style("body")}color:{NEUTRAL["text_secondary"]};'
            f'margin-top:4px;">{_html.escape(subtitle)}</div>'
        )
    return (
        f'<div style="margin-bottom:{SPACE["xl"]};">'
        f'<h1 style="{text_style("h1")}color:{NEUTRAL["text_primary"]};'
        f'margin:0;text-transform:uppercase;letter-spacing:0.02em;">'
        f'{_html.escape(title)}</h1>{sub}</div>'
    )


# ─── Badges ─────────────────────────────────────────────────────────────────

def phase_badge(phase: str, *, show_icon: bool = True) -> str:
    """Cycle phase badge (Menstrual, Follicular, etc.)."""
    icon_str = f"{phase_icon(phase)} " if show_icon else ""
    name = (phase or "—").capitalize()
    return (
        f'<span style="display:inline-flex;align-items:center;gap:4px;'
        f'padding:4px 10px;background:{phase_color(phase, "bg")};'
        f'color:{phase_color(phase, "text")};border-radius:{RADIUS["pill"]};'
        f'{text_style("label")}">{icon_str}{_html.escape(name)}</span>'
    )


def workout_badge(workout_type: str) -> str:
    """Workout type badge (Easy, Tempo, Long, etc.)."""
    wt = (workout_type or "Rest").strip()
    return (
        f'<span style="display:inline-block;padding:3px 8px;'
        f'background:{workout_color(wt, "bg")};color:{workout_color(wt, "text")};'
        f'border-radius:{RADIUS["xs"]};{text_style("label")}">'
        f'{_html.escape(wt.capitalize())}</span>'
    )


def training_phase_pill(phase: str) -> str:
    """Training phase pill (Build, Peak, Taper, Race)."""
    name = (phase or "Build").capitalize()
    return (
        f'<span style="display:inline-block;padding:4px 12px;'
        f'background:{training_phase_color(phase, "bg")};'
        f'color:{training_phase_color(phase, "text")};'
        f'border-radius:20px;{text_style("label")}">'
        f'{_html.escape(name)}</span>'
    )


def key_badge() -> str:
    """Orange KEY badge for key workouts."""
    return (
        f'<span style="background:#FF9800;color:#fff;font-size:9px;'
        f'padding:3px 8px;border-radius:4px;font-weight:600;">KEY</span>'
    )


# ─── Workout Rows ───────────────────────────────────────────────────────────

def workout_row(day: str, wtype: str, details: str = "", distance: str = "",
                *, is_key: bool = False) -> str:
    """Single workout row for plan pages."""
    wt_lower = (wtype or "rest").strip().lower()
    is_rest = wt_lower in ("rest", "", "—")

    # Left border color
    left_border = "" if is_rest else f"border-left:4px solid {workout_color(wt_lower, 'main')};"
    bg = workout_color(wt_lower, "bg") if not is_rest else NEUTRAL["bg_soft"]
    radius = f"border-radius:0 {RADIUS['sm']} {RADIUS['sm']} 0;" if not is_rest else f"border-radius:{RADIUS['sm']};"

    # Type label
    if is_rest:
        type_html = f'<span style="color:{NEUTRAL["text_muted"]};">Rest</span>'
    else:
        type_html = (
            f'<span style="font-weight:500;color:{workout_color(wt_lower, "text")};">'
            f'{_html.escape(wtype.capitalize())}</span>'
        )

    key_html = f" {key_badge()}" if is_key else ""

    details_html = ""
    if details:
        details_html = (
            f'<span style="{text_style("sm")}color:{NEUTRAL["text_body"]};'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
            f'{_html.escape(details)}</span>'
        )

    dist_html = ""
    if distance and not is_rest:
        dist_html = (
            f'<span style="{text_style("body")}font-weight:500;'
            f'color:{NEUTRAL["text_primary"]};white-space:nowrap;'
            f'margin-left:{SPACE["sm"]};">{_html.escape(distance)}</span>'
        )
    elif is_rest:
        dist_html = f'<span style="color:{NEUTRAL["text_muted"]};">—</span>'

    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:12px 14px;{left_border}{radius}background:{bg};min-width:0;">'
        f'<div style="display:flex;align-items:center;gap:10px;min-width:0;flex:1;">'
        f'<span style="{text_style("body")}font-weight:600;color:{NEUTRAL["text_primary"]};min-width:32px;">'
        f'{_html.escape(day)}</span>'
        f'{type_html}{key_html}{details_html}</div>{dist_html}</div>'
    )


# ─── Week Summary Row ───────────────────────────────────────────────────────

def week_summary_row(week_num: int, dates: str, km: str, phase: str,
                     *, is_race: bool = False) -> str:
    """Compact week row for future weeks in Plan page."""
    if is_race:
        bg = BRAND["purple"]
        text_col = "#FFFFFF"
        sub_col = "rgba(255,255,255,0.75)"
        pill = (
            f'<span style="font-size:12px;color:#FFFFFF;background:rgba(255,255,255,0.2);'
            f'padding:4px 10px;border-radius:12px;">Race</span>'
        )
    else:
        bg = NEUTRAL["bg_card"]
        text_col = NEUTRAL["text_primary"]
        sub_col = NEUTRAL["text_secondary"]
        pill = training_phase_pill(phase)

    return (
        f'<div style="background:{bg};border:1px solid {NEUTRAL["border"]};'
        f'border-radius:{RADIUS["sm"]};padding:12px 14px;margin-bottom:8px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">'
        f'<div><span style="{text_style("body")}font-weight:600;color:{text_col};">Week {week_num}</span>'
        f'<span style="{text_style("sm")}color:{sub_col};margin-left:8px;">{_html.escape(dates)}</span></div>'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<span style="{text_style("label")}color:{sub_col};">{_html.escape(km)}</span>'
        f'{pill}</div></div></div>'
    )


# ─── Metric Card ────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, subtext: str = "",
                *, highlight_color: str = "") -> str:
    """Single metric display (Resting HR, Sleep, Readiness)."""
    sub_color = highlight_color or NEUTRAL["text_secondary"]
    return (
        f'<div style="text-align:center;padding:{SPACE["md"]};">'
        f'<div style="{text_style("micro")}color:{NEUTRAL["text_secondary"]};'
        f'text-transform:uppercase;margin-bottom:4px;">{_html.escape(label)}</div>'
        f'<div style="{text_style("metric")}color:{NEUTRAL["text_primary"]};">{_html.escape(str(value))}</div>'
        f'<div style="{text_style("sm")}color:{sub_color};margin-top:2px;">{_html.escape(subtext)}</div>'
        f'</div>'
    )


# ─── Insight / Highlight Box ────────────────────────────────────────────────

def insight_banner(text: str, *, bg: str = GOLD["bg"], color: str = GOLD["text"],
                   border_color: str = "") -> str:
    """Phase insight / coach banner."""
    border = f"border-left:3px solid {border_color};" if border_color else ""
    return (
        f'<div style="background:{bg};padding:12px 14px;border-radius:{RADIUS["sm"]};'
        f'{border}{text_style("sm")}color:{color};margin-bottom:{SPACE["md"]};">'
        f'{text}</div>'
    )


def highlight_box(text: str, *, phase: str = "luteal") -> str:
    """Highlighted insight box with phase-colored left border."""
    return (
        f'<div style="padding:{SPACE["md"]} {SPACE["lg"]};'
        f'background:{phase_color(phase, "bg")};'
        f'border-left:3px solid {phase_color(phase, "main")};'
        f'border-radius:0 {RADIUS["sm"]} {RADIUS["sm"]} 0;'
        f'{text_style("body")}color:{NEUTRAL["text_primary"]};">{text}</div>'
    )


# ─── Phase Bar ──────────────────────────────────────────────────────────────

def phase_bar(segments: list, *, height: str = "8px") -> str:
    """Horizontal phase bar showing cycle phase distribution.
    segments: list of (phase_name, width_pct) tuples.
    """
    bars = ""
    for phase_name, pct in segments:
        bars += f'<div style="width:{pct}%;background:{phase_color(phase_name, "main")};"></div>'
    return (
        f'<div style="display:flex;height:{height};border-radius:4px;overflow:hidden;">'
        f'{bars}</div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:6px;">'
        f'<span style="{text_style("micro")}color:{NEUTRAL["text_secondary"]};">Mon</span>'
        f'<span style="{text_style("micro")}color:{NEUTRAL["text_secondary"]};">Sun</span></div>'
    )
