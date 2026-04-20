"""
Pulsa Design Tokens — Single source of truth for all visual constants.

Usage in app.py:
    from design_tokens import phase_color, workout_color, BRAND, NEUTRAL, SPACE, RADIUS

These map directly to CSS custom properties in app.py's :root block.
When building inline HTML, use these helpers instead of hardcoding hex values.
"""

# ─── Phase Colors ───────────────────────────────────────────────────────────
PHASE_COLORS = {
    "menstrual": {
        "main": "#E24B4A",
        "bg": "#fef2f2",
        "text": "#A32D2D",
        "icon": "\U0001fa78",  # 🩸
        "emoji": "\U0001fa78",
    },
    "follicular": {
        "main": "#1D9E75",
        "bg": "#f0fdf9",
        "text": "#085041",
        "icon": "\U0001f331",  # 🌱
        "emoji": "\U0001f331",
    },
    "ovulatory": {
        "main": "#D4537E",
        "bg": "#fdf2f8",
        "text": "#831843",
        "icon": "\U0001f338",  # 🌸
        "emoji": "\U0001f338",
    },
    "luteal": {
        "main": "#EF9F27",
        "bg": "#fefce8",
        "text": "#713f12",
        "icon": "\U0001f319",  # 🌙
        "emoji": "\U0001f319",
    },
}

def phase_color(phase: str, variant: str = "main") -> str:
    """Get phase color value. variant: 'main', 'bg', 'text'."""
    key = (phase or "luteal").strip().lower()
    return PHASE_COLORS.get(key, PHASE_COLORS["luteal"]).get(variant, "#888780")

def phase_icon(phase: str) -> str:
    """Get phase emoji."""
    key = (phase or "luteal").strip().lower()
    return PHASE_COLORS.get(key, PHASE_COLORS["luteal"]).get("icon", "")


# ─── Workout Type Colors ────────────────────────────────────────────────────
WORKOUT_COLORS = {
    "easy":      {"main": "#4CAF50", "bg": "#E8F5E9", "text": "#2E7D32"},
    "tempo":     {"main": "#FF9800", "bg": "#FFF3E0", "text": "#E65100"},
    "intervals": {"main": "#FF9800", "bg": "#FFF3E0", "text": "#E65100"},
    "long":      {"main": "#2196F3", "bg": "#E3F2FD", "text": "#1565C0"},
    "rest":      {"main": "#B4B2A9", "bg": "#F5F3F0", "text": "#888780"},
    "race":      {"main": "#FFD700", "bg": "#FFFBEB", "text": "#92400E"},
}

def workout_color(workout_type: str, variant: str = "main") -> str:
    """Get workout color. variant: 'main', 'bg', 'text'."""
    wt = (workout_type or "rest").strip().lower()
    return WORKOUT_COLORS.get(wt, WORKOUT_COLORS["rest"]).get(variant, "#888780")


# ─── Training Phase Colors (Build / Peak / Taper / Race) ────────────────────
TRAINING_PHASE_COLORS = {
    "build":  {"main": "#888780", "bg": "#F0F0EE", "text": "#5F5E5A"},
    "peak":   {"main": "#534AB7", "bg": "#EEEDFE", "text": "#3C3489"},
    "taper":  {"main": "#FF9800", "bg": "#FFF3E0", "text": "#E65100"},
    "race":   {"main": "#2D1B69", "bg": "#EEEDFE", "text": "#2D1B69"},
}

def training_phase_color(phase: str, variant: str = "main") -> str:
    """Get training phase color (Build/Peak/Taper/Race)."""
    p = (phase or "build").strip().lower()
    return TRAINING_PHASE_COLORS.get(p, TRAINING_PHASE_COLORS["build"]).get(variant, "#888780")


# ─── Brand ──────────────────────────────────────────────────────────────────
BRAND = {
    "purple":       "#2D1B69",
    "accent":       "#534AB7",
    "purple_light": "#7F77DD",
    "purple_pale":  "#EEEDFE",
    "purple_text":  "#3C3489",
}

# ─── Gold / Coach accent ────────────────────────────────────────────────────
GOLD = {
    "main":  "#D4A017",
    "muted": "#E6C98F",
    "bg":    "#FFF8E7",
    "text":  "#92690D",
}

# ─── Status ─────────────────────────────────────────────────────────────────
STATUS = {
    "success":    "#1D9E75",
    "success_bg": "#E1F5EE",
    "warning":    "#EF9F27",
    "warning_bg": "#FFF8E7",
    "error":      "#E24B4A",
    "error_bg":   "#fef2f2",
}

# ─── Neutrals ───────────────────────────────────────────────────────────────
NEUTRAL = {
    "bg_app":         "#FAF8F5",
    "bg_card":        "#FFFFFF",
    "bg_soft":        "#F5F3F0",
    "text_primary":   "#2C2C2A",
    "text_secondary": "#888780",
    "text_muted":     "#B4B2A9",
    "text_body":      "#5F5E5A",
    "border":         "#E5E4E0",
    "border_light":   "#D3D1C7",
}

# ─── Spacing (px) ───────────────────────────────────────────────────────────
SPACE = {
    "xs":  "4px",
    "sm":  "8px",
    "md":  "12px",
    "lg":  "16px",
    "xl":  "24px",
    "2xl": "32px",
    "3xl": "48px",
}

# ─── Border Radius ──────────────────────────────────────────────────────────
RADIUS = {
    "xs":   "6px",
    "sm":   "10px",
    "md":   "14px",
    "lg":   "16px",
    "pill": "999px",
}

# ─── Shadows ────────────────────────────────────────────────────────────────
SHADOW = {
    "card":     "0 1px 3px rgba(0,0,0,0.04)",
    "soft":     "0 2px 8px rgba(0,0,0,0.05)",
    "elevated": "0 4px 16px rgba(0,0,0,0.08)",
    "hero":     "0 8px 28px rgba(0,0,0,0.07)",
}

# ─── Typography ─────────────────────────────────────────────────────────────
FONT_DISPLAY = "'Barlow Condensed', sans-serif"
FONT_BODY = "'DM Sans', sans-serif"

TEXT_STYLES = {
    "h1":     f"font-size:28px;font-weight:800;font-family:{FONT_DISPLAY};",
    "h2":     f"font-size:17px;font-weight:600;font-family:{FONT_BODY};",
    "h3":     f"font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;font-family:{FONT_BODY};",
    "body":   f"font-size:14px;font-weight:400;font-family:{FONT_BODY};",
    "sm":     f"font-size:13px;font-weight:400;font-family:{FONT_BODY};",
    "label":  f"font-size:12px;font-weight:500;font-family:{FONT_BODY};",
    "metric": f"font-size:20px;font-weight:600;font-family:{FONT_BODY};",
    "micro":  f"font-size:10px;font-weight:400;font-family:{FONT_BODY};",
}

def text_style(size: str = "body") -> str:
    """Return inline CSS for text styling. size: h1, h2, h3, body, sm, label, metric, micro."""
    return TEXT_STYLES.get(size, TEXT_STYLES["body"])
