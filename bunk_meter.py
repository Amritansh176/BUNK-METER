 """
BUNK METER - Fuzzy Logic Project
--------------------------------
Question: "Should I attend today's lecture or skip it?"

Inputs (sliders):
    1) Attendance (%)                    -> danger / ok / safe
    2) Lecture importance (0-10)         -> boring / important
    3) Friends' plan temptation (0-10)   -> low / high
Output:
    Bunk Score (0-100)                   -> attend / maybe / bunk

Run:  python bunk_meter.py
Adjust the sliders; graphs and verdict will update in real time.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# =====================================================================
# PART 1: FUZZY LOGIC
# =====================================================================

def trimf(x, a, b, c):
    """Triangular membership function (a = start, b = peak, c = end)."""
    if x < a or x > c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)

# ---- Input fuzzy sets ----
INPUTS = {
    "attendance": {
        "label": "Attendance (%)", "range": (0, 100),
        "sets": {"danger": (0, 0, 75), "ok": (65, 80, 95), "safe": (85, 100, 100)},
    },
    "importance": {
        "label": "Lecture importance (0-10)", "range": (0, 10),
        "sets": {"boring": (0, 0, 6), "important": (4, 10, 10)},
    },
    "temptation": {
        "label": "Friends' plan temptation (0-10)", "range": (0, 10),
        "sets": {"low": (0, 0, 6), "high": (4, 10, 10)},
    },
}

# ---- Output fuzzy sets (Bunk Score 0-100) ----
OUTPUT_SETS = {"attend": (0, 0, 45), "maybe": (30, 50, 70), "bunk": (55, 100, 100)}

# ---- Rule base: (attendance, importance, temptation) -> output ----
RULES = [
    ("danger", "boring",    "low",  "attend"),
    ("danger", "boring",    "high", "maybe"),
    ("danger", "important", "low",  "attend"),
    ("danger", "important", "high", "attend"),
    ("ok",     "boring",    "low",  "maybe"),
    ("ok",     "boring",    "high", "bunk"),
    ("ok",     "important", "low",  "attend"),
    ("ok",     "important", "high", "maybe"),
    ("safe",   "boring",    "low",  "maybe"),
    ("safe",   "boring",    "high", "bunk"),
    ("safe",   "important", "low",  "attend"),
    ("safe",   "important", "high", "maybe"),
]


def infer(values):
    """Crisp inputs -> fuzzy score. Returns everything needed for plotting."""
    # Step 1: Fuzzification
    fuzzy = {name: {s: trimf(values[name], *abc) for s, abc in cfg["sets"].items()}
             for name, cfg in INPUTS.items()}

    # Step 2: Rule evaluation (AND = min), OR of same output = max
    fired = []
    out_strength = {k: 0.0 for k in OUTPUT_SETS}
    for a, i, t, out in RULES:
        s = min(fuzzy["attendance"][a], fuzzy["importance"][i], fuzzy["temptation"][t])
        if s > 0:
            fired.append((s, a, i, t, out))
        out_strength[out] = max(out_strength[out], s)

    # Step 3: Aggregation (clip each output set, then max)
    xs = np.linspace(0, 100, 401)
    agg = np.zeros_like(xs)
    for name, s in out_strength.items():
        shape = np.array([trimf(x, *OUTPUT_SETS[name]) for x in xs])
        agg = np.maximum(agg, np.minimum(s, shape))

    # Step 4: Defuzzification (centroid)
    score = float((xs * agg).sum() / agg.sum()) if agg.sum() > 0 else 50.0
    return fuzzy, fired, out_strength, xs, agg, score


def verdict(score):
    if score < 35:
        return "ATTEND THE LECTURE!", "#2ca02c"
    if score < 65:
        return "UNCERTAIN... Could go either way", "#ff7f0e"
    return "SKIP THE LECTURE!", "#d62728"


# =====================================================================
# PART 2: VISUALIZATION (sliders + live graphs)
# =====================================================================
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def draw_input(ax, name, value, memberships):
    cfg = INPUTS[name]
    lo, hi = cfg["range"]
    xs = np.linspace(lo, hi, 200)
    ax.clear()
    for (set_name, abc), color in zip(cfg["sets"].items(), COLORS):
        ax.plot(xs, [trimf(x, *abc) for x in xs], color=color, label=set_name)
        mu = memberships[set_name]
        if mu > 0:  # membership dot at current value
            ax.plot([value], [mu], "o", color=color, ms=8)
            ax.text(value, mu + 0.06, f"{mu:.2f}", color=color, ha="center", fontsize=8)
    ax.axvline(value, color="black", ls="--", lw=1)
    ax.set_ylim(0, 1.25)
    ax.set_xlim(lo, hi)
    ax.set_title(cfg["label"], fontsize=10)
    ax.legend(fontsize=7, loc="upper right", ncol=3)


def draw_output(ax, out_strength, xs, agg, score):
    ax.clear()
    for (name, abc), color in zip(OUTPUT_SETS.items(), COLORS):
        ax.plot(xs, [trimf(x, *abc) for x in xs], color=color, lw=1, alpha=0.5, label=name)
    ax.fill_between(xs, agg, color="#d62728", alpha=0.35, label="aggregated output")
    ax.axvline(score, color="black", lw=2)
    ax.text(score, 1.08, f"centroid = {score:.1f}", ha="center", fontsize=9, fontweight="bold")
    text, color = verdict(score)
    ax.set_title(text, fontsize=18, fontweight="bold", color=color)
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.25)
    ax.set_xlabel("Bunk Score")
    ax.legend(fontsize=7, loc="upper left")


def draw_rules(ax, fired):
    ax.clear(); ax.axis("off")
    ax.set_title("Rules currently fired (strongest first)", fontsize=10, loc="left")
    fired = sorted(fired, reverse=True)[:6]
    for k, (s, a, i, t, out) in enumerate(fired):
        ax.text(0.0, 0.88 - k * 0.15,
                f"IF att={a} AND lecture={i} AND temptation={t}  THEN {out.upper()}   [{s:.2f}]",
                fontsize=8.5, family="monospace", transform=ax.transAxes)


def main():
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("BUNK METER  (Fuzzy Logic)", fontsize=15, fontweight="bold")
    in_axes = {
        "attendance": fig.add_axes([0.06, 0.70, 0.36, 0.17]),
        "importance": fig.add_axes([0.06, 0.46, 0.36, 0.17]),
        "temptation": fig.add_axes([0.06, 0.22, 0.36, 0.17]),
    }
    ax_out = fig.add_axes([0.54, 0.52, 0.42, 0.33])
    ax_rules = fig.add_axes([0.54, 0.12, 0.42, 0.28])

    defaults = {"attendance": 80, "importance": 4, "temptation": 8}
    sliders = {}
    for k, (name, cfg) in enumerate(INPUTS.items()):
        ax_s = fig.add_axes([0.14, 0.13 - k * 0.04, 0.26, 0.025])
        lo, hi = cfg["range"]
        sliders[name] = Slider(ax_s, name, lo, hi, valinit=defaults[name])

    def update(_=None):
        values = {name: s.val for name, s in sliders.items()}
        fuzzy, fired, out_strength, xs, agg, score = infer(values)
        for name in INPUTS:
            draw_input(in_axes[name], name, values[name], fuzzy[name])
        draw_output(ax_out, out_strength, xs, agg, score)
        draw_rules(ax_rules, fired)
        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)
    update()
    return fig, sliders


if __name__ == "__main__":
    fig, sliders = main()
    plt.show()
