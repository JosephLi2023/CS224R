"""Render TurnRD per-turn credit assignment as a color-gradient LaTeX figure.

Reads a probe JSONL (one row per trajectory, schema defined in
`infra/app_probe_turnrd_v3.py`) and emits a `.tex` document showing the
most decisive successes and most-blamed failures, with each turn's
observation + action wrapped in a `\\colorbox` shaded by its per-turn
credit (green = high credit, red = high blame, grey = ~0).

Selection heuristic:
  * Success bucket: rank by max(credit) / mean(|credit|)
    (concentration of positive credit on a single turn).
  * Failure bucket: rank by |min(credit)| / mean(|credit|)
    (concentration of blame on a single turn).
  * Both buckets prefer 4-8 turn trajectories via a length tie-breaker
    (too short isn't visually interesting; too long doesn't fit).

Color mapping (per-trajectory normalized to [-1, +1] by max(|credit|)):
  +1.0 → bright green (60, 180, 75)
   0.0 → neutral grey (235, 235, 235)
  -1.0 → bright red   (215, 50, 50)
Linear interpolation in RGB. Action cells use a slightly more saturated
shade than the obs cells so the reader can distinguish them at a glance.

Usage:
  python scripts/render_turnrd_credits_latex.py \\
      --probe-jsonl /tmp/v3_R9_probe.jsonl \\
      --n-success 3 --n-failure 3 \\
      --out-tex /tmp/v3_R9_credit_demo.tex
  pdflatex -output-directory /tmp /tmp/v3_R9_credit_demo.tex

Snippet mode (no preamble — for pasting into an existing report):
  python scripts/render_turnrd_credits_latex.py \\
      --probe-jsonl /tmp/v3_R9_probe.jsonl \\
      --n-success 3 --n-failure 3 \\
      --out-tex /tmp/v3_R9_credit_demo_snippet.tex \\
      --snippet-only
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# IO + selection
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _is_success_useful(row: dict[str, Any], *, max_turns: int) -> bool:
    """Successes with at least 2 turns and no more than `max_turns` are
    eligible. The default `max_turns=12` keeps every selected success
    fully renderable on one tabular page; raise it (e.g. 40) to allow
    long marathon successes that the windowing logic will then trim.
    """
    n = int(row.get("n_turns", 0))
    return 2 <= n <= max_turns


def _is_failure_useful(row: dict[str, Any]) -> bool:
    """Failures keep a wider window because most fail by truncation
    (n_turns == max_turns). The renderer trims around the most-decisive
    turn so they still fit visually."""
    n = int(row.get("n_turns", 0))
    return n >= 2


def _credit_stats(row: dict[str, Any]) -> tuple[float, float, float, int]:
    """Return (max_credit, min_credit, mean_abs_credit, n_turns)."""
    credits = [float(t["credit"]) for t in row["turns"]]
    if not credits:
        return (0.0, 0.0, 0.0, 0)
    return (
        max(credits),
        min(credits),
        sum(abs(c) for c in credits) / len(credits),
        len(credits),
    )


def _rank_success(row: dict[str, Any]) -> tuple[float, int]:
    """Sort key for the success bucket. Higher = more decisive single turn.

    Primary: max(credit) / mean(|credit|).
    Secondary (tie-break): -|n - 6| so 4-8 turn trajectories win ties.
    """
    mx, _mn, mean_abs, n = _credit_stats(row)
    score = mx / mean_abs if mean_abs > 1e-9 else 0.0
    return (score, -abs(n - 6))


def _rank_failure(row: dict[str, Any]) -> tuple[float, int]:
    """Sort key for the failure bucket. Higher = single fatal turn."""
    _mx, mn, mean_abs, n = _credit_stats(row)
    score = abs(mn) / mean_abs if mean_abs > 1e-9 else 0.0
    return (score, -abs(n - 6))


def _fatal_position(row: dict[str, Any]) -> int:
    """Index of the most-blamed turn in a failure (argmin credit over
    turns 1..n-1). Used for diversity-based failure selection."""
    credits = [float(t["credit"]) for t in row["turns"]]
    if len(credits) <= 1:
        return 0
    return min(range(1, len(credits)), key=lambda i: credits[i])


def _pick_diverse_failures(
    pool: list[dict[str, Any]], n_pick: int
) -> list[dict[str, Any]]:
    """Pick `n_pick` failures with diverse fatal-turn positions.

    AlfWorld failures in this probe all run to truncation (n_turns ==
    max_turns), so length-based diversity isn't available. Instead we
    sort the pool by fatal-turn position (early give-up vs mid-trajectory
    close-empty-drawer vs late premature-completion) and pick evenly
    spaced from that sorted list. Within ties on fatal position, the
    existing concentration-based ranker (`_rank_failure`) breaks ties.
    """
    if n_pick <= 0 or not pool:
        return []
    # Sort by (fatal_position, -concentration_score) so that within a
    # position bucket the highest-concentration example wins.
    decorated = sorted(
        pool,
        key=lambda r: (_fatal_position(r), -_rank_failure(r)[0]),
    )
    L = len(decorated)
    if L <= n_pick:
        return decorated
    if n_pick == 1:
        return [decorated[L // 2]]
    # Evenly-spaced indices across [0, L-1].
    step = (L - 1) / (n_pick - 1)
    seen: set[int] = set()
    picks: list[dict[str, Any]] = []
    for i in range(n_pick):
        idx = round(i * step)
        if idx not in seen:
            seen.add(idx)
            picks.append(decorated[idx])
    return picks


# ---------------------------------------------------------------------------
# Color mapping
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _credit_to_rgb(
    credit_norm: float, *, darker: bool = False
) -> tuple[int, int, int]:
    """Map credit_norm ∈ [-1, +1] to RGB.

    Two-color scheme — positives only use green, negatives only use red:
      +1.0 → bright green (60, 180, 75)
       0.0 → white         (255, 255, 255)
      -1.0 → bright red    (215, 50, 50)
    More positive = more saturated green; more negative = more saturated
    red. A credit near zero is essentially white (no color cue).

    `darker=True` pulls 30% further toward the extreme color for action
    cells, so observation/action stripes within a single turn are
    visually distinguishable even at low credit magnitudes.
    """
    c = max(-1.0, min(1.0, credit_norm))
    if c >= 0:
        r = _lerp(255.0, 60.0, c)
        g = _lerp(255.0, 180.0, c)
        b = _lerp(255.0, 75.0, c)
    else:
        t = -c
        r = _lerp(255.0, 215.0, t)
        g = _lerp(255.0, 50.0, t)
        b = _lerp(255.0, 50.0, t)
    if darker:
        if c >= 0:
            r = _lerp(r, 60.0, 0.30)
            g = _lerp(g, 180.0, 0.30)
            b = _lerp(b, 75.0, 0.30)
        else:
            r = _lerp(r, 215.0, 0.30)
            g = _lerp(g, 50.0, 0.30)
            b = _lerp(b, 50.0, 0.30)
    return (int(round(r)), int(round(g)), int(round(b)))


def _rgb_to_html(rgb: tuple[int, int, int]) -> str:
    return "{:02X}{:02X}{:02X}".format(*rgb)


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------

# Characters that need escaping in LaTeX text mode. \ MUST be handled first
# (it's not in this map; we run a regex pre-pass to convert "\\" first).
_LATEX_ESCAPE = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
}


def _latex_escape(text: str, max_chars: int) -> str:
    """Escape LaTeX specials + truncate to `max_chars` with an ellipsis."""
    text = (text or "").strip()
    if not text:
        return r"\textit{<empty>}"
    if len(text) > max_chars:
        # Use ASCII "..." (not a Unicode ellipsis) to keep pdflatex happy
        # without requiring inputenc/utf8 setups.
        text = text[: max_chars - 3] + "..."
    # Convert "\\" first so we don't double-escape, then per-char rest.
    text = text.replace("\\", r"\textbackslash{}")
    out: list[str] = []
    for ch in text:
        out.append(_LATEX_ESCAPE.get(ch, ch))
    s = "".join(out)
    # Collapse whitespace — \colorbox{}{} dislikes raw newlines inside
    # its argument; we want compact one-line cells.
    s = re.sub(r"\s+", " ", s)
    return s


def _pivot_index(row: dict[str, Any]) -> int:
    """Index of the trajectory's most-meaningful turn.

    - Success (R > 0 / `success=True`) -> argmax(credit) = the turn that
      received the most positive credit ("decisive" turn).
    - Failure -> argmin(credit) = the turn that received the most
      negative credit ("fatal" turn).

    Turn 0 is excluded from the search because the renderer displays it
    as an uncolored context row with no credit (its credit is noisy --
    the V-head has no action consequence to evaluate yet).
    """
    credits = [float(t["credit"]) for t in row["turns"]]
    if len(credits) <= 1:
        return 0
    # Search over indices 1..n-1.
    candidates = range(1, len(credits))
    if bool(row.get("success")):
        return max(candidates, key=lambda i: credits[i])
    return min(candidates, key=lambda i: credits[i])


def _windowed_turns(
    row: dict[str, Any], *, max_display: int
) -> tuple[list[dict[str, Any]], int, int, bool]:
    """Return (turns_window, start_idx, total_turns, was_trimmed).

    If `max_display <= 0` or the trajectory already fits, returns all
    turns unchanged. Otherwise selects a `max_display`-wide window
    centered on the trajectory's pivot turn (most-decisive for success,
    most-fatal for failure -- see `_pivot_index`).
    """
    turns = list(row["turns"])
    total = len(turns)
    if max_display <= 0 or total <= max_display:
        return turns, 0, total, False
    pivot = _pivot_index(row)
    half = max_display // 2
    start = max(0, min(total - max_display, pivot - half))
    end = start + max_display
    return turns[start:end], start, total, True


def _build_trajectory_block(
    row: dict[str, Any],
    *,
    color_defs: list[str],
    color_defined: set[str],
    obs_max_chars: int,
    tag: str,
    max_display_turns: int,
) -> str:
    """Render one trajectory as a tabular + register its color definitions.

    `tag` is a short alphanumeric prefix (e.g. "s0", "f2") used as a
    component of each color's name. Names are scoped per-trajectory so two
    trajectories with overlapping turn indices don't clash on
    `\\definecolor`.
    """
    # Per-trajectory normalization uses turns 1..n-1 (Turn 0 is rendered
    # uncolored as a context row, so its credit doesn't participate in
    # the color saturation scale).
    all_credits = [float(t["credit"]) for t in row["turns"]]
    if len(all_credits) <= 1:
        return ""
    max_abs = max(abs(c) for c in all_credits[1:]) or 1.0

    success = bool(row["success"])
    # Pivot turn: argmax(credit) for successes (decisive) / argmin(credit)
    # for failures (fatal). Computed over the FULL trajectory so the
    # marker is correct even when the displayed window trims around it.
    pivot_idx = _pivot_index(row)
    pivot_tag_text = "decisive" if success else "fatal"

    turns_to_show, start_idx, total_turns, was_trimmed = _windowed_turns(
        row, max_display=max_display_turns
    )

    # Always render Turn 0 (the initial task observation) as an
    # uncolored context row. When the window starts later than Turn 1,
    # insert a vertical-dots ellipsis row between Turn 0 and the window
    # to make the gap explicit.
    prepend_turn0 = start_idx > 0
    show_ellipsis = start_idx > 1

    task_id = int(row["task_id"])
    final_R = float(row["final_reward"])
    label = "Success" if success else "Failure"

    lines: list[str] = []
    title = (
        r"\subsection*{Task " + str(task_id) + r" --- "
        + label + r" (R = " + f"{final_R:+.2f}" + r")}"
    )
    lines.append(title)
    if was_trimmed:
        win_lo = start_idx
        win_hi = start_idx + len(turns_to_show) - 1
        pivot_kind = "decisive" if success else "fatal"
        lines.append(
            r"\noindent{\small\textit{Showing Turn 0 + turns "
            + str(win_lo) + r"--" + str(win_hi)
            + r" of " + str(total_turns)
            + r" (window around the " + pivot_kind + r" turn).}}"
            r"\par\smallskip"
        )

    # Turn 0 first if it was trimmed out of the window.
    if prepend_turn0:
        _append_turn_zero_row(
            lines, turn0=row["turns"][0], obs_max_chars=obs_max_chars
        )
        if show_ellipsis:
            lines.append(
                r"\noindent\makebox[5em][l]{}{\small\textit{$\vdots$ "
                r"(turns " + str(1) + r"--" + str(start_idx - 1)
                + r" elided)}}\par\smallskip"
            )

    end_idx = start_idx + len(turns_to_show) - 1
    show_trailing_ellipsis = end_idx < total_turns - 1

    for offset, turn in enumerate(turns_to_show):
        ti = start_idx + offset
        # Turn 0 is rendered uncolored (no credit, no marker) regardless
        # of whether it landed inside or before the window.
        if ti == 0:
            _append_turn_zero_row(
                lines, turn0=turn, obs_max_chars=obs_max_chars
            )
            continue
        c = float(turn["credit"])
        cn = c / max_abs
        # Only the action receives the color cue (the action is what
        # incurs reward / blame; the observation is environmental
        # context the agent didn't choose).
        act_rgb = _credit_to_rgb(cn, darker=False)
        act_name = f"tr{tag}t{ti}act"
        if act_name not in color_defined:
            color_defs.append(
                r"\definecolor{" + act_name + r"}{HTML}{"
                + _rgb_to_html(act_rgb) + r"}"
            )
            color_defined.add(act_name)

        obs_text = _latex_escape(turn.get("observation_text", ""), obs_max_chars)
        act_text = _latex_escape(turn.get("action_text", ""), obs_max_chars)

        sign = "+" if c >= 0 else ""
        is_pivot = (ti == pivot_idx)
        pivot_marker = r"~\(\bigstar\)" if is_pivot else r""
        pivot_subtag = (
            r"~{\small\textit{(" + pivot_tag_text + r")}}"
            if is_pivot else r""
        )

        turn_num = int(turn.get("turn_idx", ti))
        turn_label = (
            r"\makebox[5em][l]{\textbf{Turn " + str(turn_num) + r":}}"
        )
        action_label = r"\makebox[5em][l]{\textbf{Action:}}"
        lines.append(
            r"\noindent" + turn_label + r"{\small " + obs_text + r"}\\"
        )
        lines.append(
            action_label + r"\colorbox{" + act_name
            + r"}{\small " + act_text + r"}~"
            r"{\small\textit{(credit = " + f"{sign}{c:.3f}" + r")}}"
            + pivot_marker + pivot_subtag
            + r"\par\smallskip"
        )

    if show_trailing_ellipsis:
        # Window ends before the trajectory does -- emit a trailing
        # ellipsis row showing how many turns were elided and whether
        # the trajectory ended in truncation (n_turns == max env step
        # budget) versus natural termination.
        last_omitted = total_turns - 1
        n_elided = last_omitted - end_idx
        # For failures that hit max_turns the trajectory was truncated
        # by the env step budget; phrase the ellipsis to reflect that.
        if not success:
            tail_msg = (
                r"\textit{$\vdots$ (turns " + str(end_idx + 1)
                + r"--" + str(last_omitted) + r" elided; trajectory "
                r"ran to step limit without reaching the goal)}"
            )
        else:
            tail_msg = (
                r"\textit{$\vdots$ (turns " + str(end_idx + 1)
                + r"--" + str(last_omitted) + r" elided)}"
            )
        lines.append(
            r"\noindent\makebox[5em][l]{}{\small " + tail_msg
            + r"}\par\smallskip"
        )

    lines.append(r"\medskip")
    lines.append("")
    return "\n".join(lines)


def _append_turn_zero_row(
    lines: list[str],
    *,
    turn0: dict[str, Any],
    obs_max_chars: int,
) -> None:
    """Render Turn 0 as an uncolored two-line block (no credit, no marker).

    Turn 0 carries the task's goal description and an initial observation
    of the starting environment state. Its credit is noisy (the V-head
    has no action consequence to evaluate yet) and dominates the visual
    field if colored, so we display it without a colorbox.
    """
    obs_text = _latex_escape(turn0.get("observation_text", ""), obs_max_chars)
    act_text = _latex_escape(turn0.get("action_text", ""), obs_max_chars)
    turn_label = r"\makebox[5em][l]{\textbf{Turn 0:}}"
    action_label = r"\makebox[5em][l]{\textbf{Action:}}"
    lines.append(
        r"\noindent" + turn_label + r"{\small " + obs_text
        + r"}~{\scriptsize\textit{(initial)}}\\"
    )
    lines.append(
        action_label + r"{\small " + act_text + r"}\par\smallskip"
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Render TurnRD per-turn credit assignment as a color-gradient "
            "LaTeX figure."
        )
    )
    ap.add_argument("--probe-jsonl", required=True, type=Path)
    ap.add_argument("--n-success", type=int, default=3)
    ap.add_argument("--n-failure", type=int, default=3)
    ap.add_argument("--out-tex", required=True, type=Path)
    ap.add_argument(
        "--obs-max-chars", type=int, default=600,
        help="Per-cell character cap for observation/action text.",
    )
    ap.add_argument(
        "--snippet-only", action="store_true",
        help="Emit only \\definecolor + body (no \\documentclass / preamble), "
             "for pasting into an existing report.",
    )
    ap.add_argument(
        "--max-display-turns", type=int, default=12,
        help="For trajectories longer than this, show only a window of "
             "this many turns centered on the most-decisive turn. Set to "
             "0 to disable trimming and always show every turn.",
    )
    ap.add_argument(
        "--max-success-turns", type=int, default=12,
        help="Upper bound on the success-bucket trajectory length filter. "
             "Default 12 keeps every selected success fully visible without "
             "windowing. Raise to e.g. 40 to allow long marathon successes "
             "(they will then be auto-trimmed by --max-display-turns to a "
             "window around the decisive turn). Failures keep their wider "
             "filter independently.",
    )
    ap.add_argument(
        "--failure-diversify", action="store_true",
        help="Pick failures with diverse fatal-turn positions (early / "
             "mid / late give-up patterns) instead of top-N by "
             "concentration. Useful when every failure is a truncation "
             "(n_turns == max_turns) so length-based diversity isn't "
             "available -- it shows different failure modes instead of "
             "three near-identical close-empty-drawer examples.",
    )
    args = ap.parse_args()

    rows = _load_jsonl(args.probe_jsonl)

    success_pool = [
        r for r in rows
        if r.get("success") and r["turns"]
        and _is_success_useful(r, max_turns=args.max_success_turns)
    ]
    failure_pool = [
        r for r in rows
        if not r.get("success") and r["turns"] and _is_failure_useful(r)
    ]

    success_pool.sort(key=_rank_success, reverse=True)
    failure_pool.sort(key=_rank_failure, reverse=True)

    chosen_success = success_pool[: args.n_success]
    if args.failure_diversify:
        chosen_failure = _pick_diverse_failures(failure_pool, args.n_failure)
    else:
        chosen_failure = failure_pool[: args.n_failure]

    color_defs: list[str] = []
    color_defined: set[str] = set()
    bodies: list[str] = []

    bodies.append(r"\section*{Successes}")
    if not chosen_success:
        bodies.append(
            r"\textit{No success trajectories with 2--12 turns "
            r"available in the probe JSONL.}"
        )
    for i, row in enumerate(chosen_success):
        bodies.append(_build_trajectory_block(
            row,
            color_defs=color_defs,
            color_defined=color_defined,
            obs_max_chars=args.obs_max_chars,
            tag=f"s{i}",
            max_display_turns=args.max_display_turns,
        ))

    bodies.append(r"\section*{Failures}")
    if not chosen_failure:
        bodies.append(
            r"\textit{No failure trajectories available in the probe JSONL.}"
        )
    for i, row in enumerate(chosen_failure):
        bodies.append(_build_trajectory_block(
            row,
            color_defs=color_defs,
            color_defined=color_defined,
            obs_max_chars=args.obs_max_chars,
            tag=f"f{i}",
            max_display_turns=args.max_display_turns,
        ))

    parts: list[str] = []
    if not args.snippet_only:
        parts.append(r"\documentclass{article}")
        parts.append(r"\usepackage[margin=1in]{geometry}")
        parts.append(r"\usepackage{xcolor}")
        parts.append(r"\usepackage{amssymb}")
        parts.append(r"\setlength{\fboxsep}{2pt}")
    parts.extend(color_defs)
    if not args.snippet_only:
        parts.append(r"\begin{document}")
    parts.append(
        r"\section*{TurnRD Per-Turn Credit Assignment --- v3 R9 Policy}"
    )
    parts.append(
        r"\noindent{\small\textit{Greedy K=1 rollouts on AlfWorld held-out "
        r"tasks. Each turn shows the observation on the first line and the "
        r"agent's action on the second; only the action is shaded by per-turn "
        r"credit (green = positive, red = negative) because the action is "
        r"what receives reward / blame. Color saturation is normalized per "
        r"trajectory. The most-decisive turn in successes (largest positive "
        r"credit) and the most-fatal turn in failures (largest negative "
        r"credit) are marked with $\bigstar$. Turn 0 is the initial "
        r"observation (uncolored, no credit). By construction, per-turn "
        r"credits sum to the trajectory's final reward $R$ via the "
        r"v-projection.}}"
    )
    parts.append("")
    parts.extend(bodies)
    if not args.snippet_only:
        parts.append(r"\end{document}")

    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text("\n".join(parts) + "\n")

    print(
        f"Wrote {args.out_tex} | "
        f"success={len(chosen_success)}/{args.n_success} (pool={len(success_pool)}) "
        f"failure={len(chosen_failure)}/{args.n_failure} (pool={len(failure_pool)}) "
        f"colordefs={len(color_defs)}"
    )


if __name__ == "__main__":
    main()
