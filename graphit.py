#!/usr/bin/env python3
"""
graphit.py — plot data from a single full_<N>_0 file

Usage:
    python graphit.py P1 P2 P3 P4 P5 P6 P7

    P1..P4 (0 or 1): include section
        P1 -> itree
        P2 -> mfstree
        P3 -> vamfstree
        P4 -> vamfstree (verify)
    P5 (0 or 1): if 1, square every n value before plotting (plot vs n^2)
    P6 (0..3): y-axis choice
        0 -> time
        1 -> value1
        2 -> value2
        3 -> value1 + value2
    P7 (integer): which file to read — picks full_<P7>_0
                  e.g. 2 -> full_2_0, 3 -> full_3_0, 4 -> full_4_0
    Note: the verify section has no value2, so P6 in {1, 2, 3}
          all behave like P6 = 1 for that section.

Output is saved under ./graphs/ (created if missing).
"""

import os
import re
import sys

import matplotlib.pyplot as plt
import seaborn as sns


SECTIONS = ["itree", "mfstree", "vamfstree", "vamfstree (verify)"]
LEGEND_LABELS = {
    "itree": "I-tree",
    "mfstree": "AOF-tree",
    "vamfstree": "Pruned AOF-tree",
    "vamfstree (verify)": "Verifiable Construction",
}
SECTION_COLORS = {
    "itree": "red",
    "mfstree": "orange",
    "vamfstree": "green",
    "vamfstree (verify)": "blue",
}


def parse_file(path):
    """Parse a data file.

    Returns a dict: section_name -> list of tuples (n, time, v1, v2)
    where v1 / v2 may be None if the section row has fewer columns.
    """
    with open(path) as f:
        text = f.read()

    result = {s: [] for s in SECTIONS}
    current = None

    header_re = re.compile(r"===\s*(.+?)\s*===")

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line == "---":
            continue

        m = header_re.match(line)
        if m:
            current = m.group(1).strip()
            result.setdefault(current, [])
            continue

        if current is None or ":" not in line:
            continue

        left, right = line.split(":", 1)
        try:
            n = int(left.strip())
        except ValueError:
            continue

        parts = [p.strip() for p in right.split(",") if p.strip()]
        if not parts:
            continue

        time = float(parts[0])
        v1 = float(parts[1]) if len(parts) > 1 else None
        v2 = float(parts[2]) if len(parts) > 2 else None
        result[current].append((n, time, v1, v2))

    return result


def pick_y(row, choice, is_verify):
    """Pick the y-value for a row based on the P6 selector."""
    _, time, v1, v2 = row
    if choice == 0:
        return time
    # verify section only has v1 — any of 1/2/3 collapse to v1
    if is_verify:
        return v1
    if choice == 1:
        return v1
    if choice == 2:
        return v2
    if choice == 3:
        if v1 is None or v2 is None:
            return None
        return v1 + v2
    raise ValueError(f"bad y-choice: {choice}")


def main():
    if len(sys.argv) != 8:
        print(__doc__)
        sys.exit(1)

    try:
        flags = [int(x) for x in sys.argv[1:5]]
        square_n = int(sys.argv[5]) == 1
        y_choice = int(sys.argv[6])
        file_num = int(sys.argv[7])
    except ValueError:
        print("all arguments must be integers")
        print(__doc__)
        sys.exit(1)

    if any(f not in (0, 1) for f in flags):
        print("P1..P4 must each be 0 or 1")
        sys.exit(1)
    if int(sys.argv[5]) not in (0, 1):
        print("P5 must be 0 or 1")
        sys.exit(1)
    if y_choice not in (0, 1, 2, 3):
        print("P6 must be 0, 1, 2, or 3")
        sys.exit(1)

    wanted_sections = [s for s, on in zip(SECTIONS, flags) if on]
    if not wanted_sections:
        print("no sections selected (P1..P4 all zero)")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    fname = f"full_{file_num}_0"
    fpath = os.path.join(script_dir, fname)
    if not os.path.exists(fpath):
        print(f"data file not found: {fpath}")
        sys.exit(1)

    parsed = parse_file(fpath)

    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(11, 7))

    for section in wanted_sections:
        rows = parsed.get(section, [])
        if not rows:
            continue

        is_verify = section == "vamfstree (verify)"
        xs, ys = [], []
        for row in rows:
            n = row[0]
            x = (((n * n) - n) / 2) if square_n else n
            y = pick_y(row, y_choice, is_verify)
            if y is None:
                continue
            xs.append(x)
            ys.append(y)

        if not xs:
            continue

        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            color=SECTION_COLORS[section],
            label=LEGEND_LABELS.get(section, section),
        )

    title_map = {
        0: "Construction/Verification Time",
        1: "Index Storage",
        2: "ADS Storage",
        3: "Total Storage",
    }
    x_label = "Intersections" if square_n else "Functions"
    y_label = "Time (Seconds)" if y_choice == 0 else "Storage (Bytes)"

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title_map[y_choice]}, d = {file_num}")
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()

    graphs_dir = os.path.join(script_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    prefix_map = {0: "cons", 1: "index", 2: "merkle", 3: "total"}
    letter_map = ["i", "b", "a", "v"]  # P1..P4
    letters = "".join(letter_map[i] for i, on in enumerate(flags) if on)

    out_name = f"{prefix_map[y_choice]}_{letters}_{file_num}.png"
    out_path = os.path.join(graphs_dir, out_name)
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")

    if os.environ.get("DISPLAY") or sys.platform in ("darwin", "win32"):
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
