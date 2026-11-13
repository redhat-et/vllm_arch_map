#!/usr/bin/env python3
"""Compare GPU kernel profiles from two nsys traces.

Reads nsys SQLite files, merges all kernels into one table with
side-by-side columns, sorted by max time across both runs.

Usage:
  python diff_kernels.py a.sqlite b.sqlite
  python diff_kernels.py a.sqlite b.sqlite --label-a FLASH_ATTN --label-b FLASHINFER
  python diff_kernels.py a.sqlite b.sqlite -o comparison.sqlite
  python diff_kernels.py a.sqlite b.sqlite --json -o comparison.json
"""
import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KernelStats:
    name: str
    total_ns: int
    instances: int
    avg_ns: float
    min_ns: int
    max_ns: int
    mangled_name: str = ""

    @property
    def total_ms(self):
        return self.total_ns / 1e6

    @property
    def avg_us(self):
        return self.avg_ns / 1e3


@dataclass
class MergedRow:
    name: str
    total_ns_a: int = 0
    total_ns_b: int = 0
    instances_a: int = 0
    instances_b: int = 0
    avg_ns_a: float = 0
    avg_ns_b: float = 0
    side: str = "both"  # "both", "a_only", "b_only"
    mangled_name: str = ""

    @property
    def delta_pct(self):
        if self.total_ns_a == 0:
            return None
        return (self.total_ns_b - self.total_ns_a) / self.total_ns_a * 100

    @property
    def sort_key(self):
        return max(self.total_ns_a, self.total_ns_b)


# ── SQLite loading ─────────────────────────────────────────────────

def _find_table(conn, keywords):
    tables = [
        row[0] for row in
        conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]
    for kw in keywords:
        for t in tables:
            if kw in t.upper():
                return t
    return None


def detect_workload_segments(sqlite_path, gap_sec=3, skip_warmup=True):
    """Detect workload time segments by finding gaps in kernel activity."""
    conn = sqlite3.connect(sqlite_path)
    rows = conn.execute('''
        SELECT CAST(start / 1000000000 AS INT) as sec, COUNT(*) as cnt
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY sec ORDER BY sec
    ''').fetchall()
    conn.close()

    segments = []
    seg_start = rows[0][0]
    prev = rows[0][0]
    for sec, cnt in rows[1:]:
        if sec - prev > gap_sec:
            segments.append((seg_start, prev))
            seg_start = sec
        prev = sec
    segments.append((seg_start, prev))

    if skip_warmup and len(segments) > 1:
        segments = segments[1:]
    return segments


def load_kernels(sqlite_path: str, time_range=None) -> list[KernelStats]:
    conn = sqlite3.connect(sqlite_path)

    table = _find_table(conn, ["KERNEL"])
    if not table:
        tables = [
            row[0] for row in
            conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
        raise ValueError(f"No kernel table found. Tables: {sorted(tables)}")

    cols = {row[1] for row in conn.execute(f"PRAGMA table_info('{table}')").fetchall()}

    name_col = next((c for c in ["demangledName", "shortName", "mangledName"] if c in cols), None)
    if not name_col:
        raise ValueError(f"No name column in {table}. Columns: {sorted(cols)}")

    if "end" in cols and "start" in cols:
        dur_expr = '("end" - "start")'
    elif "duration" in cols:
        dur_expr = '"duration"'
    else:
        raise ValueError(f"No duration column in {table}. Columns: {sorted(cols)}")

    string_table = _find_table(conn, ["STRINGID", "STRING"])

    time_filter = ""
    if time_range:
        start_ns = time_range[0] * 1_000_000_000
        end_ns = (time_range[1] + 1) * 1_000_000_000
        time_filter = f' WHERE k."start" >= {start_ns} AND k."start" < {end_ns}'

    if string_table:
        query = f"""
            SELECT s.value AS name,
                   COUNT(*) AS instances,
                   SUM({dur_expr}) AS total_ns,
                   AVG({dur_expr}) AS avg_ns,
                   MIN({dur_expr}) AS min_ns,
                   MAX({dur_expr}) AS max_ns
            FROM "{table}" k
            JOIN "{string_table}" s ON k."{name_col}" = s.id
            {time_filter}
            GROUP BY s.value
            ORDER BY total_ns DESC
        """
    else:
        query = f"""
            SELECT "{name_col}" AS name,
                   COUNT(*) AS instances,
                   SUM({dur_expr}) AS total_ns,
                   AVG({dur_expr}) AS avg_ns,
                   MIN({dur_expr}) AS min_ns,
                   MAX({dur_expr}) AS max_ns
            FROM "{table}" k
            {time_filter}
            GROUP BY "{name_col}"
            ORDER BY total_ns DESC
        """

    rows = conn.execute(query).fetchall()

    mangled_map = {}
    if string_table and name_col == "demangledName" and "mangledName" in cols:
        try:
            mq = f"""
                SELECT s_d.value, MIN(s_m.value)
                FROM "{table}" k
                JOIN "{string_table}" s_d ON k."demangledName" = s_d.id
                JOIN "{string_table}" s_m ON k."mangledName" = s_m.id
                {time_filter}
                GROUP BY s_d.value
            """
            mangled_map = dict(conn.execute(mq).fetchall())
        except Exception:
            pass

    conn.close()

    return [
        KernelStats(
            name=name, total_ns=int(total_ns), instances=int(instances),
            avg_ns=float(avg_ns), min_ns=int(min_ns), max_ns=int(max_ns),
            mangled_name=mangled_map.get(name, ""),
        )
        for name, instances, total_ns, avg_ns, min_ns, max_ns in rows
    ]


# ── GPU overhead extraction ───────────────────────────────────────

def load_gpu_overhead(sqlite_path: str, time_range=None) -> dict:
    """Extract memcpy, CPU runtime, and sync overhead from nsys sqlite."""
    conn = sqlite3.connect(sqlite_path)
    result = {}

    time_filter = ""
    if time_range:
        start_ns = time_range[0] * 1_000_000_000
        end_ns = (time_range[1] + 1) * 1_000_000_000
        time_filter = f' WHERE "start" >= {start_ns} AND "start" < {end_ns}'

    # Memcpy
    try:
        string_table = _find_table(conn, ["STRINGID", "STRING"])
        memcpy_table = _find_table(conn, ["ACTIVITY_KIND_MEMCPY"])
        if memcpy_table and string_table:
            tf = time_filter.replace('"start"', f'k."start"') if time_filter else ''
            rows = conn.execute(f"""
                SELECT SUM(k."end" - k."start") as total_ns,
                       COUNT(*) as cnt,
                       SUM(k.bytes) as total_bytes
                FROM "{memcpy_table}" k
                {tf}
            """).fetchone()
            if rows and rows[0]:
                result["memcpy_ms"] = round(rows[0] / 1e6, 2)
                result["memcpy_count"] = rows[1]
                result["memcpy_mb"] = round((rows[2] or 0) / 1e6, 1)
    except Exception:
        pass

    # CPU CUDA runtime calls
    try:
        runtime_table = _find_table(conn, ["ACTIVITY_KIND_RUNTIME"])
        if runtime_table and string_table:
            tf = time_filter.replace('"start"', f'r."start"') if time_filter else ''
            rows = conn.execute(f"""
                SELECT s.value as name,
                       SUM(r."end" - r."start") as total_ns,
                       COUNT(*) as cnt
                FROM "{runtime_table}" r
                JOIN "{string_table}" s ON r.nameId = s.id
                {tf}
                GROUP BY s.value
                ORDER BY total_ns DESC
                LIMIT 10
            """).fetchall()
            runtime = []
            total_runtime_ns = 0
            for name, total_ns, cnt in rows:
                clean = name.replace('_v3020', '').replace('_v7000', '').replace('_v11060', '').replace('_v11040', '')
                runtime.append({"name": clean, "total_ms": round(total_ns / 1e6, 2), "count": cnt})
                total_runtime_ns += total_ns
            result["runtime"] = runtime
            result["runtime_total_ms"] = round(total_runtime_ns / 1e6, 2)
    except Exception:
        pass

    # Sync
    try:
        sync_table = _find_table(conn, ["ACTIVITY_KIND_SYNCHRONIZATION"])
        if sync_table:
            tf = time_filter.replace('"start"', f's."start"') if time_filter else ''
            rows = conn.execute(f"""
                SELECT SUM(s."end" - s."start") as total_ns, COUNT(*) as cnt
                FROM "{sync_table}" s
                {tf}
            """).fetchone()
            if rows and rows[0]:
                result["sync_ms"] = round(rows[0] / 1e6, 2)
                result["sync_count"] = rows[1]
    except Exception:
        pass

    conn.close()
    return result


# ── Merge ──────────────────────────────────────────────────────────

def merge(kernels_a, kernels_b) -> list[MergedRow]:
    a_map = {k.name: k for k in kernels_a}
    b_map = {k.name: k for k in kernels_b}
    all_names = set(a_map) | set(b_map)

    rows = []
    for name in all_names:
        ka = a_map.get(name)
        kb = b_map.get(name)

        if ka and kb:
            side = "both"
        elif ka:
            side = "a_only"
        else:
            side = "b_only"

        mangled = ""
        if ka and ka.mangled_name:
            mangled = ka.mangled_name
        elif kb and kb.mangled_name:
            mangled = kb.mangled_name

        rows.append(MergedRow(
            name=name,
            total_ns_a=ka.total_ns if ka else 0,
            total_ns_b=kb.total_ns if kb else 0,
            instances_a=ka.instances if ka else 0,
            instances_b=kb.instances if kb else 0,
            avg_ns_a=ka.avg_ns if ka else 0,
            avg_ns_b=kb.avg_ns if kb else 0,
            side=side,
            mangled_name=mangled,
        ))

    rows.sort(key=lambda r: r.sort_key, reverse=True)
    return rows


# ── Formatting ─────────────────────────────────────────────────────

def _fmt_ms(ns):
    if ns == 0:
        return "—"
    ms = ns / 1e6
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    if ms >= 1:
        return f"{ms:.1f}ms"
    return f"{ms * 1000:.0f}µs"


def _fmt_calls(n):
    if n == 0:
        return "—"
    return f"{n:,}"


def _fmt_delta(pct):
    if pct is None:
        return ""
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"


def _short(name, maxlen=55):
    if len(name) <= maxlen:
        return name
    return name[:maxlen - 3] + "..."


def _pct(ns, total):
    if total == 0:
        return 0
    return ns / total * 100


# ── Text output ────────────────────────────────────────────────────

def print_report(rows, kernels_a, kernels_b, label_a, label_b):
    total_a = sum(k.total_ns for k in kernels_a)
    total_b = sum(k.total_ns for k in kernels_b)
    calls_a = sum(k.instances for k in kernels_a)
    calls_b = sum(k.instances for k in kernels_b)

    shared = [r for r in rows if r.side == "both"]
    only_a = [r for r in rows if r.side == "a_only"]
    only_b = [r for r in rows if r.side == "b_only"]

    # Summary
    delta_t = (total_b - total_a) / total_a * 100 if total_a else 0
    delta_c = (calls_b - calls_a) / calls_a * 100 if calls_a else 0
    print(f"\n  {'':>20} {label_a:>14}  {label_b:>14}  {'Delta':>7}")
    print(f"  {'GPU kernel time':>20} {_fmt_ms(total_a):>14}  {_fmt_ms(total_b):>14}  "
          f"{_fmt_delta(delta_t):>7}")
    print(f"  {'Total calls':>20} {calls_a:>14,}  {calls_b:>14,}  "
          f"{_fmt_delta(delta_c):>7}")
    print(f"  {'Shared kernels':>20} {len(shared):>14}  "
          f"{'Only ' + label_a: >14}: {len(only_a)}  Only {label_b}: {len(only_b)}")
    print()

    # Merged table
    la = label_a[:10]
    lb = label_b[:10]
    print(f"  {'%':>5}  {la + ' time':>10}  {la + ' calls':>10}  "
          f"{lb + ' time':>10}  {lb + ' calls':>10}  "
          f"{'Δ time':>7}  {'Δ calls':>8}  Name")
    print("  " + "─" * 120)

    for r in rows:
        pct = _pct(max(r.total_ns_a, r.total_ns_b), max(total_a, total_b))
        delta = _fmt_delta(r.delta_pct)

        if r.instances_a == 0 or r.instances_b == 0:
            dcalls = ""
        elif r.instances_a == r.instances_b:
            dcalls = "="
        else:
            dc = r.instances_b - r.instances_a
            dcalls = f"{dc:+,}"

        side_marker = ""
        if r.side == "a_only":
            side_marker = f" ◀ {la} only"
        elif r.side == "b_only":
            side_marker = f" ▶ {lb} only"
        elif r.delta_pct is not None and abs(r.delta_pct) > 20:
            side_marker = " ✦"

        print(f"  {pct:>4.1f}%  {_fmt_ms(r.total_ns_a):>10}  {_fmt_calls(r.instances_a):>10}  "
              f"{_fmt_ms(r.total_ns_b):>10}  {_fmt_calls(r.instances_b):>10}  "
              f"{delta:>7}  {dcalls:>8}  {_short(r.name)}{side_marker}")

    print("  " + "─" * 120)
    delta_total = _fmt_delta((total_b - total_a) / total_a * 100 if total_a else 0)
    print(f"  {'':>5}  {_fmt_ms(total_a):>10}  {calls_a:>10,}  "
          f"{_fmt_ms(total_b):>10}  {calls_b:>10,}  "
          f"{delta_total:>7}")
    print()


# ── SQLite output ──────────────────────────────────────────────────

def write_sqlite(rows, kernels_a, kernels_b, label_a, label_b, out_path):
    total_a = sum(k.total_ns for k in kernels_a)
    total_b = sum(k.total_ns for k in kernels_b)

    conn = sqlite3.connect(out_path)
    conn.execute("DROP TABLE IF EXISTS kernel_diff")
    conn.execute("""
        CREATE TABLE kernel_diff (
            name TEXT,
            total_ns_a INTEGER,
            total_ns_b INTEGER,
            instances_a INTEGER,
            instances_b INTEGER,
            avg_ns_a REAL,
            avg_ns_b REAL,
            delta_pct REAL,
            delta_ns INTEGER,
            side TEXT,
            pct_of_total REAL
        )
    """)
    for r in rows:
        conn.execute(
            "INSERT INTO kernel_diff VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (r.name, r.total_ns_a, r.total_ns_b, r.instances_a, r.instances_b,
             r.avg_ns_a, r.avg_ns_b, r.delta_pct,
             r.total_ns_b - r.total_ns_a, r.side,
             max(r.total_ns_a, r.total_ns_b) / max(total_a, total_b) * 100),
        )

    conn.execute("DROP TABLE IF EXISTS metadata")
    conn.execute("""
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.execute("INSERT INTO metadata VALUES ('label_a', ?)", (label_a,))
    conn.execute("INSERT INTO metadata VALUES ('label_b', ?)", (label_b,))
    conn.execute("INSERT INTO metadata VALUES ('total_ns_a', ?)", (str(total_a),))
    conn.execute("INSERT INTO metadata VALUES ('total_ns_b', ?)", (str(total_b),))

    conn.commit()
    conn.close()


# ── JSON output ────────────────────────────────────────────────────

def write_json(rows, kernels_a, kernels_b, label_a, label_b):
    total_a = sum(k.total_ns for k in kernels_a)
    total_b = sum(k.total_ns for k in kernels_b)

    merged = []
    for r in rows:
        entry = {
            "name": r.name,
            "total_ms_a": round(r.total_ns_a / 1e6, 2),
            "total_ms_b": round(r.total_ns_b / 1e6, 2),
            "instances_a": r.instances_a,
            "instances_b": r.instances_b,
            "avg_us_a": round(r.avg_ns_a / 1e3, 1),
            "avg_us_b": round(r.avg_ns_b / 1e3, 1),
            "delta_pct": round(r.delta_pct, 1) if r.delta_pct is not None else None,
            "side": r.side,
        }
        if r.mangled_name:
            entry["mangled_name"] = r.mangled_name
        merged.append(entry)

    return {
        "label_a": label_a,
        "label_b": label_b,
        "total_ms_a": round(total_a / 1e6, 2),
        "total_ms_b": round(total_b / 1e6, 2),
        "kernels": merged,
    }


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("a", help="nsys SQLite file for run A")
    parser.add_argument("b", help="nsys SQLite file for run B")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of table")
    parser.add_argument("-o", "--output",
                        help="Write to file (.sqlite or .json)")
    args = parser.parse_args()

    print(f"Loading {args.a}...")
    kernels_a = load_kernels(args.a)
    print(f"  {len(kernels_a)} kernel types, "
          f"{sum(k.instances for k in kernels_a):,} total calls")

    print(f"Loading {args.b}...")
    kernels_b = load_kernels(args.b)
    print(f"  {len(kernels_b)} kernel types, "
          f"{sum(k.instances for k in kernels_b):,} total calls")

    rows = merge(kernels_a, kernels_b)

    if args.output and args.output.endswith(".sqlite"):
        write_sqlite(rows, kernels_a, kernels_b,
                     args.label_a, args.label_b, args.output)
        print(f"Wrote {args.output}")
        print(f"  Query: SELECT * FROM kernel_diff ORDER BY pct_of_total DESC")
    elif args.output and args.output.endswith(".json"):
        data = write_json(rows, kernels_a, kernels_b, args.label_a, args.label_b)
        Path(args.output).write_text(json.dumps(data, indent=2))
        print(f"Wrote {args.output}")
    elif args.json:
        data = write_json(rows, kernels_a, kernels_b, args.label_a, args.label_b)
        print(json.dumps(data, indent=2))
    else:
        print_report(rows, kernels_a, kernels_b, args.label_a, args.label_b)


if __name__ == "__main__":
    main()
