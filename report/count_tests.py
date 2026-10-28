"""Count the preregistered primary tests in paper/tab_primary.tex and apply Holm across all studies (conservative: it
uses the within-study Holm p values, and an upper bound for entries given as < 10^-k).
usage: python report/count_tests.py [TABLE]"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
tab = open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "paper", "tab_primary.tex")).read()
rows = []
for line in tab.splitlines():
    if line.strip().startswith("%") or "&" not in line or "supported" not in line:
        continue
    c = [x.strip() for x in line.rstrip("\\ ").split("&")]
    p, out = c[4], c[5]
    m = re.search(r"10\^\{(-\d+)\}", p)
    pv = 10.0 ** int(m[1]) if m else (0.001 if "0.001" in p and "<" in p else float(re.sub(r"[^0-9.]", "", p)))
    rows.append((c[0] or rows[-1][0], c[1], pv, out == "supported"))
n, sup = len(rows), sum(r[3] for r in rows)
order = sorted(range(n), key=lambda i: rows[i][2])
run, glob_ok = 0.0, {}
for k, i in enumerate(order):
    run = max(run, min(1.0, (n - k) * rows[i][2]))
    glob_ok[i] = run < 0.05
keep = sum(1 for i in range(n) if rows[i][3] and glob_ok[i])
print(f"{n} tests, {sup} supported, {n - sup} not supported; {keep} of the {sup} survive Holm across all {n}")
for i in range(n):
    if rows[i][3] and not glob_ok[i]:
        print("  not surviving:", rows[i][0], "|", rows[i][1], rows[i][2])
