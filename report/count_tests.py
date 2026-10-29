"""Count the preregistered primary tests in paper/tab_primary.tex and apply Holm across all studies (conservative: it
uses the within-study Holm p values, and an upper bound for entries given as < 10^-k). With --mark, rewrites the
table's outcome cells: "supported$^{*}$" for supported tests that do not survive Holm across all tests.
usage: python report/count_tests.py [--mark] [TABLE]"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
mark = "--mark" in sys.argv
args = [a for a in sys.argv[1:] if a != "--mark"]
path = args[0] if args else os.path.join(HERE, "..", "paper", "tab_primary.tex")
tab = open(path).read()
rows, where = [], []
for j, line in enumerate(tab.splitlines()):
    if line.strip().startswith("%") or "&" not in line or "supported" not in line:
        continue
    c = [x.strip() for x in line.rstrip("\\ ").split("&")]
    p, out = c[4], c[5].replace("$^{*}$", "")
    where.append(j)
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
if mark:
    lines = tab.splitlines()
    for i, j in enumerate(where):
        if rows[i][3]:
            new = "supported" if glob_ok[i] else "supported$^{*}$"
            lines[j] = re.sub(r"supported(\$\^\{\*\}\$)?(\s*\\\\)", lambda m: new + m[2], lines[j])
    open(path, "w").write("\n".join(lines) + "\n")
    print("marked", sum(1 for i in range(n) if rows[i][3] and not glob_ok[i]), "rows")
