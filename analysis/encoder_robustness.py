"""Encoder robustness of the content results (exploratory): every preregistered content test, scored once with CLIP
ViT-L/14 (as preregistered) and once with SigLIP so400m-patch14-384 (siglip_probe.py), by running the same frozen
analysis on SigLIP embeddings (symlinked as *.clip.npz, text embeddings from the SigLIP text tower).
Writes a LaTeX table and a JSON of the pairs.
usage: python encoder_robustness.py RESULTS_DIR OUT_TEX OUT_JSON
"""
import json
import os
import sys

R, out_tex, out_json = sys.argv[1:4]
# study, test, label, CLIP result json, SigLIP result json, key path to the test
ROWS = [("09", "H1", r"prompt sets content: $s_{\text{class}} - s_{\text{generic}}$ (30\,s)",
         "content_out/content_results.json", "enc2_out/out_content/content_results.json", ("primary", "H1")),
        ("09", "H2", r"LongLive sink keeps content: $\Delta\kappa$ (30\,s)",
         "content_out/content_results.json", "enc2_out/out_content/content_results.json", ("primary", "H2")),
        ("09", "H3", r"Self Forcing sink keeps content: $\Delta\kappa$ (60\,s)",
         "content_out/content_results.json", "enc2_out/out_content/content_results.json", ("primary", "H3")),
        ("16", "A1", r"late return is positional: $\Delta R$",
         "s16_out/s16_results.json", "enc2_out/out_s16/s16_results.json", ("primary", "A1")),
        ("16", "A2", r"re-rotation lowers content: $\Delta\kappa$ (60\,s)",
         "s16_out/s16_results.json", "enc2_out/out_s16/s16_results.json", ("primary", "A2")),
        ("16", "A3", r"re-rotated sink keeps content: $\Delta\kappa$ (60\,s)",
         "s16_out/s16_results.json", "enc2_out/out_s16/s16_results.json", ("primary", "A3")),
        ("17", "H1", r"FramePack content relaxes: $s - s_{\text{open}}$",
         "fpc_out/fpc_results.json", "enc2_out/out_fpc/fpc_results.json", ("primary", "H1")),
        ("17", "H2", r"context prefix carries content: $\kappa$",
         "fpc_out/fpc_results.json", "enc2_out/out_fpc/fpc_results.json", ("primary", "H2")),
        ("17", "H3", r"SigLIP embedding carries content: $\kappa$",
         "fpc_out/fpc_results.json", "enc2_out/out_fpc/fpc_results.json", ("primary", "H3")),
        ("17", "H4", r"embedding adds content: $\Delta\kappa$",
         "fpc_out/fpc_results.json", "enc2_out/out_fpc/fpc_results.json", ("primary", "H4")),
        ("21", "H1", r"sink keeps content, 3 new seeds: $\Delta\kappa$ (60\,s)",
         "s21_out/s21_results.json", "enc2_out/out_s21/s21_results.json", ("primary", "H1")),
        ("21", "H2", r"re-rotation lowers content: $\Delta\kappa$ (60\,s)",
         "s21_out/s21_results.json", "enc2_out/out_s21/s21_results.json", ("primary", "H2")),
        ("21", "H3", r"re-rotated sink keeps content: $\Delta\kappa$ (60\,s)",
         "s21_out/s21_results.json", "enc2_out/out_s21/s21_results.json", ("primary", "H3")),
        ("21", "H4", r"late return is positional: $\Delta R$",
         "s21_out/s21_results.json", "enc2_out/out_s21/s21_results.json", ("primary", "H4")),
        ("19", "A1", r"MAGI-1 content relaxes: $s - s_{\text{open}}$",
         "magi19_out/magi19_results.json", "enc2_out/out_magi19/magi19_results.json", ("primary", "A1")),
        ("19", "A2", r"full history keeps more content: $\Delta\kappa$",
         "magi19_out/magi19_results.json", "enc2_out/out_magi19/magi19_results.json", ("primary", "A2")),
        ("19", "A3", r"prompt sets content: $s_{\text{class}} - s_{\text{generic}}$",
         "magi19_out/magi19_results.json", "enc2_out/out_magi19/magi19_results.json", ("primary", "A3"))]


def get(path, keys):
    p = os.path.join(R, path)
    if not os.path.exists(p):
        return None
    x = json.load(open(p))
    for k in keys:
        x = x.get(k) if isinstance(x, dict) else None
    return x


def cell(t):
    if t is None:
        return "--", "--"
    p = t.get("p_holm", t["p"])
    ptxt = "$<$0.001" if p < 0.001 else f"{p:.3f}"
    return f"{t['est']:+.2f}", ptxt + ("" if t.get("supported", True) else r"$^\dagger$")


pairs, lines = [], []
for st, h, lab, a, b, keys in ROWS:
    ta, tb = get(a, keys), get(b, keys)
    if ta is None and tb is None:
        continue
    pairs.append(dict(study=st, test=h, clip=ta, siglip=tb))
    (ea, pa), (eb, pb) = cell(ta), cell(tb)
    lines.append(f"{st} {h} & {lab} & {ea} & {pa} & {eb} & {pb} \\\\")
agree = sum(1 for x in pairs if x["clip"] and x["siglip"] and x["clip"]["supported"] == x["siglip"]["supported"])
json.dump(dict(pairs=pairs, agree=agree, n=len(pairs)), open(out_json, "w"), indent=1)
tex = r"""\begin{table}[h]
\centering\scriptsize
\caption{Content results do not depend on the scoring encoder (exploratory). Each preregistered content test rerun with
its frozen analysis on SigLIP so400m embeddings instead of CLIP ViT-L/14. Estimates in each encoder's log-odds or
$\kappa$ units; Holm $p$ within each study; $^\dagger$\,not supported. The verdict agrees for %d of %d tests.}
\label{tab:encoder}
\begin{tabular}{@{}llrrrr@{}}
\toprule
& & \multicolumn{2}{c}{CLIP} & \multicolumn{2}{c}{SigLIP} \\
Study & Test & est. & $p$ & est. & $p$ \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (agree, len(pairs), "\n".join(lines))
open(out_tex, "w").write(tex)
print(f"{agree}/{len(pairs)} verdicts agree")
for x in pairs:
    print(x["study"], x["test"], *(f"{t['est']:+.3f} {t.get('p_holm', t['p']):.3g} {t['supported']}" if t else "--"
                                   for t in (x["clip"], x["siglip"])))
