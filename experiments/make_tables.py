"""Turn results/*.json into LaTeX: paper/generated/numbers.tex (macros) and table bodies.

Every number quoted in the paper comes from here, so the paper cannot drift from the results.
"""
import os
import json
from common import ROOT, RESULTS

OUT = os.path.join(ROOT, "paper", "generated")
os.makedirs(OUT, exist_ok=True)
DS = [("syn-128", "SYN-128"), ("syn-768", "SYN-768"), ("arxiv-384", "ARXIV-384 (real)")]
TAG = {"syn-128": "A", "syn-768": "B", "arxiv-384": "C"}
DIM = {"syn-128": 128, "syn-768": 768, "arxiv-384": 384}


def load(n):
    p = os.path.join(RESULTS, f"{n}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def f2(x):
    return f"{x:.2f}"


def mname(k):
    """router / estimator key -> LaTeX-safe macro fragment (letters only)."""
    rep = {"-": " ", "+": " plus ", "%": "", "1": " one ", "2": " two ", "3": " three ", "5": " five ",
           "4": " four ", "6": " six ", "7": " seven ", "8": " eight ", "9": " nine ", "0": " zero "}
    for a, b in rep.items():
        k = k.replace(a, b)
    return "".join(w.capitalize() for w in k.split())


ROUTERS = ["always-pre", "always-post", "always-fused", "global-2way", "global-3way", "cbo-pg-hist", "rob-pg-hist",
           "cbo-exact-avi", "rob-exact-avi", "cbo-sample-1%", "rob-sample-1%", "cbo-probe", "rob-probe", "hvo-point",
           "hvo", "hvo-L1", "hvo-raw", "hvo-agnostic", "hvo+guard", "hvo-noplan", "cbo-true", "oracle",
           "rob50-hpbs+cal", "rob75-hpbs+cal", "rob95-hpbs+cal", "rob99-hpbs+cal"]
ESTS = ["pg-hist", "exact-avi", "sample-1%", "probe", "hpbs-cp", "hpbs-m256", "hpbs-L1", "hpbs", "hpbs+cal",
        "hpbs-L1+cal", "hpbs-cp+cal", "hpbs-m256+cal"]


def macros(R):
    m = []
    def mac(name, val):
        m.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    for n, _ in DS:
        o = R.get(n)
        if o is None:
            continue
        t = TAG[n]
        c, r = o["card"], o["routing"]["simd"]
        for e in ESTS:
            d = c[e]
            for q, qn in (("p50", "Med"), ("p95", "Pnf"), ("p99", "Pnn"), ("mean", "Mean")):
                mac(f"Q{mname(e)}{qn}{t}", f2(d["all"][q]))
            for reg in ("anti", "indep", "pos"):
                mac(f"Q{mname(e)}{reg.capitalize()}{t}", f2(d[reg]["p99"]))
            mac(f"Q{mname(e)}Rare{t}", f2(d["rare"]["p99"]) if d["rare"] else "--")
            mac(f"Q{mname(e)}Us{t}", f"{d['time_us']:.0f}")
            mac(f"Q{mname(e)}MB{t}", f"{d['bytes'] / 2**20:.1f}")
        for k in ROUTERS:
            d = r[k]
            for q, qn in (("p50", "Med"), ("p95", "Pnf"), ("p99", "Pnn"), ("mean", "Mean")):
                mac(f"R{mname(k)}{qn}{t}", f2(d["lat_ms"][q]))
            mac(f"R{mname(k)}Acc{t}", f"{100 * d['plan_acc']:.1f}")
            mac(f"R{mname(k)}RegMean{t}", f2(d["regret"]["mean"]))
            mac(f"R{mname(k)}RegPnn{t}", f2(d["regret"]["p99"]))
            mac(f"R{mname(k)}Rec{t}", f"{d['recall']:.3f}")
            mac(f"R{mname(k)}Us{t}", f"{d['plan_us']:.0f}")
        mac(f"Fired{t}", str(r["hvo+guard"]["fired"]))
        best = min(("global-2way", "global-3way"), key=lambda k: r[k]["lat_ms"]["mean"])
        mac(f"RBestGlobalMean{t}", f2(r[best]["lat_ms"]["mean"]))
        mac(f"RBestGlobalName{t}", "two-way" if best == "global-2way" else "three-way")
        cm_ = o["cost_model"]
        mac(f"ScanRatio{t}", f"{cm_['scalar']['scan_ns_per_vec'] / cm_['simd']['scan_ns_per_vec']:.1f}")
        mac(f"ScanRatioIEight{t}", f"{cm_['simd']['scan_ns_per_vec'] / cm_['i8']['scan_ns_per_vec']:.1f}")
        mac(f"MeanGainVsGlobal{t}", f"{100 * (1 - r['hvo']['lat_ms']['mean'] / r[best]['lat_ms']['mean']):.0f}")
        mac(f"PlanShareLOne{t}", f"{100 * r['hvo-L1']['plan_us'] / r['hvo']['plan_us']:.0f}")
        mac(f"EtaHvo{t}", f2(r["eta_train"]["hpbs+cal"]))
        mac(f"EtaProbe{t}", f2(r["eta_train"]["probe"]))
        mac(f"KappaMed{t}", f2(r["kappa_cal"]["p50"]))
        mac(f"KappaPnf{t}", f2(r["kappa_cal"]["p95"]))
        mac(f"ModelRegPnn{t}", f2(r["hvo"]["model_regret"]["p99"]))
        mac(f"nTest{t}", str(o["n_test"]))
        mac(f"nTrain{t}", str(o["n_train"]))
        mac(f"nRare{t}", str(o["n_rare"]))
        for reg in ("anti", "indep", "pos"):
            mac(f"n{reg.capitalize()}{t}", str(o["regime_counts"][reg]))
        cm = o["cost_model"]
        for pre in ("simd", "scalar", "i8"):
            mac(f"Scan{mname(pre)}{t}", f"{cm[pre]['scan_ns_per_vec']:.1f}")
        cs = cm["simd"]
        mac(f"RelNs{t}", f"{cs['rel_ns_per_row_term']:.2f}")
        mac(f"RelFixed{t}", f"{cs['rel_fixed_us']:.0f}")
        mac(f"PostGamma{t}", f"{cs['post']['gamma']:.2f}")
        mac(f"FusedGamma{t}", f"{cs['fused']['gamma']:.2f}")
        mac(f"FusedBeta{t}", f"{cs['fused_beta']:.2f}")
        mac(f"FusedCf{t}", f"{cs['fused_cf']:.2f}")
        mac(f"CalParams{t}", str(o["calibrator"]["params"]))
        bl = {}
        for line in open(os.path.join(ROOT, "data", n, "build_log.json")):
            bl.update(json.loads(line))
        mac(f"BuildHnsw{t}", f"{bl['hnsw_build_s']:.0f}")
        mac(f"BuildHpbs{t}", f"{bl['hpbs_build_s']:.0f}")
        mac(f"SketchPct{t}", f"{100 * bl['hpbs_bytes'] / (o['N'] * DIM[n] * 4):.2f}")
        mac(f"CalUs{t}", f"{o['calibrator']['forward_us']:.1f}")
        # noise sweep spot values
        for x in o["noise"]:
            tag = mname(str(x["eta"]).replace(".", "p"))
            mac(f"NoiseModel{tag}{t}", f2(x["model"]["p99"]))
            mac(f"NoiseRobust{tag}{t}", f2(x["robust_model"]["p99"]))
            mac(f"NoiseBound{tag}{t}", f2(x["bound_tight"]))
            mac(f"NoiseViol{tag}{t}", str(x["violations_general"] + x["violations_tight"] + x["violations_robust"]))
        mac(f"NoiseViolTotal{t}", str(sum(x["violations_general"] + x["violations_tight"] + x["violations_robust"]
                                          for x in o["noise"])))
        mac(f"NoiseChecks{t}", str(sum(x["n"] for x in o["noise"])))
    return "\n".join(m) + "\n"


def card_table(R):
    rows = []
    names = [("pg-hist", "1-D hist.\\ + AVI (PG-style)"), ("exact-avi", "Exact $s_g$, $s_{\\text{loc}}{=}s_g$ (Milvus-style)"),
             ("sample-1\\%", None), ("probe", "HNSW probe, $P{=}256$ (Exqutor-style)"),
             ("hpbs-cp", "HPBS, cross-polytope cells ($m{=}256$)"), ("hpbs-m256", "HPBS, $k$-means $m{=}256$"),
             ("hpbs-L1", "HPBS, $L{=}1$"), ("hpbs-L1+cal", "HPBS, $L{=}1$ + calibrator"),
             ("hpbs", "HPBS ($L{=}4$, $m{=}1024$), raw"), ("hpbs+cal", "\\textbf{HPBS + calibrator}")]
    for n, label in DS:
        o = R.get(n)
        if o is None:
            continue
        rows.append(f"\\multicolumn{{12}}{{l}}{{\\textit{{{label}}}}} \\\\")
        for key, nm in names:
            k = key.replace("\\%", "%")
            c = o["card"][k]
            nm = nm or "Uniform sample 1\\%"
            cells = []
            for reg in ("anti", "indep", "pos"):
                d = c[reg]
                cells += [f2(d["p50"]), f2(d["p95"]), f2(d["p99"])]
            rare = c["rare"]["p99"] if c["rare"] else float("nan")
            rows.append(f"{nm} & " + " & ".join(cells) + f" & {f2(rare)} & {c['time_us']:.0f} \\\\")
        rows.append("\\midrule")
    return "\n".join(rows[:-1]) + "\n"


def e2e_table(R):
    order = [("always-pre", "Always pre-filter"), ("always-post", "Always post-filter"),
             ("always-fused", "Always fused"), ("global-2way", "Global $s_g$ threshold (2-way)"),
             ("global-3way", "Global $s_g$ thresholds (3-way)"), ("cbo-pg-hist", "CM + 1-D histograms"),
             ("cbo-exact-avi", "CM + exact $s_g$, AVI"), ("cbo-sample-1%", "CM + 1\\% sample"),
             ("cbo-probe", "CM + HNSW probe"), ("rob-probe", "CM + HNSW probe, robust"),
             ("hvo-point", "HVO, point routing"), ("hvo", "\\textbf{HVO (robust)}"),
             ("cbo-true", "\\emph{CM + true selectivities}"), ("oracle", "\\emph{Oracle}")]
    rows = []
    for n, label in DS:
        o = R.get(n)
        if o is None:
            continue
        rows.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{label}}}}} \\\\")
        for k, nm in order:
            d = o["routing"]["simd"][k]
            l = d["lat_ms"]
            rows.append(f"{nm} & {f2(l['mean'])} & {f2(l['p50'])} & {f2(l['p95'])} & {f2(l['p99'])} & "
                        f"{d['recall']:.3f} & {100 * d['plan_acc']:.1f} & {f2(d['regret']['mean'])} & "
                        f"{f2(d['regret']['p99'])} & {d['plan_us']:.0f} \\\\")
        rows.append("\\midrule")
    return "\n".join(rows[:-1]) + "\n"


def hw_table(R):
    rows = []
    for n, label in DS:
        o = R.get(n)
        if o is None:
            continue
        for pre, nm in (("scalar", "Scalar FP32"), ("simd", "NEON FP32"), ("i8", "NEON INT8")):
            cm = o["cost_model"][pre]
            r = o["routing"][pre]
            ph = {x["sloc"]: x["sg_star"] for x in o["phase"][pre]}
            rows.append(f"{label} & {nm} & {cm['scan_ns_per_vec']:.1f} & {100 * ph[0.003]:.1f} & {100 * ph[0.01]:.1f} & "
                        f"{r['oracle']['split'][0]} & {r['hvo']['split'][0]} & {f2(r['hvo']['lat_ms']['mean'])} & "
                        f"{f2(r['hvo']['lat_ms']['p99'])} & {f2(r['oracle']['lat_ms']['mean'])} \\\\")
        rows.append("\\midrule")
    return "\n".join(rows[:-1]) + "\n"


def abl_table(R):
    rows = []
    items = [("hvo", "HVO (robust, $L{=}4$, calibrated)"), ("hvo-point", "\\quad point routing (no interval)"),
             ("hvo-raw", "\\quad without calibrator"), ("hvo-L1", "\\quad $L{=}1$ partition"),
             ("hvo-agnostic", "\\quad hardware-agnostic cost (distance counts)"),
             ("hvo+guard", "\\quad + runtime early-exit guard"), ("rob50-hpbs+cal", "\\quad interval $q{=}0.50$"),
             ("rob75-hpbs+cal", "\\quad interval $q{=}0.75$"), ("rob95-hpbs+cal", "\\quad interval $q{=}0.95$"),
             ("rob99-hpbs+cal", "\\quad interval $q{=}0.99$"), ("hvo-noplan", "\\quad planning time excluded")]
    for k, nm in items:
        cells = []
        for n, _ in DS:
            o = R.get(n)
            if o is None:
                cells += ["--"] * 4
                continue
            d = o["routing"]["simd"][k]
            cells += [f2(d["lat_ms"]["mean"]), f2(d["lat_ms"]["p99"]), f"{100 * d['plan_acc']:.1f}", f2(d["regret"]["p99"])]
        rows.append(nm + " & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows) + "\n"


def noise_coords(R, n="syn-128"):
    o = R.get(n)
    if o is None:
        return ""
    out = {}
    for key in ("bound_general", "bound_tight"):
        out[key] = " ".join(f"({x['eta']},{x[key]:.3f})" for x in o["noise"])
    for key in ("model", "measured", "robust_model", "robust_measured"):
        out[key] = " ".join(f"({x['eta']},{x[key]['p99']:.3f})" for x in o["noise"])
        out[key + "max"] = " ".join(f"({x['eta']},{x[key]['max']:.3f})" for x in o["noise"])
    out["boundrobust"] = " ".join(f"({x['eta']},{min(x['bound_robust'], 1e4):.3f})" for x in o["noise"])
    out["boundtight"] = " ".join(f"({x['eta']},{min(x['bound_tight'], 1e4):.3f})" for x in o["noise"])
    out["boundgeneral"] = " ".join(f"({x['eta']},{min(x['bound_general'], 1e4):.3f})" for x in o["noise"])
    return "\n".join(f"\\def\\noise{k.replace('_', '')}{{{v}}}" for k, v in out.items()) + "\n"


def fig_coords(R):
    """pgfplots coordinate lists for the binned and phase figures."""
    lines = []
    for n, _ in DS:
        o = R.get(n)
        if o is None:
            continue
        t = TAG[n]
        for key in ("pre", "fused", "post", "hvo", "oracle"):
            lines.append(f"\\def\\bin{key}{t}{{" + " ".join(f"({b['mid']:.6g},{b[key] * 1e3:.4g})" for b in o["bins"]) + "}")
        for e in ("pg-hist", "sample-1%", "probe", "hpbs+cal"):
            nm = mname(e)
            lines.append(f"\\def\\binq{nm}{t}{{" + " ".join(f"({b['mid']:.6g},{b['q90_' + e]:.4g})" for b in o["bins"]) + "}")
        for pre in ("simd", "scalar", "i8"):
            pts = [(sl, g) for sl, g, _ in o["phase_curve"][pre]]
            lines.append(f"\\def\\phase{mname(pre)}{t}{{" + " ".join(f"({min(max(g, 1e-5), 1):.6g},{sl:.6g})" for sl, g in pts) + "}")
        for k in ("always-pre", "always-post", "global-2way", "cbo-probe", "rob-probe", "hvo-point", "hvo", "oracle"):
            d = o["routing"]["simd"][k]
            lines.append(f"\\def\\lat{mname(k)}{t}{{{d['lat_ms']['mean']:.3f}/{d['lat_ms']['p99']:.3f}}}")
    return "\n".join(lines) + "\n"


def main():
    R = {n: load(n) for n, _ in DS}
    open(os.path.join(OUT, "numbers.tex"), "w").write(macros(R) + noise_coords(R))
    open(os.path.join(OUT, "card_rows.tex"), "w").write(card_table(R))
    open(os.path.join(OUT, "e2e_rows.tex"), "w").write(e2e_table(R))
    open(os.path.join(OUT, "hw_rows.tex"), "w").write(hw_table(R))
    open(os.path.join(OUT, "abl_rows.tex"), "w").write(abl_table(R))
    open(os.path.join(OUT, "figdata.tex"), "w").write(fig_coords(R))
    print("wrote", os.listdir(OUT))


if __name__ == "__main__":
    main()
