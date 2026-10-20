"""Turn results/*.json into LaTeX: paper/generated/numbers.tex (macros) and table bodies.

Every number quoted in the paper comes from here, so the paper cannot drift from the results.
"""
import os
import json
from common import ROOT, RESULTS

OUT = os.path.join(ROOT, "paper", "generated")
os.makedirs(OUT, exist_ok=True)
DS = [("arxiv-384", "ARXIV-384 (real)"), ("amazon-384", "AMAZON-384 (real)"), ("syn-128", "SYN-128"), ("syn-768", "SYN-768")]
TAG = {"syn-128": "A", "syn-768": "B", "arxiv-384": "C", "amazon-384": "D"}
DIM = {"syn-128": 128, "syn-768": 768, "arxiv-384": 384, "amazon-384": 384}


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
           "hvo", "hvo-recall", "hvo-L1", "hvo-raw", "hvo-agnostic", "hvo+guard", "hvo-noplan", "cbo-true", "oracle",
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
            mac(f"R{mname(k)}Low{t}", f"{100 * d['low_recall']:.1f}")
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
             ("hpbs-cp", "HPBS, cross-polytope cells"), ("hpbs", "HPBS raw ($L{=}4$, $m{=}1024$)"),
             ("hpbs-L1+cal", "HPBS, $L{=}1$ + calibrator"), ("hpbs+cal", "\\textbf{HPBS + calibrator}")]
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


def e2e_table(R, dss):
    order = [("always-pre", "Always pre-filter"), ("always-post", "Always post-filter (resumable)"),
             ("always-fused", "Always fused (ACORN-1)"), ("global-2way", "Global $s_g$ threshold"),
             ("cbo-pg-hist", "CM + 1-D histograms"), ("cbo-probe", "CM + HNSW probe"),
             ("rob-probe", "CM + HNSW probe, robust"), ("hvo-point", "HVO, point routing"),
             ("hvo", "\\textbf{HVO (robust)}"), ("hvo-recall", "HVO, recall-aware"),
             ("cbo-true", "\\emph{CM + true selectivities}"), ("oracle", "\\emph{Oracle}")]
    rows = []
    for n, label in DS:
        o = R.get(n)
        if o is None or n not in dss:
            continue
        rows.append(f"\\multicolumn{{11}}{{l}}{{\\textit{{{label}}}}} \\\\")
        for k, nm in order:
            d = o["routing"]["simd"][k]
            l = d["lat_ms"]
            rows.append(f"{nm} & {f2(l['mean'])} & {f2(l['p50'])} & {f2(l['p95'])} & {f2(l['p99'])} & "
                        f"{d['recall']:.3f} & {100 * d['low_recall']:.1f} & {100 * d['plan_acc']:.1f} & "
                        f"{f2(d['regret']['mean'])} & {f2(d['regret']['p99'])} & {d['plan_us']:.0f} \\\\")
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
             ("hvo-recall", "\\quad + recall constraint"), ("hvo-raw", "\\quad without calibrator"),
             ("hvo-L1", "\\quad $L{=}1$ partition"), ("hvo-agnostic", "\\quad hardware-agnostic cost (distance counts)"),
             ("hvo+guard", "\\quad + runtime early-exit guard"), ("rob75-hpbs+cal", "\\quad interval $q{=}0.75$"),
             ("rob95-hpbs+cal", "\\quad interval $q{=}0.95$"), ("hvo-noplan", "\\quad planning time excluded")]
    for k, nm in items:
        cells = []
        for n, _ in DS:
            o = R.get(n)
            if o is None:
                cells += ["--"] * 3
                continue
            d = o["routing"]["simd"][k]
            cells += [f2(d["lat_ms"]["mean"]), f"{d['recall']:.3f}", f2(d["regret"]["p99"])]
        rows.append(nm + " & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows) + "\n"


def pg_table(P):
    """PostgreSQL + pgvector configurations (test split)."""
    order = [("stock", "pgvector 0.8.7, defaults"), ("iter", "pgvector 0.8.7, iterative scan"),
             ("exqutor", "Exqutor (pgvector 0.7.1 + patches)"), ("pre", "Forced pre-filter"),
             ("post", "Forced iterative post-filter"), ("hvo_norec", "\\textbf{HVO hook}"),
             ("hvo", "\\textbf{HVO hook, recall-aware}"), ("oracle", "\\emph{Oracle}")]
    rows = []
    for key, nm in order:
        cells = []
        for n in ("arxiv-384", "amazon-384"):
            d = P.get(n, {}).get(key)
            if d is None:
                cells += ["--"] * 5
                continue
            cells += [f2(d["lat_ms"]["mean"]), f2(d["lat_ms"]["p50"]), f2(d["lat_ms"]["p99"]), f"{d['recall']:.3f}",
                      f"{100 * d['low_recall']:.1f}"]
        rows.append(nm + " & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows) + "\n"


def k_table(RK):
    """k sweep: mean latency, recall and P99 regret of HVO vs. the best global router and the oracle."""
    rows = []
    for n, label in DS:
        for k in (1, 10, 100):
            o = RK.get((n, k))
            if o is None:
                continue
            r = o["routing"]["simd"]
            best = min(("global-2way", "global-3way"), key=lambda x: r[x]["lat_ms"]["mean"])
            cells = [label if k == 1 else "", str(k)]
            for key in ("hvo", best, "rob-probe", "cbo-true", "oracle"):
                cells.append(f2(r[key]["lat_ms"]["mean"]))
            cells += [f"{r['hvo']['recall']:.3f}", f"{r['oracle']['recall']:.3f}", f2(r["hvo"]["regret"]["p99"]),
                      f2(r[best]["regret"]["p99"]), f2(o["card"]["hpbs+cal"]["all"]["p99"]), f2(o["card"]["probe"]["all"]["p99"])]
            rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    return "\n".join(rows[:-1]) + "\n"


def transfer_table(TR):
    names = [n for n, _ in DS]
    lab = dict(DS)
    rows = []
    for src in names + ["none"]:
        cells = [("no calibrator" if src == "none" else lab[src].replace(" (real)", ""))]
        for tgt in names:
            d = [x for x in TR["pairs"] if x["source"] == src and x["target"] == tgt]
            cells.append(f"{f2(d[0]['qerr']['p99'])} / {f2(d[0]['lat_ms']['mean'])}" if d else "--")
        rows.append(" & ".join(cells) + " \\\\")
    return "\n".join(rows) + "\n"


def impl_table(R, R1):
    """First- vs. second-generation graph plans (k = 10)."""
    rows = []
    for n, label in DS:
        o, o1 = R.get(n), R1.get(n)
        if o is None or o1 is None:
            continue
        a, a1 = o["routing"]["simd"], o1["routing"]["simd"]
        cells = [label]
        for d in (a1["always-post"], a["always-post"], a1["always-fused"], a["always-fused"]):
            cells += [f2(d["lat_ms"]["mean"]), f2(d["lat_ms"]["p99"]), f"{d['recall']:.3f}"]
        cells += [f2(a1["hvo"]["lat_ms"]["mean"]), f2(a["hvo"]["lat_ms"]["mean"])]
        rows.append(" & ".join(cells) + " \\\\")
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
    R1 = {n: load(n + "_v1") for n, _ in DS}
    RK = {(n, k): load(n if k == 10 else f"{n}_k{k}") for n, _ in DS for k in (1, 10, 100)}
    P = load("pg") or {}
    TR = load("transfer") or {"pairs": []}
    NAT = {n: load(n + "_native") for n, _ in DS}
    m = macros(R) + noise_coords(R) + extra_macros(R, RK, P, TR, NAT)
    open(os.path.join(OUT, "numbers.tex"), "w").write(m)
    open(os.path.join(OUT, "card_rows.tex"), "w").write(card_table(R))
    open(os.path.join(OUT, "e2e_real_rows.tex"), "w").write(e2e_table(R, ("arxiv-384", "amazon-384")))
    open(os.path.join(OUT, "e2e_syn_rows.tex"), "w").write(e2e_table(R, ("syn-128", "syn-768")))
    open(os.path.join(OUT, "hw_rows.tex"), "w").write(hw_table(R))
    open(os.path.join(OUT, "abl_rows.tex"), "w").write(abl_table(R))
    open(os.path.join(OUT, "pg_rows.tex"), "w").write(pg_table(P))
    open(os.path.join(OUT, "k_rows.tex"), "w").write(k_table(RK))
    open(os.path.join(OUT, "transfer_rows.tex"), "w").write(transfer_table(TR))
    open(os.path.join(OUT, "impl_rows.tex"), "w").write(impl_table(R, R1))
    open(os.path.join(OUT, "figdata.tex"), "w").write(fig_coords(R))
    print("wrote", sorted(os.listdir(OUT)))


def extra_macros(R, RK, P, TR, NAT):
    m = []
    def mac(name, val):
        m.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    for n in ("arxiv-384", "amazon-384"):
        t = TAG[n]
        for key in ("stock", "iter", "exqutor", "pre", "post", "hvo", "hvo_norec", "oracle"):
            d = P.get(n, {}).get(key)
            if d is None:
                continue
            nm = mname(key.replace("_", " "))
            mac(f"PG{nm}Mean{t}", f2(d["lat_ms"]["mean"]))
            mac(f"PG{nm}Med{t}", f2(d["lat_ms"]["p50"]))
            mac(f"PG{nm}Pnn{t}", f2(d["lat_ms"]["p99"]))
            mac(f"PG{nm}Rec{t}", f"{d['recall']:.3f}")
            mac(f"PG{nm}Low{t}", f"{100 * d['low_recall']:.1f}")
            if "plan_us" in d:
                mac(f"PG{nm}Us{t}", f"{d['plan_us']['p50']:.0f}")
                mac(f"PG{nm}Acc{t}", f"{100 * d['plan_acc']:.1f}")
                mac(f"PG{nm}PostShare{t}", f"{100 * d['post_share']:.0f}")
    for n, _ in DS:
        t = TAG[n]
        x = NAT.get(n)
        if x:
            mac(f"NativeUs{t}", f"{x['hvo']['native_us']:.0f}")
            mac(f"NativeAgree{t}", f"{100 * x['hvo']['agreement']:.1f}")
        for k in (1, 100):
            o = RK.get((n, k))
            if o is None:
                continue
            r = o["routing"]["simd"]
            best = min(("global-2way", "global-3way"), key=lambda x: r[x]["lat_ms"]["mean"])
            kk = "One" if k == 1 else "Hundred"
            mac(f"K{kk}HvoMean{t}", f2(r["hvo"]["lat_ms"]["mean"]))
            mac(f"K{kk}GlobalMean{t}", f2(r[best]["lat_ms"]["mean"]))
            mac(f"K{kk}OracleMean{t}", f2(r["oracle"]["lat_ms"]["mean"]))
            mac(f"K{kk}Gain{t}", f"{100 * (1 - r['hvo']['lat_ms']['mean'] / r[best]['lat_ms']['mean']):.0f}")
    for p_ in TR.get("pairs", []):
        if p_["source"] in ("syn-128",) and p_["target"] in ("arxiv-384", "amazon-384"):
            mac(f"TrSynToReal{TAG[p_['target']]}Q", f2(p_["qerr"]["p99"]))
            mac(f"TrSynToReal{TAG[p_['target']]}Mean", f2(p_["lat_ms"]["mean"]))
        if p_["source"] == p_["target"]:
            mac(f"TrSelf{TAG[p_['target']]}Q", f2(p_["qerr"]["p99"]))
            mac(f"TrSelf{TAG[p_['target']]}Mean", f2(p_["lat_ms"]["mean"]))
        if p_["source"] == "none":
            mac(f"TrNone{TAG[p_['target']]}Q", f2(p_["qerr"]["p99"]))
            mac(f"TrNone{TAG[p_['target']]}Mean", f2(p_["lat_ms"]["mean"]))
    return "\n".join(m) + "\n"


if __name__ == "__main__":
    main()
