"""Fit calibrator + cost model on the train split, evaluate estimators and routers on the test split.

usage: python experiments/analyze.py syn-128
writes results/<name>.json
"""
import sys
import json
import time
import warnings
from common import *  # noqa: F401,F403
import numpy as np
from scipy.optimize import least_squares, nnls
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")
Z = 2.326          # z_{1-delta}, delta = 0.01
EF0, CAP = 64, 1 << 17
PLANS = ["pre", "fused", "post"]
RECALL_OK = 0.9


# ------------------------------------------------------------------------------ loading
def load_all(name):
    G = load(name, QFILE)
    E = load(name, EFILE)
    R = load(name, PFILE)
    fk, pk = PLAN_KEYS
    n = len(R)
    A = dict(sg=G["sg"][:n], sloc=G["sloc"][:n], tau=np.full(n, DATASETS[name][0] / K, float),
             split=G["split"][:n],
             nterm=np.array([p.n_terms() for p in G["P"][:n]], float),
             join=np.array([bool(p.dim_ins) for p in G["P"][:n]], float))
    A["rho"] = rho_gls(A["sloc"], A["sg"]); A["regime"] = regime(A["rho"])
    A["kernels"] = [v for v in ("simd", "scalar", "i8") if f"pre_{v}" in R[0]]
    for v in A["kernels"]:
        A[f"t_pre_{v}"] = np.array([r[f"pre_{v}"]["t"] for r in R])
        A[f"rec_pre_{v}"] = np.array([r[f"pre_{v}"]["recall"] for r in R])
    A["t_rel"] = np.array([r["pre_simd"]["t_rel"] for r in R])
    A["n_scan"] = np.array([r["pre_simd"]["n_scan"] for r in R], float)
    for p, key in (("post", pk), ("fused", fk)):      # plan set chosen by HVO_PLANS
        A[f"t_{p}"] = np.array([r[key]["t"] for r in R])
        A[f"rec_{p}"] = np.array([r[key]["recall"] for r in R])
        A[f"ndis_{p}"] = np.array([sum(x.get("ndis", 0) for x in r[key]["rounds"]) for r in R], float)
    A["rounds_post"] = [r[pk]["rounds"] for r in R]
    A["est"] = {k: np.array([[row[k][0], row[k][1], row[k][2]] for row in E["rows"][:n]])
                for k in E["rows"][0] if not k.endswith("_feat")}
    A["feat"] = {k[:-5]: np.array([row[k] for row in E["rows"][:n]]) for k in E["rows"][0] if k.endswith("_feat")}
    A["bytes"] = E["bytes"]
    return A


def plan_times(A, pre="simd"):
    t = np.stack([A[f"t_pre_{pre}"], A["t_fused"], A["t_post"]], 1)
    rec = np.stack([A[f"rec_pre_{pre}"], A["rec_fused"], A["rec_post"]], 1)
    return t, rec


# ------------------------------------------------------------------------ calibrator
def floor_s(s, tau):
    return np.clip(s, 1.0 / tau, 1.0)


class Calibrator:
    def __init__(self, seed=0):
        self.m = MLPRegressor(hidden_layer_sizes=(16, 16), alpha=1e-3, max_iter=3000, early_stopping=True,
                              validation_fraction=0.15, random_state=seed)

    def fit(self, F, s_raw, s_true, tau):
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-6
        y = np.log(floor_s(s_true, tau)) - np.log(floor_s(s_raw, tau))
        self.m.fit((F - self.mu) / self.sd, y)
        W = self.m.coefs_; b = self.m.intercepts_
        self.params = sum(w.size for w in W) + sum(x.size for x in b)
        return self

    def forward(self, f):
        """Plain numpy forward pass (what the optimizer runs per query)."""
        h = (f - self.mu) / self.sd
        for i, (w, b) in enumerate(zip(self.m.coefs_, self.m.intercepts_)):
            h = h @ w + b
            if i < len(self.m.coefs_) - 1:
                h = np.maximum(h, 0)
        return h

    def apply(self, F, s_raw, tau):
        return np.clip(floor_s(s_raw, tau) * np.exp(self.forward(F).ravel()), 1.0 / tau, 1.0)

    def time_us(self, F):
        t = time.perf_counter()
        for f in F[:200]:
            self.forward(f[None, :])
        return (time.perf_counter() - t) / min(200, len(F)) * 1e6


# ------------------------------------------------------------------------ recall model
class RecallModel:
    """P(Recall@k >= RECALL_OK) for each graph plan, logistic in features available at planning time
    (log s_loc estimate, log s_g estimate, #terms, join). Pre-filtering is exact and always admissible."""

    def fit(self, F, rec_ok):
        from sklearn.linear_model import LogisticRegression
        self.m = []
        for j in range(rec_ok.shape[1]):
            y = rec_ok[:, j].astype(int)
            if y.min() == y.max():
                self.m.append(float(y[0])); continue
            self.m.append(LogisticRegression(C=1.0, max_iter=1000).fit(F, y))
        return self

    def proba(self, F):
        return np.stack([np.full(len(F), m) if isinstance(m, float) else m.predict_proba(F)[:, 1]
                         for m in self.m], 1)

    def coefs(self):
        return [(m, None) if isinstance(m, float) else (float(m.intercept_[0]), m.coef_[0].tolist()) for m in self.m]


def recall_features(sg_hat, sl_hat, nterm, join):
    return np.stack([np.log(np.maximum(sl_hat, 1e-7)), np.log(np.maximum(sg_hat, 1e-7)), nterm, join], 1)


# ------------------------------------------------------------------------ cost model
def k_exp(s, k=K):
    """Candidate horizon of post-filtering: the k-th match sits at rank k / s_loc."""
    return np.minimum(k / np.clip(s, 1e-7, 1), CAP)


def n_rounds(need):
    """Doubling rounds an adaptive plan runs: budgets EF0 * 2^i, i = 0..j, until budget >= need."""
    need = np.minimum(np.maximum(need, EF0), CAP)
    return np.ceil(np.log2(need / EF0) - 1e-9).astype(int)


def staircase(p, need, extra=0):
    """sum_{i=0}^{j+extra} (b0 + b1 * (EF0 2^i)^gamma), the cost of an adaptive graph plan."""
    b0, b1, g = np.exp(p[0]), np.exp(p[1]), p[2]
    j = np.minimum(n_rounds(need) + extra, int(np.log2(CAP / EF0)))
    # closed form of the geometric sum
    r = 2.0 ** g
    return (j + 1) * b0 + b1 * EF0 ** g * (r ** (j + 1) - 1) / (r - 1)


class CostModel:
    """Hardware-calibrated cost model (seconds). Functional forms follow Section 5 of the paper;
    constants are fitted on the calibration (train) split using the true selectivities.

    pre   : C_rel + a0 + c_scan * s_g N
    post  : staircase over budgets 64 * 2^i until the budget reaches k_exp(s_loc) (Eq. k_exp)
    fused : C_rel + staircase until ef reaches c_f * k / s_loc, plus one confirmation round
    """

    def fit(self, A, tr, N, pre="simd"):
        self.N, self.pre = N, pre
        M = np.stack([np.ones(tr.sum()), N * A["nterm"][tr], N * A["join"][tr]], 1)
        y = A["t_rel"][tr]
        self.rel = nnls(M / y[:, None], np.ones(len(y)))[0]
        tp = A[f"t_pre_{pre}"][tr] - self.rel_cost(A["nterm"][tr], A["join"][tr])
        M = np.stack([np.ones(tr.sum()), A["sg"][tr] * N], 1)
        w = 1.0 / A[f"t_pre_{pre}"][tr]
        self.pre_c = nnls(M * w[:, None], tp * w)[0]
        s = floor_s(A["sloc"][tr], A["tau"][tr])

        def fit_stair(need, t, extra):
            def res(p):
                return np.log(staircase(p, need, extra)) - np.log(t)
            x0 = np.clip([np.log(t.min() / 2), np.log(t.mean() / need.mean()), 0.9], -39, 39)
            r = least_squares(res, x0=x0, bounds=([-40, -40, 0.3], [40, 40, 2.5]))
            return r.x, float(np.mean(r.fun ** 2))
        self.post_p, _ = fit_stair(k_exp(s), A["t_post"][tr], 0)
        tf = np.maximum(A["t_fused"][tr] - self.rel_cost(A["nterm"][tr], A["join"][tr]), 1e-6)
        best = None
        for cf in (0.25, 0.5, 1, 2, 4):
            for beta in (1.0, 1.25, 1.5, 1.75, 2.0):
                p_, err = fit_stair(cf * (K / s) ** beta, tf, 1)
                if best is None or err < best[3]:
                    best = (cf, beta, p_, err)
        self.cf, self.beta, self.fused_p, _ = best
        # ACORN stops adaptively (most queries cheap, a few explode), which no smooth staircase fits:
        # the fitted variable term collapses to zero. For it we use a monotone (isotonic) regression of
        # time on the k-th-match horizon, fitted in linear time so that it predicts the expected cost.
        self.iso = None
        if PLANSET == "v2":
            from sklearn.isotonic import IsotonicRegression
            x = np.log(np.minimum(K / s, CAP))
            ir = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, tf)
            self.iso = (ir.X_thresholds_.astype(float), ir.y_thresholds_.astype(float))
            ir2 = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, np.maximum(A["ndis_fused"][tr], 1))
            self.iso_ndis = (ir2.X_thresholds_.astype(float), ir2.y_thresholds_.astype(float))
        # hardware-agnostic variant: same plan shapes, but every distance evaluation costs the same
        self.ndis_post, _ = fit_stair(k_exp(s), np.maximum(A["ndis_post"][tr], 1), 0)
        self.ndis_fused, _ = fit_stair(self.fused_need(s), np.maximum(A["ndis_fused"][tr], 1), 1)
        return self

    def fused_cost(self, s, agnostic=False):
        if self.iso is not None:
            xs, ys = self.iso_ndis if agnostic else self.iso
            return np.interp(np.log(np.minimum(K / s, CAP)), xs, ys)
        return staircase(self.ndis_fused if agnostic else self.fused_p, self.fused_need(s), 1)

    def fused_need(self, s):
        return self.cf * (K / s) ** self.beta

    def rel_cost(self, nterm, join):
        return self.rel[0] + self.rel[1] * self.N * nterm + self.rel[2] * self.N * join

    def costs(self, sg, sloc, tau, nterm, join, agnostic=False):
        s = floor_s(sloc, tau)
        if agnostic:
            return np.stack([sg * self.N, self.fused_cost(s, agnostic=True),
                             staircase(self.ndis_post, k_exp(s), 0)], 1)
        rel = self.rel_cost(nterm, join)
        pre = rel + self.pre_c[0] + self.pre_c[1] * sg * self.N
        fused = rel + self.fused_cost(s)
        post = staircase(self.post_p, k_exp(s), 0)
        return np.stack([pre, fused, post], 1)

    def describe(self):
        f = lambda p: dict(b0_us=np.exp(p[0]) * 1e6, b1_ns=np.exp(p[1]) * 1e9, gamma=p[2])
        return dict(rel_ns_per_row_term=self.rel[1] * 1e9, rel_ns_per_row_join=self.rel[2] * 1e9,
                    rel_fixed_us=self.rel[0] * 1e6, pre_fixed_us=self.pre_c[0] * 1e6,
                    scan_ns_per_vec=self.pre_c[1] * 1e9, post=f(self.post_p), fused=f(self.fused_p),
                    fused_cf=self.cf, fused_beta=self.beta, ndis_post=f(self.ndis_post), ndis_fused=f(self.ndis_fused),
                    fused_isotonic=self.iso is not None,
                    fused_iso_points=0 if self.iso is None else len(self.iso[0]))


# ------------------------------------------------------------------------------ metrics
def pct(x, ps=(50, 90, 95, 99)):
    return {f"p{p}": float(np.percentile(x, p)) for p in ps} | {"mean": float(np.mean(x)), "max": float(np.max(x))}


BOOT_B, BOOT_SEED = 2000, 0


def boot_ci(x, y=None):
    """95% percentile-bootstrap interval of mean(x) over queries (the experimental unit), or, with y,
    of the paired relative gain 1 - mean(x)/mean(y). B = 2000 resamples, fixed seed."""
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(x), size=(BOOT_B, len(x)))
    s = x[idx].mean(1) if y is None else 1.0 - x[idx].mean(1) / y[idx].mean(1)
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def oracle(t, rec):
    """Fastest plan whose recall@k >= RECALL_OK; if no plan qualifies, the highest-recall plan."""
    tt = np.where(rec >= RECALL_OK, t, np.inf)
    c = tt.argmin(1)
    none = ~np.isfinite(tt.min(1))
    c[none] = rec[none].argmax(1)
    return c, t[np.arange(len(c)), c]


def evaluate(choice, t, rec, t_plan, opt_t, opt_c, extra=None):
    n = len(choice)
    lat = t[np.arange(n), choice] + t_plan
    if extra is not None:
        lat = extra + t_plan
    reg = lat / opt_t
    return dict(lat_ms=pct(lat * 1e3), lat_ci95=boot_ci(lat * 1e3),
                lat_cdf=[float(v) for v in np.percentile(lat * 1e3, np.arange(1, 100))], _lat=lat,
                recall=float(rec[np.arange(n), choice].mean()),
                low_recall=float((rec[np.arange(n), choice] < RECALL_OK).mean()),
                plan_acc=float((choice == opt_c).mean()), regret=pct(reg),
                split=[int((choice == j).sum()) for j in range(3)], plan_us=float(np.median(t_plan) * 1e6))


def lam_stair(eta, gamma=1.0, beta=1.0):
    """Lemma 1: sensitivity of a doubling (staircase) plan cost to a multiplicative error eta in s_loc
    when the budget scales as (k/s)^beta and a round with budget e costs b0 + b1 e^gamma:
    Lambda(eta) = 1 + delta * 2^(gamma * delta), delta = ceil(beta * log2 eta)."""
    eta = np.asarray(eta, float)
    d = np.ceil(beta * np.log2(np.maximum(eta, 1.0)) - 1e-12)
    return 1.0 + d * 2.0 ** (gamma * d)


def main(name):
    A = load_all(name)
    N = DATASETS[name][0]
    tr, te = A["split"] == "train", A["split"] == "test"
    out = dict(dataset=name, N=N, n_train=int(tr.sum()), n_test=int(te.sum()))
    out["regime_counts"] = {r: int((A["regime"][te] == r).sum()) for r in ("anti", "indep", "pos")}
    out["sg_range"] = [float(A["sg"].min()), float(A["sg"].max())]

    # ---------------- calibrator
    cals, s_cals = {}, {}
    for sk in A["feat"]:
        c_ = Calibrator().fit(A["feat"][sk][tr], A["est"][sk][tr, 1], A["sloc"][tr], A["tau"][tr])
        cals[sk] = c_
        s_cals[sk] = c_.apply(A["feat"][sk], A["est"][sk][:, 1], A["tau"])
    raw = A["est"]["hpbs"]
    cal, s_cal = cals["hpbs"], s_cals["hpbs"]
    cal_us = cal.time_us(A["feat"]["hpbs"])
    out["calibrator"] = dict(params=int(cal.params), forward_us=cal_us)

    # ---------------- cardinality accuracy (test split)
    card = {}
    names = ["pg-hist", "exact-avi", "sample-1%", "probe", "hpbs-cp", "hpbs-m256", "hpbs-L1", "hpbs"]
    est_sloc = {n_: A["est"][n_][:, 1] for n_ in names}
    est_time = {n_: A["est"][n_][:, 2] for n_ in names}
    for sk in s_cals:
        est_sloc[sk + "+cal"] = s_cals[sk]
        est_time[sk + "+cal"] = A["est"][sk][:, 2] + cal_us * 1e-6
    for n_, s in est_sloc.items():
        qe = qerr(s[te], A["sloc"][te], A["tau"][te])
        d = {"all": pct(qe)}
        for r in ("anti", "indep", "pos"):
            m = A["regime"][te] == r
            d[r] = pct(qe[m]) if m.any() else None
        sgn = n_.replace("+cal", "")
        d["sg_qerr"] = pct(qerr(A["est"][sgn][te, 0], A["sg"][te], N))
        d["time_us"] = float(np.median(est_time[n_][te]) * 1e6)
        d["bytes"] = int(A["bytes"].get(sgn, 0))
        # rare local matches
        m = A["sloc"][te] < 1e-3
        d["rare"] = pct(qe[m]) if m.sum() > 5 else None
        card[n_] = d
    out["card"] = card
    out["n_rare"] = int((A["sloc"][te] < 1e-3).sum())

    # ---------------- cost model
    out["routing"] = {}
    for pre in A["kernels"]:
        CM = CostModel().fit(A, tr, N, pre=pre)
        t, rec = plan_times(A, pre)
        opt_c, opt_t = oracle(t, rec)
        tt, rr, oc, ot = t[te], rec[te], opt_c[te], opt_t[te]
        args = (A["tau"][te], A["nterm"][te], A["join"][te])
        R = {}
        R["oracle"] = evaluate(oc, tt, rr, 0 * ot, ot, oc)
        def cbo(sg, sl, tplan, agnostic=False):
            C = CM.costs(sg, sl, *args, agnostic=agnostic)
            return evaluate(C.argmin(1), tt, rr, tplan, ot, oc), C
        R["cbo-true"], _ = cbo(A["sg"][te], A["sloc"][te], 0 * ot)
        def robust(sg, sl, tplan, eta):
            """Minimax routing over s_loc in [sl/eta, sl*eta]; plan costs are monotone in s_loc."""
            lo = CM.costs(sg, np.clip(sl / eta, 1e-7, 1), *args)
            hi = CM.costs(sg, np.clip(sl * eta, 1e-7, 1), *args)
            C = np.maximum(lo, hi)
            return evaluate(C.argmin(1), tt, rr, tplan, ot, oc), C

        def eta_of(est_all, q=0.9):
            """Width of the uncertainty interval: the q-quantile of the estimator's train Q-error."""
            return float(np.quantile(qerr(est_all[tr], A["sloc"][tr], A["tau"][tr]), q))

        sg_hat = {k_: A["est"][k_.replace("+cal", "")][:, 0] for k_ in est_sloc}
        R["eta_train"] = {}
        for k_ in ("hpbs+cal", "hpbs-L1+cal", "hpbs", "probe", "exact-avi", "pg-hist", "sample-1%"):
            pt, _ = cbo(sg_hat[k_][te], est_sloc[k_][te], est_time[k_][te])
            e90 = eta_of(est_sloc[k_])
            rb, Cr = robust(sg_hat[k_][te], est_sloc[k_][te], est_time[k_][te], e90)
            R["eta_train"][k_] = e90
            R[f"pt-{k_}"], R[f"rob-{k_}"] = pt, rb
            if k_ == "hpbs+cal":
                C_hvo = Cr
            for q in (0.5, 0.75, 0.95, 0.99):
                if k_ in ("hpbs+cal", "hpbs-L1+cal", "probe"):
                    R[f"rob{int(q * 100)}-{k_}"], _ = robust(sg_hat[k_][te], est_sloc[k_][te], est_time[k_][te],
                                                             eta_of(est_sloc[k_], q))
        # named routers used in the paper
        R["hvo"] = R["rob-hpbs+cal"]
        R["hvo-point"] = R["pt-hpbs+cal"]
        R["hvo-raw"] = R["rob-hpbs"]
        R["hvo-L1"] = R["rob-hpbs-L1+cal"]
        R["hvo-agnostic"], _ = cbo(raw[te, 0], s_cal[te], est_time["hpbs+cal"][te], agnostic=True)
        # recall-aware HVO: graph plans are admissible only if P(recall >= RECALL_OK) >= TAU_REC
        TAU_REC = 0.9
        Ftr = recall_features(raw[tr, 0], s_cal[tr], A["nterm"][tr], A["join"][tr])
        Fte = recall_features(raw[te, 0], s_cal[te], A["nterm"][te], A["join"][te])
        RM = RecallModel().fit(Ftr, rec[tr][:, 1:] >= RECALL_OK)
        ok = np.concatenate([np.ones((te.sum(), 1)), RM.proba(Fte)], 1) >= TAU_REC
        C_rec = np.where(ok, C_hvo, np.inf)
        R["hvo-recall"] = evaluate(C_rec.argmin(1), tt, rr, est_time["hpbs+cal"][te], ot, oc)
        R["hvo-recall"]["tau"] = TAU_REC
        R["recall_model"] = RM.coefs()
        if pre == "simd":
            planner = dict(
                eta=float(R["eta_train"]["hpbs+cal"]), tau_rec=TAU_REC, recall_model=RM.coefs(),
                cal=dict(mu=cal.mu.tolist(), sd=cal.sd.tolist(), W=[w.tolist() for w in cal.m.coefs_],
                         b=[b.tolist() for b in cal.m.intercepts_]),
                cost=dict(rel=CM.rel.tolist(), pre_c=CM.pre_c.tolist(), post=list(map(float, CM.post_p)),
                          fused=list(map(float, CM.fused_p)), cf=float(CM.cf), beta=float(CM.beta),
                          iso_x=[] if CM.iso is None else CM.iso[0].tolist(),
                          iso_y=[] if CM.iso is None else CM.iso[1].tolist()),
                test_idx=np.flatnonzero(te).tolist(), choice_hvo=C_hvo.argmin(1).tolist(),
                choice_hvo_recall=C_rec.argmin(1).tolist())
            with open(rfile(name).replace(".json", "_planner.json"), "w") as f:
                json.dump(planner, f)
        for b_ in ("pg-hist", "exact-avi", "sample-1%", "probe"):
            R[f"cbo-{b_}"] = R[f"pt-{b_}"]
        for j, p in enumerate(PLANS):
            R[f"always-{p}"] = evaluate(np.full(te.sum(), j), tt, rr, 0 * ot, ot, oc)
        # global-selectivity threshold routers, thresholds tuned on the train split (generous to them)
        sg_tr, sg_te = A["sg"][tr], A["sg"][te]
        t_tr = t[tr]
        grid = np.logspace(-4, 0, 41)
        best = min(grid, key=lambda th: t_tr[np.arange(tr.sum()), np.where(sg_tr < th, 0, 2)].mean())
        R["global-2way"] = evaluate(np.where(sg_te < best, 0, 2), tt, rr, 0 * ot, ot, oc)
        R["global-2way"]["threshold"] = float(best)
        best3 = min(((a, b) for a in grid for b in grid if b >= a),
                    key=lambda ab: t_tr[np.arange(tr.sum()), np.where(sg_tr < ab[0], 0, np.where(sg_tr < ab[1], 1, 2))].mean())
        R["global-3way"] = evaluate(np.where(sg_te < best3[0], 0, np.where(sg_te < best3[1], 1, 2)), tt, rr, 0 * ot, ot, oc)
        R["global-3way"]["thresholds"] = [float(x) for x in best3]
        if pre == "simd":
            # per-query record of the test split, for the worked examples in the paper (examples.py)
            eta_h = R["eta_train"]["hpbs+cal"]
            sl_h, sg_h = est_sloc["hpbs+cal"][te], sg_hat["hpbs+cal"][te]
            lo_c = CM.costs(sg_h, np.clip(sl_h / eta_h, 1e-7, 1), *args)
            hi_c = CM.costs(sg_h, np.clip(sl_h * eta_h, 1e-7, 1), *args)
            true_c = CM.costs(A["sg"][te], A["sloc"][te], *args)
            ex = dict(plans=list(PLANS), eta=eta_h, global_threshold=float(best), idx=np.flatnonzero(te).tolist(),
                      sg=A["sg"][te].tolist(), sloc=A["sloc"][te].tolist(), tau=A["tau"][te].tolist(),
                      t=tt.tolist(), rec=rr.tolist(), oracle=oc.tolist(),
                      global_choice=np.where(sg_te < best, 0, 2).tolist(), hvo_choice=C_hvo.argmin(1).tolist(),
                      sloc_raw=raw[te, 1].tolist(), sloc_hat=sl_h.tolist(), sg_hat=sg_h.tolist(),
                      sloc_probe=est_sloc["probe"][te].tolist(), cost_lo=lo_c.tolist(), cost_hi=hi_c.tolist(),
                      cost_true=true_c.tolist(), plan_s=est_time["hpbs+cal"][te].tolist())
            with open(rfile(name).replace(".json", "_examples.json"), "w") as f:
                json.dump(ex, f)
        # runtime early-exit guard (replayed from the recorded first round of the post plan)
        ch = C_hvo.argmin(1)
        lat = tt[np.arange(len(ch)), ch].copy()
        fired = 0
        idx_te = np.flatnonzero(te)
        for i, gi in enumerate(idx_te):
            if ch[i] != 2:
                continue
            r0 = A["rounds_post"][gi][0]
            if r0["nvalid"] >= K or len(A["rounds_post"][gi]) == 1:
                continue
            s_obs = max(r0["nvalid"], 0.5) / max(r0["ncand"], 1)
            if s_obs < 0.5 * s_cal[gi]:
                c_now = CM.costs(np.array([raw[gi, 0]]), np.array([s_obs]), np.array([A["tau"][gi]]),
                                 np.array([A["nterm"][gi]]), np.array([A["join"][gi]]))[0]
                if c_now[0] < c_now[2] - r0["t"]:
                    lat[i] = r0["t"] + tt[i, 0]; fired += 1
        R["hvo+guard"] = evaluate(ch, tt, rr, est_time["hpbs+cal"][te], ot, oc, extra=lat)
        R["hvo-noplan"] = evaluate(ch, tt, rr, 0 * ot, ot, oc)
        R["hvo+guard"]["fired"] = fired
        # model regret of HVO's choices (cost model evaluated at the true selectivities)
        C_true = CM.costs(A["sg"][te], A["sloc"][te], *args)
        mreg = C_true[np.arange(len(ch)), ch] / C_true.min(1)
        R["hvo"]["model_regret"] = pct(mreg)
        # calibration error of the cost model on the test split, best plan only
        kap = np.exp(np.abs(np.log(C_true[np.arange(len(oc)), oc] / ot)))
        R["kappa_cal"] = pct(kap)
        best = min(("global-2way", "global-3way"), key=lambda k_: R[k_]["lat_ms"]["mean"])
        for v in ("hvo", "hvo-L1", "hvo-recall"):
            x, y = R[v]["_lat"], R[best]["_lat"]
            R[v]["gain_vs_global"] = {"vs": best, "gain": float(1 - x.mean() / y.mean()), "ci95": boot_ci(x, y)}
        for d_ in R.values():
            if isinstance(d_, dict):
                d_.pop("_lat", None)
        out["routing"][pre] = R
        out.setdefault("cost_model", {})[pre] = CM.describe()
        if pre == "simd":
            CM_simd, t_simd, rec_simd, C_hvo_simd = CM, t, rec, C_hvo

    # ---------------- noise injection: regret vs. multiplicative error (Theorem 2)
    CM, t, rec = CM_simd, t_simd, rec_simd
    opt_c, opt_t = oracle(t, rec)
    args = (A["tau"][te], A["nterm"][te], A["join"][te])
    C_true = CM.costs(A["sg"][te], A["sloc"][te], *args)
    rng = np.random.default_rng(0)
    noise = []
    def lam_fused(eta):
        """Sensitivity of the fused cost to a factor-eta error in s: closed form for a staircase (Lemma 1),
        computed numerically for the isotonic model (ignoring C_rel >= 0 only makes it larger)."""
        if CM.iso is None:
            return float(lam_stair(eta, CM.fused_p[2], CM.beta))
        xs, ys = CM.iso
        g = np.linspace(xs[0] - 1, xs[-1] + 1, 400)
        f = lambda x: np.interp(x, xs, ys)
        d = np.log(max(eta, 1.0))
        return float(max(1.0, np.max(np.maximum(f(g + d) / f(g), f(g) / f(g + d)))))
    lam = lambda eta: max(float(lam_stair(eta, CM.post_p[2], 1.0)), lam_fused(eta))
    for eta in [1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]:
        mr, me, viol_g, viol_t, rr_, rme, viol_r = [], [], 0, 0, [], [], 0
        for rep in range(20):
            sgn = rng.choice([-1, 1], size=te.sum())
            s_hat = np.clip(A["sloc"][te] * eta ** sgn, 1e-7, 1)
            ch = CM.costs(A["sg"][te], s_hat, *args).argmin(1)
            m_reg = C_true[np.arange(len(ch)), ch] / C_true.min(1)
            lat = t[te][np.arange(len(ch)), ch]
            mr.append(m_reg); me.append(lat / opt_t[te])
            # robust (minimax) routing over the interval [s_hat/eta, s_hat*eta], which contains s_loc
            Cr = np.maximum(CM.costs(A["sg"][te], np.clip(s_hat / eta, 1e-7, 1), *args),
                            CM.costs(A["sg"][te], np.clip(s_hat * eta, 1e-7, 1), *args))
            chr_ = Cr.argmin(1)
            r_reg = C_true[np.arange(len(chr_)), chr_] / C_true.min(1)
            rr_.append(r_reg); rme.append(t[te][np.arange(len(chr_)), chr_] / opt_t[te])
            opt_pre = C_true.argmin(1) == 0
            viol_r += int((r_reg[opt_pre] > 1 + 1e-9).sum()) + int((r_reg > lam(eta ** 2) * (1 + 1e-9)).sum())
            L = np.full(len(ch), lam(eta))
            viol_g += int((m_reg > L ** 2 * (1 + 1e-9)).sum())
            pre_inv = (ch == 0) | (C_true.argmin(1) == 0)
            viol_t += int((m_reg[pre_inv] > L[pre_inv] * (1 + 1e-9)).sum())
        mr, me, rr_, rme = map(np.concatenate, (mr, me, rr_, rme))
        noise.append(dict(eta=eta, model=pct(mr), measured=pct(me), robust_model=pct(rr_), robust_measured=pct(rme),
                          bound_robust=lam(eta ** 2), violations_robust=viol_r,
                          bound_tight=lam(eta), bound_general=lam(eta) ** 2,
                          violations_general=viol_g, violations_tight=viol_t, n=len(mr)))
    out["noise"] = noise

    # ---------------- phase boundary (Theorem 1) from the fitted model, per kernel
    pb = {}
    for pre in A["kernels"]:
        CMp = CostModel().fit(A, tr, N, pre=pre)
        bnd = []
        for sl in [0.001, 0.003, 0.01, 0.02, 0.05, 0.1, 0.3]:
            gs = np.logspace(-4, 0, 400)
            C = CMp.costs(gs, np.full_like(gs, sl), np.full_like(gs, 1e9), np.full_like(gs, 2), np.zeros_like(gs))
            w = C.argmin(1)
            first_not_pre = np.flatnonzero(w != 0)
            bnd.append(dict(sloc=sl, sg_star=float(gs[first_not_pre[0]]) if len(first_not_pre) else 1.0,
                            winners=[int((w == j).sum()) for j in range(3)]))
        pb[pre] = bnd
    out["phase"] = pb
    # phase curves for the figure: s_g* (pre vs best graph plan) along a dense s_loc grid, per kernel
    curves = {}
    sl_grid = np.logspace(np.log10(K / N), 0, 60)
    gs = np.logspace(np.log10(K / N), 0, 600)
    for pre in A["kernels"]:
        CMp = CostModel().fit(A, tr, N, pre=pre)
        cur = []
        for sl in sl_grid:
            C = CMp.costs(gs, np.full_like(gs, sl), np.full_like(gs, N / K), np.full_like(gs, 2), np.zeros_like(gs))
            w = C.argmin(1)
            nz = np.flatnonzero(w != 0)
            g_star = float(gs[nz[0]]) if len(nz) else 1.0
            graph = int(np.bincount(w[w != 0], minlength=3)[1:].argmax() + 1) if len(nz) else 0
            cur.append((float(sl), g_star, graph))
        curves[pre] = cur
    out["phase_curve"] = curves
    # binned statistics by true local selectivity (test split)
    CM, t, rec = CM_simd, t_simd, rec_simd
    opt_c, opt_t = oracle(t, rec)
    ch = C_hvo_simd.argmin(1)
    lat_hvo = t[te][np.arange(len(ch)), ch] + est_time["hpbs+cal"][te]
    edges = np.logspace(np.log10(K / N), 0, 13)
    sl_te = A["sloc"][te]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sl_te >= lo) & (sl_te < hi * (1 + 1e-9))
        if m.sum() < 5:
            continue
        b = dict(lo=float(lo), hi=float(hi), mid=float(np.sqrt(lo * hi)), n=int(m.sum()),
                 pre=float(np.median(t[te][m, 0])), fused=float(np.median(t[te][m, 1])),
                 post=float(np.median(t[te][m, 2])), hvo=float(np.median(lat_hvo[m])),
                 oracle=float(np.median(opt_t[te][m])))
        for n_ in ("pg-hist", "sample-1%", "probe", "hpbs+cal"):
            b["q_" + n_] = float(np.median(qerr(est_sloc[n_][te][m], sl_te[m], A["tau"][te][m])))
            b["q90_" + n_] = float(np.percentile(qerr(est_sloc[n_][te][m], sl_te[m], A["tau"][te][m]), 90))
        bins.append(b)
    out["bins"] = bins
    with open(rfile(name), "w") as f:
        json.dump(out, f, indent=1, default=float)
    return out


def brief(o):
    print(o["dataset"], "regimes", o["regime_counts"], "rare", o["n_rare"])
    for n_, d in o["card"].items():
        print(f"  {n_:10s} all p50 {d['all']['p50']:.2f} p95 {d['all']['p95']:.2f} p99 {d['all']['p99']:.2f}"
              f" | anti p99 {d['anti']['p99'] if d['anti'] else float('nan'):.2f} pos p99 {d['pos']['p99'] if d['pos'] else float('nan'):.2f}"
              f" | {d['time_us']:.0f}us {d['bytes']/1e6:.2f}MB")
    print("  cost model", json.dumps(o["cost_model"]["simd"], default=lambda x: round(float(x), 3)))
    for pre in ("simd",):
        for n_, d in o["routing"][pre].items():
            if "lat_ms" not in d:
                continue
            print(f"  {n_:16s} p50 {d['lat_ms']['p50']:.2f} p95 {d['lat_ms']['p95']:.2f} p99 {d['lat_ms']['p99']:.2f} "
                  f"mean {d['lat_ms']['mean']:.2f} acc {d['plan_acc']*100:.1f} reg mean {d['regret']['mean']:.2f} "
                  f"p99 {d['regret']['p99']:.2f} rec {d['recall']:.3f} split {d['split']}")


if __name__ == "__main__":
    brief(main(sys.argv[1]))
