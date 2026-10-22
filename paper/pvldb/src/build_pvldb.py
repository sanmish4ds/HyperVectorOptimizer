"""Build paper/pvldb/main.tex (12-page PVLDB version) from paper/hvo.tex (full version).

usage (from paper/):
  python3 pvldb/src/build_pvldb.py "$PWD" pvldb/src/pre.tex pvldb/src/settings.tex \
          pvldb/src/abl_extra.tex pvldb/src/disc_machine.tex
then in pvldb/: pdflatex main; bibtex main; pdflatex main; pdflatex main
"""
import re
import sys

PAPER = sys.argv[1]
S = open(f"{PAPER}/hvo.tex").read()


def cut(a, b, block=None):
    T = S if block is None else block
    i = T.index(a)
    j = T.index(b, i + len(a))
    return T[i:j]


def env(label, block=None):
    T = S if block is None else block
    i = T.index("\\label{%s}" % label)
    st, e = max((T.rfind("\\begin{%s}" % e, 0, i), e) for e in ("figure*", "figure", "table*", "table"))
    return T[st:T.index("\\end{%s}" % e, i) + len("\\end{%s}" % e)]


def drop_env(block, label):
    return block.replace(env(label, block), "")


def rep(block, o, n):
    assert o in block, o[:70]
    return block.replace(o, n)


def drop_para(block, start, new=""):
    """Remove the paragraph (or list item) that begins with `start`, up to the next blank line."""
    i = block.index(start)
    j = block.find("\n\n", i)
    j = len(block) if j < 0 else j
    return block[:i] + new + block[j:]


pre = open(sys.argv[2]).read()   # preamble with title and author
abstract = cut("\\begin{abstract}", "\\keywords{")
body = abstract + "\n\\maketitle\n\n%%% do not modify the following VLDB block %%\n%%% VLDB block start %%%\n\\vldbtopmatter\n%%% VLDB block end %%%\n\n"

intro = cut("\\section{Introduction}", "\\section{Background and Related Work}")
intro = rep(intro, "(Sections~\\ref{sec:costs}--\\ref{sec:theory})",
            "(Sections~\\ref{sec:costs}--\\ref{sec:theory}; proofs in the extended version~\\cite{hvoextended})")
intro = drop_para(intro, "Section~\\ref{sec:related} reviews related work")
body += intro

rel = cut("\\section{Background and Related Work}", "\\section{Filtered Vector Search, by Example}")
rel = drop_env(rel, "tab:related")
rel = rep(rel, "\\emph{Summary.} Table~\\ref{tab:related} compares the approaches. HVO differs", "\\emph{Summary.} HVO differs")
body += rel

ex = cut("\\section{Filtered Vector Search, by Example}", "\\section{HyperVector-Optimizer}")
ex = drop_env(ex, "tab:notation")
ex = rep(ex, " Table~\\ref{tab:notation} lists the notation.", "")
ex = drop_env(ex, "fig:phase")
ex = ex[:ex.index("\\emph{Where each plan wins.}")] + (
    "\\emph{Where each plan wins.} Writing the plan costs of Section~\\ref{sec:costs} as functions of $\\sg$ and $\\sloc$ "
    "shows that filtering first is cheapest only below a boundary $\\sg^*(\\sloc)$. Because checking the filter column by "
    "column already costs time proportional to the table size, a graph plan wins for every $\\sg$ once $\\sloc$ exceeds "
    "roughly $1$--$2\\%$, so the boundary is nearly flat and the global selectivity says little about the right plan. The "
    "extended version~\\cite{hvoextended} states and proves this as a theorem and plots the boundary.\n\n")
body += ex

m = cut("\\section{HyperVector-Optimizer}", "\\subsection{The Three Plans}")
m = rep(m, "The proof is in Appendix~\\ref{app:hpbs}.", "The proof is in the extended version~\\cite{hvoextended}.")
m = rep(m, "twelve features of the raw estimate and the query (Appendix~\\ref{app:hpbs})",
        "twelve features of the raw estimate and the query~\\cite{hvoextended}")
body += m
body += r"""\subsection{The Three Plans}\label{sec:plans}
All plans run on one thread. \emph{Filter first} ($\Ppre$) checks the filter column by column and compares the query with each matching row in a C loop (NEON SIMD). \emph{Search first} ($\Ppost$) is a \emph{resumable} scan over a faiss~\cite{douze2024faiss} HNSW graph ($M=32$, $ef_{\text{construction}}=200$) in the spirit of \texttt{pgvector}'s iterative scan: it returns rows in batches of $k, k, 2k, 4k, \ldots$ in roughly decreasing similarity, checks each batch against the filter, and continues from where it stopped until $k$ rows have passed (at most $2^{17}$). \emph{Filter while searching} ($\Pfus$) is ACORN-1~\cite{patel2024acorn} ($M=32$, $M_\beta=64$), which doubles its search budget from 64 until it has $k$ results and a doubled budget returns the same set. Running ACORN at one million vectors required fixing two memory bugs in its reference code; the fix has been submitted upstream. Because each plan keeps going until it has its answer, its running time does not depend on the optimizer's estimate.

"""
costs = cut("\\subsection{Predicting Plan Costs from Measurements}", "\\subsection{Choosing a Plan Robustly}")
costs = drop_para(costs, "\\emph{Fitting the constants.}",
                  "\\emph{Fitting the constants.} All constants are fitted on the training queries from their \\emph{true} "
                  "selectivities and measured times, the linear ones by non-negative least squares and the staircase ones by "
                  "nonlinear least squares in log space ($\\gamma\\in[0.3,2.5]$). Exponents above one are common, because a "
                  "graph search's priority queues make large budgets disproportionately expensive. ACORN's cost could not be "
                  "fitted by a smooth staircase, so we model it with a monotone (isotonic) curve in $h_k$.")
costs = drop_para(costs, "\\emph{Counting instead of measuring.}")
body += costs
rt = cut("\\subsection{Choosing a Plan Robustly}", "\\subsection{Guarantees}")
rt = drop_env(rt, "alg:hvo")
rt = rep(rt, "\\emph{Native planner.} The whole decision (Algorithm~2) runs in one C function,",
         "\\emph{Native planner.} The whole decision---sketch estimate, calibrator, recall check and minimax costing---runs in one C function,")
rt = drop_para(rt, "\\emph{Runtime guard.}")
body += rt
th = cut("\\subsection{Guarantees}", "\\subsection{Integration into PostgreSQL}")
th = re.sub(r"\\begin\{proof\}.*?\\end\{proof\}\n?", "", th, flags=re.S)
th = drop_para(th, "The factor is $1$ when")
th = drop_para(th, "Theorem~\\ref{thm:robust} is the formal reason")
th = drop_para(th, "With columnar predicate evaluation, $C(\\Ppre)\\ge c_1N$")
th = rep(th, "All statements hold query by query.",
         "All statements hold query by query; proofs are in the extended version~\\cite{hvoextended}.")
body += th
body += cut("\\subsection{Integration into PostgreSQL}", "\\subsection{Assumptions and Implementation}")

body += cut("\\section{Evaluation}", "\\subsection{Evaluation Settings}")
body += open(sys.argv[3]).read()   # condensed settings
est = cut("\\subsection{Estimation Accuracy}", "\\begin{table*}[!t]\n\\footnotesize\n\\renewcommand{\\arraystretch}{0.95}")
est = rep(est, " An earlier design of the cells (Appendix~\\ref{app:hpbs}) is the least accurate version everywhere.", "")
est = drop_para(est, "\\emph{Values shared by many rows.}",
                "\\emph{Values shared by many rows.} $57\\%$ of Amazon products have price 0; giving such values their own "
                "histogram bin (in the sketch and, as a most-common-value list, in the baseline) was necessary for accurate estimates.")
body += est + env("tab:cardinality") + "\n\n" + env("fig:card") + "\n\n" + env("tab:e2e") + "\n\n" + env("tab:pg") + "\n\n"
e2e = cut("\\subsection{End-to-End Latency}", "\\subsection{Inside PostgreSQL}")
for lab in ("fig:cdf", "fig:split"):
    e2e = drop_env(e2e, lab)
e2e = rep(e2e, "Tables~\\ref{tab:e2e} and~\\ref{tab:e2e_syn} report latency, recall and regret for every router at $k=10$. Figure~\\ref{fig:cdf} shows the full distribution of query times on the real data, Figure~\\ref{fig:split} which plans each router chooses, and Figure~\\ref{fig:card}(c) where the plans cross on SYN-128.",
          "Table~\\ref{tab:e2e} reports latency, recall and regret for every router on the real data at $k=10$ (the synthetic-data table is in the extended version~\\cite{hvoextended}). Figure~\\ref{fig:cdf} shows the full distribution of query times, and Figure~\\ref{fig:card}(c) where the plans cross on SYN-128.")
e2e = drop_para(e2e, "\\emph{Which queries are routed differently.}",
                "\\emph{Which queries are routed differently.} HVO and the global rule send similar shares of queries to each "
                "plan; the difference is \\emph{which} queries go where. Both match the fast search-first plan on most queries, "
                "but HVO avoids the wrongly routed rare-match queries that make up the global rule's slowest cases (Figure~\\ref{fig:cdf}).")
e2e = drop_para(e2e, "ACORN's cost could not be fitted by any smooth staircase:")
i = e2e.index("\\emph{What still limits HVO.}"); j = e2e.index("\\end{itemize}", i) + len("\\end{itemize}")
e2e = e2e[:i] + "\\emph{What still limits HVO.} Planning time (the native planner takes $\\NativeUsC$--$\\NativeUsB\\,\\mu$s, which matters when the best plan takes a few hundred microseconds) and the cost model's own error (a median factor of $\\KappaMedC$ for the best plan on ARXIV-384) explain most of the remaining gap to the cost model with true selectivities." + e2e[j:]
body += e2e + env("fig:cdf") + "\n\n"
pgres = cut("\\subsection{Inside PostgreSQL}", "\\subsection{Effect of $k$}")
pgres = drop_env(pgres, "fig:pg")
pgres = rep(pgres, ", and Figure~\\ref{fig:pg} plots each setup's recall against its latency.", ".")
body += pgres
kk = cut("\\subsection{Effect of $k$}", "\\subsection{Does the Calibrator Transfer?}")
kk = drop_env(kk, "fig:gainci")
kk = rep(kk, "Table~\\ref{tab:k} and Figure~\\ref{fig:gainci} vary $k$.", "Figure~\\ref{fig:gainci} varies $k$.")
body += kk + env("fig:gainci") + "\n\n"
abl = cut("\\subsection{Ablations and Compare Loops}", "\\subsection{Regret under Injected Estimation Error}")
for lab in ("tab:hw", "fig:sweep", "tab:ablation", "tab:k", "tab:transfer", "tab:impl"):
    if "\\label{%s}" % lab in abl:
        abl = drop_env(abl, lab)
abl = drop_para(abl, "\\emph{Compare loops.}")
abl = rep(abl, "\\subsection{Ablations and Compare Loops}\\label{sec:eval_hw}", "\\subsection{Ablations, Transfer and Regret}\\label{sec:eval_hw}")
abl = rep(abl, "    \\item \\emph{Width of the safety range} (Figure~\\ref{fig:sweep}).", "    \\item \\emph{Width of the safety range.}")
abl = rep(abl, cut("    \\item \\emph{Runtime guard.}", "\\end{itemize}", abl), "")
body += abl.rstrip() + "\n" + open(sys.argv[4]).read() + env("tab:ablation") + "\n\n"
disc = cut("\\section{Discussion}", "\\section{Conclusion}")
disc = drop_para(disc, "\\emph{One machine, one graph library.}", open(sys.argv[5]).read().strip())
disc = drop_para(disc, "\\emph{ACORN.}")
disc = drop_para(disc, "\\emph{PostgreSQL integration.}")
disc = drop_para(disc, "\\emph{Statistical scope.}")
disc = drop_para(disc, "\\emph{Small $k$ and planning time.}")
disc = drop_para(disc, "\\emph{Not evaluated.}")
disc = disc.replace(" (3) Where probing the index is cheap and matches crowd around the query, as in AMAZON-384, a probe-based estimate is competitive; HPBS is most valuable when matches are rare near the query and when planning must not touch the index. (4) Inside", " (3) Inside")
assert "(3) Inside" in disc
body += disc
body += r"""\section{Conclusion}\label{sec:conclusion}
Choosing a plan for a filtered vector query requires knowing how far the search must travel before it finds $k$ matches. We defined that distance as the $k$-th-match local selectivity, estimated it with a compact summary that never touches the index, and turned the estimate into a plan choice with measured costs and proven guarantees. HVO cuts average latency on arXiv by $\GainHvoC\%$ (95\% interval $\GainLoHvoC$--$\GainHiHvoC\%$) and the slowest queries several-fold on both real datasets, though its average gain on Amazon and synthetic data is not statistically clear, and inside PostgreSQL it returns more correct results than forcing \texttt{pgvector}'s iterative scan while being faster. Testing other machines, graph libraries and real query logs is future work.

"""
body += "\\bibliographystyle{ACM-Reference-Format}\n\\bibliography{refs}\n\n\\end{document}\n"
body = body.replace("(Section~\\ref{sec:eval_regret})", "(Section~\\ref{sec:eval_hw})").replace("(Section~\\ref{sec:eval_transfer})", "(Section~\\ref{sec:eval_hw})")
body = body.replace("(Tables~\\ref{tab:e2e} and~\\ref{tab:e2e_syn})", "(Table~\\ref{tab:e2e})")
for lab, envname in (("fig:cdf", "figure*"), ("fig:gainci", "figure")):
    i = body.index("\\label{%s}" % lab); st = body.rfind("\\begin{%s}" % envname, 0, i)
    body = body[:st] + body[st:i].replace("height=5.0cm", "height=4.3cm") + body[i:]
out = pre + body
assert "Chukkapalli" not in out
body = re.sub(r"\n{3,}", "\n\n", out)
open(f"{PAPER}/pvldb/main.tex", "w").write(body)
print("written", body.count("\n"), "lines")
