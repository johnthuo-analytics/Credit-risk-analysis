"""Render three Power BI-style dashboard pages as PNGs for the README.

Run:  python -m src.dashboard_images   (needs data/powerbi/ from src.export_powerbi)
These are static previews that mirror the layout described in docs/POWERBI.md.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

from . import config as C

NAVY, BLUE, RED, GREEN, AMBER, GREY, BG = "#1d3557", "#2a6f97", "#c8553d", "#6a994e", "#e9b44c", "#8d99ae", "#f4f6fa"
BAND_COL = {"Low": GREEN, "Medium": AMBER, "High": RED}
PB = C.ROOT / "data" / "powerbi"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#cfd6e4",
                     "axes.labelcolor": "#4a4a4a", "xtick.color": "#4a4a4a", "ytick.color": "#4a4a4a"})


def header(fig, title, sub):
    fig.patch.set_facecolor(BG)
    fig.text(0.03, 0.955, title, fontsize=22, fontweight="bold", color=NAVY)
    fig.text(0.03, 0.925, sub, fontsize=10.5, color=GREY)
    fig.text(0.97, 0.955, "Demo data (synthetic)", fontsize=9, color=GREY, ha="right")


def card(fig, x, y, w, h, label, value, color=NAVY):
    ax = fig.add_axes([x, y, w, h]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.06",
                                fc="white", ec="#e1e6f0", transform=ax.transAxes))
    ax.add_patch(plt.Rectangle((0, 0.04), 0.018, 0.92, fc=color, transform=ax.transAxes))
    ax.text(0.09, 0.66, value, fontsize=21, fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.09, 0.22, label, fontsize=10, color=GREY, transform=ax.transAxes)


def panel(ax, title):
    ax.set_facecolor("white")
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=NAVY, pad=10)
    ax.grid(axis="y", color="#eef1f6", zorder=0)


def pct(ax, axis="y"):
    fmt = lambda v, _: f"{v:.0%}"
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def page1(d, bs):
    fig = plt.figure(figsize=(16, 9))
    header(fig, "Credit Risk - Executive Overview", "Portfolio health, branch risk and risk-band mix")
    tot_paid, outst = d.Total_Paid.sum(), d.Outstanding_Balance.sum()
    kpis = [("Total loans", f"{len(d):,}", NAVY), ("Disbursed", f"{d.Loan_Amount.sum()/1e6:,.0f}M", BLUE),
            ("Outstanding", f"{outst/1e6:,.0f}M", BLUE), ("Default rate", f"{d.Default_Flag.mean():.1%}", RED),
            ("Collection efficiency", f"{tot_paid/(tot_paid+outst):.1%}", GREEN)]
    for i, (l, v, c) in enumerate(kpis):
        card(fig, 0.03 + i * 0.1915, 0.775, 0.18, 0.12, l, v, c)
    gs = GridSpec(1, 3, figure=fig, left=0.05, right=0.97, bottom=0.09, top=0.70, wspace=0.28)
    avg = d.Default_Flag.mean()
    ax = fig.add_subplot(gs[0]); panel(ax, "Default rate by branch")
    cols = [RED if v > avg else BLUE for v in bs.default_rate]
    ax.bar(bs.Branch, bs.default_rate, color=cols, zorder=3)
    ax.axhline(avg, color=NAVY, ls="--", lw=1.2); ax.text(len(bs) - 0.5, avg + 0.003, f"Portfolio {avg:.1%}", ha="right", fontsize=9, color=NAVY)
    pct(ax); ax.tick_params(axis="x", rotation=30)
    ax = fig.add_subplot(gs[1]); panel(ax, "Outstanding balance by branch (M)")
    o = bs.sort_values("outstanding")
    ax.barh(o.Branch, o.outstanding / 1e6, color=BLUE, zorder=3)
    for y, v in enumerate(o.outstanding / 1e6): ax.text(v + 0.3, y, f"{v:.1f}", va="center", fontsize=9)
    ax.grid(axis="x", color="#eef1f6", zorder=0); ax.grid(axis="y", visible=False)
    ax = fig.add_subplot(gs[2]); panel(ax, "Loans by risk band"); ax.grid(False)
    cnt = d.Risk_Band.value_counts().reindex(["Low", "Medium", "High"])
    ax.pie(cnt, colors=[BAND_COL[b] for b in cnt.index], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.38, edgecolor="white"), labels=[f"{b}\n{v/len(d):.0%}" for b, v in cnt.items()],
           textprops=dict(fontsize=10))
    return fig


def page2(d, imp):
    fig = plt.figure(figsize=(16, 9))
    header(fig, "Credit Risk - What Drives Default", "Default rate by borrower profile, and what the model relies on")
    gs = GridSpec(2, 3, figure=fig, left=0.05, right=0.97, bottom=0.07, top=0.88, wspace=0.28, hspace=0.42)
    def rate_bar(pos, col, order, title, color):
        ax = fig.add_subplot(gs[pos]); panel(ax, title)
        r = d.groupby(col, observed=True).Default_Flag.mean().reindex(order)
        ax.bar(r.index, r.values, color=color, zorder=3); pct(ax)
        for x, v in enumerate(r.values): ax.text(x, v + 0.006, f"{v:.0%}", ha="center", fontsize=9)
    rate_bar((0, 0), "Score_Band", ["Very poor", "Poor", "Fair", "Good", "Excellent"], "Default rate by credit score band", RED)
    rate_bar((0, 1), "Cycle_Band", ["0", "1", "2-3", "4-6", "7+"], "Default rate by loan cycle", BLUE)
    rate_bar((0, 2), "Age_Band", ["<=25", "26-35", "36-45", "46-55", "56+"], "Default rate by age band", GREEN)
    ax = fig.add_subplot(gs[1, :2]); panel(ax, "Loan-to-income vs predicted default probability")
    s = d[d.Loan_to_Income < 12].sample(900, random_state=1)
    ax.scatter(s.Loan_to_Income, s.Predicted_PD, c=s.Risk_Band.map(BAND_COL), s=16, alpha=.7, zorder=3)
    ax.set_xlabel("Loan-to-income"); ax.set_ylabel("Predicted PD"); pct(ax)
    ax.grid(axis="x", color="#eef1f6", zorder=0)
    ax = fig.add_subplot(gs[1, 2]); panel(ax, "Model feature importance")
    i = imp[imp.Importance > 0].sort_values("Importance")
    ax.barh(i.Feature, i.Importance, color=NAVY, zorder=3); ax.set_xlabel("Drop in AUC")
    ax.grid(axis="x", color="#eef1f6", zorder=0); ax.grid(axis="y", visible=False)
    return fig


def page3(d, cp, mc):
    fig = plt.figure(figsize=(16, 9))
    header(fig, "Credit Risk - Collections & Model", "Who to chase first, risk mix by branch, and how the models compare")
    gs = GridSpec(2, 2, figure=fig, left=0.05, right=0.97, bottom=0.07, top=0.88, wspace=0.22, hspace=0.38)
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_title("Top 10 collection targets (largest balance, partly paid)", loc="left", fontsize=11.5, fontweight="bold", color=NAVY)
    t = cp.head(10)[["Client_ID", "Branch", "Loan_Officer", "Outstanding_Balance", "paid_share"]].copy()
    t["Outstanding_Balance"] = t.Outstanding_Balance.map("{:,.0f}".format); t["paid_share"] = (t.paid_share * 100).map("{:.0f}%".format)
    t.columns = ["Client", "Branch", "Officer", "Outstanding", "Paid"]
    tb = ax.table(cellText=t.values, colLabels=t.columns, loc="upper center", cellLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.55)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#e6ebf3")
        if r == 0: cell.set_facecolor(NAVY); cell.set_text_props(color="white", fontweight="bold")
        else: cell.set_facecolor("white" if r % 2 else "#f4f6fa")
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "Share of loans in each risk band, by branch"); ax.grid(False)
    m = pd.crosstab(d.Branch, d.Risk_Band, normalize="index")[["Low", "Medium", "High"]]
    im = ax.imshow(m.values, cmap="Blues", aspect="auto", vmin=0, vmax=.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(m.columns); ax.set_yticks(range(len(m))); ax.set_yticklabels(m.index)
    for r in range(m.shape[0]):
        for c in range(3): ax.text(c, r, f"{m.values[r, c]:.0%}", ha="center", va="center", fontsize=10, color="white" if m.values[r, c] > .33 else "#222")
    for s in ax.spines.values(): s.set_visible(False)
    ax = fig.add_subplot(gs[1, 0]); panel(ax, "Model comparison (hold-out test set)")
    x = np.arange(len(mc)); w = .27
    for k, (col, lab, colr) in enumerate([("test_auc", "AUC", NAVY), ("test_gini", "Gini", BLUE), ("test_ks", "KS", AMBER)]):
        ax.bar(x + (k - 1) * w, mc[col], w, label=lab, color=colr, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(mc.model); ax.legend(frameon=False, ncol=3, loc="upper right"); ax.set_ylim(0, 1)
    ax = fig.add_subplot(gs[1, 1]); panel(ax, "Predicted vs actual default rate (calibration by band)")
    g = d.groupby("Risk_Band").agg(pred=("Predicted_PD", "mean"), act=("Default_Flag", "mean")).reindex(["Low", "Medium", "High"])
    x = np.arange(3)
    ax.bar(x - .18, g.pred, .36, label="Predicted", color=BLUE, zorder=3); ax.bar(x + .18, g.act, .36, label="Actual", color=NAVY, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(g.index); pct(ax); ax.set_ylim(0, 0.5); ax.legend(frameon=False, loc="upper left")
    return fig


def main():
    d = pd.read_csv(PB / "fact_loans.csv")
    bs = pd.read_csv(PB / "dim_branch_summary.csv").sort_values("default_rate", ascending=False)
    imp = pd.read_csv(PB / "feature_importance.csv")
    cp = pd.read_csv(PB / "collection_priority.csv")
    mc = pd.read_csv(PB / "model_comparison.csv")
    C.ROOT.joinpath("images").mkdir(exist_ok=True)
    for name, fig in [("dashboard_1_overview", page1(d, bs)), ("dashboard_2_drivers", page2(d, imp)),
                      ("dashboard_3_collections_model", page3(d, cp, mc))]:
        fig.savefig(C.ROOT / "images" / f"{name}.png", dpi=110, facecolor=fig.get_facecolor())
        plt.close(fig)
    print("Saved 3 dashboard pages to images/")


if __name__ == "__main__":
    main()
