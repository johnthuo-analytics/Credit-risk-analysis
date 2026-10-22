"""Portfolio EDA figures.  Run:  python -m src.eda"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import config as C
from .data import clean, load_raw
from .portfolio import branch_summary, concentration_hhi, cycle_summary

sns.set_theme(style="whitegrid")


def main():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = clean(load_raw())
    bs = branch_summary(df)
    bs.to_csv(C.REPORTS_DIR / "branch_summary.csv")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.barplot(x=bs.index, y=bs["default_rate"], ax=ax[0], color="#c8553d")
    ax[0].axhline(df["Default_Flag"].mean(), ls="--", color="k", label="Portfolio average")
    ax[0].set(title="Default rate by branch", ylabel="Default rate", xlabel="")
    ax[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax[0].legend()
    sns.barplot(x=bs.index, y=bs["outstanding"] / 1e6, ax=ax[1], color="#2a6f97")
    ax[1].set(title="Outstanding balance by branch (millions)", ylabel="Millions", xlabel="")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "branch_risk.png", dpi=140); plt.close(fig)

    cs = cycle_summary(df)
    ax = cs["default_rate"].plot.bar(figsize=(6, 4), color="#6a994e", rot=0)
    ax.set(title="Default rate by loan cycle", xlabel="Previous loans", ylabel="Default rate")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "cycle_default.png", dpi=140); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=df, x="Loan_Status", y="Credit_Score", ax=ax[0], palette=["#6a994e", "#c8553d"], hue="Loan_Status", legend=False)
    sns.boxplot(data=df[df.Loan_to_Income < 12], x="Loan_Status", y="Loan_to_Income", ax=ax[1], palette=["#6a994e", "#c8553d"], hue="Loan_Status", legend=False)
    ax[0].set_title("Credit score vs outcome"); ax[1].set_title("Loan-to-income vs outcome")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "drivers.png", dpi=140); plt.close(fig)

    corr = df[C.NUMERIC_FEATURES + ["Default_Flag"]].corr()
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Correlation matrix"); plt.tight_layout()
    plt.savefig(C.FIG_DIR / "correlation.png", dpi=140); plt.close()

    print(bs.round(3).to_string())
    print(f"\nBranch concentration (HHI of outstanding): {concentration_hhi(df):.3f}")
    print(cs.round(3).to_string())


if __name__ == "__main__":
    main()
