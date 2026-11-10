"""Reproduce both numerical figures from the deposited fold outputs."""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results"
mpl.rcParams.update({"font.family":"DejaVu Sans", "font.size":8.8, "axes.spines.top":False,
                     "axes.spines.right":False, "pdf.fonttype":42, "savefig.dpi":240})
colors = {"Treatment":"#7b4268", "Dilution":"#176b78"}


def example():
    fits = pd.read_csv(OUT / "default_folds.csv")
    cell = fits.query("dataset == 'orchard' and target == 'column' and fold == 4").set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.15), sharex=True, layout="constrained")
    for j, ax in enumerate(axes):
        ax.set_title(("(a)  Package output","(b)  After local refinement")[j], loc="left", fontsize=10)
        for y, model in enumerate(["Dilution","Treatment"]):
            row = cell.loc[model]
            x = row.mse if j == 0 else row.reference_loss
            label = f"{x:.4f}"
            if j:
                ax.plot([row.after_lower,row.after_upper],[y,y], color=colors[model], lw=3)
                ax.vlines([row.after_lower,row.after_upper],y-.09,y+.09,color=colors[model],lw=1.1)
                label += f"   ({int(row.repair_steps)} " + ("step)" if row.repair_steps == 1 else "steps)")
            ax.scatter(x,y,color=colors[model],s=30,marker="o" if model=="Dilution" else "D",zorder=4)
            ax.annotate(label,(x,y),xytext=(0,10),textcoords="offset points",
                        ha="center", color="#26323a",fontsize=8)
        ax.set(xlim=(.051,.157), ylim=(-.38,1.55),yticks=[0,1],yticklabels=["Dilution","Treatment"],
               xlabel="Held-out mean squared loss",xticks=[.06,.08,.10,.12,.14])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y",length=0)
        ax.grid(axis="x",lw=.5,color="#d6dee0",zorder=0)
    fig.savefig(OUT / "Fig1.pdf")
    plt.close(fig)


def geometry():
    g = pd.read_csv(OUT / "training_geometry.csv.gz").query("accepted and same_face and in_ball and usable")
    ratio = g.half_width/g.loss_error
    fig, axes = plt.subplots(1,2, figsize=(7.45,2.95),sharey=True, layout="constrained")
    for ax, x, title, xlabel in zip(axes,[g.hessian_condition,g.alignment],
            ["(a)  Free-coordinate Hessian","(b)  Direction of the refitting error"],
            ["Condition number",r"Directional alignment $|\cos\alpha|$"]):
        ax.scatter(x,ratio,s=5,alpha=.26,linewidths=0,color="#226777",rasterized=True)
        ax.set_yscale("log")
        ax.set_title(title,loc="left",fontsize=10)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y",color="#d6dee0",lw=.55)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Interval half-width / observed loss error")
    axes[0].set_ylim(.84,7500)
    axes[1].set_xscale("log")
    xx=np.geomspace(g.alignment.min(),1.,150)
    axes[1].plot(xx,1/xx,lw=.9,color="#874e47",ls="--",label=r"$1/|\cos\alpha|$")
    axes[1].legend(loc="upper right",frameon=False)
    fig.savefig(OUT / "FigAuditGeometry.pdf")
    plt.close(fig)

if __name__ == "__main__":
    example()
    geometry()
