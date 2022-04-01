""" This file contains code for generating panels of the figures.

Each function should correspond to a panel in a figure in the main text or
supplemental material.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mp_colors
from matplotlib.patches import Polygon
from matplotlib.patches import FancyArrowPatch

import numpy as np
import pandas as pd
from skimage import transform
from sklearn.linear_model import LinearRegression
try:
    from IPython.display import display
except ImportError:
    display = print
    
from . import plots
from . import data_loading
from .plots import mpl_italics

matplotlib.rc("xtick", labelsize=6)
matplotlib.rc("ytick", labelsize=6)
matplotlib.rc("font", family="DejaVu Sans", size=7)


@plots.subpanel
def kms_with_unexpected_wt_variability(ax_obj, verbose=True):
    """Plot the survival functions of the original set of WT strains that
    showed some variability in lifespan.
    """
    fit_df = data_loading.get_variable_wt_survival_curves()
    fit_df = fit_df[fit_df.strain.isin(data_loading.original_variable_wt_strains)]
    fit_df["color"] = "#808080"

    # Printing some info to include in the figure legend about the nember
    # of observations in each curve.
    if verbose:
        print(f"Lifespan fits for {len(fit_df.strain.unique())} strains.")
        at_risk = fit_df[["strain", "entrance"]].groupby("strain").max()
        print("Observations per strain:")
        display(at_risk.sort_values("entrance"))
        print("Mean Observations:")
        display(at_risk.median())

    plots.km_wt_curve(fit_df, ax_obj=ax_obj)
    ax_obj.set_xlabel("# of Divisions")
    ax_obj.set_ylabel("KM - Estimate")
    ax_obj.set_title("Variable WT Lifespan Estimates")
    return ax_obj


@plots.subpanel
def polymorphism_analysis_snps_and_rdna(ax_obj):
    """Create a graphic representing the distribution of polymorphisms in the
    strains with variable lifespan.

    Each column represents a strain, ordered from shortest to longest RLS.
    Each row represents a sequence polymorphism.
        -The top section are SNPs, indels, and other deviations from the
         reference sequence. Ordered, from top to bottom, by prevalence in this
         collection of strains. To avoid genetic variants that were only present
         in a few or nearly all strains -- unlikely to explain the observed
         continuous variation in lifespans -- we excluded:
             * SNPs that were present in only 1 strain.
             * SNPs that were present in all but 1 strain.
             * CNVs that were present in <= 2 strains.
             * CNSs that were present in all but 2 strains.
        - The bottom section represents CNV variations of chunks in the yeast
          genome. Ordered, from top to bottom, by location in the genome.
    """
    table = data_loading.get_variable_wt_genetic_vars()
    cnv_table = (
        table.reset_index()
        .set_index("c")
        .loc["CNV"]
        .reset_index()
        .set_index(["CHROM", "LOCUS", "c"])
    )
    snp_table = (
        table.reset_index()
        .set_index("c")
        .loc["SNP"]
        .reset_index()
        .set_index(["CHROM", "LOCUS", "c"])
    )

    # filter out things that only occur in one/two stains
    snp_table = snp_table[
        (snp_table.sum(axis=1) > 1) & (snp_table.sum(axis=1) < len(snp_table) - 1)
    ]
    cnv_bool = cnv_table.apply(lambda x: ((x > 1.1) | (x < 0.9)) & (x > 0))
    cnv_table = cnv_table[
        (cnv_bool.sum(axis=1) > 2) & (cnv_bool.sum(axis=1) < len(cnv_bool) - 2)
    ]
    # get RLS stats for sorting columns
    lifespan_stats = data_loading.get_wt_unexpected_variability_lifespan_stats()
    lifespan_stats = lifespan_stats.groupby("strain").mean().reset_index()
    sorted_lifespans = lifespan_stats.sort_values("median")[["strain", "median"]]

    sorted_columns_snps = snp_table[list(sorted_lifespans.strain)]
    sorted_columns_cnvs = cnv_table[list(sorted_lifespans.strain)]

    img_snps = sorted_columns_snps.to_numpy()
    img_cnvs = sorted_columns_cnvs.to_numpy()

    # repeat to make pixels in image larger
    repeat = 2
    img_snps = img_snps.repeat(repeat, axis=1)
    img_cnvs = img_cnvs.repeat(repeat, axis=1)

    fig = plt.gcf()
    ax_obj.axis("off")
    left, bottom, width, height = ax_obj.get_position().bounds

    h_cnvs = height * img_snps.shape[0] / (img_snps.shape[0] + img_cnvs.shape[0])
    h_snps = height * img_cnvs.shape[0] / (img_snps.shape[0] + img_cnvs.shape[0])
    gap = 0.01
    # create sub-axes to place these plots onto
    # axes for images:
    scale = 0.75
    offset = 0.1
    ax1 = fig.add_axes((left + offset, bottom + h_snps, width * scale, h_cnvs))
    ax2 = fig.add_axes((left + offset, bottom - gap, width * scale, h_snps))
    # axes for color maps:
    ax3 = fig.add_axes(
        (left + 0.03, bottom + height * 3 / 4, width * 0.05, height * 0.15)
    )
    ax4 = fig.add_axes(
        (left + 0.03, bottom + height * 1 / 4, width * 0.05, height * 0.25)
    )
    # plot the data and the colormaps
    plot_snps = ax1.pcolormesh(img_snps, vmin=-0.1, vmax=1, cmap="binary")
    plot_cnvs = ax2.pcolormesh(img_cnvs, vmin=0.65, vmax=1.3, cmap="viridis")
    snp_bar = fig.colorbar(
        plot_snps, cax=ax3, ticklocation="left", boundaries=[-0.1, 0, 2]
    )
    cnv_bar = fig.colorbar(plot_cnvs, cax=ax4, ticklocation="left")

    # format and annotate the plots with relevant features
    snp_bar.set_ticks([-0.05, 1])
    snp_bar.set_ticklabels(["REF", "VAR"])
    ax3.set_title("SNP", fontsize="medium")

    cnv_bar.set_ticks([0.75, 1, 1.25])
    ax4.set_title("Norm. CN", fontsize="medium")

    ax1.vlines(-0.5, 0, img_snps.shape[0])
    ax1.text(
        -0.5,
        (img_snps.shape[0] - 1) / 2,
        "SNPs",
        rotation=90,
        va="center",
        ha="right",
    )

    ax2.vlines(-0.5, 0, img_cnvs.shape[0])
    ax2.text(
        -0.5,
        (img_cnvs.shape[0] - 1) / 2,
        "CNVs",
        rotation=90,
        va="center",
        ha="right",
    )
    # Find row that is in the middle of the rDNA locus
    rdna_chrom, rnda_coords = "Chr12", 467000
    rdna_row = (
        -7,
        cnv_table.reset_index()[
            (cnv_table.reset_index().CHROM == rdna_chrom)
            & (cnv_table.reset_index().LOCUS == rnda_coords)
        ].index[0]
        - 0.5,
    )
    arrow_b = (-2, rdna_row[1])
    arrow_a = (-6, rdna_row[1])

    arrow = FancyArrowPatch(
        arrow_a,
        arrow_b,
        arrowstyle="-[, widthB=5, lengthB=1",
        alpha=0.6,
        lw=2,
        clip_on=False,
    )
    # Create a slightly stretched axis to add the annotations to.

    fig = plt.gcf()
    left, bottom, width, height = ax2.get_position().bounds
    ax3 = fig.add_axes((left, bottom, width * 1.1, height), label="for annotations")
    ax3.axis("off")
    ax3.set_ylim(ax2.get_ylim())
    ax3.set_xlim(ax2.get_xlim())
    ax3.text(*rdna_row, "In rDNA locus", va="center", ha="right")
    ax3.add_patch(arrow)

    ax2.text(
        img_cnvs.shape[1] / 2,
        -3,
        "RLS",
        verticalalignment="center",
        horizontalalignment="center",
    )
    ax2.arrow(
        *(1, -1),
        *(img_cnvs.shape[1] - 4, 0),
        head_width=0.65,
        width=0.25,
        color="k",
        clip_on=False,
    )

    ax_obj.set_title("Distribution of Genetic Variations", loc="left")
    for ax_ in (ax1, ax2):
        ax_.set_yticks([])
        ax_.set_yticklabels([])
        ax_.set_xticks([])
        ax_.set_xticklabels([])
        ax_.axis("off")

    return ax_obj


@plots.subpanel
def rdcn_vs_rls_of_unexpected_wt_variability(ax_obj, verbose=True):
    """ Plot rDNA CN vs RLS for original WT strains with variable lifespan."""
    lifespan_stats = data_loading.get_wt_unexpected_variability_lifespan_stats()
    if verbose:
        display(lifespan_stats)
    for _, strain_df in lifespan_stats.groupby("strain"):
        plots.plot_scatter(
            strain_df,
            ax_obj=ax_obj,
            xs="rdnacn",
            ys="median",
            ymin="lower_ci",
            ymax="upper_ci",
            color=plots.r2c(float(strain_df.iloc[0].rdnacn)),
            markersize=4,
        )
    plots.format_rls_by_rdna_ax(ax_obj)
    ax_obj.set_title("")
    ax_obj.set_ylim(0, 30)
    ax_obj.set_xlim(50, 175)
    ax_obj.set_title("rDNA CN vs RLS")

    fig = plt.gcf()
    left, bottom, width, height = ax_obj.get_position().bounds
    colorbar_ax = fig.add_axes(
        (
            left + width - width * 0.25,
            bottom + height * 0.15,
            width * 0.05,
            height * 0.2,
        )
    )
    colorbar_plotted_ax = plots.add_on_color_bar(
        cax=colorbar_ax,
        val_lim=(75, 175),
        c_lim=(plots.r2c.min, plots.r2c.max),
        cmap=plots.r2c.color_map_name,
    )
    colorbar_plotted_ax.set_yticks([75, 125, 175])
    colorbar_ax.set_title("rDNA CN", fontsize="medium")
    return ax_obj


@plots.subpanel
def kms_of_rdnacn_panel(ax_obj, verbose=True):
    """Plot the survival functions of the panel of strains that were selected/
    engineered to have rDNA CN ranging from ~50 to ~250.
    """
    fit_df = data_loading.get_variable_wt_survival_curves()
    if verbose:
        print(f"Lifespan fits for {len(fit_df.strain.unique())} strains.")
        at_risk = fit_df[["strain", "entrance"]].groupby("strain").max()
        print("Observations per strain:")
        display(at_risk.sort_values("entrance"))
        print("Mean Observations:")
        display(at_risk.median())
    fit_df = data_loading.add_standard_rdnacn_to_table(fit_df)
    plots.km_wt_curve(fit_df, ax_obj=ax_obj)

    ax_obj.set_xlabel("# of Divisions")
    ax_obj.set_ylabel("KM - Estimate")
    ax_obj.set_xlim(0, 80)
    ax_obj.set_title("Variable Lifespan Estimates in rDNA CN panel")

    fig = plt.gcf()
    left, bottom, width, height = ax_obj.get_position().bounds
    colorbar_ax = fig.add_axes(
        (
            left + 0.04,
            bottom + 0.01,
            0.01,
            0.07,
        )
    )
    colorbar_plotted_ax = plots.add_on_color_bar(
        cax=colorbar_ax,
        val_lim=(75, 250),
        c_lim=(plots.r2c.min, plots.r2c.max),
        cmap=plots.r2c.color_map_name,
    )
    colorbar_plotted_ax.set_yticks([75, 150, 225])
    colorbar_ax.set_title("rDNA CN", fontsize="medium")

    return ax_obj


@plots.subpanel
def rdnacn_vs_rls_scatter_plot(ax_obj):
    """Plot rDNA CN vs RLS of the panel of strains that were selected/
    engineered to have rDNA CN ranging from ~50 to ~250.
    """
    lifespan_stats = data_loading.get_lifespan_summary()
    wt_stats = lifespan_stats[lifespan_stats.KO_gene == "-"]
    plots.add_wt_to_plot(ax_obj, wt_stats, alpha=1)
    plots.format_rls_by_rdna_ax(ax_obj)
    ax_obj.set_title("")
    ax_obj.set_ylim(0, 40)
    ax_obj.set_xlim(50, 250)
    ax_obj.set_title("rDNA CN vs RLS")

    fig = plt.gcf()
    left, bottom, width, height = ax_obj.get_position().bounds
    colorbar_ax = fig.add_axes(
        (
            left + width - 0.04,
            bottom + 0.01,
            0.01,
            0.07,
        )
    )

    colorbar_plotted_ax = plots.add_on_color_bar(
        cax=colorbar_ax,
        val_lim=(75, 250),
        c_lim=(plots.r2c.min, plots.r2c.max),
        cmap=plots.r2c.color_map_name,
    )
    colorbar_plotted_ax.set_yticks([75, 150, 225])
    colorbar_ax.set_title("rDNA CN", fontsize="medium")

    return ax_obj


@plots.subpanel
def wgs_vs_chef_for_rdnacn_analysis(ax_obj):
    """Plot rDNA CN estimates calculated from WGS sequencing data vs those
    made from CHEF gels.
    """
    table = data_loading.get_chef_vs_wgs_rdancn_estimates()
    ax_obj.scatter(
        x=table["CHEF rDNAcn Estimate"],
        y=table["WGS rDNAcn Estimate"],
    )
    ax_obj.set_ylabel("WGS rDNA CN Estimate")
    ax_obj.set_xlabel("CHEF-Gel rDNA CN Estimate")

    xcoord = table["CHEF rDNAcn Estimate"].values.reshape(-1, 1)
    ycoord = table["WGS rDNAcn Estimate"].values

    reg = LinearRegression().fit(xcoord, ycoord)

    slope = reg.coef_[0]
    intercept = reg.intercept_
    r2_ = reg.score(xcoord, ycoord)

    xvals = np.arange(min(xcoord), max(xcoord))
    yvals = xvals * slope + intercept

    ax_obj.plot(xvals, yvals, alpha=0.5, color="k")
    sign = "-" if intercept < 0 else "+"
    intercept = abs(intercept)

    ax_obj.text(
        x=xcoord.min(),
        y=ycoord.max() + 20,
        s=f"WGS = CHEF * {slope:.02f} {sign} {intercept:.02f} \nR**2={r2_:.2f}",
        fontsize="x-small",
    )
    ax_obj.set_yticks([25, 75, 125, 175, 225, 275])
    ax_obj.set_ylim(0, 300)
    ax_obj.set_xticks([25, 75, 125, 175, 225, 275])
    ax_obj.set_xlim(0, 300)

    return ax_obj


@plots.subpanel
def wgs_vs_chef_for_rdnacn_gel(ax_obj):
    """Plot CHEF gel of strains with varying rDNA CN. Annotated with scale for
    both array size in megabases and number of repeats.

    Samples are loaded in order of increasing size predicted from WGS.
    Note that the array bands in lanes 10 and 11 did not resolve well because
    these were outside of the optimized range for resolution by CHEF. Arrows
    indicate minor band that migrated at/near expected.
    """
    # Load and display gel
    img = data_loading.get_chef_gel_image()
    ax_obj.imshow(
        img, clim=(np.percentile(img, 5), np.percentile(img, 95)), cmap="binary"
    )
    # Add annotations/scales
    chef_gel_details = data_loading.ChefGelDetails()
    offset_to_well = 60  # pixels from crop to well start
    ladder_locs = np.copy(chef_gel_details.ladder_pix) + offset_to_well

    # from Bio-Rad's website, omitted some bands
    formatted_ladder_labels = [f"{i:.2f} Mb" for i in chef_gel_details.ladder_mbs]
    ax_obj.set_yticks(ladder_locs)
    ax_obj.set_yticklabels(formatted_ladder_labels)
    ax_obj.set_xticks([])
    ax2 = ax_obj.secondary_yaxis("right")  # for copy# annotations

    cns = [75, 125, 160, 200, 250]
    locs = [
        chef_gel_details.rdnacn2pix(copy_number) + offset_to_well for copy_number in cns
    ]
    formatted_labels = [f"{int(round(label))} CN" for label in cns]
    ax2.set_yticks(locs)
    ax2.set_yticklabels(formatted_labels)

    # Add arrow annotations for poorly resolved bands
    ax_obj.arrow(
        410,  # Start
        chef_gel_details.rdnacn2pix(260) + offset_to_well,  # Y
        dx=30,
        dy=30,  # dy
        width=2,
        head_width=8,
        head_length=10,
        color="r",
    )
    ax_obj.arrow(
        458,  # Start
        chef_gel_details.rdnacn2pix(280) + offset_to_well,  # Y
        dx=30,
        dy=30,  # dy
        width=2,
        head_width=8,
        head_length=10,
        color="r",
    )
    return ax_obj


@plots.subpanel
def rdnacn_vs_rls_scatter_plot_in_different_media(ax_obj):
    """Plot rDNA CN vs RLS of the panel of strains that were grown and aged in
    different environments/media.

    rDNA CN estimates were estimated by WGS samples prepared from the same
    cultures as used for lifespan experiment.
    """

    lifespan_stats = data_loading.get_lifespan_summary_for_different_medias()
    plots.plot_scatter(
        lifespan_stats,
        ax_obj=ax_obj,
        xs="rdnacn",
        ys="median",
        ymin="lower_ci",
        ymax="upper_ci",
        groupby="media",
    )
    plots.format_rls_by_rdna_ax(ax_obj)
    ax_obj.set_title("")
    ax_obj.set_ylim(0, 40)
    ax_obj.set_xlim(50, 250)
    ax_obj.set_title("rDNA CN vs RLS")
    ax_obj.legend()
    return ax_obj


@plots.subpanel
def rdnacn_vs_rls_scatter_plot_in_by4743(ax_obj):
    """Plot rDNA CN vs RLS of the panel of strains generated from a standard
    WT strain, BY4743. This strain has been used in a majority of the
    previously published yeast aging literature.

    Other WT strains from this study are also plotted w/ lower alpha for reference.
    """

    lifespan_stats = data_loading.get_lifespan_summary()
    wt_stats = lifespan_stats[lifespan_stats.KO_gene == "-"]
    plots.add_wt_to_plot(ax_obj, wt_stats, alpha=1)

    lifespan_stats = data_loading.get_lifespan_summary_for_by_strains()
    plots.plot_scatter(
        lifespan_stats,
        ax_obj=ax_obj,
        xs="rdnacn",
        ys="median",
        ymin="lower_ci",
        ymax="upper_ci",
        color="k",
        alpha=1,
        markeredgecolor="none",
    )

    plots.format_rls_by_rdna_ax(ax_obj)
    ax_obj.set_title("")
    ax_obj.set_ylim(0, 40)
    ax_obj.set_xlim(50, 300)
    ax_obj.set_title("rDNA CN vs RLS")
    return ax_obj


@plots.subpanel
def growth_rate_vs_rdnacn(ax_obj):
    """Plot culture growth rate (displayed as doubling time) as a function of
    rDNA CN.
    """
    growth_rates = data_loading.get_panel_growth_rates().reset_index()
    growth_rates["rdnacn"] = growth_rates["rdnacn"].astype(float)
    for xcoord, ycoord in zip(
        growth_rates["rdnacn"], growth_rates["doubling_time_minutes"]
    ):
        ax_obj.plot(
            xcoord, ycoord, linestyle="none", color=plots.r2c(xcoord), marker="o"
        )
    ax_obj.set_ylim(85, 110)
    ax_obj.set_xlim(63.75, 245.25)
    ax_obj.set_ylabel("Doubling Time (minutes)")
    ax_obj.set_xlabel("rDNA CN")
    ax_obj.set_title("Doubling Time vs rDNA CN")

    ax_obj.annotate(  # note that this does no play well with resizing figure. be careful
        "rDNA CN range\naffecting RLS",
        xy=(110, 96),
        arrowprops=dict(
            arrowstyle="-[, widthB=6.5, lengthB=.2",
            facecolor="black",
            alpha=0.6,
            shrinkB=15,
        ),
        xytext=(110, 102),  # Manually chosen,
        ha="center",
        fontsize=8,
    )
    # Calculating some stats to report

    xcoord = growth_rates.reset_index()["rdnacn"].values.reshape(-1, 1)
    ycoord = growth_rates["doubling_time_minutes"].values
    reg = LinearRegression().fit(xcoord, ycoord)
    ax_obj.plot(
        np.linspace(70, 240, 100),
        np.linspace(70, 240, 100) * reg.coef_[0] + reg.intercept_,
        color="k",
        alpha=0.2,
    )
    print(reg.coef_)

    growth_rates = growth_rates.reset_index()
    growth_rates = growth_rates[growth_rates.rdnacn < 150]
    xcoord = growth_rates["rdnacn"].values.reshape(-1, 1)
    ycoord = growth_rates["doubling_time_minutes"].values
    reg = LinearRegression().fit(xcoord, ycoord)
    ax_obj.plot(
        np.linspace(70, 150, 100),
        np.linspace(70, 150, 100) * reg.coef_[0] + reg.intercept_,
        color="b",
        alpha=0.2,
    )
    print(reg.coef_)
    return ax_obj


@plots.subpanel
def cell_size_vs_rdnacn(ax_obj):
    """Plot cell size (median diameter µm) as a function of rDNA CN.

    Volumes were measured on a Coulter counter from same strain on 3 different
    days (3 different exponentially growing cultures).

    Y-axis limits for this figure were set based upon approximately 95% of the
    cells size difference observed in the studies of the yeast deletion
    collection.

    """
    cell_sizes = data_loading.get_panel_cell_size()
    dates = ["feb_5", "feb_10", "feb_24"]
    markers = ["o", "s", "D"]

    for date, marker in zip(dates, markers):
        for xcoord, ycoord in zip(cell_sizes["rdnacn"], cell_sizes[date]):
            ax_obj.plot(
                xcoord,
                ycoord,
                linestyle="none",
                color=plots.r2c(xcoord),
                marker=marker,
            )
    # Out of frame points for the legend
    for label, mark in zip(range(len(markers)), markers):
        label = f"rep. {label+1}"
        ax_obj.scatter(
            -1, -1, marker=mark, label=label, facecolors="none", edgecolors="k"
        )
    ax_obj.legend()
    ax_obj.set_ylim(3.5, 6)
    ax_obj.set_xlim(63.75, 245.25)
    ax_obj.legend()
    ax_obj.set_ylabel("Median Cell Diameter (µm)")
    ax_obj.set_xlabel("rDNA CN")
    ax_obj.set_title("Cell Size vs rDNA CN")
    return ax_obj


@plots.subpanel
def large_model(ax_obj):
    """Load and display large model figure that was generated/drawn elsewhere."""
    img = plt.imread(os.path.join(data_loading.data_dir, "model large.png"))
    ax_obj.imshow(img)
    ax_obj.axis("off")
    return ax_obj


@plots.subpanel
def rnaseq_vs_rls_volcano(ax_obj):
    """ Plot volcano plot of gene expression changes as a function of RLS. """
    results = data_loading.get_panel_rnaseq_deseq_vs_rls_table().apply(
        pd.to_numeric, axis=1
    )
    sig_cutoff = 1e-5
    sig = results[results.padj < sig_cutoff]
    nonsig = results[results.padj >= sig_cutoff]

    ax_obj.scatter(
        nonsig.log2FoldChange,
        np.log10(nonsig.padj) * -1,
        alpha=0.05,
        color="k",
        edgecolor=None,
    )
    ax_obj.scatter(
        sig.log2FoldChange,
        np.log10(sig.padj) * -1,
        alpha=0.4,
        edgecolor=None,
        color="red",
    )
    ax_obj.set_ylabel("-log10(padj)")
    ax_obj.set_xlabel("FC / RLS (a.u.)")
    ax_obj.set_xlim(-0.1, 0.1)
    ax_obj.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])

    ax_obj.set_title("Gene Expression Changes")
    # Annotations
    xcoord, ycoord = results.loc["SIR2"][["log2FoldChange", "padj"]].to_numpy()
    ycoord = np.log10(ycoord) * -1

    ax_obj.annotate(
        mpl_italics("SIR2"),
        xy=(xcoord, ycoord),
        arrowprops=dict(arrowstyle="->", facecolor="black", alpha=0.6, shrinkB=5),
        xytext=(0.0, 6),  # Manually chosen
        fontsize=8,
    )
    return ax_obj


@plots.subpanel
def atacseq_at_sir2_locus(ax_obj):
    """ Plot ATAC-seq insertion density at the SIR2 locus. """
    # load copy number for applying appropriate color
    cn_df = data_loading.get_rdnacn_measurements()
    cn_df = cn_df[pd.isnull(cn_df.media)]

    # load atac-seq insertion density at SIR2 locus
    roi_data = data_loading.get_atac_seq_counts_at_sir2()
    smoothed = roi_data.rolling(50).mean()  # smooth will a rolling window of 50bp
    smoothed = smoothed.multiply(1e7)  # scale data

    # create a few sub-axes to plot data/annotations
    data_ax = ax_obj
    gene_ax = data_ax.twinx()
    gene_ax.set_ylim(0, 1)

    # add genes at this locus. coordinates were obtained from SGD.
    sir2_start, sir2_stop = 378445, 376757
    nat1_start, nat1_stop = 381438, 378874
    prp11_start, prp11_stop = 376480, 375680

    plots.add_gene(
        sir2_start, sir2_stop, mpl_italics("SIR2"), gene_ax, y_loc=0.88, fc="gray"
    )
    plots.add_gene(
        nat1_start, nat1_stop, mpl_italics("NAT1"), gene_ax, y_loc=0.88, fc="white"
    )
    plots.add_gene(
        prp11_start, prp11_stop, mpl_italics("PRP11"), gene_ax, y_loc=0.88, fc="white"
    )

    # plot data
    for strain in roi_data.columns:
        copy_number = cn_df.loc[strain].rdnacn
        data_ax.plot(
            smoothed[strain],
            c=plots.r2c(copy_number),
            alpha=0.6,
            label="{strain}-{int(round(copy_number))}",
        )

    # format ax_obj
    gene_ax.set_yticks([])
    data_ax.set_ylabel("Relative Insertion Density (a.u.)")
    data_ax.set_xlabel("Chromosome Position -- Chr4")
    data_ax.set_title(
        "Chromatin Accessibility Changes at " + mpl_italics("SIR2") + " Locus"
    )

    # add annotation of UAF binding site estimated from Iida et al (2019).
    gene_ax.axvline(sir2_start + 200, c="k")
    gene_ax.axvline(sir2_start + 350, c="k")
    gene_ax.annotate(
        "UAF Binding Site",
        xy=(sir2_start + 200, 0.5),
        arrowprops=dict(
            facecolor="black",
            shrink=0.1,
            width=2,
            headwidth=8,
            alpha=0.6,
        ),
        xytext=(sir2_start - 945, 0.3),  # Manually chosen
        fontsize=plt.rcParams.get("axes.titlesize"),
        horizontalalignment="center",
        verticalalignment="center",
    )

    data_ax.set_xlim(min(roi_data.index), max(roi_data.index))

    fig = plt.gcf()
    left, bottom, width, height = ax_obj.get_position().bounds
    colorbar_ax = fig.add_axes(
        (
            left + 0.08,
            bottom + 0.1,
            0.01,
            0.07,
        )
    )
    colorbar_plotted_ax = plots.add_on_color_bar(
        cax=colorbar_ax,
        val_lim=(75, 250),
        c_lim=(plots.r2c.min, plots.r2c.max),
        cmap=plots.r2c.color_map_name,
    )
    colorbar_plotted_ax.set_yticks([75, 150, 225])
    colorbar_ax.set_title("rDNA CN", fontsize="medium")

    return data_ax


@plots.subpanel
def atacseq_at_uaf_binding_site(ax_obj):
    """ Plot ATAC-seq insertion density at the UAF binding site upstream of SIR2"""
    cn_df = data_loading.get_rdnacn_measurements()
    cn_df = cn_df[pd.isnull(cn_df.media)]

    roi_data = data_loading.get_atac_seq_counts_at_sir2()

    data_ax = ax_obj
    gene_ax = data_ax.twinx()

    smoothed = roi_data.rolling(50).mean()
    smoothed = smoothed.multiply(1e7)

    for strain in roi_data.columns:
        copy_number = cn_df.loc[strain].rdnacn
        data_ax.plot(
            smoothed[strain],
            c=plots.r2c(copy_number),
            alpha=0.6,
            label="{strain}-{int(round(copy_number))}",
        )

    gene_ax.set_yticks([])
    data_ax.set_ylabel("Relative Insertion Density (a.u.)")
    data_ax.set_xlabel("Chromosome Position -- Chr4")
    data_ax.set_title("Accessibility at UAF Binding Site")
    sir2_start = 378445
    gene_ax.axvline(sir2_start + 200, c="k")  # annotated UAF binding site
    gene_ax.axvline(sir2_start + 350, c="k")
    data_ax.set_xlim(sir2_start + 150, sir2_start + 400)
    return data_ax


@plots.subpanel
def rdna_expression(ax_obj):
    """Plot expression of several rRNA genes as a function of rDNA CN.

    Libraries were prepared from non-polyA-selected RNA.
    """
    results = pd.DataFrame(data_loading.get_nonselected_counts())
    rdna_results = pd.DataFrame(
        results[
            results.index.str.contains("rdn", case=False)  # rDNA genes
            & ~results.index.str.contains(
                "cut", case=False
            )  # but not their overlapping features
        ]
    )

    rdna_results = rdna_results.assign(
        gene=rdna_results.reset_index()["index"]
        .str.extract("([0-9a-zA-Z]+)")[0]
        .to_list()
    )
    rdna_results = pd.DataFrame(rdna_results.sum())
    rdna_results.loc["gene"] = "All rRNAs"

    rdna_results = rdna_results.T.set_index("gene").T
    rdna_measurements = data_loading.get_rdnacn_measurements()
    rdna_measurements = rdna_measurements[pd.isnull(rdna_measurements.media)]
    rdna_measurements = rdna_measurements[["rdnacn"]]
    rdna_results = rdna_results.merge(
        rdna_measurements,
        left_index=True,
        right_index=True,
        how="left",  # make sure we don't drop rows that we don't have rDNAcn for
    )
    ax_obj.plot(
        rdna_results["rdnacn"],
        rdna_results["All rRNAs"],
        label="All rRNAs",
        marker="o",
        linestyle="None",
    )
    ax_obj.set_ylim(
        rdna_results["All rRNAs"].min() - rdna_results["All rRNAs"].std() * 3,
        rdna_results["All rRNAs"].max() + rdna_results["All rRNAs"].std() * 3,
    )
    ax_obj.legend()
    ax_obj.set_xlabel("rDNA CN")
    ax_obj.set_ylabel("Normalized Counts")
    ax_obj.set_title("Reads Mapping To Genes in Unselected RNA")
    return ax_obj


@plots.subpanel
def old_cell_erc_blot(ax_obj):
    """Plot annotated image of southern blot of ERC levels from aged cells.

    Arrows  1-4  highlight  bands  of Extrachromosomal rDNA Circles (ERCs),
    arrow A highlights the chromosomal rDNA array.

    Southern blots probed for rDNA and NPR2 (loading control).

    Samples  were  obtained  from  wild-type  and fob1∆ strains  with  variable
    rDNA  CN  aged  for  24  hrs.
    """
    fig = plt.gcf()

    img = data_loading.get_old_cell_erc_blot()
    # show image
    ax_obj.imshow(img, cmap="binary", clim=(200, 32000))
    ax_obj.set_yticks([])
    ax_obj.set_xticks([])
    # create ax_obj to add annotations to
    ann_ax = fig.add_axes(ax_obj.get_position())
    ann_ax.set_ylim(0, 1)
    ann_ax.set_xlim(0, 1)
    # annotate rDNA probe
    ann_ax.axvline(-0.03, 0.05, 0.95, clip_on=False, color="k", linewidth=0.7)
    ann_ax.text(
        x=-0.07,
        y=0.5,
        s="rDNA",
        rotation=90,
        horizontalalignment="left",
        verticalalignment="center",
        fontsize=plt.rcParams.get("axes.titlesize"),
    )
    # annotate genotypes
    ann_ax.axhline(
        y=1.17, xmin=0.02, xmax=0.49, clip_on=False, color="k", linewidth=0.7
    )
    ann_ax.axhline(
        y=1.17, xmin=0.51, xmax=0.96, clip_on=False, color="k", linewidth=0.7
    )

    ann_ax.text(
        x=0.51 / 2,
        y=1.19,
        s="WT",
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=plt.rcParams.get("axes.titlesize"),
    )

    ann_ax.text(
        x=(0.51 + 0.96) / 2,
        y=1.19,
        s=mpl_italics("fob1∆"),
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=plt.rcParams.get("axes.titlesize"),
    )
    
    # annotate rDNA CN increasing from left to right on gel
    coords = np.asarray([[0, 1.02], [0.49, 1.02], [0.49, 1.12]])

    patch = Polygon(coords, color="#a6a6a6", clip_on=False)
    ann_ax.add_patch(patch)
    patch = Polygon(coords + [0.50, 0], color="#a6a6a6", clip_on=False)
    ann_ax.add_patch(patch)
    ann_ax.text(0.32, 1.022, "rDNA CN", ha="center", va="bottom")
    ann_ax.text(0.32 + 0.5, 1.022, "rDNA CN", ha="center", va="bottom")
    # annotate band species
    arrow_locs = [0.62, 0.52, 0.32, 0.08]
    arrow_labels = ["2", "3", "A", "4"]
    for loc, label in zip(arrow_locs, arrow_labels):
        ann_ax.annotate(
            label,
            xy=(1, loc),
            arrowprops=dict(
                facecolor="black",
                width=4,
                headwidth=8,
                headlength=8,
                alpha=0.6,
                edgecolor="none",
                shrink=0.1,
            ),
            xytext=(1.1, loc),  # Manually chosen
            fontsize=plt.rcParams.get("axes.titlesize"),
            ha="center",
            va="center",
        )

    loc = 0.81
    ann_ax.annotate(
        "1",
        xy=(1, loc),
        arrowprops=dict(
            arrowstyle="-[, widthB=1.15, lengthB=0",
            facecolor="black",
            alpha=0.6,
            lw=4,
            clip_on=False,
            shrinkB=4,
        ),
        xytext=(1.1, loc),  # Manually chosen
        fontsize=plt.rcParams.get("axes.titlesize"),
        ha="center",
        va="center",
    )

    ann_ax.axis("off")
    # add image of control gel
    img = data_loading.get_old_cell_npr2_blot()
    left, bottom, width, height = ax_obj.get_position().bounds
    ax2 = fig.add_axes([left, bottom - (height * 0.7), width, height])
    ax2.imshow(img, cmap="binary", clim=(0.01, 0.15), clip_on=False)
    
    # Add scale bar for migration distances
    pix_per_cm = 50
    len_pix = 1 * pix_per_cm
    ax2.hlines(
        y=img.shape[0]-10,
        xmin=img.shape[1]-len_pix-10,
        xmax=img.shape[1]-10
    )
    
    ax2.set_yticks([])
    ax2.set_xticks([])
    # annotate NPR2 probe
    ann_ax2 = fig.add_axes(ax2.get_position())
    ann_ax2.axis("off")
    ann_ax2.set_ylim(0, 1)
    ann_ax2.set_xlim(0, 1)
    ann_ax2.axvline(-0.03, 0.05, 0.95, clip_on=False, color="k", linewidth=0.7)
    ann_ax2.text(
        x=-0.07,
        y=0.5,
        s="NPR2",
        rotation=90,
        horizontalalignment="left",
        verticalalignment="center",
        fontsize=plt.rcParams.get("axes.titlesize"),
    )
    return ax_obj


@plots.subpanel
def young_cell_erc_blot(ax_obj):
    """Plot annotated image of southern blot of ERC levels in DNA prepared from
    young cells.
    """

    img1 = data_loading.get_young_erc_blot_rdna_probe()
    img2 = data_loading.get_young_erc_blot_npr2_probe()
    # resize to make the same width as the rDNA probed image
    img2 = transform.resize(
        img2, [int(round(img2.shape[0] * img1.shape[1] / img2.shape[1])), img1.shape[1]]
    )

    fig = plt.gcf()
    ax_obj.axis("off")
    left, bottom, width, height = ax_obj.get_position().bounds

    height1 = height * img1.shape[0] / (img1.shape[0] + img2.shape[0])
    height2 = height * img2.shape[0] / (img1.shape[0] + img2.shape[0])
    ax1 = fig.add_axes((left, bottom + height2, width, height1))
    ax2 = fig.add_axes((left, bottom, width, height2))
    ax1.imshow(
        img1, clim=(np.percentile(img1, 5), np.percentile(img1, 95)), cmap="binary"
    )
    ax2.imshow(
        img2, clim=(np.percentile(img2, 5), np.percentile(img2, 95)), cmap="binary"
    )
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticks([])

    ann_ax = fig.add_axes(ax1.get_position())
    ann_ax.axis("off")

    ann_ax.set_ylim(0, 1)
    ann_ax.set_xlim(0, 1)
    ann_ax.axvline(-0.03, 0.05, 0.95, clip_on=False, color="k", linewidth=0.7)
    ann_ax.text(
        x=-0.09,
        y=0.5,
        s="rDNA",
        rotation=90,
        horizontalalignment="left",
        verticalalignment="center",
        fontsize=plt.rcParams.get("axes.titlesize"),
    )

    ann_ax.axhline(
        y=1.17, xmin=0.02, xmax=0.45, clip_on=False, color="k", linewidth=0.7
    )
    ann_ax.axhline(
        y=1.17, xmin=0.46, xmax=0.92, clip_on=False, color="k", linewidth=0.7
    )
    ann_ax.axhline(
        y=1.17, xmin=0.93, xmax=0.99, clip_on=False, color="k", linewidth=0.7
    )

    ann_ax.axhline(
        y=1.31, xmin=0.02, xmax=0.92, clip_on=False, color="k", linewidth=0.7
    )
    ann_ax.axhline(
        y=1.31, xmin=0.93, xmax=0.99, clip_on=False, color="k", linewidth=0.7
    )

    ann_ax.text(
        x=(0.02 + 0.45) / 2,
        y=1.19,
        s="WT",
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=plt.rcParams.get("axes.titlesize"),
    )

    ann_ax.text(
        x=(0.93 + 0.99) / 2,
        y=1.19,
        s="WT",
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=plt.rcParams.get("axes.titlesize"),
    )

    ann_ax.text(
        x=(0.46 + 0.92) / 2,
        y=1.19,
        s=mpl_italics("fob1∆"),
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=plt.rcParams.get("axes.titlesize"),
    )

    ann_ax.text(
        x=(0.02 + 0.92) / 2,
        y=1.33,
        s="Young",
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=plt.rcParams.get("axes.titlesize"),
    )

    ann_ax.text(
        x=(0.93 + 0.99) / 2,
        y=1.33,
        s="O",
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=plt.rcParams.get("axes.titlesize"),
    )

    coords = np.asarray([[0, 1.02], [0.46, 1.02], [0.46, 1.12]])

    patch = Polygon(coords, color="#a6a6a6", clip_on=False)
    ann_ax.add_patch(patch)
    patch = Polygon(coords + [0.465, 0], color="#a6a6a6", clip_on=False)
    ann_ax.add_patch(patch)

    ann_ax.text(0.33, 1.01, "rDNA CN", ha="center", va="bottom")
    ann_ax.text(0.33 + 0.46, 1.01, "rDNA CN", ha="center", va="bottom")
    arrow_locs = [0.62, 0.52, 0.32, 0.08]
    arrow_labels = ["2", "3", "A", "4"]
    for loc, label in zip(arrow_locs, arrow_labels):
        ann_ax.annotate(
            label,
            xy=(1, loc),
            arrowprops=dict(
                facecolor="black",
                width=4,
                headwidth=8,
                headlength=8,
                alpha=0.6,
                shrink=0.1,
                edgecolor="none",
            ),
            xytext=(1.15, loc),  # Manually chosen
            fontsize=plt.rcParams.get("axes.titlesize"),
            ha="center",
            va="center",
        )

    loc = 0.81
    ann_ax.annotate(
        "1",
        xy=(1, loc),
        arrowprops=dict(
            arrowstyle="-[, widthB=1.15, lengthB=0",
            facecolor="black",
            alpha=0.6,
            shrinkB=4,
            lw=4,
            clip_on=False,
        ),
        xytext=(1.15, loc),  # Manually chosen
        fontsize=plt.rcParams.get("axes.titlesize"),
        ha="center",
        va="center",
    )

    ann_ax2 = fig.add_axes(ax2.get_position())
    ann_ax2.axis("off")
    ann_ax2.set_ylim(0, 1)
    ann_ax2.set_xlim(0, 1)
    ann_ax2.axvline(-0.03, 0.05, 0.95, clip_on=False, color="k", linewidth=0.7)
    ann_ax2.text(
        x=-0.08,
        y=0.5,
        s="NPR2",
        rotation=90,
        horizontalalignment="left",
        verticalalignment="center",
        fontsize=plt.rcParams.get("axes.titlesize"),
    )
    return ax_obj


@plots.subpanel
def quantification_of_the_erc_blot_scatter(ax_obj):
    """ Plot quantified ERC levels (all species) as a function or rDNA CN."""
    gels = data_loading.get_erc_blot_quantifications()  # load data
    old_cell_gel = gels[gels.gel == "old_samples"]  # get data just from old cells

    # prepare WT data
    wt_lanes = old_cell_gel[old_cell_gel.genotype == "WT"]
    wt_colors = [plots.r2c(copy_number) for copy_number in wt_lanes["rdnacn"]]
    itbl = zip(wt_lanes["rdnacn"], wt_lanes["normalized ERCs"], wt_colors)
    for xcoord, ycoord, color in itbl:
        ax_obj.scatter(xcoord, ycoord, color=color, marker="o")

    # prepare fob1∆ data
    fob1_lanes = old_cell_gel[old_cell_gel.genotype == "fob1∆"]
    fob1_colors = [plots.r2c_magma(copy_number) for copy_number in fob1_lanes["rdnacn"]]
    itbl = zip(fob1_lanes["rdnacn"], fob1_lanes["normalized ERCs"], fob1_colors)
    for xcoord, ycoord, color in itbl:
        ax_obj.scatter(xcoord, ycoord, color=color, marker="D")

    # format axis
    ax_obj.set_ylabel("Normalized Total ERC Levels (a.u.)")
    ax_obj.set_xlabel("rDNA CN")
    ax_obj.set_title("ERC Quantification")
    ax_obj.set_xlim(50, 210)
    ax_obj.set_ylim(0, 40)

    # Create  a legend with the appropriate marker shape for each strain
    # plot out of frame points for the legend
    for label, marker in zip(("WT", mpl_italics("fob1∆")), ("o", "D")):
        ax_obj.scatter(
            -1, -1, marker=marker, label=label, facecolors="none", edgecolors="k"
        )
    ax_obj.legend()

    fig = plt.gcf()
    left, bottom, width, height = ax_obj.get_position().bounds
    h = 0.07
    colorbar_ax = fig.add_axes(
        (
            left + width * 0.8,
            bottom + height * 0.45,
            0.01,
            h,
        )
    )
    colorbar_plotted_ax = plots.add_on_color_bar(
        cax=colorbar_ax,
        val_lim=(75, 250),
        c_lim=(plots.r2c.min, plots.r2c.max),
        cmap=plots.r2c.color_map_name,
    )
    colorbar_plotted_ax.set_yticks([75, 150, 225])
    colorbar_ax.set_title("rDNA CN", fontsize="medium")

    fob1_colorbar_ax = fig.add_axes(
        (
            left + width * 0.8 + 0.015,
            bottom + height * 0.45,
            0.01,
            h,
        )
    )
    fob1_colorbar_plotted_ax = plots.add_on_color_bar(
        cax=fob1_colorbar_ax,
        val_lim=(75, 250),
        c_lim=(plots.r2c_magma.min, plots.r2c_magma.max),
        cmap=plots.r2c_magma.color_map_name,
    )
    fob1_colorbar_plotted_ax.set_yticks([])

    return ax_obj


@plots.subpanel
def quantification_of_young_erc_blot_scatter(ax_obj):
    """Plot quantified ERC levels (all species) as a function or rDNA CN in
    young cells. One aged WT sample is added for reference.
    """

    gels = data_loading.get_erc_blot_quantifications()
    young_gel = gels[gels.gel == "young_samples"]

    # prepare WT data
    wt_lanes = young_gel[young_gel.genotype == "WT"]
    wt_young_lanes = wt_lanes[wt_lanes.age == "young"]
    wt_colors = [plots.r2c(copy_number) for copy_number in wt_young_lanes["rdnacn"]]
    itbl = zip(wt_young_lanes["rdnacn"], wt_young_lanes["normalized ERCs"], wt_colors)
    for xcoord, ycoord, color in itbl:
        ax_obj.scatter(xcoord, ycoord, color=color, marker="o")

    wt_old_lanes = wt_lanes[wt_lanes.age == "old"]
    wt_colors = [plots.r2c(copy_number) for copy_number in wt_old_lanes["rdnacn"]]
    itbl = zip(wt_old_lanes["rdnacn"], wt_old_lanes["normalized ERCs"], wt_colors)
    for xcoord, ycoord, color in itbl:
        ax_obj.scatter(xcoord, ycoord, color=color, marker="x")

    # prepare fob1∆ data
    fob1_lanes = young_gel[young_gel.genotype == "fob1∆"]
    fob1_colors = [plots.r2c_magma(copy_number) for copy_number in fob1_lanes["rdnacn"]]
    itbl = zip(fob1_lanes["rdnacn"], fob1_lanes["normalized ERCs"], fob1_colors)
    for xcoord, ycoord, color in itbl:
        ax_obj.scatter(xcoord, ycoord, color=color, marker="D")

    # format axis
    ax_obj.set_ylabel("Normalized Total ERC Levels (a.u.)")
    ax_obj.set_xlabel("rDNA CN")
    ax_obj.set_title("ERC Quantification - Young Cells")
    ax_obj.set_xlim(50, 210)
    ax_obj.set_ylim(3, 50)
    ax_obj.set_yscale("log")

    # Create  a legend with the appropriate marker shape for each strain
    # plot out of frame points for the legend
    for label, marker in zip(("WT", "WT (old)", mpl_italics("fob1∆")), ("o", "x", "D")):
        if marker != "x":
            ax_obj.scatter(
                -1, -1, marker=marker, label=label, facecolors="none", edgecolors="k"
            )
        else:
            ax_obj.scatter(
                -1, -1, marker=marker, label=label, facecolors="k", edgecolors="k"
            )
    ax_obj.legend()
    return ax_obj


@plots.subpanel
def hazard_of_erc_related_death(ax_obj):
    """Plot hazard of cells dying from different modes-of-death.

    Data collected from WT strains are in the top row.
    Data collected from fob1∆ strain in the bottom row.
    Hazard of ERC-associated death is in the left column.
    Hazard of ERC-independent death is in the right column.
    """
    fig = plt.gcf()

    mode_of_death_hazards = data_loading.get_mode_of_death_hazards()

    # Create 4 panels on which to plot different categories ...
    left, bottom, width, height = ax_obj.get_position().bounds
    gap = 50  # for spacing sub-axes
    left += width / 5
    bottom += height / gap

    ax_ul = fig.add_axes(
        (left, bottom + height / 2, height / 2 - height / gap, width / 2 - width / gap)
    )
    ax_ur = fig.add_axes(
        (
            left + width / 2,
            bottom + height / 2,
            height / 2 - height / gap,
            width / 2 - width / gap,
        )
    )
    ax_ll = fig.add_axes(
        (left, bottom, height / 2 - height / gap, width / 2 - width / gap)
    )
    ax_lr = fig.add_axes(
        (left + width / 2, bottom, height / 2 - height / gap, width / 2 - width / gap)
    )

    # Plot different data categories
    subset = mode_of_death_hazards[
        (mode_of_death_hazards.KO_gene == "WT")
        & (mode_of_death_hazards.hazard_of == "round_death")
    ]
    plots.plot_hazard(ax_ul, subset, plots.r2c)
    subset = mode_of_death_hazards[
        (mode_of_death_hazards.KO_gene == "WT")
        & (mode_of_death_hazards.hazard_of == "elongated_death")
    ]
    plots.plot_hazard(ax_ur, subset, plots.r2c)
    subset = mode_of_death_hazards[
        (mode_of_death_hazards.KO_gene == "fob1")
        & (mode_of_death_hazards.hazard_of == "round_death")
    ]
    plots.plot_hazard(ax_ll, subset, plots.r2c_magma)
    subset = mode_of_death_hazards[
        (mode_of_death_hazards.KO_gene == "fob1")
        & (mode_of_death_hazards.hazard_of == "elongated_death")
    ]
    plots.plot_hazard(ax_lr, subset, plots.r2c_magma)

    # format and annotated axes
    ax_ul.set_xticks([])
    ax_ur.set_xticks([])
    ax_ur.set_yticks([])
    ax_lr.set_yticks([])

    ax_ul.set_yticks([0.05, 0.1, 0.15])
    ax_ll.set_yticks([0.05, 0.1, 0.15])
    ax_ll.set_xticks([0, 10, 20, 30, 40])
    ax_lr.set_xticks([0, 10, 20, 30, 40])

    ax_ul.set_ylabel("Hazard")
    ax_ll.set_ylabel("Hazard")
    ax_ll.set_xlabel("# of Divisions")
    ax_lr.set_xlabel("# of Divisions")

    ax_ul.set_title("rDNA Independent \nMode-of-Death")
    ax_ur.set_title("rDNA Dependent \nMode-of-Death")

    ax_obj.text(
        x=0,
        y=0.75,
        s="WT",
        rotation=90,
        fontsize=plt.rcParams.get("axes.titlesize"),
        ha="center",
        va="center",
    )
    ax_obj.text(
        x=0,
        y=0.25,
        s=mpl_italics("fob1∆"),
        rotation=90,
        fontsize=plt.rcParams.get("axes.titlesize"),
        ha="center",
        va="center",
    )
    ax_obj.axvline(0.02, 0.05, 0.45, clip_on=False, color="k", linewidth=0.7)
    ax_obj.axvline(0.02, 0.55, 0.95, clip_on=False, color="k", linewidth=0.7)
    ax_obj.axis("off")

    return ax_obj


@plots.subpanel
def fob1d_vs_rdna_scatter_plot(ax_obj):
    """ Plot rDNA CN vs RLS for fob1∆ strains."""
    mutant_name = "fob1"
    lifespan_stats = data_loading.get_lifespan_summary()
    wt_stats = lifespan_stats[lifespan_stats.KO_gene == "-"]
    mutant_stats = lifespan_stats[lifespan_stats.KO_gene == mutant_name]
    plots.add_wt_to_plot(ax_obj, wt_stats, alpha=1)
    plots.add_fob1_to_plot(ax_obj, mutant_stats)
    plots.format_rls_by_rdna_ax(ax_obj)
    ax_obj.set_ylim(0, 55)
    ax_obj.set_xlim(50, 275)

    # Out of frame points for the legend
    for label, marker in zip(("WT", mpl_italics("fob1∆")), ("o", "D")):
        ax_obj.scatter(
            -1, -1, marker=marker, label=label, facecolors="none", edgecolors="k"
        )
    handle, label = ax_obj.get_legend_handles_labels()
    handle, label = handle[-2:], label[-2:]
    ax_obj.legend(handle, label)
    return ax_obj


@plots.subpanel
def get_young_erc_blot_rdna_probe(ax_obj):
    """Plot image of ERC southern blot of samples from young cells."""
    img = data_loading.get_young_erc_blot_rdna_probe()
    ax_obj.imshow(
        img, clim=(np.percentile(img, 5), np.percentile(img, 95)), cmap="binary"
    )
    ax_obj.set_yticks([])
    ax_obj.set_xticks([])
    return ax_obj


@plots.subpanel
def get_young_erc_blot_npr2_probe(ax_obj):
    """Plot image of NPR2 southern blot of samples from young cells."""
    img = data_loading.get_young_erc_blot_npr2_probe()
    ax_obj.imshow(
        img, clim=(np.percentile(img, 5), np.percentile(img, 95)), cmap="binary"
    )
    ax_obj.set_yticks([])
    ax_obj.set_xticks([])
    return ax_obj


@plots.subpanel
def mutant_vs_rdnacn_scatter_plot(mutant_name, ax_obj):
    """Plot rDNA CN vs RLS for mutant strain.

    WT strains in background.
    """
    lifespan_stats = data_loading.get_lifespan_summary()
    wt_stats = lifespan_stats[lifespan_stats.KO_gene == "-"]
    mutant_stats = lifespan_stats[lifespan_stats.KO_gene == mutant_name]
    plots.add_wt_to_plot(ax_obj, wt_stats, fixed_color="k")
    plots.add_mutant_to_plot(ax_obj, mutant_stats)
    plots.format_rls_by_rdna_ax(ax_obj)
    plots.add_mutant_legend(ax_obj, mutant_name)
    return ax_obj


# a bit silly, but I want each figure panel to have its own function
@plots.subpanel
def hda2d_vs_rdnacn_scatter_plot(ax_obj):
    """Plot rDNA CN vs RLS for hda2∆ strains.

    WT strains in background.
    """
    return mutant_vs_rdnacn_scatter_plot(mutant_name="hda2", ax_obj=ax_obj)


@plots.subpanel
def gpa2d_vs_rdnacn_scatter_plot(ax_obj):
    """Plot rDNA CN vs RLS for gpa2∆ strains.

    WT strains in background.
    """
    return mutant_vs_rdnacn_scatter_plot(mutant_name="gpa2", ax_obj=ax_obj)


@plots.subpanel
def idh1d_vs_rdnacn_scatter_plot(ax_obj):
    """Plot rDNA CN vs RLS for idh1∆ strains.

    WT strains in background.
    """
    return mutant_vs_rdnacn_scatter_plot(mutant_name="idh1", ax_obj=ax_obj)


@plots.subpanel
def rpl13ad_vs_rdnacn_scatter_plot(ax_obj):
    """Plot rDNA CN vs RLS for rpl13A∆ strains.

    WT strains in background.
    """
    return mutant_vs_rdnacn_scatter_plot(mutant_name="rpl13A", ax_obj=ax_obj)


@plots.subpanel
def ubp8d_vs_rdnacn_scatter_plot(ax_obj):
    """Plot rDNA CN vs RLS for ubp8∆ strains.

    WT strains in background.
    """
    return mutant_vs_rdnacn_scatter_plot(mutant_name="ubp8", ax_obj=ax_obj)


@plots.subpanel
def ubr2d_vs_rdnacn_scatter_plot(ax_obj):
    """Plot rDNA CN vs RLS for ubr2∆ strains.

    WT strains in background.
    """
    return mutant_vs_rdnacn_scatter_plot(mutant_name="ubr2", ax_obj=ax_obj)


@plots.subpanel
def tor1d_vs_rdnacn_scatter_plot(ax_obj):
    """Plot rDNA CN vs RLS for tor1∆ strains.

    WT strains in background.
    """
    return mutant_vs_rdnacn_scatter_plot(mutant_name="tor1", ax_obj=ax_obj)


@plots.subpanel
def statistic_of_deletion_effects(ax_obj):
    """Plot effects of ∆s, rDNA CN, and ∆s:rDNA CN on hazard.

    Effects are estimated from fitting a Cox Proportional Hazard model
    with these as factors in the model. Additional description of the model
    can be found in the materials and methods section.
    """
    stats = data_loading.get_mutant_lifespans_cph_stats()
    stats = stats.sort_values("coef")

    # format names to make the plotting prettier
    new_names = {
        "rdnacn_1": "rDNA CN",
        "KO_gene_gpa2": "gpa2∆",
        "rdnacn_1:KO_gene_gpa2": "rDNA CN:gpa2∆",
        "KO_gene_hda2": "hda2∆",
        "rdnacn_1:KO_gene_hda2": "rDNA CN:hda2∆",
        "KO_gene_idh1": "idh1∆",
        "rdnacn_1:KO_gene_idh1": "rDNA CN:idh1∆",
        "KO_gene_rpl13A": "rpl13A∆",
        "rdnacn_1:KO_gene_rpl13A": "rDNA CN:rpl13A∆",
        "KO_gene_tor1": "tor1∆",
        "rdnacn_1:KO_gene_tor1": "rDNA CN:tor1∆",
        "KO_gene_ubp8": "ubp8∆",
        "rdnacn_1:KO_gene_ubp8": "rDNA CN:ubp8∆",
        "KO_gene_ubr2": "ubr2∆",
        "rdnacn_1:KO_gene_ubr2": "rDNA CN:ubr2∆",
        "KO_gene_fob1": "fob1∆",
        "rdnacn_1:KO_gene_fob1": "rDNA CN:fob1∆",
    }
    stats = stats.assign(
        covariate_formatted=list(pd.Series(stats.index).replace(new_names))
    )
    # get table of just the deletions on hazard
    mutant_effect = (
        stats[
            ~stats.covariate_formatted.str.contains("rDNA CN")
            & ~stats.covariate_formatted.str.contains("experiment_")
            & ~stats.covariate_formatted.str.contains("rdnacn_2")
        ]
        .sort_values("coef")
        .set_index("covariate_formatted")
    )
    # get table of deletions interacting with rDNA CN
    mutant_interaction = (
        stats[stats.covariate_formatted.str.contains("rDNA CN") & stats.coef > 0]
        .sort_values("coef", ascending=True)
        .set_index("covariate_formatted")
    )
    # create sub axes for each type of data table
    fig = plt.gcf()
    ax_obj.axis("off")
    left, bottom, width, height = ax_obj.get_position().bounds
    gap = 0.03
    width1 = width * len(mutant_effect) / (len(mutant_effect) + len(mutant_interaction))
    width2 = (
        width * len(mutant_interaction) / (len(mutant_effect) + len(mutant_interaction))
    )
    ax1 = fig.add_axes((left, bottom, width1 - gap / 2, height * 0.9))
    ax2 = fig.add_axes(
        (left + width1 + gap / 2, bottom, width2 - gap / 2, height * 0.9)
    )

    tables = [mutant_effect, mutant_interaction]
    axes = [ax1, ax2]
    titles = ["KO effect on Hazard at 150 rDNA copies", "rDNA CN Interactions"]
    # Plot data with CIs
    for ax_, stats, title in zip(axes, tables, titles):
        ax_.set_xlim(0, len(stats)+1)
        ax_.set_xticks(range(1, len(stats) + 1))
        formatted_labels = [plots.format_gene_for_mpl_italics(i) for i in stats.index]
        ax_.set_xticklabels(formatted_labels, rotation=70, ha="right")
        ax_.set_title(title)
        max_value = max(stats["coef upper 95%"].max(), 0.02)
        min_value = min(stats["coef lower 95%"].min(), -0.02)
        if ax_ is ax1:
            ax_.set_ylim(max_value, min_value)
        else:
            # ax_.set_ylim(min_value, max_value)
            ax_.set_ylim(max_value, min_value)

        for xcoord, (_, row) in enumerate(stats.iterrows()):
            if row.name == "rDNA CN":
                color = "k"
            elif "rDNA CN" in row.name:
                color = "blue"
            else:
                color = "red"
            if row["coef lower 95%"] < 0 < row["coef upper 95%"]:
                fillstyle = "none"
                markeredgecolor = color
            else:
                fillstyle = "full"
                markeredgecolor = "none"

            ax_.plot(
                xcoord + 1,
                row.coef,
                marker="o",
                fillstyle=fillstyle,
                color=color,
                markeredgecolor=markeredgecolor,
                alpha=0.7,
            )
            ax_.vlines(
                x=xcoord + 1,
                ymin=row["coef lower 95%"],
                ymax=row["coef upper 95%"],
                color=color,
                alpha=0.7,
            )
        if ax_ is ax1:
            ax_.set_ylabel("log(HR) (95% CI)")
        else:
            ax_.yaxis.tick_right()
        ax_.axhline(0, linestyle="--", alpha=0.3, color="k")
    return ax_obj


def _lifespan_validation(ax_obj, metric="frame_stop", ylim=360):
    """ Method for plotting a confusion matrix of model performance."""
    fig = plt.gcf()
    ax_obj.axis("off")

    left, bottom, width, height = ax_obj.get_position().bounds

    fig = plt.gcf()

    gap = 4.5

    left += width / 5
    bottom += height / gap

    ax_ul = fig.add_axes(
        (left, bottom + height / 2, width / 2 - width / gap, height / 2 - height / gap),
        facecolor="#4aa96c66",
    )
    ax_ur = fig.add_axes(
        (
            left + width / 2,
            bottom + height / 2,
            width / 2 - width / gap,
            height / 2 - height / gap,
        ),
        facecolor="#f55c4766",
    )
    ax_ll = fig.add_axes(
        (left, bottom, width / 2 - width / gap, height / 2 - height / gap),
        facecolor="#f55c4766",
    )
    ax_lr = fig.add_axes(
        (left + width / 2, bottom, width / 2 - width / gap, height / 2 - height / gap),
        facecolor="#4aa96c66",
    )

    table = data_loading.get_annotation_comparison()
    table = data_loading.add_standard_rdnacn_to_table(table)
    table = data_loading.add_simple_genotype(table)
    table = table.assign(color=table.apply(plots.parse_color, axis=1))

    plots.cm_scatter(
        ax_ul, table[table.is_death_annotated & table.is_death_predicted], metric
    )
    plots.cm_scatter(
        ax_ll, table[table.is_death_annotated & ~table.is_death_predicted], metric
    )
    plots.cm_scatter(
        ax_ur, table[~table.is_death_annotated & table.is_death_predicted], metric
    )
    plots.cm_scatter(
        ax_lr, table[~table.is_death_annotated & ~table.is_death_predicted], metric
    )

    for ax_ in [ax_ul, ax_ur, ax_ll, ax_lr]:
        ax_.set_ylim(0, ylim)
        ax_.set_xlim(0, ylim)
    ax_ul.set_title("Annotated: D | Predicted: D", fontsize="small")
    ax_ll.set_title("Annotated: D | Predicted: C", fontsize="small")
    ax_ur.set_title("Annotated: C | Predicted: D", fontsize="small")
    ax_lr.set_title("Annotated: C | Predicted: C", fontsize="small")
    return ax_obj


@plots.subpanel
def lifespan_prediction_validation_tls(ax_obj):
    """Plot accuracy of event detection model on human annotated data.

    The quadrant indicates if the model and human annotated/predicted the same
    type of event.

    The scatter plots in each quadrant indicate the frame at which the event was
    annotated/predicted.

    Colors indicate the rDNA CN and genotype of that cell and are consistent with
    the figures in the rest of the manuscript.
    """
    return _lifespan_validation(ax_obj, metric="frame_stop", ylim=396)


@plots.subpanel
def lifespan_prediction_validation_bud_cnt(ax_obj):
    """Plot accuracy of event detection model on human annotated data, combined
    with the performance of the bud counting model.

    The quadrant indicates if the model and human annotated/predicted the same
    type of event.

    The scatter plots in each quadrant indicate how many budding events (age)
    were annotated/predicted.

    Colors indicate the rDNA CN and genotype of that cell and are consistent with
    the figures in the rest of the manuscript.
    """
    return _lifespan_validation(ax_obj, metric="bud_cnt", ylim=60)


@plots.subpanel
def mod_validation_grouped_scatter(ax_obj):
    """Plot accuracy of mode-of-death model on human annotated data.

    The categorical human annotation is on the x-axis.
    The probability of an rDNA-dependent MoD is on the y-axis.

    Colors indicate the rDNA CN and genotype of that cell and are consistent with
    the figures in the rest of the manuscript.
    """
    table = data_loading.get_mod_annotation_comparison()
    table = data_loading.add_standard_rdnacn_to_table(table)
    table = data_loading.add_simple_genotype(table)
    table = table.assign(rdnacn_color=table.apply(plots.parse_color, axis=1))

    table["annotated_event_type"] = table["annotated_event_type"].map(
        {1: "Round Death", 2: "Elongated Death", 3: "Other", 4: "Other"}
    )
    plots.prism_style_column_scatter(
        table,
        groupby=("annotated_event_type"),
        y="mode_of_death_elongated_prob",
        color="rdnacn_color",
        ax_obj=ax_obj,
        alpha=0.3,
        edgecolor="none",
    )
    xlim = ax_obj.get_xlim()
    ax_obj.fill_between(
        x=range(0, 10),
        y1=0.2,
        y2=0.8,
        color="#bbbfca66",
    )
    ax_obj.set_xlim(xlim)
    ax_obj.set_xlabel("Mode-of-Death Annotation")
    ax_obj.set_ylabel("Mode-of-Death Prediction Score")
    ax_obj.annotate(
        "Predicted rDNA\nDependent MoD",
        xy=(xlim[1] - 0.05, 0.9),
        arrowprops=dict(
            facecolor="black",
            alpha=0.7,
            arrowstyle="-[, widthB=1.8",
            relpos=(0, 0.5),
        ),
        xytext=(xlim[1] + 0.2, 0.9),  # Manually chosen
        fontsize="medium",
        va="center",
        ha="left",
    )
    ax_obj.annotate(
        "Predicted rDNA\nIndependent MoD",
        xy=(xlim[1] - 0.05, 0.1),
        arrowprops=dict(
            facecolor="black",
            alpha=0.7,
            arrowstyle="-[, widthB=1.8",
            relpos=(0, 0.5),
        ),
        xytext=(xlim[1] + 0.2, 0.1),  # Manually chosen
        fontsize="medium",
        va="center",
        ha="left",
    )
    return ax_obj


@plots.subpanel
def mod_validation_confusion_matrix(ax_obj):
    """Create a table representing the accuracy of mode-of-death model on human
    annotated data.
    """
    table = data_loading.get_mod_annotation_comparison()
    table = table.rename(
        columns={
            "annotated_event_type": "Annotated:",
            "mode_of_death_prediction": "Predicted:",
        }
    )
    table["Annotated:"] = table["Annotated:"].map(
        {
            1: "Round",
            2: "Elongated",
            3: "Other",
            4: "Other",
        }
    )
    table["Predicted:"] = table["Predicted:"].map(
        {
            "r": "Round",
            "e": "Elongated",
            "u": "Other",
            None: "Other",
        }
    )

    table = (
        pd.DataFrame(
            table.groupby(["Predicted:", "Annotated:"]).size().reset_index(name="Count")
        )
        .pivot(index="Predicted:", columns="Annotated:", values="Count")
        .fillna(0)
        .astype(int)
    )
    mpl_table = ax_obj.table(
        cellText=table.values,
        colLabels=table.columns,
        rowLabels=table.index,
        fontsize="8",
        cellLoc="center",
        bbox=[0.25, 0.17, 0.6, 0.7],
        colColours=["#bbbfca66" for i in range(len(table.columns))],
        rowColours=["#bbbfca66" for i in range(len(table))],
    )
    ax_obj.axis("off")
    ax_obj.set_ylim(0, 1)
    ax_obj.set_xlim(0, 1)
    ax_obj.text(
        0.5,
        0.95,
        "Annotated MoD:",
        fontsize=plt.rcParams.get("axes.titlesize"),
        ha="center",
        va="center",
    )
    mpl_table.axes.text(
        0.05,
        0.45,
        "Predicted MoD:",
        fontsize=plt.rcParams.get("axes.titlesize"),
        ha="center",
        va="center",
        rotation=90,
    )
    return ax_obj


@plots.subpanel
def lifespan_prediction_validation_confusion_matrix(ax_obj):
    """Create a table representing the accuracy of death/washout event detection
    model on human annotated data.
    """
    table = data_loading.get_annotation_comparison()
    columns = ["is_death_predicted", "is_death_annotated"]
    table = table[columns]
    table = table.replace(True, "Death").replace(False, "Censored")
    table = (
        table.groupby(columns)
        .size()
        .reset_index(name="count")
        .pivot(index="is_death_predicted", columns="is_death_annotated", values="count")
    )
    table = table[["Death", "Censored"]].loc[["Death", "Censored"]]

    mpl_table = ax_obj.table(
        cellText=table.values,
        colLabels=table.columns,
        rowLabels=table.index,
        cellLoc="center",
        bbox=[0.15, 0.2, 0.9, 0.7],
        colColours=["#bbbfca66" for i in range(len(table.columns))],
        rowColours=["#bbbfca66" for i in range(len(table))],
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(6)
    ax_obj.axis("off")
    mpl_table.axes.text(
        0.63,
        0.95,
        "Annotated:",
        fontsize=plt.rcParams.get("axes.titlesize"),
        ha="center",
        va="center",
        transform=ax_obj.transAxes,
    )
    mpl_table.axes.text(
        -0.2,
        0.44,
        "Predicted:",
        fontsize=plt.rcParams.get("axes.titlesize"),
        ha="left",
        va="center",
        rotation=90,
        transform=ax_obj.transAxes,
    )
    ax_obj.set_title("Death/Censored Model Confusion Matrix")
    return ax_obj


@plots.subpanel
def effect_on_genotype_on_event_model_performance_table(ax_obj):
    """Create a table representing the calculated effects of rDNA CN and fob1∆
    status on the accuracy of the event detection model.
    """
    table = data_loading.get_statistics_on_event_model_performance()

    def format_row_names(i):
        return {"rdnacn": "rDNA CN", "KO_gene_fob1": "fob1∆"}.get(i, i)

    table.index = table.index.set_names(["index"])
    table = table.assign(
        new_index=list(table.reset_index()["index"].apply(format_row_names))
    ).set_index("new_index")
    table[table.columns] = table[table.columns].applymap(lambda x: f"{x:.03f}")
    mpl_table = ax_obj.table(
        cellText=table.values,
        colLabels=table.columns,
        rowLabels=[plots.format_gene_for_mpl_italics(i) for i in table.index],
        cellLoc="center",
        bbox=[0.15, 0.1, 0.9, 0.8],
        colColours=["#bbbfca66" for i in range(len(table.columns))],
        rowColours=["#bbbfca66" for i in range(len(table))],
    )
    ax_obj.axis("off")
    mpl_table.axes.text(
        0.5,
        0.95,
        "Effects on Event Model Accuracy:",
        fontsize=plt.rcParams.get("axes.titlesize"),
        ha="center",
        va="center",
        transform=ax_obj.transAxes,
    )
    return ax_obj


@plots.subpanel
def effect_on_genotype_on_mod_model_performance_table(ax_obj):
    """Create a table representing the calculated effects of rDNA CN, fob1∆, or
    and ∆ status on the accuracy of the mode-of-death model.
    """
    table = data_loading.get_statistics_on_mod_model_performance()

    def format_row_names(i):
        return {"rdnacn": "rDNA CN", "KO": "∆ (any gene)"}.get(i.strip(), i)

    table.index = table.index.set_names(["index"])
    table = table.assign(
        new_index=list(table.reset_index()["index"].apply(format_row_names))
    ).set_index("new_index")
    table[table.columns] = table[table.columns].applymap(lambda x: f"{x:.03f}")
    mpl_table = ax_obj.table(
        cellText=table.values,
        colLabels=table.columns,
        rowLabels=table.index,
        cellLoc="center",
        bbox=[0.05, 0.1, 0.9, 0.75],
        colColours=["#bbbfca66" for i in range(len(table.columns))],
        rowColours=["#bbbfca66" for i in range(len(table))],
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(6)
    ax_obj.axis("off")
    mpl_table.axes.text(
        0.5,
        0.95,
        "Effects on MoD Model Accuracy:",
        fontsize=plt.rcParams.get("axes.titlesize"),
        ha="center",
        va="center",
        transform=ax_obj.transAxes,
    )
    return ax_obj


@plots.subpanel
def prediction_vs_annotation_km_curves(ax_obj):
    """Plot the survival functions generated from automated predictions or
    human annotations. (Subset of data that has been annoated.)
    """
    fits = data_loading.get_surv_fits_of_annotated_data()
    for label, sdf in fits.groupby("label"):
        plots.plt_km_curve(sdf, ax_obj, label=label)
    ax_obj.legend()
    ax_obj.set_xlabel("# of Divisions")
    ax_obj.set_ylabel("KM - Estimate")
    ax_obj.set_title("Annotations vs Predictions")
    ax_obj.set_xlim(0, 60)
    return ax_obj


@plots.subpanel
def mod_example_images(ax_obj):
    """ Show annotated examples of different mode-of-death morphologies."""
    fig = plt.gcf()
    ax_obj.axis("off")
    left, bottom, width, height = ax_obj.get_position().bounds
    gap = 0.07
    ax1 = fig.add_axes((left, bottom, width / 2 - gap / 2, height * 0.9))
    ax2 = fig.add_axes(
        (left + width / 2 + gap / 2, bottom, width / 2 - gap / 2, height * 0.9)
    )

    img = data_loading.get_elongated_death_example()[28:-28, 28:-28]
    ax1.imshow(img, cmap="gray", clim=(0, np.percentile(img, 99.5)))
    img = data_loading.get_round_death_example()[28:-28, 28:-28]
    ax2.imshow(img, cmap="gray", clim=(0, np.percentile(img, 99.5)))
    ax1.axis("off")
    ax2.axis("off")
    ax1.set_title("Elongated / \nrDNA Dependent MoD")
    ax2.set_title("Round / \nrDNA Independent MoD")
    return ax_obj


@plots.subpanel
def mod_model_arch(ax_obj):
    """Create a schematic of the mode-of-death model architecture.

    Each box represents a layer in the model.
    """
    block_texts = [
        "Input: NxLx1x72x72",
        "Reshape: (NxL)x1x72x72",
        "Convolution Blocks \n (ResNet 50)",
        "Fully Connected",
        "3 X Dilated ResBlocks",
        "Convolution (1x1 / 1)",
        "Convolution (3x1 / 1)",
        "MaxPool /2",
        "Convolution (3x1 / 1)",
        "Convolution (3x1 / 1)",
        "MaxPool /2",
        "Convolution (3x1 / 1)",
        "Convolution (3x1 / 1)",
        "MaxPool /2",
        "Mean (dim2)",
        "Fully Connected",
        "Softmax",
    ]
    block_colors = [
        [234, 234, 234],
        [190, 208, 246],
        [168, 208, 151],
        [240, 240, 240],
        [250, 224, 194],
        [168, 208, 151],
        [168, 208, 151],
        [202, 147, 175],
        [168, 208, 151],
        [168, 208, 151],
        [202, 147, 175],
        [168, 208, 151],
        [168, 208, 151],
        [202, 147, 175],
        [240, 240, 240],
        [240, 240, 240],
        [255, 239, 193],
    ]
    assert len(block_texts) == len(block_colors)
    block_colors = np.asarray(block_colors)
    block_colors = block_colors / 255
    bbox_args = dict(boxstyle="round", fc="0.8")
    arrow_args = dict(arrowstyle="<-", shrinkB=8)

    for i, (text, color) in enumerate(zip(block_texts, block_colors)):
        color = mp_colors.to_hex(color)
        bbox_args.update(fc=mp_colors.to_hex(color) + "66")
        if i == 0:
            arrow_args.update(visible=False)
        else:
            arrow_args.update(visible=True)
        ax_obj.annotate(
            text,
            xytext=(0.5, (i + 1)),
            xy=(0.5, i),
            ha="center",
            va="center",
            fontsize="medium",
            bbox=bbox_args,
            arrowprops=arrow_args,
            clip_on=False,
        )
    ax_obj.set_ylim(len(block_colors), 0)
    ax_obj.axis("off")
    return ax_obj


@plots.subpanel
def micromanipulation_comparison(ax_obj):
    micro_fits = data_loading.get_surv_fits_by_micromanipulation()
    ylm_fits = data_loading.get_ylm_surv_fits_for_strains_tested_by_micromanipulation()

    cn = data_loading.get_rdnacn_measurements()
    cn = cn[cn.media == "YPD"]
    assay = "microdissection"
    for cidx, (strain, sdf) in enumerate(micro_fits.groupby("strain")):
        label = f"{assay}| rDNA CN: {int(round(cn.loc[strain].rdnacn))} ({strain})"
        plots.plt_km_curve(sdf, ax_obj, label=label, linestyle="-.", color=f"C{cidx}")
    assay = "microfluidics"
    for cidx, (strain, sdf) in enumerate(ylm_fits.groupby("strain")):
        label = f"{assay}     | rDNA CN: {int(round(cn.loc[strain].rdnacn))} ({strain})"
        plots.plt_km_curve(
            sdf,
            ax_obj,
            label=label,
            linestyle="-",
            color=f"C{cidx}",
        )
    ax_obj.legend()
    ax_obj.set_xlabel("# of Divisions")
    ax_obj.set_ylabel("KM - Estimate")
    ax_obj.set_title("Microdissection vs Microfluidics")
    ax_obj.set_xlim(0, 60)
    return ax_obj
