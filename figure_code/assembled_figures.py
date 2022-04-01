""" Used to assemble completed figures for manuscript.

Calls methods in figure_panels.py to add plots/images to multi-panel plots that
are specified here.
"""

import matplotlib.pyplot as plt

from . import figure_panels as panels
from . import plots

VERBOSE = False


def figure_1():
    """rDNA copy number explains a majority of the variability in
    Replicative Lifespan (RLS) estimates.

     A)Kaplan-Meier estimates for 13 wild-type strains derived from the same
     parent show significant variability (> 400 observations per curve, median
     of ~1000 observations per curve, 95% confidence interval
     represented by shaded region).

     B)Copy number variations in the rDNA  locus correlate with RLS.
     Polymorphisms identified by whole-genome sequencing data are grouped into
     Single Nucleotide Polymorphisms (SNPs) and Copy Number Variations (CNVs) of
     2kb genome bins. 41 polymorphisms occurring in at least 2 strains are
     shown. Strains are ordered by increasing median RLS from left to right.
     Color map for SNPs is grey (absent; REF) and black (present; VAR). Color
     map for CNVs is from low (dark blue) to high (yellow).

     C)rDNA copy number (rDNA CN) correlates with median RLS. Color map
     indicates rDNA CN and is maintained throughout the rest of the figures.
     Each point represents a unique lifespan experiment (microfluidic channel)
     with 400 - 600 cells.  95% confidence intervals around median RLS estimate
     are represented as vertical bars.

     D)Kaplan-Meier estimates for an extended panel of 54 wild-type strains with
     variable rDNA CN (> 400 observations per curve, median
     ~1150 observations per curve).

     E)Correlation of median RLS and rDNA CN for the same strains as in D (n =
     319 - 653 per microfluidic channel, median 593 cells).

    """
    w_inches = 7.1  # 2 columns
    h_inches = 7.1
    fig = plt.figure(figsize=(w_inches, h_inches), facecolor="w", dpi=300)

    # Placement L ,  B,   W,   H
    a_rect = [0.1, 0.575, 0.2625, 0.35]
    b_rect = [0.425, 0.575, 0.2, 0.35]
    c_rect = [0.75, 0.575, 0.2, 0.35]
    d_rect = [0.1, 0.1, 0.35, 0.35]
    e_rect = [0.6, 0.1, 0.35, 0.35]

    a_ax = fig.add_axes(a_rect)
    b_ax = fig.add_axes(b_rect)
    c_ax = fig.add_axes(c_rect)
    d_ax = fig.add_axes(d_rect)
    e_ax = fig.add_axes(e_rect)

    panels.kms_with_unexpected_wt_variability(a_ax)
    panels.polymorphism_analysis_snps_and_rdna(b_ax)
    panels.rdcn_vs_rls_of_unexpected_wt_variability(c_ax, verbose=VERBOSE)
    panels.kms_of_rdnacn_panel(d_ax, verbose=VERBOSE)
    panels.rdnacn_vs_rls_scatter_plot(e_ax)

    plots.label_panel(a_ax, "A", label_position=(-0.15, 1.05))
    plots.label_panel(b_ax, "B", label_position=(-0.15, 1.05))
    plots.label_panel(c_ax, "C", label_position=(-0.15, 1.05))
    plots.label_panel(d_ax, "D", label_position=(-0.15, 1.05))
    plots.label_panel(e_ax, "E", label_position=(-0.12, 1.05))

    ## Add some additional annotations for each row

    annotation_ax = fig.add_axes([0, 0, 1, 1], facecolor=(1, 1, 1, 0))
    annotation_ax.hlines(
        0.5,
        0,
        1,
        color="k",
        clip_on=False,
        linewidth=annotation_ax.spines["left"].get_linewidth(),
    )
    annotation_ax.set_ylim(0, 1)
    annotation_ax.set_xlim(0, 1)

    annotation_ax.text(
        0.02,
        0.75,
        "Spontaneous Variation",
        rotation=90,
        verticalalignment="center",
        horizontalalignment="center",
        fontsize="x-large",
    )

    annotation_ax.text(
        0.02,
        0.25,
        "Experimental Variation",
        rotation=90,
        verticalalignment="center",
        horizontalalignment="center",
        fontsize="x-large",
    )
    annotation_ax.axis("off")
    return fig


def sup_fig_1():
    """Whole genome-sequencing (WGS) accurately measures rDNA CN.

    A) Contour-clamped homogeneous electric field electrophoresis (CHEF) and
    Southern blot against rDNA on 11 wild-type strains. CHEF settings were
    optimized for a range of 0.44 - 2.3 Mb. Red arrows identify the
    chromosomal rDNA array beyond this range.

    B) Correlation between CHEF-derived and WGS-derived CN estimates.
    """
    w_inches = 4.5  # 1 column
    h_inches = 2.25
    fig = plt.figure(figsize=(w_inches, h_inches), facecolor="w", dpi=300)
    rect = 0.1, 0.1, 0.4, 0.7
    ax1 = fig.add_axes(rect)
    rect = 0.68, 0.2, 0.3, 0.6
    ax2 = fig.add_axes(rect)
    panels.wgs_vs_chef_for_rdnacn_gel(ax1)
    panels.wgs_vs_chef_for_rdnacn_analysis(ax2)
    plots.label_panel(ax1, "A")
    plots.label_panel(ax2, "B")
    return fig


def sup_fig_2():
    """rDNA affects RLS in multiple media types.

    Correlation of median RLS and rDNA CN on wild-type strains in indicated
    media types (SC: Synthetic Complete; YNB: Minimal Medium; YPD: Rich Medium).
    rDNA CN was measured by WGS on the same culture that was used for the
    lifespan experiment (n >539 per lifespan estimate). RLS was higher in YNB
    and lower in YPD compared to SC.

    """
    w_inches = 4.5
    h_inches = 4.5
    fig = plt.figure(figsize=(w_inches, h_inches), facecolor="w", dpi=300)
    rect = 0.17, 0.17, 0.75, 0.75
    ax1 = fig.add_axes(rect)
    panels.rdnacn_vs_rls_scatter_plot_in_different_media(ax1)
    return fig


def sup_fig_3():
    """rDNA CN affects RLS in a commonly used strain.

    Correlation of median RLS and rDNA CN on strains derived from BY4743 (black
    circles; n >570 per lifespan). For comparison, same data as in Fig.1E is
    shown (colored circles).

    """

    w_inches = 4.5
    h_inches = 4.5
    fig = plt.figure(figsize=(w_inches, h_inches), facecolor="w", dpi=300)
    rect = 0.17, 0.17, 0.75, 0.75
    ax1 = fig.add_axes(rect)
    panels.rdnacn_vs_rls_scatter_plot_in_by4743(ax1)
    return fig


def figure_2():
    """Chromosomal rDNA CN modulates lifespan via ERC levels.

    A) Southern blots probed for rDNA and NPR2 (loading control).
    Samples were obtained from wild-type and fob1∆ strains
    with variable rDNA CN aged for 24 hrs. Arrows 1-4 highlight bands of
    Extrachromosomal rDNA Circles (ERCs), arrow A highlights the chromosomal
    rDNA array.

    B) Quantification of ERC levels on blot from A shows anti-correlation
    between rDNA CN and ERC levels. ERC accumulation is suppressed by deletion
    of FOB1.

    C) Hazard estimates for 51 wild-types and 20 fob1∆ strains
    derived from microfluidics experiments show increased hazard risk of rDNA
    dependent mode of death (MoD) with low rDNA CN. This effect is suppressed
    by fob1∆. Color maps, indicating rDNA CN, are the same as
    in B. (>400 cells per hazard estimate, shaded regions represent 95%
    confidence interval).

    D) fob1∆ eliminates the correlation between rDNA CN and
    RLS ( >400 cells per median lifespan estimate, vertical bars represent
    95% confidence interval)}
    """
    w_inches = 7.1  # 2 columns
    h_inches = 5.5
    fig = plt.figure(figsize=(w_inches, h_inches), facecolor="w", dpi=300)

    # Placement L ,  B,   W,   H
    a_rect = 0.05, 0.6, 0.5, 0.3
    b_rect = 0.67, 0.57, 0.28, 0.33
    c_rect = 0.05, 0.056, 0.36, 0.36
    d_rect = 0.5 + 0.05 * 2, 0.056, 0.36, 0.36

    a_ax = fig.add_axes(a_rect)
    b_ax = fig.add_axes(b_rect)
    c_ax = fig.add_axes(c_rect)
    d_ax = fig.add_axes(d_rect)

    panels.old_cell_erc_blot(a_ax)
    panels.quantification_of_the_erc_blot_scatter(b_ax)
    panels.hazard_of_erc_related_death(c_ax)
    panels.fob1d_vs_rdna_scatter_plot(d_ax)

    plots.label_panel(a_ax, "A")
    plots.label_panel(b_ax, "B")
    plots.label_panel(c_ax, "C")
    plots.label_panel(d_ax, "D")

    return fig


def sup_fig_5():
    """ERC levels in young strains with variable rDNA CN.

    A) Southern blots probed for rDNA and NPR2 (loading control). Samples were
    obtained from log phase wild-type and fob1∆ cells with variable rDNA CN.
    Right-most lane is a control sample from wild-type cells aged for 24 hr
    (same as Fig. 3A, lane 1, "O"). Arrows 1-4 highlight bands of
    Extrachromosomal rDNA Circles (ERCs), arrow A highlights the chromosomal
    rDNA array.

    B) Quantification of ERC levels on blot from A.

    """
    w_inches = 7.1  # 2 columns
    h_inches = 2.25
    fig = plt.figure(figsize=(w_inches, h_inches), facecolor="w", dpi=300)
    rect = 0.1, 0.05, 0.3, 0.78
    ax1 = fig.add_axes(rect)
    rect = 0.55, 0.15, 0.4, 0.73
    ax2 = fig.add_axes(rect)

    panels.young_cell_erc_blot(ax1)
    panels.quantification_of_young_erc_blot_scatter(ax2)

    plots.label_panel(ax1, "A", label_position=(-0.05, 1.15))
    plots.label_panel(ax2, "B", label_position=(-0.05, 1.10))

    return fig


def sup_fig_6():
    """Validation of machine-learning based Mode of death (MoD) classification.

    A) Example images of cells experiencing “Elongated” or “Round” MoD,
    defined by their (near) terminal bud morphology.

    B) Model architecture for MoD classifier.

    C) Confusion matrix summarizing the performance of the MoD classifier. The
    resulting classes obtained from manual/human annotations ("Annotated MoD")
    vs automated/machine predictions ("Predicted MoD") are shown for a subset
    of lifespan movies (total n = 403).

    D) Plot showing raw prediction values for cells that were manually
    categorized as “Elongated” or “Round” mode-of-death. Prediction values
    above 0.8 were categorized as “Elongated”/”rDNA Dependent”. Mean (solid
    line), median (dashed line), quartiles (dotted line) are shown. Points are
    colored using the same color map to indicate rDNA CN consistent with the
    other figures (e.g. Fig. 2B).

    E) Table summarizing the effect of rDNA CN and mutations on MoD model
    performance. rDNA CN, nor gene deletions (∆) had a detectable
    effect on model performance. Effects on model accuracy were estimated by
    fitting logit ordinal regression with rDNA CN and gene deletions as
    factors in the model. Model definitions are available in provided code
    (see Data Availability section).

    """
    w_inches = 7.1  # 2 columns
    h_inches = 7.25

    fig = plt.figure(figsize=(w_inches, h_inches), facecolor="w", dpi=300)

    rect = 0.05, 0.7, 0.3, 0.3
    ax1 = fig.add_axes(rect)
    rect = 0.0, 0.05, 0.25, 0.7
    ax2 = fig.add_axes(rect)
    rect = 0.5, 0.75, 0.5, 0.2
    ax3 = fig.add_axes(rect)
    rect = 0.4, 0.445, 0.4, 0.275
    ax4 = fig.add_axes(rect)
    rect = 0.37, 0.05, 0.5, 0.2
    ax5 = fig.add_axes(rect)

    panels.mod_example_images(ax1)
    panels.mod_model_arch(ax2)
    panels.mod_validation_confusion_matrix(ax3)
    panels.mod_validation_grouped_scatter(ax4)
    panels.effect_on_genotype_on_mod_model_performance_table(ax5)

    plots.label_panel(ax1, "A", label_position=(-0.1, 0.86))
    plots.label_panel(ax2, "B", label_position=(0.1, 0.98))
    plots.label_panel(ax3, "C", label_position=(-0.05, 1.15))
    plots.label_panel(ax4, "D", label_position=(-0.05, 1.15))
    plots.label_panel(ax5, "E", label_position=(-0.1, 1.05))

    return fig


def figure_3():
    """SIR2 accessibility and expression correlate with rDNA CN and Replicative
    Lifespan (RLS).

      A) Chromatin accessibility at the locus as derived by ATAC-Seq. The UAF
      binding site upstream of SIR2 shows increased accessibility with
      decreasing rDNA CN. Color map is the same as in Fig 1.

      B) Magnification of chromatin accessibility changes at the UAF binding
      site from A.

      C) SIR2 gene expression is strongly upregulated with increasing RLS. Gene
      expression as a function of RLS is plotted (FC/RLS) vs the adjusted
      p-value (-log10(padj)).

    """

    w_inches = 5  # 2 columns
    h_inches = 5
    fig = plt.figure(figsize=(w_inches, h_inches), facecolor="w", dpi=300)

    a_rect = 0.09, 0.475 + 0.05 * 2, 0.9, 0.375
    b_rect = 0.09, 0.1, 0.4, 0.35
    c_rect = 0.59, 0.1, 0.4, 0.35

    a_ax = fig.add_axes(a_rect)
    b_ax = fig.add_axes(b_rect)
    c_ax = fig.add_axes(c_rect)

    panels.atacseq_at_sir2_locus(a_ax)
    panels.atacseq_at_uaf_binding_site(b_ax)
    panels.rnaseq_vs_rls_volcano(c_ax)

    plots.label_panel(a_ax, "A")
    plots.label_panel(b_ax, "B", label_position=(-0.15, 1.05))
    plots.label_panel(c_ax, "C", label_position=(-0.15, 1.05))

    return fig


def sup_fig_7():
    """rDNA CN and growth rate are weakly correlated.

    Growth rates of 14 wild-type strains with variable rDNA CN were measured
    from batch cultures in SC medium. Doubling times were calculated from an
    exponential fit. Y-axis were set to encompass approximately 95% of the
    growth rate difference observed in studies of the yeast deletion collection.
    While there is a weak positive correlation between rDNA CN and growth rate,
    this is unlikely to explain the observed differences in RLS, given the small
    magnitude of the changes in growth rate.
    """

    w_inches = 4.5  # 1 column
    h_inches = 4.5
    fig = plt.figure(figsize=(w_inches, h_inches), facecolor="w", dpi=300)
    rect = 0.15, 0.15, 0.75, 0.75
    ax1 = fig.add_axes(rect)
    panels.growth_rate_vs_rdnacn(ax1)
    return fig


def sup_fig_8():
    """rDNA CN and cell size are not correlated.

    Median cell sizes of 12 wild-type strains with variable rDNA CN were
    measured in triplicate from batch cultures in SC medium using a Coulter
    Counter. Y-axis limits were set to encompass approximately 95% of the cells
    size difference observed in studies of the yeast deletion collection.

    """
    w_inches = 4.5  # 1 column
    h_inches = 4.5
    fig = plt.figure(figsize=(w_inches, h_inches), facecolor="w", dpi=300)
    rect = 0.15, 0.15, 0.75, 0.75
    ax1 = fig.add_axes(rect)
    panels.cell_size_vs_rdnacn(ax1)
    return fig


def sup_fig_9():
    """Total rRNA expression is constant across variable rDNA CNs.

    RNA was prepared for RNA sequencing without polyA enrichment or
    ribo-depletion. The fraction of reads mapping to various rRNAs are plotted
    against rDNA CN.

    """
    w_inches = 4.5  # 1 column
    h_inches = 4.5
    fig = plt.figure(figsize=(w_inches, h_inches), facecolor="w", dpi=300)
    rect = 0.15, 0.15, 0.75, 0.75
    ax1 = fig.add_axes(rect)
    panels.rdna_expression(ax1)
    return fig


def figure_4():
    """rDNA CN affects the lifespan of known modulators of aging.

    A-G) Correlation of median RLS and rDNA CN for wild-type and mutant strains
    as indicated (>300 cells per lifespan estimate; median = 593). Vertical
    gray line indicates rDNA CN of 150.

    H) Cox Proportional Hazard model testing the effect of mutations on the
    hazard at 150 rDNA CN (left panel) and on the correlation between rDNA CN
    and RLS (right panel). Left: negative coefficients indicate a reduction in
    hazard due to the mutation at an rDNA CN of 150. Vertical bars represent
    95% confidence interval, empty circles indicate that the estimated range
    overlaps with 0. Right: More positive interaction terms indicate a stronger
    correlation between rDNA CN and RLS in indicated mutant background.
    Coefficients are measured per mutation, or per unit change for rDNA CN.
    """
    w_inches = 7.1  # 2 columns
    h_inches = 7.25
    fig = plt.figure(figsize=(w_inches, h_inches), facecolor="w", dpi=300)
    grid = fig.add_gridspec(nrows=3, ncols=3, wspace=0.35, hspace=0.35)
    a_ax = fig.add_subplot(grid[0, 0])
    b_ax = fig.add_subplot(grid[0, 1])
    c_ax = fig.add_subplot(grid[0, 2])

    d_ax = fig.add_subplot(grid[1, 0])
    e_ax = fig.add_subplot(grid[1, 1])
    f_ax = fig.add_subplot(grid[1, 2])

    g_ax = fig.add_subplot(grid[2, 0])
    h_ax = fig.add_subplot(grid[2, 1:])

    panels.hda2d_vs_rdnacn_scatter_plot(a_ax)
    panels.ubp8d_vs_rdnacn_scatter_plot(b_ax)
    panels.ubr2d_vs_rdnacn_scatter_plot(c_ax)
    panels.rpl13ad_vs_rdnacn_scatter_plot(d_ax)
    panels.gpa2d_vs_rdnacn_scatter_plot(e_ax)
    panels.idh1d_vs_rdnacn_scatter_plot(f_ax)
    panels.tor1d_vs_rdnacn_scatter_plot(g_ax)
    panels.statistic_of_deletion_effects(h_ax)

    plots.label_panel(a_ax, "A")
    plots.label_panel(b_ax, "B")
    plots.label_panel(c_ax, "C")
    plots.label_panel(d_ax, "D")
    plots.label_panel(e_ax, "E")
    plots.label_panel(f_ax, "F")
    plots.label_panel(g_ax, "G")
    plots.label_panel(h_ax, "H")

    return fig


def sup_fig_10():
    """Validation of two-component machine learning lifespan prediction model.

    A) Lifespan estimates generated on a subset of lifespan movies from
    manual/human annotations or computer-vision predictions. There was no
    difference in the estimated survival functions.

    B) Confusion matrix summarizing the performance of the "event" classifier
    on a subset of the movies used in this study. The resulting classes
    obtained from manual/human annotations ("Annotated") vs automated/machine
    predictions ("Predicted") are shown for a subset of lifespan movies.
    Lifespan observations were classified as observing the cell death
    ("Death") or not observing the cell death ("Censored"). Typically, cells
    were censored because the cell washed out of the observation trap before
    the end of its lifespan.

    C) Combined confusion-matrix / scatter plots indicating the accuracy of
    when each event (D = death, C = censored) was annotated/predicted.
    Annotated frame indexes are on the x-axis, predicted frame indexes are on
    the y-axis. Points are colored using color map to indicate genotype and
    rDNA CN, similar to Fig 2B. Most points fall near 1:1 line, indicating
    good model performance. Data points off the line are unlikely to alter the
    final lifespan estimate.

    D) Table summarizing rDNA CN and fob1∆ effects on event model accuracy.
    Neither rDNA CN or fob1∆ had a significant effect on event model
    performance. Effects on model accuracy were estimated by fitting logit
    ordinal regression.

    E) Combined confusion-matrix / scatter plots indicating the accuracy of
    the bud counting model on a subset of the movies used in this study.
    Annotated number of buds prior to event are on the x-axis, predicted
    number of buds prior to event are on the y-axis.

    """
    w_inches = 7.1  # 2 columns
    h_inches = 7.1
    fig = plt.figure(figsize=(w_inches, h_inches), facecolor="w", dpi=300)

    ax0 = fig.add_axes([0.15, 0.8, 0.2, 0.15])
    ax1 = fig.add_axes([0.6, 0.75, 0.3, 0.2])

    ax2 = fig.add_axes([0.07, 0.4, 0.4, 0.3])
    ax3 = fig.add_axes([0.6, 0.46, 0.25, 0.25])

    ax4 = fig.add_axes([0.07, 0.05, 0.4, 0.3])

    panels.prediction_vs_annotation_km_curves(ax0)
    panels.lifespan_prediction_validation_confusion_matrix(ax1)
    panels.lifespan_prediction_validation_tls(ax2)
    panels.effect_on_genotype_on_event_model_performance_table(ax3)
    panels.lifespan_prediction_validation_bud_cnt(ax4)

    plots.label_panel(ax0, "A", label_position=(-0.2, 1.1))
    plots.label_panel(ax1, "B", label_position=(-0.2, 1.1))
    plots.label_panel(ax2, "C", label_position=(0.05, 1.1))
    plots.label_panel(ax3, "D")
    plots.label_panel(ax4, "E", label_position=(0.05, 1.1))

    return fig


def figure_5():
    """rDNA CN affects lifespan through known regulation of established aging
    factors.

    Model showing how differences in rDNA CN affect lifespan. In cells with low
    rDNA CN, Sir2 expression is low, which in turn causes ERC accumulation and
    shortened lifespan.
    """

    w_inches = 7.1  # 2 columns
    h_inches = 7.25

    fig = plt.figure(figsize=(w_inches, h_inches), facecolor="w", dpi=300)
    ax0 = fig.add_axes([0, 0, 1, 1])
    ax0.axis("off")
    panels.large_model(ax0)

    return fig


def sup_fig_4():
    """"""

    w_inches = 4.5  # 1 column
    h_inches = 4.5
    fig = plt.figure(figsize=(w_inches, h_inches), facecolor="w", dpi=300)
    rect = 0.15, 0.15, 0.75, 0.75
    ax1 = fig.add_axes(rect)
    panels.micromanipulation_comparison(ax1)

    return fig
