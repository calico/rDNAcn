""" This file contains code that is useful for plotting & visualizing the data.
"""
import re

import matplotlib.pyplot as plt
import matplotlib.colors as mp_colors
import matplotlib.cm as cm
import numpy as np

from . import generic_plots


# Color maps
class R2C:
    """ Utility for converting an rDNA CN into a matplotlib color."""
    def __init__(self, min_cn=50, max_cn=175, color_map="viridis"):
        self.min = min_cn
        self.max = max_cn
        self.color_map_name = color_map
        self.color_map = [
            mp_colors.to_rgb(c)
            for c in plt.get_cmap(self.color_map_name)(np.linspace(0, 1, 100))
        ]
        self.mappable = cm.ScalarMappable(
            cmap=self.color_map_name, norm=mp_colors.Normalize(min_cn, max_cn)
        )

    def get_color(self, rdnacn):
        """ Provide a rDNA CN and get a color """
        idx = int(round(100 * (rdnacn - self.min) / (self.max - self.min)))
        idx = max(0, idx)
        idx = min(99, idx)
        return self.color_map[idx]

    def __call__(self, rdnacn):
        return self.get_color(rdnacn)


# Create universal color generators
r2c_viridis = R2C()
r2c = r2c_viridis  # default
r2c_magma = R2C(color_map="magma", max_cn=250)


def add_on_color_bar(cax, val_lim=None, c_lim=None, cmap="viridis"):
    data = np.tile(np.linspace(*val_lim, 1000), [1, 1]).transpose()
    cax.pcolormesh(data, cmap=cmap, vmin=c_lim[0], vmax=c_lim[1])
    cax.axis("off")
    ax2 = cax.twinx()
    ax2.set_ylim()
    ax2.set_ylim(*val_lim)
    ax2.yaxis.tick_left()
    return ax2


@generic_plots.subpanel  # Decorator that automatically creates axis to plot on
def plt_km_curve(fit_df, ax_obj, *args, **kwargs):
    """Plot a km curve from a dataframe onto a matplotlib axis.

    dataframe must have 'KM_estimate', 'KM_estimate_lower_0.95' and
    'KM_estimate_upper_0.95' columns. These should be the default of the
    `survival_function_` of lifelines.KaplanMeierFitter.

    *args and *kwargs will be passed to `ax_obj.plot()`

    """
    fit_df = fit_df.sort_index()
    ax_obj.plot(fit_df["KM_estimate"], *args, **kwargs)
    kwargs.pop("label", None)  # Don't label the CI shaded area
    ax_obj.fill_between(
        x=fit_df.index,
        y1=fit_df["KM_estimate_lower_0.95"],
        y2=fit_df["KM_estimate_upper_0.95"],
        alpha=0.25,
        **kwargs,
    )
    return ax_obj


@generic_plots.subpanel
def km_wt_curve(lifespan_fit_df, ax_obj):
    """Plot a km curve from a dataframe onto a matplotlib axis w/ WT colormap.
    If 'color' is a column in the dataframe, this will override the normal
    WT colormap.
    """
    for strain, strain_df in lifespan_fit_df.groupby("strain", sort=False):
        if "color" in strain_df.columns:
            color = strain_df.iloc[0].color
        elif "rdnacn" in strain_df.columns:
            color = r2c(float(strain_df.iloc[0].rdnacn))
        else:
            color = None
        plt_km_curve(
            strain_df,
            ax_obj=ax_obj,
            label=strain,
            color=color,
        )
    return ax_obj


# These next two methods are very specific to this project and could likely
# be generalized a bit better. could address later?


@generic_plots.subpanel
def add_wt_to_plot(ax_obj, wt_stats, alpha=0.1, fixed_color=None):
    """Plots rDNA CN vs RLS w/ the "WT" colormap onto the provided matplotlib
    axis (ax_obj).
    """
    for _, strain_df in wt_stats.groupby("strain"):
        if fixed_color is None:
            color = r2c(float(strain_df.iloc[0].rdnacn))
        else:
            color = fixed_color
        generic_plots.plot_scatter(
            strain_df,
            ax_obj=ax_obj,
            xs="rdnacn",
            ys="median",
            ymin="lower_ci",
            ymax="upper_ci",
            alpha=alpha,
            markeredgecolor="none",
            color=color,
        )
    return ax_obj


@generic_plots.subpanel
def add_fob1_to_plot(ax_obj, mutant_stats):
    """Plots rDNA CN vs RLS w/ the "fob1∆" (magma) colormap and marker (diamond)
    onto the provided matplotlib axis (ax_obj).
    """
    for _, strain_df in mutant_stats.groupby("strain"):
        generic_plots.plot_scatter(
            strain_df,
            ax_obj=ax_obj,
            xs="rdnacn",
            ys="median",
            ymin="lower_ci",
            ymax="upper_ci",
            marker="D",
            color=r2c_magma(float(strain_df.iloc[0].rdnacn)),
        )


@generic_plots.subpanel
def add_mutant_to_plot(ax_obj, mutant_stats):
    """Plots rDNA CN vs RLS w/ the "mutant" (red) color and marker (diamond)
    onto the provided matplotlib axis (ax_obj).
    """
    generic_plots.plot_scatter(
        mutant_stats,
        ax_obj=ax_obj,
        xs="rdnacn",
        ys="median",
        ymin="lower_ci",
        ymax="upper_ci",
        color="#eb5e0b",
        markersize=4,
        marker="D",
    )
    ax_obj.axvline(x=150, alpha=0.07, color="k")


@generic_plots.subpanel
def format_rls_by_rdna_ax(ax_obj):
    """Format the matplotlib axis for consistent appearance.

    Adjusts x & y limits and labels.
    Removes any legend that might have previously been added.
    """
    ax_obj.set_ylim(0, 50)
    ax_obj.set_xlim(50, 275)
    ax_obj.set_xlabel("rDNA CN")
    ax_obj.set_ylabel("Median RLS estimate")
    legend = ax_obj.get_legend()
    if legend:
        legend.remove()


@generic_plots.subpanel
def plot_hazard(ax_obj, subset, colormapper):
    """ Method to help with plotting hazard functions with correct colormap."""
    subset = subset.sort_values("rdnacn")
    for _, strain_df in subset.groupby("strain", sort=False):
        strain_df = strain_df.sort_values("age")
        color = colormapper(strain_df.iloc[0].rdnacn)
        cropped_df = strain_df[strain_df.at_risk > 25]
        ax_obj.plot(
            cropped_df["age"],
            cropped_df["differenced-NA_estimate"],
            color=color,
        )
        ax_obj.fill_between(
            x=cropped_df["age"],
            y1=cropped_df["NA_estimate_lower_0.95"],
            y2=cropped_df["NA_estimate_upper_0.95"],
            alpha=0.05,
            color=color,
        )
        ax_obj.set_ylim(0, 0.15)
        ax_obj.set_xlim(0, 50)


@generic_plots.subpanel
def add_gene(gene_start, gene_stop, gene_name, ax_obj, y_loc=0.1, **kwargs):
    """ Add a gene object to a visualization of a genetic locus. """
    width = 0.2
    ax_obj.arrow(
        gene_start,  # Start
        y_loc,  # Y
        (gene_stop - gene_start) * 0.9,  # dx
        0,  # dy
        width=width,
        head_width=width,
        head_length=abs(gene_start - gene_stop) * 0.1,
        **kwargs,
    )
    ax_obj.text(
        (gene_start + gene_stop) / 2,
        y_loc,
        gene_name,
        horizontalalignment="center",
        verticalalignment="center",
        clip_on=True,
        fontsize=8,
        fontstyle="italic",
    )


def add_mutant_legend(ax_obj, mutant_name):
    """ Method for adding legend with correct marker shape w/ mutant name."""
    # Out of frame points for the legend
    for label, marker in zip(("WT", mpl_italics(f"{mutant_name}∆")), ("o", "D")):
        ax_obj.scatter(
            -1, -1, marker=marker, label=label, facecolors="none", edgecolors="k"
        )
    handle, label = ax_obj.get_legend_handles_labels()
    handle, label = handle[-2:], label[-2:]
    ax_obj.legend(handle, label, framealpha=0)


def format_gene_for_mpl_italics(text):
    """ Method for formating gene names with italics in matplotlib."""
    gene = re.search("([a-zA-Z0-9]+∆)", text)
    if gene:
        text = text.replace(gene.group(1), mpl_italics(gene.group(1)))
    return text


def mpl_italics(text):
    """Method for wrapping a string with the characters needed to format with
    italics."""
    return r"$\it" + f"{text}$"


def cm_scatter(ax_obj, df_obj, metric="bud_cnt"):
    """ Method for plotting a scatter plot onto a quadrant of a scatter plot."""

    xcoord = f"{metric}_annotated"
    ycoord = f"{metric}_predicted"
    if "color" not in df_obj.columns:
        color = "k"
    else:
        color = df_obj["color"]
    ax_obj.scatter(df_obj[xcoord], df_obj[ycoord], alpha=0.5, c=color, marker=".")
    ax_obj.plot(range(500), color="k", alpha=0.3)
    ax_obj.set_ylabel(ycoord, fontsize="small")
    ax_obj.set_xlabel(xcoord, fontsize="small")


def parse_color(row):
    """Method for taking a pandas series with 'KO_gene' and rDNA CN, and returning
    the correct color"""
    if row.KO_gene == "-":  # WT
        color = mp_colors.to_hex(r2c_viridis(row.rdnacn))
    elif row.KO_gene == "fob1":
        color = mp_colors.to_hex(r2c_magma(row.rdnacn))
    else:
        color = "k"
    return color
