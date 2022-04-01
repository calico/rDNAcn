import inspect
from functools import wraps

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd


def subpanel(f):
    """Convenience tool to help when testing functions that plot.

    Will check if the function was expecting `ax_obj` as a positional
    argument or as a kwargs.

    If it was not provided (or left as ax_obj=None) then it will create
    a figure and ax_obj to plot on.

    This might not be totally general, so use with care...

    If you notice anything strange with the layouts, please make sure that
    you de-decorate your function and check it's behavior and make sure
    that this is not causing any issues.
    """
    sig = inspect.signature(f)

    @wraps(f)
    def _ax_wrapper(*args, **kwargs):
        if len(args) > len(sig.parameters):
            raise TypeError(f"{f.__name__}() was provided too many arguments!")
        parsed_args = dict(zip(sig.parameters, args))
        for k, v in kwargs.items():
            if k in parsed_args:
                raise TypeError(
                    f"{f.__name__}() got multiple values for argument '{k}'"
                )
            parsed_args.update({k: v})
        missing_args = set(sig.parameters.keys()) - set(parsed_args.keys())
        for k in missing_args:
            if sig.parameters.get(k).default == inspect._empty:  # no default
                if k == "ax_obj":
                    parsed_args.update({"ax_obj": None})
        if "ax_obj" in parsed_args:
            if parsed_args["ax_obj"] is None:
                fig, ax_obj = plt.subplots()
                parsed_args["ax_obj"] = ax_obj
        return f(**parsed_args)

    return _ax_wrapper


def label_panel_decorator(f):
    @wraps(f)
    def _wrapper(*args, **kwargs):
        panel_letter = kwargs.pop("panel_letter", None)
        label_position = kwargs.pop("label_position", None)
        plotted_figure = f(*args, **kwargs)
        assert hasattr(plotted_figure, "plot"), type(
            plotted_figure
        )  # duck type axes object
        if panel_letter is not None:
            return label_panel(plotted_figure, panel_letter, label_position)
        else:
            return plotted_figure

    return _wrapper


def label_panel(ax_obj, letter, label_position=None):
    """Add the letter in the wasted space to the upper left"""
    if label_position is None:
        label_position = (-0.075, 1.05)
    text = ax_obj.text(
        *label_position,
        letter,
        fontsize=10,
        fontweight="bold",
        transform=ax_obj.transAxes,
    )
    return ax_obj


@subpanel
def prism_style_column_scatter(df, groupby, *args, ax_obj=None, y="y", **kwargs):
    """Make a prism Style column scatter plot.

    Args:
        df -- dataframe
        groupby -- how to group the data into different columns
        *args -- args are passed to plt.plot()
    Kargs:
        y_val_column <str> -- column to use as the y axis
        ax_obj <axes> object -- axes to plot onto
        **kwargs -- kwargs are pass to plt.plot
    """
    assert y in df.columns

    to_plot = pd.DataFrame(df)
    to_plot = to_plot.assign(x_val=to_plot.groupby(groupby).ngroup() + 1)
    jitter_min = -0.25
    jitter_max = 0.25
    to_plot = to_plot.assign(
        x_val_jitter=to_plot.x_val + np.random.uniform(-0.25, 0.25, len(to_plot))
    )
    if ax_obj is None:
        fig, ax_obj = plt.subplots()
    if "marker" not in kwargs:
        kwargs.update(marker="o")

    if ("color" in kwargs) | ("c" in kwargs):
        kwargs.update(color=kwargs.pop("c", kwargs.get("color")))
        if kwargs.get("color") == "group":
            kwargs.update(
                color=to_plot.groupby(groupby)
                .ngroup()
                .apply(
                    lambda i: list(colors.BASE_COLORS.keys())[
                        i % len(list(colors.BASE_COLORS.keys()))
                    ]
                )
            )
        elif kwargs.get("color") in to_plot.columns:
            kwargs.update(color=to_plot[kwargs.get("color")])

    # Plot the scatter points
    ax_obj.scatter(
        to_plot.x_val_jitter,
        to_plot[y],
        *args,
        **kwargs,
    )

    # Add the mean/median/quartile bars
    groups = []
    for group, gdf in to_plot.groupby(groupby):
        mean = gdf[y].mean()
        median = gdf[y].median()
        lower_quart = np.percentile(gdf[y], 25)
        upper_quart = np.percentile(gdf[y], 75)
        ax_obj.hlines(
            mean,
            xmin=gdf.iloc[0].x_val + jitter_min,
            xmax=gdf.iloc[0].x_val + jitter_max,
            alpha=0.8,
        )
        ax_obj.hlines(
            median,
            xmin=gdf.iloc[0].x_val + jitter_min,
            xmax=gdf.iloc[0].x_val + jitter_max,
            linestyle="-.",
            alpha=0.7,
        )
        ax_obj.hlines(
            lower_quart,
            xmin=gdf.iloc[0].x_val + jitter_min,
            xmax=gdf.iloc[0].x_val + jitter_max,
            linestyle=":",
            alpha=0.5,
        )
        ax_obj.hlines(
            upper_quart,
            xmin=gdf.iloc[0].x_val + jitter_min,
            xmax=gdf.iloc[0].x_val + jitter_max,
            linestyle=":",
            alpha=0.5,
        )
        groups.append(group)
    ax_obj.set_xticks(range(1, len(groups) + 1))
    ax_obj.set_xticklabels(groups, rotation=70)
    ax_obj.set_ylabel(y)
    return ax_obj


@subpanel
def _plot_scatter(
    df,
    ax_obj,
    *args,
    xs="rdnacn",
    ys="median",
    ymin="lower_ci",
    ymax="upper_ci",
    **kwargs,
):
    """
    Plot a scatter plot with error bars from a pandas dataframe.
    """
    ax_obj = _plot_scatter_points(df, ax_obj, *args, xs=xs, ys=ys, **kwargs)
    if (ymin is not None) and (ymax is not None):
        ax_obj = _plot_scatter_error_bars(
            df, ax_obj, *args, xs=xs, ymin=ymin, ymax=ymax, **kwargs
        )
    ax_obj.set_xlabel(xs)
    ax_obj.set_ylabel(ys)
    return ax_obj


@subpanel
def _plot_scatter_points(df, ax_obj, *args, xs="rdnacn", ys="median", **kwargs):
    """
    Plot scatter plot markers/points.
    """
    if "marker" not in kwargs:
        kwargs.update(marker="o")
    if "linestyle" not in kwargs:
        kwargs.update(linestyle="None")
    ax_obj.plot(
        df[xs].astype(float),
        df[ys].astype(float),
        *args,
        **kwargs,
    )
    return ax_obj


@subpanel
def _plot_scatter_error_bars(
    df, ax_obj, *args, xs="rdnacn", ymin="lower_ci", ymax="upper_ci", **kwargs
):
    """
    Plot scatter plot error bars.
    """
    if ("cs" not in kwargs) and ("colors") not in kwargs:
        assert len(ax_obj.lines), "No markers on plot"
        kwargs.update(colors=ax_obj.lines[-1].get_color())
    kwargs.pop("label", None)
    kwargs.pop("markersize", None)
    kwargs.pop("markeredgecolor", None)
    kwargs.pop("marker", None)

    ax_obj.vlines(
        df[xs].astype(float),
        ymin=df[ymin].astype(float),
        ymax=df[ymax].astype(float),
        *args,
        **kwargs,
    )
    return ax_obj


@subpanel
def plot_scatter(df, ax_obj, *args, groupby="strain", **kwargs):
    """Generate a scatter plot from a pandas dataframe:

    args:
        > xs <str>  | name of column to use for x values.
        > ys <str>  | name of column to use for y values.
        > ymin <str>| name of column to use for bottom of error bar.
        > ymax <str>| name of column to use for top of error bar.
        > should accept many of matplotlib arguments used to format plot and vlines (
            e.g. `alpha`, etc)
    groupby <str> | name of the column to group the plot by

    Defaults to arguments used to generate scatter plots of rDNAcn vs median RLS. (See plot scatter)
    """
    for grouped, group_df in df.groupby(groupby):
        ax_obj = _plot_scatter(group_df, ax_obj, *args, label=grouped, **kwargs)
    ax_obj.legend()
    return ax_obj
