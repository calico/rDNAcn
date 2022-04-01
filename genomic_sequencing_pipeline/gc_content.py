""" Utilities for calculating and correcting for GC biases. """
from multiprocessing.pool import Pool


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from Bio import SeqIO
from Bio import SeqUtils
from tqdm import tqdm

from . import rdna


def _calculate_chromosome_gc_content(rec, window_size=200):
    """ Method to be applied to seqrec object to calculate gc content."""
    chrom_gc_stats = []
    rec_len = len(rec)
    for coord in range(0, rec_len):
        start = max(0, coord - window_size // 2)
        end = min(rec_len, coord + window_size // 2)
        gc_content = SeqUtils.GC(rec[start:end].seq)
        chrom_gc_stats.append(
            {"chromosome": rec.id, "position": coord, "gc_content": gc_content}
        )
    return pd.DataFrame(chrom_gc_stats)


def calculate_genome_gc_content(
    ref_genome_path="/home/nate/data/sequencing/ref/SacCer3.fasta",
):
    """ Calculate the GC content of a genome. """
    all_dfs = []
    print("Calculating gc_content...")
    recs = list(SeqIO.parse(ref_genome_path, format="fasta"))
    with Pool() as pool:
        all_dfs = pool.map(_calculate_chromosome_gc_content, recs)
    all_dfs = pd.concat(all_dfs)
    return all_dfs


def load_genome_gc_content(
    path="/home/nate/data/SacCer3_GC_Content_200bp_sliding_window.csv",
):
    """ Load the GC content that has previously been calculated """
    return pd.read_csv(path).drop("Unnamed: 0", axis=1)


def get_genome_gc_content():
    """ Get the genome GC content (for yeast). If pre-calculated load that,
    otherwise recalculate. """
    try:
        return load_genome_gc_content()
    except FileNotFoundError:
        return calculate_genome_gc_content()


def plot_gc_content(gc_df, ymin=50, ymax=600):
    """ Plot the GC content of the genome"""
    ncols = 4
    nrows = 4
    kb_to_exclude = 20  ## excluded the telomeric regions
    fig, axes = plt.subplots(figsize=(10, 10), nrows=nrows, ncols=ncols)
    for i, (chrm, cdf) in enumerate(gc_df.groupby("chromosome")):
        to_plot = cdf.iloc[kb_to_exclude * 1000 : -kb_to_exclude * 1000]
        row, col = i // ncols, i % ncols
        ax_obj = axes[row, col]
        ax_obj.set_title(chrm)
        ax_obj.hexbin(
            x=to_plot["gc_content"],
            y=to_plot["depth"],
            gridsize=25,
            extent=(20, 60, ymin, ymax),
            cmap="inferno",
        )
        ax_obj.axis([20, 60, ymin, ymax])
    fig.tight_layout()
    plt.show()


def merge_gc_content(df_obj, gc_content=None):
    """ Merge the gc content of the genome with a dataframe containing
    columns of 'chromosome'. and 'position'.
    """
    assert "chromosome" in df_obj.columns
    assert "position" in df_obj.columns
    if gc_content is None:
        gc_content = get_genome_gc_content()
    return df_obj.merge(gc_content, on=["chromosome", "position"])


def calculate_slope(df_obj, xcol="rDNA Copies", ycol="median"):
    """
    Calculate simple linear regression of two columns of a DataFrame.

    Returns m (slope), b(intercept), r2
    """
    df_obj = df_obj[[xcol, ycol]]
    df_obj = df_obj.replace([np.inf, -np.inf], np.nan).dropna()
    x_vals = df_obj[xcol].to_numpy()
    y_vals = df_obj[ycol].to_numpy()

    if df_obj.empty:
        return None, None, None
    regr = stats.linregress(x_vals, y_vals)
    slope = float(regr.slope)
    intercept = float(regr.intercept)
    r_squared = float(regr.rvalue ** 2)
    return slope, intercept, r_squared


def smooth(x, window_len=11, window="hanning"):
    # Taken from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    # pylint: disable=C # we are leaving the style of the code as found
    # pylint: disable=W
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    NOTE: length(output) != length(input), to correct this:
    return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    # print(len(s))
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")
    return y


def plot_coverage_distribution(depth_report, ax_obj=None):
    """ Plot the coverage distribution across the genome """
    if ax_obj is None:
        _, ax_obj = plt.subplots(figsize=(15, 3))
    count = 0
    xticks = []
    xtick_labels = []
    for chrm, cdf in depth_report.groupby("chromosome", sort=False):
        chr_length = max(cdf.position) / 1000
        window = 1000
        ax_obj.plot(
            cdf.position[:-1] / 1000 + count, smooth(cdf.depth, window)[:-window]
        )
        xticks.append(count + chr_length // 2)
        xtick_labels.append(chrm)
        count += chr_length
    ax_obj.set_ylim(0, depth_report.depth.median() * 4)
    ax_obj.set_xticks(xticks)
    ax_obj.set_ylabel("Depth")
    ax_obj.set_xticklabels(xtick_labels, rotation=75)
    return ax_obj


def get_nonrepetive_genome(path="/home/nate/data/SacCer3_NonRepetitivePositions.csv"):
    """ Load a file/csv with the coordinates of the non-repetitive elements of the
    yeast genome.

    `SacCer3_NonRepetitivePositions.csv` is a file with two columns:
        1) chromosome
        2) coordinate

    This specifies all the regions of the genome that are not repetive.

    This was created by taking the output of .rdna.open_depth_report, and filter
    regions that were <2x median coverage.

    e.g. :
        ```
        from . import rdna
        rep = rdna.open_depth_report(analysis_dir, 1)
        rep = rep[rep.depth < (rep.depth.median()*2)]
        rep[['chromosome', 'position']].to_csv(
            'SacCer3_NonRepetitivePositions.csv',
            index=False
        )
        ```
    """
    return pd.read_csv(path)


class GcReport:
    """ Get a report of the GC bias of a sequenced genome """

    def __init__(self, report_dir, number_samples=None):
        self.pbar = None
        self.results = []
        self.report_dir = report_dir
        self.number_samples = number_samples
        self.gc_report = get_genome_gc_content()
        self.nonrepetive_genome = get_nonrepetive_genome()
        self.report_df = None

    @staticmethod
    def _calculate_gc_bias(args):
        """ calculate the apparent bias of a sample """
        report_dir, sample_number = args  # unpack args
        gc_report = get_genome_gc_content()
        nonrepetive_genome = get_nonrepetive_genome()
        depth_report = merge_gc_content(
            rdna.open_depth_report(report_dir, sample_number), gc_report
        )

        depth_report = nonrepetive_genome.merge(
            depth_report, on=("chromosome", "position")
        )
        gc_bias = calculate_slope(depth_report, "gc_content", "depth")
        return {
            "sample_number": sample_number,
            "gc_slope": gc_bias[0],
            "gc_intercept": gc_bias[1],
            "gc_r2: ": gc_bias[2],
        }

    def _handle_gc_bias_results(self, result):
        """ Handle the result returns by `_calculate_gc_bias` """
        if result is not None:
            self.results.append(result)
        self.pbar.update(1)

    def generate(self):
        """ Generate report of GC bias for all the samples """
        with tqdm(total=self.number_samples) as self.pbar, Pool(4) as pool:
            to_process = [
                (self.report_dir, i) for i in range(1, self.number_samples + 1)
            ]
            for result in pool.imap_unordered(
                self._calculate_gc_bias, to_process, chunksize=1
            ):
                self._handle_gc_bias_results(result)
        self.report_df = pd.DataFrame(self.results).set_index("sample_number")
        return self.report_df
