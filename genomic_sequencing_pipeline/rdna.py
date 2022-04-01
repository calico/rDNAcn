""" Utilities for calculating rDNA copy number for WGS data """
import os
import warnings
from multiprocessing.pool import ThreadPool

import pandas as pd
import numpy as np
from tqdm import tqdm


def calculate_rdna_copies(row):
    """ Calculate the number of rDNA repeats based upon wgs coverage """
    n_repeats_in_reference = 2
    return (row.rdna_coverage / row.chr12_baseline) * n_repeats_in_reference


def _get_depth_on_chr12(coverage_map, start=451575, stop=468931):
    """ Get the depth on a region of Chr12 with specified start/stop"""
    coverage_map = coverage_map[coverage_map.chromosome == "Chr12"]
    coverage = coverage_map[
        (start < coverage_map.position) & (coverage_map.position < stop)
    ]
    return coverage.depth.median()


def get_rdna_depth(coverage_map, rdna_start=451575, rdna_stop=468931):
    """ Get the depth on a region of Chr12 in the rDNA"""
    return _get_depth_on_chr12(coverage_map, rdna_start, rdna_stop)


def get_chr12_baseline1(coverage_map):
    """ Get the depth on a region of Chr12 known to be single copy"""
    return _get_depth_on_chr12(coverage_map, 760000, 790000)


def get_chr12_baseline2(coverage_map):
    """ Get the depth on a region of Chr12 known to be single copy"""
    return _get_depth_on_chr12(coverage_map, 160000, 190000)


def open_depth_report(report_dir, sample_number):
    """
    Looks for a .depth file created by output of `samtools depth`.
    """
    path = os.path.join(report_dir, str(sample_number), f"{sample_number}_sorted.depth")
    if not os.path.exists(path):
        warnings.warn(f"Expected .depth file does not exist: {path}")
        return None
    coverage_map = pd.read_csv(
        path, names=["chromosome", "position", "depth"], sep="\t"
    )
    return coverage_map


class rDNAReport:
    """ Tool for creating a report of rDNA CN from the output of the
    sequencing pipeline.
    """

    def __init__(self, report_dir, number_samples=None):
        """
        report_dir<str> = directory containing each sample folder
        number_samples<int> = number of samples to expect, can be inferred
            if now extra folders were created.
        """
        self.rdna_list = []
        self.rdna_df = pd.DataFrame()
        self.report_dir = report_dir
        self.number_samples = number_samples
        if self.number_samples is None:
            self.number_samples = len(
                [
                    i
                    for i in os.listdir(self.report_dir)
                    if os.path.isdir(os.path.join(self.report_dir, i))
                ]
            )
        self.pbar = None

    def _handle_depth_report_result(self, res):
        """ Handle the result of `_calculate_rdna`"""
        self.rdna_list.append(res)
        if self.pbar is not None:
            self.pbar.update(1)

    def open_depth_report(self, sample_number=1):
        """Open the depth report of a specific sample number.

        Assumes default file structure of the sequencing pipeline.
        """
        path = os.path.join(
            self.report_dir, str(sample_number), f"{sample_number}_sorted.depth"
        )
        if not os.path.exists(path):
            warnings.warn(f"Expected .depth file does not exist: {path}")
            return None
        coverage_map = pd.read_csv(
            path, names=["chromosome", "position", "depth"], sep="\t"
        )
        return coverage_map

    def _calculate_rdna(self, sample_number):
        """ Calculate the rDNA CN for a specific sample number """
        coverage_map = self.open_depth_report(sample_number)
        if coverage_map is None:
            return {"sample_number": sample_number}
        results_dict = {
            "sample_number": sample_number,
            "rdna_coverage": get_rdna_depth(coverage_map),
            "chr12_baseline": np.mean(
                [get_chr12_baseline1(coverage_map), get_chr12_baseline2(coverage_map),]
            ),
        }
        return results_dict

    def generate(self):
        """ Calculate the rDNA CN for all samples """
        with tqdm(total=self.number_samples) as self.pbar, ThreadPool(4) as tpool:
            for i in range(1, self.number_samples + 1):
                tpool.apply_async(
                    self._calculate_rdna,
                    (i,),
                    callback=self._handle_depth_report_result,
                )
            tpool.close()
            tpool.join()
        self.rdna_df = pd.DataFrame(self.rdna_list)
        self.rdna_df["rDNA Copies"] = self.rdna_df.apply(calculate_rdna_copies, axis=1)
        return self.rdna_df.set_index("sample_number").sort_index()
