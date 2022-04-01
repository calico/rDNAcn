""" This module contains utilities to help with loading various form of data
associated with this project.
"""
import os
from io import StringIO
import warnings

try:
    from IPython.display import display
except ImportError:
    display = print


import pandas as pd
import numpy as np
import tifffile
from skimage import transform
from sklearn.linear_model import LinearRegression
from statsmodels.miscmodels.ordinal_model import OrderedModel
import lifelines

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
data_file = os.path.join(data_dir, "rDNA CN Manuscript Supp Data Tables.xlsx")


class DataFileLoader:
    """Load the data file only once, but only when you first need it """

    def __init__(self):
        self._data = None

    def get(self, key):
        """ Get an item, load the file if it hasnt been already  """
        if self._data is None:
            self._load_data()

        assert (
            key in self._data
        ), f"No data table name {key}! Only {list(self._data.keys())}"
        return pd.DataFrame(self._data.get(key)).copy()  # Always get a copy

    def __getitem__(self, key):
        return self.get(key)

    def _load_data(self):
        if self._data is None:
            self._data = dict(
                pd.read_excel(data_file, sheet_name=None, engine="openpyxl")
            )


data = DataFileLoader()

# List of strains in which the original observation of lifespan variability was made:
original_variable_wt_strains = [
    "CGY19.02",
    "CGY18.60",
    "CGY18.62",
    "CGY19.04",
    "CGY18.61",
    "CGY19.03",
    "CGY19.12",
    "CGY18.59",
    "CGY18.72",
    "CGY19.14",
    "CGY19.13",
    "CGY18.71",
    "CGY18.73",
]

strain_alias_mapping = {
    # there are a few strains that are referred to by their alias
    # in the results database.
    "CGY23.73": "SMY10046_2",
    "CGY23.74": "SMY10046_3",
    "CGY23.75": "SMY10046_4",
    "CGY23.77": "SMY10047_1",
    "CGY23.78": "SMY10047_2",
    "CGY23.79": "SMY10047_3",
    "CGY23.80": "SMY10047_4",
    "CGY23.81": "SMY10047_5",
}


def get_elongated_death_example():
    """ Load image of elongated (ERC/rDNA-associated) death """
    return tifffile.imread(os.path.join(data_dir, "mode-of-death-example-images.tif"))[
        0, :, :  # elongated is first slice
    ]


def get_round_death_example():
    """ Load image of round (ERC/rDNA-independent) death """
    return tifffile.imread(os.path.join(data_dir, "mode-of-death-example-images.tif"))[
        1, :, :  # round is second slice
    ]


def get_panel_growth_rates():
    """ Get table of growth rates from strains with different rdNA CN."""
    od_measurements = data.get("Panel Growth Rates")
    od_measurements = (
        od_measurements.T.set_index(0).T.set_index("Strain").apply(pd.to_numeric)
    )
    log_od_measurements = od_measurements.apply(np.log2)
    log_od_measurements = log_od_measurements.loc[6:14]  # good exp range
    log_od_measurements.index = log_od_measurements.index.astype(int)
    measurement_times = log_od_measurements.index.values.reshape(-1, 1) * 60
    ods = log_od_measurements.values
    reg = LinearRegression().fit(measurement_times, ods)
    table = (
        pd.DataFrame(1 / reg.coef_, columns=["doubling_time_minutes"])
        .assign(strain=od_measurements.columns)
    )
    return add_standard_rdnacn_to_table(table).set_index('strain')


def get_surv_fits_of_annotated_data():
    """ Calculate KM fits for annotated lifespan data. """
    table = get_annotation_comparison()
    table = add_standard_rdnacn_to_table(table)
    table = add_simple_genotype(table)

    kmf = lifelines.KaplanMeierFitter()
    kmf.fit(table["bud_cnt_predicted"], table["is_death_predicted"])
    prediction_fit = pd.concat(
        (kmf.event_table, kmf.survival_function_, kmf.confidence_interval_), axis=1
    ).assign(label="prediction")
    kmf.fit(table["bud_cnt_annotated"], table["is_death_annotated"])
    annotation_fit = pd.concat(
        (kmf.event_table, kmf.survival_function_, kmf.confidence_interval_), axis=1
    ).assign(label="annotation")
    return pd.concat((prediction_fit, annotation_fit))


def get_mod_annotation_comparison():
    """ Get table comparing mode-of-death predictions to hand-annotated data. """
    sheet_name = "Supp ML Validation -- MOD Model"
    table = data.get(sheet_name)
    table = (
        table.set_index(
            [
                "mode_of_death_prediction",
                "strain",
            ]
        )
        .apply(pd.to_numeric)
        .reset_index()
        .set_index("catcher_id")
    )
    return table


def get_statistics_on_event_model_performance():
    """Get a table of statistics testing the effect of rDNA CN and
    genotype on YLM event model performance.
    """
    table = get_annotation_comparison()
    table = add_standard_rdnacn_to_table(table)
    table = add_simple_genotype(table)
    table = add_additional_stats(table)

    stats_columns = [
        "bud_cnt_diff",
        "bud_cnt_abs_diff",
        "bud_cnt_percent_difference",
        "tls_diff",
        "tls_abs_diff",
        "tls_percent_difference",
    ]
    cat_columns = ["is_death_annotated", "is_death_predicted"]
    var_columns = ["KO_gene", "rdnacn"]
    table = table[var_columns + cat_columns + stats_columns].sort_values(
        ["KO_gene", "rdnacn"]
    )
    table = (
        pd.get_dummies(table, columns=["KO_gene"])
        .reset_index(drop=True)
        .drop("KO_gene_-", axis=1)
    )
    table["event_predicted_correctly"] = (
        table["is_death_annotated"] == table["is_death_predicted"]
    )

    mod_log = OrderedModel(
        table["event_predicted_correctly"],
        table[["rdnacn", "KO_gene_fob1"]],
        distr="logit",
    )

    res_log = mod_log.fit(method="bfgs", disp=False)
    tables = res_log.summary().tables
    results_table = pd.DataFrame(pd.read_csv(StringIO(tables[1].as_csv()), index_col=0))
    results_table = results_table.rename(
        columns={c: c.strip() for c in results_table.columns}
    )
    results_table = results_table.iloc[:-1]
    return results_table


def get_statistics_on_mod_model_performance():
    """Get a table of statistics testing the effect of rDNA CN and
    genotype on mode-of-death model performance.
    """

    table = get_mod_annotation_comparison()
    table = add_standard_rdnacn_to_table(table)
    table = add_simple_genotype(table)

    cat_columns = ["mode_of_death_prediction", "annotated_event_type"]
    var_columns = ["KO_gene", "rdnacn"]
    table = table[var_columns + cat_columns].sort_values(["KO_gene", "rdnacn"])
    table["KO"] = table.KO_gene.apply(lambda v: 0 if v in ["-", "fob1"] else 1)
    table["fob1∆"] = table.KO_gene.apply(lambda v: 1 if v == "fob1" else 0)
    print("MOD N annotated of each genotype:")
    display(table.KO_gene.value_counts())

    table["mode_of_death_annotation"] = table.annotated_event_type.apply(
        lambda a: {1: "r", 2: "e"}.get(a, "u")
    )

    table["event_predicted_correctly"] = (
        table["mode_of_death_annotation"] == table["mode_of_death_prediction"]
    )

    mod_log = OrderedModel(
        table["event_predicted_correctly"],
        table[["rdnacn", "KO", "fob1∆"]],
        distr="logit",
    )

    res_log = mod_log.fit(method="bfgs", disp=False)
    tables = res_log.summary().tables
    results_table = pd.read_csv(StringIO(tables[1].as_csv()), index_col=0)
    results_table = pd.DataFrame(results_table)
    results_table = results_table.rename(
        columns={c: c.strip() for c in results_table.columns}
    )
    results_table = results_table.iloc[:-1]  # remove False/True row
    return results_table


def add_additional_stats(table):
    """ Add additional metrics to table comparing annotations/predictions."""
    table["bud_cnt_diff"] = table["bud_cnt_predicted"] - table["bud_cnt_annotated"]
    table["bud_cnt_abs_diff"] = table.bud_cnt_diff.apply(abs)
    table["tls_diff"] = table["tls_predicted"] - table["tls_annotated"]
    table["tls_abs_diff"] = table.tls_diff.apply(abs)
    table["bud_cnt_percent_difference"] = (
        table["bud_cnt_abs_diff"]
        / ((table["bud_cnt_predicted"] + table["bud_cnt_annotated"]) / 2)
        * 100
    )
    table["tls_percent_difference"] = (
        table["tls_abs_diff"]
        / ((table["tls_predicted"] + table["tls_annotated"]) / 2)
        * 100
    )
    return table


def get_annotation_comparison():
    """ Get table comparing predictions to hand-annotated data. """

    sheet_name = "Supp ML Validation -- Lifespan "
    table = data.get(sheet_name)
    # Fixing the datatypes
    table["age_at_frame_predicted"] = table["age_at_frame_predicted"].str.split(",")
    table["age_at_frame_annotated"] = table["age_at_frame_annotated"].str.split(",")
    table["age_at_frame_predicted"] = table["age_at_frame_predicted"].apply(
        lambda a: [int(i) for i in a]
    )
    table["age_at_frame_annotated"] = table["age_at_frame_annotated"].apply(
        lambda a: [int(i) for i in a]
    )
    cols = [
        "index",
        "id_predicted",
        "catcher_id",
        "frame_start_predicted",
        "frame_stop_predicted",
        "bud_cnt_predicted",
        "score",
        "cv_model_used_id",
        "z",
        "c",
        "channel",
        "processor_version",
        "tls_predicted",
        "culture_id",
        "position_number",
        "mode_of_death_elongated_prob",
        "mode_of_death_model_version",
        "channel_location_x_um",
        "channel_location_y_um",
        "id_annotated",
        "frame_start_annotated",
        "frame_stop_annotated",
        "bud_cnt_annotated",
        "tls_annotated",
    ]
    table[cols] = table[cols].apply(pd.to_numeric, errors="coerce", axis=1)
    return table


def get_young_erc_blot_rdna_probe():
    """ Load/crop image of young cell ERC blot (rDNA probe)."""
    path = os.path.join(data_dir, "20210424_rDNA_36H-[Phosphor].tif")
    img = tifffile.imread(path)
    img = transform.rotate(img, 91.5)
    img = img[760:-670, 750:-896]
    return img


def get_young_erc_blot_npr2_probe():
    """ Load/crop image of young cell ERC blot (NPR2/control probe)."""
    path = os.path.join(data_dir, "20210420_npr2_1HR-[Phosphor].tif")
    img = tifffile.imread(path)
    img = transform.rotate(img, -85)
    img = img[750:-200, 370:-340]
    return img


def get_chef_gel_image():
    """ Load and crop image of CHEF gel."""
    path = os.path.join(data_dir, "2020-03-06 12hr 59min - CHEF - rdna probe.tif")
    img = tifffile.imread(path)
    img = img[380:-320, 290:-210]
    return img


class ChefGelDetails:
    """ Details of CHEF gel ladder """

    def __init__(self):
        self.ladder_pix = np.asarray(
            [59, 207, 260, 329, 384, 463]
        )  # Measured in fiji from well
        self.ladder_mbs = [
            2.7,
            2.35,
            1.81,
            1.66,
            1.37,
            1.05,
        ]  # From BioRad's website for H. wingei ladder

        ladder_reg = LinearRegression().fit(
            self.ladder_pix.reshape(-1, 1), self.ladder_mbs
        )
        ladder_slope = ladder_reg.coef_[0]
        ladder_intercept = ladder_reg.intercept_
        # self.ladder_r2 = ladder_reg.score(self.ladder_pix.reshape(-1, 1), self.ladder_mbs)
        # (mb * 1000kb/mb - CEN_EXTRA - TEL_EXTRA  / kb/copy)
        self.pix2mbs = lambda p: p * ladder_slope + ladder_intercept
        self.pix2rdnacn = lambda p: (self.pix2mbs(p) * 1000 - 8.8 - 30.9) / 9.1
        self.rdnacn2pix = (
            lambda c: ((c * 9.1 + 9.8 + 30.9) / 1000 - ladder_intercept) / ladder_slope
        )


def get_chef_vs_wgs_rdancn_estimates():
    """ Get a table comparing WGS vs CHEF gel rDNA CN estimates """
    chef_gel_details = ChefGelDetails()
    sample_pix = [  # Measured in fiji from the well
        585,  # For each well, how many pixels down was the band?
        545,
        463,
        436,
        409,
        390,
        384,
        378,
        345,
        233,
        179,
    ]
    sample_estimates = [chef_gel_details.pix2rdnacn(p) for p in sample_pix]
    sequencing_estimates = data.get("Supp Sequenced CHEF Samples")
    sequencing_estimates["CHEF rDNAcn Estimate"] = sample_estimates
    return sequencing_estimates


def get_nonselected_counts():
    """Get counts for RNA levels (not polyA selected) in strains with varying
    rDNA CN.
    
    
    File is available at GEO (GSE193600).
    """
    path = os.path.join(data_dir, "GSE193599_noSelect_NORM_counts.txt")
    tpm_table = pd.read_csv(path, sep="\t")
    return tpm_table


def get_lifespan_summary_for_by_strains():
    """Get a table of median lifespan estimates for comparing to BY strain
    background.
    """
    sheet_name = "Supp Summarized BY Lifespans"
    lifespan_summary = data.get(sheet_name)
    lifespan_summary = lifespan_summary.apply(pd.to_numeric, errors="ignore", axis=1)
    return add_standard_rdnacn_to_table(lifespan_summary)


def get_lifespan_summary_for_different_medias():
    """Get a table of median lifespan estimates for comparing strains
    grown/aged in different environments.
    """
    sheet_name = "Supp Summarized Media Lifespans"
    lifespan_summary = data.get(sheet_name)
    lifespan_summary = lifespan_summary.apply(pd.to_numeric, errors="ignore", axis=1)
    return lifespan_summary


def get_lifespan_summary():
    """ Get a table of median lifespan estimates. """
    sheet_name = "Summarized Lifespans"
    lifespan_summary = data.get(sheet_name)
    lifespan_summary = lifespan_summary.apply(pd.to_numeric, errors="ignore", axis=1)
    return lifespan_summary


def get_wt_unexpected_variability_lifespan_stats():
    """ Get a table of summarized lifespans for original WT strains."""
    all_statistics = get_lifespan_summary()
    return (
        all_statistics[all_statistics.strain.isin(original_variable_wt_strains)]
        .drop("KO_gene", axis=1)
        .set_index(["experiment", "strain"])
        .apply(pd.to_numeric, axis=1)
        .reset_index()
    )


def get_variable_wt_survival_curves():
    """ Get table of survival functions from panel of WT strains. """
    sheet_name = "WT Lifespan Survival Fits"
    return (
        data.get(sheet_name)
        .set_index(["strain"])
        .apply(pd.to_numeric, axis=1)
        .reset_index()
        .set_index("index")
    )


def get_mode_of_death_hazards():
    """Get a table representing the hazard of each mode of death for each strain."""
    sheet_name = "Mode-of-Death Hazards"
    return add_standard_rdnacn_to_table((
        data.get(sheet_name)
        .set_index(["strain", "KO_gene", "hazard_of"])
        .apply(pd.to_numeric, axis=1)
        .reset_index()
    ))


def get_experiments_for_by_strains():
    """Get table of experiments/channels to use for comparing to BY strain
    background.
    """
    experiments = data.get("Supp BY Lifespan Experiments")
    return experiments


def get_erc_blot_quantifications():
    """Get a table representing the quantification of ERCs from the images
    of the southern blots.
    """
    sheet_name = "Panel ERC quantification"
    table = (
        data.get(sheet_name)
        .set_index(["age", "genotype", "gel", "strain"])
        .apply(pd.to_numeric, axis=1)
        .reset_index()
    )
    cn_df = get_rdnacn_measurements()
    cn_df = cn_df[pd.isnull(cn_df.media)][["rdnacn"]]
    table = table.merge(cn_df, right_index=True, left_on="strain", how="left")
    if np.any(pd.isnull(table.rdnacn)):
        missing = table[pd.isnull(table.rdnacn)].drop_duplicates(
            subset=["strain", "rdnacn"]
        )
        warnings.warn("Missing data!")
        display(missing)
    return table


def get_experiments_for_different_medias():
    """Get table of experiments/channels to use for comparing strains grown/aged
    in different environments.
    """
    return data.get("Supp Media Lifespan Experiments")


def get_old_cell_erc_blot():
    """ Load/crop the old cell ERC blot."""
    path = os.path.join(data_dir, "20210325_rDNA_ON-[Phosphor].tif")
    img = tifffile.imread(path)
    img = np.rot90(img)
    img = img[
        1450:-550,
        560:-432,
    ]
    img = np.rot90(img, k=2)
    return img


def get_old_cell_npr2_blot():
    """ Load/crop the old cell NPR2 blot """
    path = os.path.join(data_dir, "20210325_NPR2_ON-[Phosphor].tif")
    img = tifffile.imread(path)
    img = np.rot90(img)
    img = transform.rotate(img, -4)
    img = img[630:-540, 228:-294]
    return img


def get_atac_seq_counts_at_sir2():
    """Get a table for the density of ATACseq insertions at/near the SIR2 locus.
    
    File is available at GEO (GSE193600).
    """
    path = os.path.join(data_dir, "GSE193598_ATAC_counts_table.hdf5")
    count_table = pd.read_hdf(path)
    normalized_count_table = count_table / count_table.sum()
    sir2_locus_data = normalized_count_table.loc["Chr4"].loc[
        376757 - 1000:378445 + 1000
    ]
    return sir2_locus_data


def get_mutant_lifespans_cph_stats():
    """ Get a table for how the lifespans are affected by ∆s or rDNAcn."""
    return (
        data.get("Mutant Lifespans CPH Stats")
        .set_index("covariate")
        .apply(pd.to_numeric, axis=1)
    )


def get_variable_wt_genetic_vars():
    """ Get a table of loci that vary in panel of original WT strains. """
    return (
        data.get("Variable WT Genetic Vars")
        .set_index(["CHROM", "c"])
        .apply(pd.to_numeric, axis=1)
        .reset_index()
        .set_index(["CHROM", "LOCUS", "c"])
    )


def get_panel_rnaseq_deseq_vs_rls_table():
    """Get a table of DESeq results for how rDNAcn affects gene expression."""
    return data.get("RNAseq-DeSEQ (gene v rls)").set_index("gene").replace("NA", np.nan)


def get_panel_cell_size():
    """Get a table of cell size measurements for a panel of strains with
    different rDNA copy numbers.
    """
    table = data.get("Panel Cell Size").set_index("strain").apply(pd.to_numeric)
    table = table[table.index.notnull()]
    table = table[[c for c in table.columns if "Unnamed" not in c]]
    return add_standard_rdnacn_to_table(table.reset_index()).set_index('strain')


def get_rdnacn_measurements():
    """Get a table of rDNAcn measurements.

    Load measurements from file of all strains that we have rDNAcn
    measurements of.


    Returns a pandas dataframe. The strain name is the index column. The two
    data columns are:
        -'rdnacn' -- the number of rDNA array repeats measured for this strain.
        -'media' -- the media in which the sample was prepared.
            'SC'| 'YNB' |'YPD' - Samples prepared from low-OD logarithmically
                growing cultures in the specified media for ~24 hours.
            'None' - Samples were prepared by scraping cells from patch of cells
                grown on YPD plates.
    """
    sheet_name = "Strain Table"
    cn_df = data.get(sheet_name)
    # some samples were sequenced twice for the same condition, take the mean
    cn_df = cn_df.replace(np.nan, "None")
    cn_df = cn_df.reset_index().groupby(["strain", "media"]).mean().reset_index()
    cn_df = cn_df.replace("None", np.nan)
    cn_df["rdnacn"] = cn_df.rdnacn.round().astype(int)
    return cn_df.set_index("strain")[["rdnacn", "media"]]


def get_experiments_to_use():
    """ Get a table of experiments to use that have been QC'd. """
    return data.get("Lifespan Experiments")


def add_standard_rdnacn_to_table(table):
    """Add the column "rdnacn" to a dataframe.

    Args:
        `table` -- pandas dataframe with the column 'strain'
    Returns:
        `table` but with an a new column, 'rdnacn'

    Will raise a warning and display the columns with no standard rDNA CN
    measurements.

    Will raise `AssertionError` if there is no 'strain' column in `table`.
    """
    assert "strain" in table.columns
    rdna_measurements = get_rdnacn_measurements()
    # only get "standard" measurements, not in a specific media
    rdna_measurements = rdna_measurements[pd.isnull(rdna_measurements.media)]
    # throw out unneeded extra columns
    rdna_measurements = rdna_measurements[["rdnacn"]]
    # merge this column to the table provided
    table = table.merge(
        rdna_measurements,
        left_on="strain",
        right_index=True,
        how="left",  # make sure we dont drop rows that we don't have rDNAcn for
    )
    # Warn if we don't have an rDNAcn measurement for one of the strains in the
    # table that was provided
    if "rdnacn" in table.columns:
        if np.any(pd.isnull(table.rdnacn)):
            missing = table[pd.isnull(table.rdnacn)].drop_duplicates(
                subset=["strain", "rdnacn"]
            )
            warnings.warn("Missing data!")
            display(sorted(missing.strain.unique()))
    return table


def add_simple_genotype(table):
    """Add a column "KO_gene" to a table based on strain name.

    Strain must appear in the table returned by `get_experiments_to_use`,
    or else its KO_gene will be assigned 'n/a'.

    Args:
        `table` -- pandas dataframe with the column 'strain'
    Returns:
        `table` but with an a new column, 'KO_gene'

    Will raise `AssertionError` if there is no 'strain' column in `table`.
    """
    assert "strain" in table.columns
    exp_table = get_experiments_to_use()

    table = table.merge(
        exp_table[["Strain", "KO_gene"]].drop_duplicates().set_index("Strain"),
        left_on="strain",
        right_index=True,
        how="left",
    )
    return table


def get_surv_fits_by_micromanipulation():
    """Load lifespan observations made from micromanipulation.
    Observations are then fit using `KaplanMeierFitter` (grouped
    by strain) and returned as a dataframe.
    """
    observations = data.get("Supp Micromanipulation Data")

    surv_fits = []
    for strain, strain_df in observations.groupby("strain"):
        kmf = lifelines.KaplanMeierFitter()
        kmf.fit(strain_df["age"].astype(int), strain_df["event"])
        strain_fit = pd.concat(
            (kmf.event_table, kmf.survival_function_, kmf.confidence_interval_), axis=1
        ).assign(strain=strain)
        surv_fits.append(strain_fit)
    return pd.concat(surv_fits)


def get_ylm_surv_fits_for_strains_tested_by_micromanipulation():
    """Load lifespan observations made from microfluidics.
    Data is filtered to just the strains that were also used in micromanipulation
    experiment and to data collected in YPD. Observations are
    then fit using `KaplanMeierFitter` (grouped by strain) and
    returned as a dataframe.
    """
    strains = ["CGY24.09", "CGY24.17"]
    path = os.path.join(data_dir, "media_lifespans_obs.csv")
    if not os.path.exists(path):
        raise UserWarning("Please unzip `data/raw_data_archive.zip`")
    observations = pd.read_csv(path)
    observations = observations[
        observations.media.str.contains("YPD") & observations.strain.isin(strains)
    ]
    surv_fits = []
    for strain, strain_df in observations.groupby("strain"):
        kmf = lifelines.KaplanMeierFitter()
        kmf.fit(
            strain_df["bud_cnt"].astype(int), strain_df["event_type"].str.contains("x")
        )
        strain_fit = pd.concat(
            (kmf.event_table, kmf.survival_function_, kmf.confidence_interval_), axis=1
        ).assign(strain=strain)
        surv_fits.append(strain_fit)
    return pd.concat(surv_fits)


def load_lifespan_observations():
    # Load table of observations
    path = os.path.join(data_dir, "lifespan_observation_table.csv")
    obs = pd.read_csv(path, index_col=0)

    # fix a couple strains that were referred to by an alias
    obs["strain"] = obs.strain.replace(strain_alias_mapping)
    obs = add_simple_genotype(obs)
    obs = add_standard_rdnacn_to_table(obs)
    return obs


def load_lifespan_observations_by():
    # Load table of observations
    path = os.path.join(data_dir, "BY_lifespans_obs.csv")
    if not os.path.exists(path):
        raise UserWarning("Please unzip `data/raw_data_archive.zip`")
    obs = pd.read_csv(path, index_col=0)

    # fix a couple strains that were referred to by an alias
    obs["strain"] = obs.strain.replace(strain_alias_mapping)
    obs = add_simple_genotype(obs)
    obs = add_standard_rdnacn_to_table(obs)
    return obs


def load_lifespan_observations_media():
    # Load table of observations
    path = os.path.join(data_dir, "media_lifespans_obs.csv")
    if not os.path.exists(path):
        raise UserWarning("Please unzip `data/raw_data_archive.zip`")
    obs = pd.read_csv(path, index_col=0)

    # fix a couple strains that were referred to by an alias
    obs["strain"] = obs.strain.replace(strain_alias_mapping)
    obs = add_simple_genotype(obs)
    obs = add_standard_rdnacn_to_table(obs)
    return obs


def calculate_cph_stats():
    """ Recalculate Proportional Hazards of rDNA CN, ∆s, and experiment. """
    obs = load_lifespan_observations()
    # keep only the columns we care about for this
    df = pd.DataFrame(
        obs[
            [
                "bud_cnt",
                "event_type",
                "KO_gene",
                "rdnacn",
                "experiment",
                "channel",
                "strain",
            ]
        ]
    )
    df = df.assign(
        cluster_col=df.apply(lambda r: f"{r.experiment}-{r.channel}-{r.strain}", axis=1)
    )

    # create 1-hot encoding of ∆s and experiments
    new_obs_table = (
        pd.get_dummies(df, columns=["KO_gene", "experiment"])
        .reset_index(drop=True)
        .drop("KO_gene_-", axis=1)
    )

    # Observed events contain `x` in event_type column
    new_obs_table["E"] = new_obs_table.event_type.str.contains("x")

    # Collapse replicate rows and weight them to increase performance
    new_obs_table_weights = new_obs_table.copy()
    new_obs_table_weights["weights"] = 1.0
    new_obs_table_weights = (
        new_obs_table_weights.groupby(new_obs_table.columns.tolist())["weights"]
        .sum()
        .reset_index()
    )
    new_obs_table = new_obs_table_weights

    # Add a little noise to the data, otherwise the model
    # complains about having replicate rows. This was more
    # important before I started using the "weight" option
    # above, but leaving is here because it does not really
    # affect the restult too much and there is some error in
    # our lifespan measurements anyway.
    new_obs_table["T"] = new_obs_table["bud_cnt"] + (
        np.random.random(len(new_obs_table)) - 0.5
    )

    # Model as two components (# repeats > 150 & # repeats < 150)
    knot = 150
    new_obs_table["rdnacn_1"] = new_obs_table.rdnacn.apply(lambda r: min(knot, r) - 150)
    new_obs_table["rdnacn_2"] = new_obs_table.rdnacn.apply(lambda r: max(0, r - knot))

    cph = lifelines.CoxPHFitter(penalizer=0.1)
    exp_factors = " + ".join([f"experiment_{c}" for c in obs.experiment.unique()])
    cph.fit(
        new_obs_table,
        "T",
        "E",
        cluster_col="cluster_col",
        show_progress=True,
        weights_col="weights",
        formula=(
            # rDNA CN factors
            f"rdnacn_1 + rdnacn_2"
            # experiment factors
            f" + {exp_factors}"
            # ∆ factors
            " + KO_gene_gpa2"
            " + KO_gene_hda2"
            " + KO_gene_idh1"
            " + KO_gene_rpl13A"
            " + KO_gene_tor1"
            " + KO_gene_ubp8"
            " + KO_gene_ubr2"
            " + KO_gene_fob1"
            # interaction terms
            f"+ rdnacn_1:KO_gene_fob1"
            f"+ rdnacn_2:KO_gene_fob1"
            f"+ rdnacn_1:KO_gene_hda2"
            f"+ rdnacn_2:KO_gene_hda2"
            f"+ rdnacn_1:KO_gene_ubp8"
            f"+ rdnacn_2:KO_gene_ubp8"
        ),
    )
    return cph.summary.reset_index()[
        ["covariate", "coef", "coef lower 95%", "coef upper 95%"]
    ]


def calculate_survival_fits_of_all_strains():
    """Calculate the survival functions of all strains.

    Data will be grouped by strain (pooling all replicates).
    """
    observations = load_lifespan_observations()
    return calculate_survival_fits(
        observations, groupby=("strain", "rdnacn", "KO_gene")
    )


def calculate_survival_fits(df, groupby=("strain",), metric="bud_cnt"):
    """
    Take a dataframe containing lifespan predictions and calculate the survival fits.
    args:
    -df <DataFrame> -- Table of results to summarize
    -groupby <tuple> -- Tuple containing column headers on how to group the dataframe.
    -metric <str> -- Column name to use for lifespan
    """
    surv_fits = []
    groupby = list(groupby)
    for groupby_vals, sdf in df.groupby(groupby):
        if len(groupby) == 1:
            groupby_vals = [groupby_vals]
        kmf = lifelines.KaplanMeierFitter()
        kmf.fit(sdf[metric].astype(int), sdf["event_type"].str.contains("x"))
        strain_fit = pd.concat(
            (kmf.event_table, kmf.survival_function_, kmf.confidence_interval_), axis=1
        ).assign(**dict(zip(groupby, groupby_vals)))
        surv_fits.append(strain_fit)
    return pd.concat(surv_fits).reset_index().rename(columns={"index": metric})


def caculate_median_lifespans_of_all_strains():
    """Calculate the median lifespan of all lifespan observations.
    Grouped by experiment channel (strains may appear more than
    once (medias, replicates, etcs).
    """
    obs = pd.concat(
        (
            load_lifespan_observations(),
            load_lifespan_observations_by(),
            load_lifespan_observations_media(),
        )
    )
    medians = calculate_median_lifespans(
        obs, groupby=("experiment", "strain", "rdnacn", "media", "channel")
    )
    return medians


def calculate_median_lifespans(
    df, metric="bud_cnt", groupby=("experiment", "strain", "channel"), min_n=50
):
    """
    Take a dataframe containing lifespan predictions and calculate the median lifespan.
    args:
    -df <DataFrame> -- Table of results to summarize
    -metric <str> -- Column name to use for lifespan
    -groupby <tuple> -- Tuple containing column headers on how to group the dataframe.
    -min_n <int> -- Mininum number of predictions required to calculate a median.
    """

    rls = []
    groupby = list(groupby)
    assert metric in df.columns, f"{metric} is not a column in `df`!"
    for groupby_vals, sdf in df.groupby(groupby, sort=False):
        if sdf.shape[0] < min_n:
            print(f"Not sufficient data for {groupby_vals}")
            continue
        kmf = lifelines.KaplanMeierFitter()
        kmf.fit(sdf[metric], sdf.event_type.str.contains("x"))
        median = kmf.median_ if hasattr(kmf, "median_") else kmf._median
        low95, high95 = lifelines.utils.median_survival_times(
            kmf.confidence_interval_
        ).values[0]
        if len(groupby) == 1:
            groupby_vals = [groupby_vals]
        rls.append(
            dict(
                **dict(zip(groupby, groupby_vals)),
                **{"median": median, "lower_ci": low95, "upper_ci": high95},
            )
        )
    rls = pd.DataFrame(rls)[list(groupby) + ["median", "lower_ci", "upper_ci"]]
    return rls


def calculate_mode_of_death_hazards_for_all_strains():
    """Calculate the hazards of different"""
    obs = pd.concat(
        (
            load_lifespan_observations(),
            load_lifespan_observations_by(),
            load_lifespan_observations_media(),
        )
    )
    hazards = calculate_mode_of_death_hazards(
        obs,
        groupby=(
            "strain",
            "rdnacn",
            "media",
        ),
    )
    return hazards


def calculate_mode_of_death_hazards(
    df,
    metric="bud_cnt",
    groupby=("experiment", "strain", "rdnacn", "channel", "rdnacn"),
):
    """
    Take a dataframe containing lifespan predictions and calculate the the hazards of different
    modes-of-death.
    args:
    -df <DataFrame> -- Table of results to summarize
    -metric <str> -- Column name to use for lifespan
    -groupby <tuple> -- Tuple containing column headers on how to group the dataframe.
    """
    all_data = []
    groupby = list(groupby)
    it = tqdm(df.groupby(groupby)) if tqdm is not None else df.groupby(groupby)
    for groupby_vals, sdf in it:
        if len(groupby) == 1:
            groupby_vals = [groupby_vals]
        for event_type in ("elongated_death", "round_death", "any_death"):
            all_data.append(
                _fit_hazard(
                    durations=sdf[metric],
                    observed=(
                        sdf.event_type.str.contains("x")
                        if event_type == "any_death"
                        else sdf.mode_of_death_prediction == event_type[0]
                    ),
                )
                .assign(**dict(zip(groupby, groupby_vals)), hazard_of=event_type)
                .reset_index()
            )
    return pd.concat(all_data, axis=0)


def _fit_hazard(durations, observed):
    """Fit hazards with the NelsonAllanFitter and
    return the smoothed hazards and the CIs.
    """
    naf = lifelines.NelsonAalenFitter()
    naf.fit(durations=durations, event_observed=observed)
    return pd.concat(
        (
            naf.event_table["at_risk"],
            naf.smoothed_hazard_(5),
            naf.smoothed_hazard_confidence_intervals_(5),
        ),
        axis=1,
    )
