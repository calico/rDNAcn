""" Tool for creating scripts to align fastqs

This is provided mostly for documenation and might need considerable adaptation
to get it to work in another environment.
"""
import os
import subprocess

import pandas as pd
import xmltodict


def complete_template(path, variables):
    """ Open a template file @ `path` and complete with keys in
    `variables`<dict>
    """
    with open(path, "r", encoding="ascii") as template_file:
        template_text = template_file.read()
    completed_text = template_text.format(**variables)
    return completed_text


class Pipeline:
    """ Tools for generating scripts for WGS analysis compatible with
    SLURM computing clusters.
    """

    # pylint: disable=too-many-instance-attributes
    # This is only complicate the code to fix this right now

    def __init__(self, working_dir=None, bcl_dir=None, sample_sheet=None):
        """ C"""

        self.nthread = 4
        self.ref_genome = "/home/adamw/seq_data/ref/SacCer3.fasta"
        self.email = "nate@calicolabs.com"

        self.working_dir = working_dir
        self.bcl_dir = bcl_dir
        self.sample_sheet = sample_sheet  ## relative to wd or absolute
        self.slurm_job_id_dmux = None
        self.run_params = None
        self.slurm_job_id_align = None
        self.dmux_held = False
        self.fastq_dir = None

        self.sample_sheet_df = pd.DataFrame(
            pd.read_csv(
                self._get_samplesheet_fullpath(), skiprows=self._infer_headerlines()
            )
        )
        self.sample_sheet_df["sample_number"] = range(
            1, self.sample_sheet_df.shape[0] + 1
        )
        self.sample_sheet_df = self.sample_sheet_df.set_index("sample_number")

    def demux(self, start_options=None, wait=False):
        """ Create and submit the jobs to demux the sample. """
        self._dmux_create_folders()
        self._dmux_create_jobs()
        self._dmux_submit_jobs(start_options, wait)

    def align(self):
        """ Create and submit the jobs to align the samples. """
        self._handle_split_lane_samples()
        self._organize_fastqs()
        self._align_create_jobs()
        self._align_submit_jobs()

    def _handle_split_lane_samples(self):
        """ Handle if the same sample was split into two different lanes.

        We only need one reference of the sample (unique 'Sample_ID') in the
        sample sheet for alignement, as they should have been demuxed into a
        one set of fastq files (with our setting).
        """
        self.sample_sheet_df = self.sample_sheet_df.drop_duplicates("Sample_ID")
        self.sample_sheet_df = (
            self.sample_sheet_df.reset_index()
            .assign(sample_number=range(1, self.sample_sheet_df.shape[0] + 1))
            .set_index("sample_number")
        )

    def _dmux_create_folders(self):
        """ Create folders for demux output """
        self.fastq_dir = os.path.join(self.working_dir, "fastqs")
        os.makedirs(self.fastq_dir, exist_ok=True)
        os.makedirs(os.path.join(self.fastq_dir, "Reports"), exist_ok=True)
        os.makedirs(os.path.join(self.fastq_dir, "Stats"), exist_ok=True)

    def _dmux_create_jobs(self):
        """ Create the jobs scripts for demuxing these samples """
        args = dict(
            working_dir=self.working_dir,
            bcl_dir=self.bcl_dir,
            email=self.email,
            sample_sheet=self.sample_sheet,
            read_length=self._get_read_length(),
            index_read_length=self._get_index_read_length(),
        )
        dest = os.path.join(self.working_dir, "demux.sbatch")
        template_path = os.path.join(
            os.path.dirname(__file__), "templates", "demux.sbatch.tmp"
        )
        with open(dest, "w", encoding="ascii") as tmp:
            tmp.write(complete_template(template_path, args))

    def _dmux_submit_jobs(self, start_options=None, wait=False):
        """ Submit the jobs to to the cluster. """
        start_options = getattr(self, "dmux_job_start_options", "")
        cmd = ["sbatch"]
        if isinstance(start_options, str) and start_options:
            cmd.append(start_options)
        if wait:
            cmd.append("--hold")
            dmux_held = True
        else:
            dmux_held = False
        cmd += ["--parsable", os.path.join(self.working_dir, "demux.sbatch")]
        self.slurm_job_id_dmux = (
            subprocess.run(cmd, stdout=subprocess.PIPE, check=False)
            .stdout.decode("ascii")
            .strip()
        )
        print("CMD submitted:", " ".join(cmd))
        if dmux_held:
            dest = os.path.join(self.working_dir, "release_demux.sh")
            subprocess.run(["nohup", "bash", dest, "&"], check=False)
        return self.slurm_job_id_dmux

    def read_run_params(self):
        """ Read run params to find the settings to demux the samples. """
        with open(os.path.join(self.bcl_dir, "RunInfo.xml"), encoding="ascii") as fin:
            self.run_params = xmltodict.parse(fin.read())
        return self.run_params

    def _get_read_length(self):
        """ Infer the read length """
        # return 150
        if not hasattr(self, "run_params"):
            self.read_run_params()
        return int(self.run_params["RunInfo"]["Run"]["Reads"]["Read"][0]["@NumCycles"])

    def _get_index_read_length(self):
        """ Infer the index read length """
        if not hasattr(self, "run_params"):
            self.read_run_params()
        read_length = int(
            self.run_params["RunInfo"]["Run"]["Reads"]["Read"][1]["@NumCycles"]
        )
        barcode_length = len(self.sample_sheet_df["index"].iloc[0])
        assert read_length >= barcode_length, "Incorrect run settings detected!!"
        if read_length != barcode_length:
            read_length = f"{barcode_length}n{read_length-barcode_length}"
        return read_length

    def _organize_fastqs(self):
        """ Organize fastqs into where our utility expects them to be. In each
            sample folder -- achieving this with a symlink.
        """
        for sample_number, row in self.sample_sheet_df.iterrows():
            for read in ["R1", "R2"]:
                fastq_file_name = (
                    f"{row.Sample_Name}_S{sample_number}_{read}_001.fastq.gz"
                )
                path = [self.working_dir, "fastqs"]
                project = getattr(row, "Sample_Project", None)
                if project is not None:
                    path.append(project)
                sample_id = getattr(row, "Sample_ID", None)
                if sample_id is not None:
                    path.append(sample_id)
                path.append(fastq_file_name)
                src = os.path.join(*path)
                os.makedirs(os.path.dirname(src), exist_ok=True)
                if not os.path.exists(src):  ## touch the file so we can make symlinks
                    with open(src, "a", encoding="ascii"):
                        pass
                dst_sample_folder = os.path.join(self.working_dir, f"{sample_number}")
                os.makedirs(dst_sample_folder, exist_ok=True)
                dst = os.path.join(dst_sample_folder, fastq_file_name)
                if os.path.islink(dst):
                    os.remove(dst)
                os.symlink(src=src, dst=dst)

    def _infer_headerlines(self):
        """ Infer the number of header lines in the sample sheet """
        with open(self._get_samplesheet_fullpath(), encoding="ascii") as sample_sheet:
            for i, line in enumerate(sample_sheet.readlines()):
                if "[Data]" in line:
                    return i + 1
        return 0

    def _get_samplesheet_fullpath(self):
        """ Test if the path provided was enough to find the file. Otherwise
        look in the specific working directory
        """
        return (
            self.sample_sheet
            if os.path.isfile(self.sample_sheet)
            else os.path.join(self.working_dir, self.sample_sheet)
        )

    def generate_sample_file_names(self, sample_number, row, include_lane=False):
        """ Given a row of a dataframe -- generate the expected file names. """
        pfx = os.path.join(self.working_dir, f"{sample_number}", f"{sample_number}")
        fastq_f = f"{row.Sample_Name}_S{sample_number}_R1_001.fastq.gz"
        fastq_r = f"{row.Sample_Name}_S{sample_number}_R2_001.fastq.gz"
        if include_lane:
            fastq_f = (
                f"{row.Sample_Name}_S{sample_number}_L{row.Lane:03d}_R1_001.fastq.gz"
            )
            fastq_r = (
                f"{row.Sample_Name}_S{sample_number}_L{row.Lane:03d}_R2_001.fastq.gz"
            )
        f_fastq = os.path.join(self.working_dir, f"{sample_number}", fastq_f)
        r_fastq = os.path.join(self.working_dir, f"{sample_number}", fastq_r)
        return dict(
            f_fastq=f_fastq,
            r_fastq=r_fastq,
            f_fastq_trimmed=f_fastq.replace(".fastq.gz", "_trimmed.fastq"),
            r_fastq_trimmed=r_fastq.replace(".fastq.gz", "_trimmed.fastq"),
            singles=f"{pfx}_singles.fastq",
            bam=f"{pfx}_aligned.bam",
            sorted_bam=f"{pfx}_sorted.bam",
            depth=f"{pfx}_sorted.depth",
        )

    def _align_create_jobs(self, include_lane=False):
        """ Create the job scripts for aligning the reads """
        order = [
            "f_fastq",
            "r_fastq",
            "f_fastq_trimmed",
            "r_fastq_trimmed",
            "singles",
            "bam",
            "sorted_bam",
            "depth",
        ]
        pd.DataFrame(
            [
                self.generate_sample_file_names(sample_number, row, include_lane)
                for sample_number, row in self.sample_sheet_df.iterrows()
            ]
        )[order].to_csv(
            os.path.join(self.working_dir, "align_file_names.tsv"),
            sep="\t",
            header=False,
            index=False,
        )
        dest = os.path.join(self.working_dir, "align.sbatch")
        template_path = os.path.join(
            os.path.dirname(__file__), "templates", "align.sbatch.tmp"
        )
        args = dict(
            email=self.email,
            wd=self.working_dir,
            nthread=self.nthread,
            n_samples=self.sample_sheet_df.shape[0],
            ref_location=self.ref_genome,
            filename_file=os.path.join(self.working_dir, "align_file_names.tsv"),
        )
        with open(dest, "w", encoding="ascii") as tmp:
            tmp.write(complete_template(template_path, args))
        os.makedirs(os.path.join(self.working_dir, "job_output"), exist_ok=True)

    def _align_submit_jobs(self):
        """ Submit the align jobs to the cluster """
        sbatch = os.path.join(self.working_dir, "align.sbatch")
        cmd = ["sbatch", "--parsable"]
        demux_job_id = getattr(self, "slurm_job_id_dmux", None)
        if demux_job_id is not None:
            cmd.append(f"--dependency=afterok:{demux_job_id}")
        cmd.append(sbatch)
        self.slurm_job_id_align = (
            subprocess.run(cmd, stdout=subprocess.PIPE, check=False)
            .stdout.decode("ascii")
            .strip()
        )
