"""Script for submitting jobs."""

import os
import subprocess
import sys

MAIN_FILE = "algorithms/iql.py"
START_SEED = 456
NUM_SEEDS = 10
NUM_GPUS = 1
NUM_CPUS_PER_TASK = 16
PARTITION = "learnai4rl"


def _main() -> None:
    log_dir = "logs/"
    logfile_prefix = "corl__None"
    job_name = logfile_prefix
    args_and_flags_str = " ".join(sys.argv[1:])
    return _submit_job(job_name, log_dir, logfile_prefix, args_and_flags_str)


def _submit_job(job_name: str, log_dir: str, logfile_prefix: str,
                args_and_flags_str: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logfile_pattern = os.path.join(log_dir, f"{logfile_prefix}__%j.log")
    assert logfile_pattern.count("None") == 1
    logfile_pattern = logfile_pattern.replace("None", "%a")
    mystr = (f"#!/bin/bash\npython {MAIN_FILE} {args_and_flags_str} "
             f"--seed $SLURM_ARRAY_TASK_ID")
    temp_run_file = "temp_run_file.sh"
    assert not os.path.exists(temp_run_file)
    with open(temp_run_file, "w", encoding="utf-8") as f:
        f.write(mystr)
    cmd = (f"sbatch --time=99:00:00 --partition={PARTITION} "
           f"--gres=gpu:{NUM_GPUS} --cpus-per-task {NUM_CPUS_PER_TASK} "
           f"--job-name={job_name} --array={START_SEED}-{START_SEED+NUM_SEEDS-1} "
           f"-o {logfile_pattern} {temp_run_file}")
    print(f"Running command: {cmd}")
    output = subprocess.getoutput(cmd)
    os.remove(temp_run_file)
    if "command not found" in output:
        raise Exception("Are you logged into the cluster?")


if __name__ == "__main__":
    _main()
