import json
import sys

suffix = sys.argv[1]
data_type = sys.argv[2]
template = f"sbatch_configs/eval/retrieve_{data_type}_template.sh"
template = open(template, "r").read()


with open(f"sbatch_configs/eval/retrieve_{data_type}.sh", "w") as f:
    f.write(template.replace("[suffix]", suffix))