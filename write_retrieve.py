import json
import sys

suffix = sys.argv[1]
data_type = sys.argv[2]
if len(sys.argv) > 3:
    data_name = sys.argv[3]
    assert data_type == 'berds'
    assert data_name in ["arguana_generated", "kialo", "opinionqa"]
template = f"sbatch_configs/eval/retrieve_{data_type}_template.sh"
template = open(template, "r").read()


with open(f"sbatch_configs/eval/retrieve_{data_type}.sh", "w") as f:
    if data_type == 'berds':
        f.write(template.replace("[suffix]", suffix).replace("[data_name]", data_name))
    else:
        f.write(template.replace("[suffix]", suffix))