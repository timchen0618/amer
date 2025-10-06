#!/bin/bash

# ## Linear Data Generation
# PYTHONPATH=. python src_data_generation/generate_linear.py --dimensions 1024 --train-queries 20000 --test-queries 1000 --corpus-size 1000000  --output-dir data/linear
# PYTHONPATH=. python src_data_generation/generate_linear.py --dimensions 1024 --train-queries 20000 --test-queries 1000 --corpus-size 1000000 --multiple-query-distributions --output-dir data/linear_multi_query
# PYTHONPATH=. python src_data_generation/generate_linear.py --dimensions 1024 --train-queries 20000 --test-queries 1000 --corpus-size 1000000 --multiple-query-distributions --ood-distribution --output-dir data/linear_ood
## MLP Data Generation
PYTHONPATH=. python src_data_generation/generate_mlps.py --dimensions 1024 --train-queries 20000 --test-queries 1000 --corpus-size 1000000 --output-dir data/mlps
PYTHONPATH=. python src_data_generation/generate_mlps.py --dimensions 1024 --train-queries 20000 --test-queries 1000 --corpus-size 1000000 --multiple-query-distributions --output-dir data/mlps_multi_query
PYTHONPATH=. python src_data_generation/generate_mlps.py --dimensions 1024 --train-queries 20000 --test-queries 1000 --corpus-size 1000000 --multiple-query-distributions --ood-distribution --output-dir data/mlps_ood

# Create input dataset
cd ../
python create_input_dataset.py gaussian_synthetic