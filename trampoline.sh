#!/bin/bash

pip install prettytable
pip install terminaltables
pip install fastargs
pip install pandas
pip install pyyaml
pip install tqdm

python /home/c02hane/CISPA-projects/neuron_pruning-2024/lottery-ticket-harness/harness.py --config /home/c02hane/CISPA-projects/neuron_pruning-2024/lottery-ticket-harness/"$1"
