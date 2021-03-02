#!/usr/bin/env bash
python3 driver_script.py "awb/arctic_a0094" hyper_cmp 2>/dev/null
python3 driver_script.py "bdl/arctic_a0017" hyper_cmp 2>/dev/null
python3 driver_script.py "jmk/arctic_a0067" hyper_cmp 2>/dev/null
python3 driver_script.py "rms/arctic_a0382" hyper_cmp 2>/dev/null
python3 driver_script.py "slt/arctic_b0041" hyper_cmp 2>/dev/null

python3 driver_script.py "awb/arctic_a0094" hyper_free 2>/dev/null
python3 driver_script.py "bdl/arctic_a0017" hyper_free 2>/dev/null
python3 driver_script.py "jmk/arctic_a0067" hyper_free 2>/dev/null
python3 driver_script.py "rms/arctic_a0382" hyper_free 2>/dev/null
python3 driver_script.py "slt/arctic_b0041" hyper_free 2>/dev/null

 # Hyper files must contain "hyper"; all other files are input files
python3 post_script.py \
    hyper_cmp hyper_free \
    "awb/arctic_a0094" \
    "bdl/arctic_a0017" \
    "jmk/arctic_a0067" \
    "rms/arctic_a0382" \
    "slt/arctic_b0041"
