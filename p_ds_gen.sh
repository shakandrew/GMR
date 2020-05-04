#!/usr/bin/env bash
python parallel_ds_creation.py tracks.csv 0 400 train01 training &
python parallel_ds_creation.py tracks.csv 400 800 train02 training &
python parallel_ds_creation.py tracks.csv 800 1200 train03 training &
python parallel_ds_creation.py tracks.csv 1200 1600 train04 training &
python parallel_ds_creation.py tracks.csv 1600 2000 train05 training &
python parallel_ds_creation.py tracks.csv 2000 2400 train06 training &
python parallel_ds_creation.py tracks.csv 2400 2800 train07 training &
python parallel_ds_creation.py tracks.csv 2800 3200 train08 training &
python parallel_ds_creation.py tracks.csv 3200 3600 train09 training &
python parallel_ds_creation.py tracks.csv 3600 4000 train10 training &
python parallel_ds_creation.py tracks.csv 4000 4400 train11 training &
python parallel_ds_creation.py tracks.csv 4400 4800 train12 training &
python parallel_ds_creation.py tracks.csv 4800 5200 train13 training &
python parallel_ds_creation.py tracks.csv 5200 5600 train14 training &
python parallel_ds_creation.py tracks.csv 5600 6000 train15 training &
python parallel_ds_creation.py tracks.csv 6000 6400 train16 training &

python parallel_ds_creation.py tracks.csv 0 800 valid01 valid &
python parallel_ds_creation.py tracks.csv 0 800 test01 test &
