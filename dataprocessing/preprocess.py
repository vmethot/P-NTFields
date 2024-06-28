import sys

sys.path.append('.')

import multiprocessing as mp
import os
from glob import glob
from multiprocessing import Pool

import numpy as np

import configs.config_loader as cfg_loader
import dataprocessing.voxelized_pointcloud_sampling as voxelized_pointcloud_sampling
from dataprocessing.convert_to_scaled_off import to_off
from dataprocessing.speed_sampling_gpu import sample_speed

cfg = cfg_loader.get_config()

def multiprocess(func, paths):
    if cfg.num_cpus == -1:
        num_cpus = mp.cpu_count()
    else:
        num_cpus = cfg.num_cpus
    p = Pool(num_cpus)
    p.map(func, paths)
    p.close()
    p.join()

def main():
    print(cfg.data_dir)
    print(cfg.input_data_glob)

    print('Finding raw files for preprocessing.')
    paths = glob( "./"+cfg.data_dir + cfg.input_data_glob)
    print(paths)
    paths = sorted(paths)

    chunks = np.array_split(paths,cfg.num_chunks)
    paths = chunks[cfg.current_chunk]

    print('Start scaling.')
    multiprocess(to_off, paths)

    print('Start speed sampling.')
    for path in paths:
        print(path)
        sample_speed(path, cfg.num_samples, cfg.num_dim)

    print('Start voxelized pointcloud sampling.')
    voxelized_pointcloud_sampling.init(cfg)
    multiprocess(voxelized_pointcloud_sampling.voxelized_pointcloud_sampling, paths)

if __name__ == '__main__':
    main()