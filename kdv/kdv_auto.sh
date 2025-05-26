#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --partition=p100,4090
#SBATCH -J kdv_auto
#SBATCH --output=/data/home/liuyulong/auto_discovery/kdv/%x-%j.out  # 标准输出文件
#SBATCH --error=/data/home/liuyulong/auto_discovery/kdv/%x-%j.err   # 标准错误输出文件

# 激活 Conda 环境
source /data/home/liuyulong/miniconda3/bin/activate base

# 启动Python脚本
srun -n 1 python /data/home/liuyulong/auto_discovery/kdv/kdv.py