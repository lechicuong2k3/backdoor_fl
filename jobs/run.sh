#!/bin/bash

# you have to cd to your workdir first
cd /vinserver_user/21thinh.dd/FedBackdoor/source

# another thing, you have to specify the full path of your python (you can determine this by running `which python` in the sandbox container
/home/admin/miniconda3/envs/21thinh.dd/bin/python -u run_centralized.py