#!/bin/bash

# Enable persistence mode for all GPUs
sudo nvidia-smi -pm 1

# Set the power limit for each GPU
sudo nvidia-smi -i 0 -pl 250  # Set GPU0 to 250 Watts
sudo nvidia-smi -i 1 -pl 250  # Set GPU1 to 250 Watts
