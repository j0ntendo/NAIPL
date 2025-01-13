#!/bin/bash

# Enable persistence mode for all GPUs
sudo nvidia-smi -pm 1

# Set the power limit for each GPU
sudo nvidia-smi -i 0 -pl 250  # Set GPU0 to 250 Watts
sudo nvidia-smi -i 1 -pl 250  # Set GPU1 to 250 Watts
#!/bin/bash

# Enable persistence mode for all GPUs
sudo nvidia-smi -pm 1

# Set the power limit for each GPU
# Replace <power_limit_gpu0> and <power_limit_gpu1> with the desired power limits in watts.
# Example: For GPU0, you might set it to 250 watts, and for GPU1, perhaps 250 watts as well.

sudo nvidia-smi -i 0 -pl <power_limit_gpu0>
sudo nvidia-smi -i 1 -pl <power_limit_gpu1>
