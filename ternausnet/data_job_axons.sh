# Runs the scripts necessary to go from the labeled data to a test-train-val split 
# of original images and their masks (with axons)

#!/bin/sh
python3 label_to_mask_axons.py
python3 prepare_data.py --num_classes 2