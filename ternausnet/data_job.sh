# Runs the scripts necessary to go from the labeled data to a test-train-val split of original images and their masks

#!/bin/sh
python3 unorder_data.py
python3 label_to_mask.py
python3 prepare_data.py
