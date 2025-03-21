#!/bin/bash
echo "---Setting up the environment of VidKV---"
# Install required Python packages
# pip install lmms-eval
# pip instll seaborn
# # Do not remove the following
# pip uninstall lmms-eval

# Install required Transformers with our vidkv
cd transformers
pip install .
cd ..

echo "Environment setup completed successfully!"
