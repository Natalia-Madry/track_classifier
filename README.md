# Track Classification
Graph-based classification of particle tracks in the CERN LHCb experiment.
The goal of this project is to distinguish real downstream tracks from fake (ghost) tracks using graph neural networks.

Each single track is represented as a graph and classified as:

- 1 - real (true) track 
- 0 - fake (ghost) track

## Requirements

- Python 3.8+  
- PyTorch  
- PyTorch Geometric  
- NumPy  
- Pandas  
- Matplotlib  

## Install dependencies:
pip install torch torch-geometric numpy pandas matplotlib

## How to Run:
### Run training and evaluation:

python3 train.py


## Output

- Trained GNN classifier
- Probability distributions for real and fake tracks
- Classification metrics (precision, recall, confusion matrix)
