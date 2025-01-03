# Interpretable-Sequence-Clustering-Tree

Source Code for Interpretable sequence clustering https://www.sciencedirect.com/science/article/pii/S0020025524013677

## Running
1. First, compile the cpp file on your Linux in order to generate the fast Random Projection Generator.
2. Run ISCT to get the clustering results

We recommend using Pypy, which may give exponential speedups on larger datasets

We added the tree structure visualization, to use it on linux please first 

```
sudo apt-get install graphviz
export PATH=$PATH:/usr/local/bin
source ~/.bashrc
```


## Dependencies
- Python 3.9.16 (Pypy)
- NumPy 1.24.3
- Scikit-learn 1.2.2
- Pandas 2.0.1
- Prefixspan 0.5.2


## Visualization

ISCT could provide you with a highly concise and short clustering tree, taking poineer as example:
<img width="408" alt="image" src="https://github.com/jd445/Interpretable-Sequence-Clustering-Tree/assets/65555729/5a0a465f-0d7d-4d5c-a149-9ceb927abed9">
