# Interpretable-Sequence-Clustering-Tree

Source Code for Interpretable sequence clustering https://www.sciencedirect.com/science/article/pii/S0020025524013677



## Running


We added the tree structure visualization, to use it on linux please first 

```
sudo apt-get install graphviz
export PATH=$PATH:/usr/local/bin
source ~/.bashrc
```
We recommend using Pypy, which may give exponential speedups on larger datasets

## Usage

### Random Projection on Python version

```
pip install ISCT
```


<!-- code -->
```python

from ISCT import ISCT
sequences = [
    ['a', 'g', 't', 't', 'c'], 
    ['a', 't', 'g', 'g', 't', 'u', 't'], 
    ['a', 'c', 't', 'u', 'u', 'a', 'a'], 
    ['a', 'c', 'a', 'u', 'a', 't', 'c', 't'], 
    ['a', 'g', 'g', 'c', 'a', 'a', 'c'], 
    ['a', 'c', 'g', 'g', 'c', 'c', 'a', 'a']]
isct = ISCT(num_clusters = 3, visulization_name = 'test', min_sample=2) # if visulization_name is provided, it generates the "test.pdf".
y_pred = isct.fit(sequences)
print(y_pred)
```


<!-- 
### Random Projection  CPP version

1. First, compile the cpp file on your Linux in order to generate the fast Random Projection Generator.
2. Run ISCT_cpp.py to get the clustering results -->





## Dependencies
- Python 3.9.16 (Pypy)
- NumPy 1.24.3
- Scikit-learn 1.2.2
- Pandas 2.0.1
- Prefixspan 0.5.2


## Visualization

ISCT could provide you with a highly concise and short clustering tree, taking poineer as example:
<img width="408" alt="image" src="https://github.com/jd445/Interpretable-Sequence-Clustering-Tree/assets/65555729/5a0a465f-0d7d-4d5c-a149-9ceb927abed9">
