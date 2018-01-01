Link Prediction Experiments
============

**This repository contains a series of machine learning experiments for [link prediction](https://www.cs.cornell.edu/home/kleinber/link-pred.pdf) within social networks.** We first implement and apply a variety of link prediction methods to each of the ego networks contained within the [SNAP Facebook dataset](https://snap.stanford.edu/data/egonets-Facebook.html) and to various [random networks](https://networkx.github.io/documentation/networkx-1.10/reference/generators.html) generated using [networkx](https://networkx.github.io/), and then calculate and compare the ROC AUC and Average Precision scores of each method.

### Link Prediction Methods Tested:
* [(Variational) Graph Auto-Encoders](https://arxiv.org/abs/1611.07308): An end-to-end trainable convolutional neural network model for unsupervised learning on graphs
* [Node2Vec/DeepWalk](http://snap.stanford.edu/node2vec/): A skip-gram based approach to learning node embeddings from random walks within a given graph
* [Spectral Clustering](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html): Using spectral embeddings to create node representations from an adjacency matrix
* [Baseline Indexes](https://networkx.github.io/documentation/networkx-1.10/reference/algorithms.link_prediction.html): Adamic-Adar, Jaccard Coefficient, Preferential Attachment


## Requirements
* python 2.7
* [TensorFlow](https://www.tensorflow.org/install/) (1.0 or later)
* [networkx](https://networkx.github.io/)
* [gensim](https://radimrehurek.com/gensim/install.html)
* [scikit-learn](http://scikit-learn.org/stable/)
* [scipy](https://www.scipy.org/_)
* [jupyter notebook](http://jupyter.org/install.html)

## Pre-Use Installation

```bash
python setup.py install
```


## Included Files

### Network Data
* `facebook/`: Original [Facebook ego networks](https://snap.stanford.edu/data/egonets-Facebook.html) dataset, with added .allfeats files (with both ego and alter features)
* `fb-processed/`: Pickle dumps of (adjacency_matrix, feature_matrix) tuples for each ego network, and for combined network
* `process-ego-networks.py`: Script used to process raw Facebook data and generate pickle dumps
* `process-combined-network.py`: Script used to combine Facebook ego networks and generate complete network pickle dump

### Annotated IPython Notebooks
* `link-prediction-baselines.ipynb`: Adamic-Adar, Jaccard Coefficient, Preferential Attachment
* `spectral-clustering.ipynb`: Using spectral embeddings for link prediction
* `node2vec.ipynb`: Skip-gram based representation learning for node/edge embeddings
* `graph-vae.ipynb`: (Variational) Graph Autoencoder, learns node embeddings to recreate adjacency matrix

### Link Prediction Helper Scripts
* `link_prediction_scores.py`: Utility functions for running various link prediction tests

### Full Link Prediction Experiments
* `nx-graph-experiments.ipynb`: Run all link prediction tests on various types of random networks (Erdos-Renyi, Barabasi-Albert, etc.)
* `fb-graph-experiments.ipynb`: Run all link prediction tests on each Facebook ego network
