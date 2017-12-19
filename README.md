Social Network VAE
============

This is a TensorFlow implementation of the (Variational) Graph Auto-Encoder model as described in Thomas Kipf's paper: T. N. Kipf, M. Welling, [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) (2016)

**In this repository, we are interested in applying GAEs to social networks,
primarily for the task of link prediction.** We compare the Graph Auto-Encoder link prediction results to those of existing methods: baseline indexes (Adamic-Adar, Jaccard Coefficient, Preferential Attachment), spectral clustering, and [node2vec](http://snap.stanford.edu/node2vec/).

### Background:

Graph Auto-Encoders (GAEs) are end-to-end trainable neural network models for unsupervised learning, clustering and link prediction on graphs. 

![(Variational) Graph Auto-Encoder](figure.png)

GAEs are based on Graph Convolutional Networks (GCNs), a recent class of models for end-to-end (semi-)supervised learning on graphs:

T. N. Kipf, M. Welling, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), ICLR (2017). 

A high-level introduction is given in Thomas Kipf's blog post:

Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)

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
cd gae
python setup.py install
```


## Included Files

### Annotated IPython Notebooks
* `link-prediction-baselines.ipynb`: Adamic-Adar, Jaccard Coefficient, Preferential Attachment
* `spectral-clustering.ipynb`: Using spectral embeddings for link prediction
* `node2vec.ipynb`: skip-gram based representation learning for node/edge embeddings
* `graph-vae.ipynb`: (variational) graph autoencoder, learns node embeddings

### Link Prediction Experiment Scripts
* `link_prediction_scores.py`: various functions for running link prediction tests

## Data

In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), and
* an N by D feature matrix (D is the number of features per node) -- optional

Have a look at the included notebooks/scripts for usage examples.

In this forked repository, we load social network data (Facebook). The original dataset
can be found here: https://snap.stanford.edu/data/egonets-Facebook.html.

The original GAE code also loads citation network data (Cora, Citeseer or Pubmed). The original datasets can be found here: http://linqs.cs.umd.edu/projects/projects/lbc/ and here (in a different format): https://github.com/kimiyoung/planetoid
