import networkx as nx
import pandas as pd
import pickle
import numpy as np
from gae.preprocessing import mask_test_edges

RANDOM_SEED = 0

### ---------- Load in FB Graphs ---------- ###
FB_EGO_USERS = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]
fb_graphs = {} # Dictionary to store all FB ego network graphs

# Read in each FB Ego graph
	# Store graphs in dictionary as (adj, features) tuples
for user in FB_EGO_USERS:
    network_dir = './fb-processed/{0}-adj-feat.pkl'.format(user)
    with open(network_dir, 'rb') as f:
        adj, features = pickle.load(f)
    
    # Store in dict
    fb_graphs[user] = (adj, features)
    
# Read in combined FB graph
combined_dir = './fb-processed/combined-adj-sparsefeat.pkl'
with open(combined_dir, 'rb') as f:
    adj, features = pickle.load(f)
    fb_graphs['combined'] = (adj, features)


### ---------- Generate Train-Test Splits ---------- ###
FRAC_EDGES_HIDDEN = [0.25, 0.5, 0.75]
TRAIN_TEST_SPLITS_FOLDER = './train-test-splits/'

# TODO = ['fb-combined-0.75-hidden']

# Iterate over fractions of edges to hide
for frac_hidden in FRAC_EDGES_HIDDEN:
    val_frac = 0.1
    test_frac = frac_hidden - val_frac
    
    # Iterate over each graph
    for g_name, graph_tuple in fb_graphs.iteritems():
        adj = graph_tuple[0]
        feat = graph_tuple[1]
        
        current_graph = 'fb-{}-{}-hidden'.format(g_name, frac_hidden)
        
        # if current_graph in TODO:
        print "Current graph: ", current_graph

        np.random.seed(RANDOM_SEED)
        
        # Run all link prediction methods on current graph, store results
        train_test_split = mask_test_edges(adj, test_frac=test_frac, val_frac=val_frac,
            verbose=True)

        file_name = TRAIN_TEST_SPLITS_FOLDER + current_graph + '.pkl'

        # Save split
        with open(file_name, 'wb') as f:
            pickle.dump(train_test_split, f, protocol=2)


