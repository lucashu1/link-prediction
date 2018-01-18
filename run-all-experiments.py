import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import link_prediction_scores as lp
import pickle
import os

NUM_REPEATS = 10
RANDOM_SEED = 0
FRAC_EDGES_HIDDEN = [0.25, 0.5, 0.75]

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



### ---------- Create Random NetworkX Graphs ---------- ###
# Dictionary to store all nx graphs
nx_graphs = {}

# Small graphs
N_SMALL = 200
nx_graphs['er-small'] = nx.erdos_renyi_graph(n=N_SMALL, p=.02, seed=RANDOM_SEED) # Erdos-Renyi
nx_graphs['ws-small'] = nx.watts_strogatz_graph(n=N_SMALL, k=5, p=.1, seed=RANDOM_SEED) # Watts-Strogatz
nx_graphs['ba-small'] = nx.barabasi_albert_graph(n=N_SMALL, m=2, seed=RANDOM_SEED) # Barabasi-Albert
nx_graphs['pc-small'] = nx.powerlaw_cluster_graph(n=N_SMALL, m=2, p=.02, seed=RANDOM_SEED) # Powerlaw Cluster
nx_graphs['sbm-small'] = nx.random_partition_graph(sizes=[N_SMALL/10]*10, p_in=.1, p_out=.01, seed=RANDOM_SEED) # Stochastic Block Model

# Larger graphs
N_LARGE = 1000
nx_graphs['er-large'] = nx.erdos_renyi_graph(n=N_LARGE, p=.01, seed=RANDOM_SEED) # Erdos-Renyi
nx_graphs['ws-large'] = nx.watts_strogatz_graph(n=N_LARGE, k=3, p=.1, seed=RANDOM_SEED) # Watts-Strogatz
nx_graphs['ba-large'] = nx.barabasi_albert_graph(n=N_LARGE, m=2, seed=RANDOM_SEED) # Barabasi-Albert
nx_graphs['pc-large'] = nx.powerlaw_cluster_graph(n=N_LARGE, m=2, p=.02, seed=RANDOM_SEED) # Powerlaw Cluster
nx_graphs['sbm-large'] = nx.random_partition_graph(sizes=[N_LARGE/10]*10, p_in=.05, p_out=.005, seed=RANDOM_SEED) # Stochastic Block Model

# Remove isolates from random graphs
for g_name, nx_g in nx_graphs.iteritems():
    isolates = nx.isolates(nx_g)
    if len(isolates) > 0:
        for isolate_node in isolates:
            nx_graphs[g_name].remove_node(isolate_node)



### ---------- Run Link Prediction Tests ---------- ###
for i in range(NUM_REPEATS):

    ## ---------- FACEBOOK ---------- ###
    fb_results = {}

    # Check existing experiment results, increment file number by 1
    past_results = os.listdir('./results/')
    experiment_num = 0
    experiment_file_name = 'fb-experiment-{}-results.pkl'.format(experiment_num)
    while (experiment_file_name in past_results):
        experiment_num += 1
        experiment_file_name = 'fb-experiment-{}-results.pkl'.format(experiment_num)

    FB_RESULTS_DIR = './results/' + experiment_file_name
    TRAIN_TEST_SPLITS_FOLDER = './train-test-splits/'

    # Iterate over fractions of edges to hide
    for frac_hidden in FRAC_EDGES_HIDDEN:
        val_frac = 0.1
        test_frac = frac_hidden - val_frac
        
        # Iterate over each graph
        for g_name, graph_tuple in fb_graphs.iteritems():
            adj = graph_tuple[0]
            feat = graph_tuple[1]
            
            experiment_name = 'fb-{}-{}-hidden'.format(g_name, frac_hidden)
            print "Current experiment: ", experiment_name

            # # TODO: remove this!
            # if experiment_name !='fb-combined-0.25-hidden' and \
            #     experiment_name != 'fb-combined-0.5-hidden' and \
            #     experiment_name != 'fb-combined-0.75-hidden':
            #     continue

            train_test_split_file = TRAIN_TEST_SPLITS_FOLDER + experiment_name + '.pkl'
            
            # Run all link prediction methods on current graph, store results
            fb_results[experiment_name] = lp.calculate_all_scores(adj, feat, \
                                                         test_frac=test_frac, val_frac=val_frac, \
                                                         random_state=RANDOM_SEED, verbose=2,
                                                         train_test_split_file=train_test_split_file)

            # Save experiment results at each iteration
            with open(FB_RESULTS_DIR, 'wb') as f:
                pickle.dump(fb_results, f, protocol=2)
            
    # Save final experiment results
    with open(FB_RESULTS_DIR, 'wb') as f:
        pickle.dump(fb_results, f, protocol=2)



    ### ---------- NETWORKX ---------- ###
    nx_results = {}

    # Check existing experiment results, increment file number by 1
    past_results = os.listdir('./results/')
    experiment_num = 0
    experiment_file_name = 'nx-experiment-{}-results.pkl'.format(experiment_num)
    while (experiment_file_name in past_results):
        experiment_num += 1
        experiment_file_name = 'nx-experiment-{}-results.pkl'.format(experiment_num)

    NX_RESULTS_DIR = './results/' + experiment_file_name

    # Iterate over fractions of edges to hide
    for frac_hidden in FRAC_EDGES_HIDDEN:
        val_frac = 0.1
        test_frac = frac_hidden - val_frac
        
        # Iterate over each graph
        for g_name, nx_g in nx_graphs.iteritems():
            adj = nx.adjacency_matrix(nx_g)
            
            experiment_name = 'nx-{}-{}-hidden'.format(g_name, frac_hidden)
            print "Current experiment: ", experiment_name
            
            # Run all link prediction methods on current graph, store results
            nx_results[experiment_name] = lp.calculate_all_scores(adj, \
                                                         test_frac=test_frac, val_frac=val_frac, \
                                                         random_state=RANDOM_SEED, verbose=0)

            # Save experiment results at each iteration
            with open(NX_RESULTS_DIR, 'wb') as f:
                pickle.dump(nx_results, f, protocol=2)
            
    # Save final experiment results
    with open(NX_RESULTS_DIR, 'wb') as f:
        pickle.dump(nx_results, f, protocol=2)

