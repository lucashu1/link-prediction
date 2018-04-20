from __future__ import print_function
import networkx as nx
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from math import log

##################################
######### READ EDGE LIST #########
##################################

print('Reading edgelist')

# Read combined edge-list
twitter_edges_dir = './twitter/twitter_combined.txt'
edges_f = open(twitter_edges_dir)

# Parse edgelist into directed graph
twitter_g = nx.read_edgelist(edges_f, nodetype=int, create_using=nx.DiGraph())
print('Num. weakly connected components: ', nx.number_weakly_connected_components(twitter_g))

# print('Saving adjacency matrix')

# Get adjacency matrix
adj = nx.adjacency_matrix(twitter_g)

# Save adjacency matrix
with open('./twitter/twitter-combined-adj.pkl', 'wb') as f:
    pickle.dump(adj, f)



##################################
##### VISUALIZATIONS, STATS ######
##################################

# Generate visualization
def save_visualization(g, file_name, title):
    plt.figure(figsize=(18,18))
    degrees = g.in_degree()
    
    # Draw networkx graph -- scale node size by log(degree+1)
    nx.draw_spring(g, with_labels=False, 
                   linewidths=2.0,
                   nodelist=degrees.keys(),
                   node_size=[log(degree_val+1) * 100 for degree_val in degrees.values()], \
                   node_color='r',
                   arrows=True)
    
    # Create black border around node shapes
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000")
    
#     plt.title(title)
    plt.savefig(file_name)
    plt.clf()

# Save network stats to .txt file
def save_network_statistics(g):
    stats = {}
    stats['num_weakly_connected_components'] = nx.number_weakly_connected_components(g)
    stats['num_strongly_connected_components'] = nx.number_strongly_connected_components(g)
    stats['num_nodes'] = nx.number_of_nodes(g)
    stats['num_edges'] = nx.number_of_edges(g)
    stats['density'] = nx.density(g)
    try:
        stats['avg_clustering_coef'] = nx.average_clustering(g)
    except:
        stats['avg_clustering_coef'] = None # not defined for directed graphs
    stats['avg_degree'] = sum(g.degree().values()) / float(stats['num_nodes'])
    stats['transitivity'] = nx.transitivity(g)
    try:
        stats['diameter'] = nx.diameter(g)
    except:
        stats['diameter'] = None # unconnected --> infinite path length between connected components

    with open('./network-statistics/twitter-combined-statistics.txt', 'wb') as f:
        for stat_name, stat_value in stats.iteritems():
            f.write(stat_name + ': ' + str(stat_value) + '\n')

print('Generating network visualization')
# save_visualization(g=twitter_g, file_name='./visualizations/twitter-combined-visualization.pdf', title='Twitter Combined Network')

print('Calculating and saving network statistics')
# save_network_statistics(twitter_g)
