# Combine all ego networks (including node features) and store in (adj, features) tuple
# Adapted from: https://github.com/jcatw/snap-facebook

#!/usr/bin/env python
import networkx as nx
import numpy as np
import glob
import os, os.path
import math
import pickle
import scipy.sparse as sp

pathhack = os.path.dirname(os.path.realpath(__file__))

feat_file_name = "%s/facebook/feature_map.txt" % (pathhack,)
feature_index = {}  #numeric index to name
inverted_feature_index = {} #name to numeric index
network = nx.Graph()
ego_nodes = []

def parse_featname_line(line):
    line = line[(line.find(' '))+1:]  # chop first field
    split = line.split(';')
    name = ';'.join(split[:-1]) # feature name
    index = int(split[-1].split(" ")[-1]) #feature index
    return index, name

def load_features():
    # may need to build the index first
    if not os.path.exists(feat_file_name):
        feat_index = {}
        # build the index from facebook/*.featnames files
        featname_files = glob.iglob("%s/facebook/*.featnames" % (pathhack,))
        for featname_file_name in featname_files:
            featname_file = open(featname_file_name, 'r')
            for line in featname_file:
                # example line:
                # 0 birthday;anonymized feature 376
                index, name = parse_featname_line(line)
                feat_index[index] = name
            featname_file.close()
        keys = feat_index.keys()
        keys.sort()
        out = open(feat_file_name,'w')
        for key in keys:
            out.write("%d %s\n" % (key, feat_index[key]))
        out.close()
        
    # index built, read it in (even if we just built it by scanning)
    global feature_index
    global inverted_feature_index
    index_file = open(feat_file_name,'r')
    for line in index_file:
        split = line.strip().split(' ')
        key = int(split[0])
        val = split[1]
        feature_index[key] = val
    index_file.close()

    for key in feature_index.keys():
        val = feature_index[key]
        inverted_feature_index[val] = key

def load_nodes():
    assert len(feature_index) > 0, "call load_features() first"
    global network
    global ego_nodes

    # get all the node ids by looking at the files
    ego_nodes = [int(x.split("/")[-1].split('.')[0]) for x in glob.glob("%s/facebook/*.featnames" % (pathhack,))]
    node_ids = ego_nodes

    # parse each node
    for node_id in node_ids:
        featname_file = open("%s/facebook/%d.featnames" % (pathhack,node_id), 'r')
        feat_file     = open("%s/facebook/%d.feat"      % (pathhack,node_id), 'r')
        egofeat_file  = open("%s/facebook/%d.egofeat"   % (pathhack,node_id), 'r')
        edge_file     = open("%s/facebook/%d.edges"     % (pathhack,node_id), 'r')

        # 0 1 0 0 0 ...
        ego_features = [int(x) for x in egofeat_file.readline().split(' ')]

        # Add ego node if not already contained in network
        if not network.has_node(node_id):
            network.add_node(node_id)
            network.node[node_id]['features'] = np.zeros(len(feature_index))
        
        # parse ego node
        i = 0
        for line in featname_file:
            key, val = parse_featname_line(line)
            # Update feature value if necessary
            if ego_features[i] + 1 > network.node[node_id]['features'][key]:
                network.node[node_id]['features'][key] = ego_features[i] + 1
            i += 1

        # parse neighboring nodes
        for line in feat_file:
            featname_file.seek(0)
            split = [int(x) for x in line.split(' ')]
            node_id = split[0]
            features = split[1:]

            # Add node if not already contained in network
            if not network.has_node(node_id):
                network.add_node(node_id)
                network.node[node_id]['features'] = np.zeros(len(feature_index))

            i = 0
            for line in featname_file:
                key, val = parse_featname_line(line)
                # Update feature value if necessary
                if features[i] + 1 > network.node[node_id]['features'][key]:
                    network.node[node_id]['features'][key] = features[i] + 1
                i += 1
            
        featname_file.close()
        feat_file.close()
        egofeat_file.close()
        edge_file.close()

def load_edges():
    global network
    assert network.order() > 0, "call load_nodes() first"
    edge_file = open("%s/facebook/facebook_combined.txt" % (pathhack,),"r")
    for line in edge_file:
        # nodefrom nodeto
        split = [int(x) for x in line.split(" ")]
        node_from = split[0]
        node_to = split[1]
        network.add_edge(node_from, node_to)

def load_network():
    """
    Load the network.  After calling this function, facebook.network points to a networkx object for the facebook data.

    """
    load_features()
    load_nodes()
    load_edges()

def feature_matrix():
    n_nodes = network.number_of_nodes()
    n_features = len(feature_index)

    X = np.zeros((n_nodes, n_features))
    for i,node in enumerate(network.nodes()):
        X[i,:] = network.node[node]['features']

    return X

def universal_feature(feature_index):
    """
    Does every node have this feature?

    """
    return len([x for x in network.nodes_iter() if network.node[x]['feautures'][feature_index] > 0]) // network.order() == 1

if __name__ == '__main__':
    # print "Running tests."
    print "Loading network..."
    load_network()
    print "done."

    failures = 0
    def test(actual, expected, test_name):
        global failures  #lol python scope
        try:
            print "testing %s..." % (test_name,)
            assert actual == expected, "%s failed (%s != %s)!" % (test_name,actual, expected)
            print "%s passed (%s == %s)." % (test_name,actual,expected)
        except AssertionError as e:
            print e
            failures += 1
    
    test(network.order(), 4039, "order")
    test(network.size(), 88234, "size")
    test(round(nx.average_clustering(network),4), 0.6055, "clustering")
    print "%d tests failed." % (failures,)

    print ''

    print "Saving network..."
    adj = nx.adjacency_matrix(network)
    features = feature_matrix()
    network_tuple = (adj, sp.csr_matrix(features))

    with open("fb-processed/combined-adj-sparsefeat.pkl", "wb") as f:
        pickle.dump(network_tuple, f)

    print "Network saved!"
    
    
