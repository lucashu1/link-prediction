from __future__ import division
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.manifold import spectral_embedding
import node2vec
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
import time
import os
import tensorflow as tf
from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
import pickle
from copy import deepcopy

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input: positive test/val edges, negative test/val edges, edge score matrix
# Output: ROC AUC score, ROC Curve (FPR, TPR, Thresholds), AP score
def get_roc_score(edges_pos, edges_neg, score_matrix, apply_sigmoid=False):

    # Edge case
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None, None, None)

    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        if apply_sigmoid == True:
            preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_pos.append(score_matrix[edge[0], edge[1]])
        pos.append(1) # actual value (1 for positive)
        
    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        if apply_sigmoid == True:
            preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_neg.append(score_matrix[edge[0], edge[1]])
        neg.append(0) # actual value (0 for negative)
        
    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    roc_curve_tuple = roc_curve(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    
    return roc_score, roc_curve_tuple, ap_score

# Input: NetworkX training graph, train_test_split (from mask_test_edges)
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def adamic_adar_scores(g_train, train_test_split):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack input

    start_time = time.time()
    
    aa_scores = {}

    # Calculate scores
    aa_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.adamic_adar_index(g_train): # (u, v) = node indices, p = Adamic-Adar index
        aa_matrix[u][v] = p
        aa_matrix[v][u] = p # make sure it's symmetric
    aa_matrix = aa_matrix / aa_matrix.max() # Normalize matrix

    runtime = time.time() - start_time
    aa_roc, aa_roc_curve, aa_ap = get_roc_score(test_edges, test_edges_false, aa_matrix)

    aa_scores['test_roc'] = aa_roc
    aa_scores['test_roc_curve'] = aa_roc_curve
    aa_scores['test_ap'] = aa_ap
    aa_scores['runtime'] = runtime
    return aa_scores


# Input: NetworkX training graph, train_test_split (from mask_test_edges)
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def jaccard_coefficient_scores(g_train, train_test_split):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack input

    start_time = time.time()
    jc_scores = {}

    # Calculate scores
    jc_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.jaccard_coefficient(g_train): # (u, v) = node indices, p = Jaccard coefficient
        jc_matrix[u][v] = p
        jc_matrix[v][u] = p # make sure it's symmetric
    jc_matrix = jc_matrix / jc_matrix.max() # Normalize matrix

    runtime = time.time() - start_time
    jc_roc, jc_roc_curve, jc_ap = get_roc_score(test_edges, test_edges_false, jc_matrix)

    jc_scores['test_roc'] = jc_roc
    jc_scores['test_roc_curve'] = jc_roc_curve
    jc_scores['test_ap'] = jc_ap
    jc_scores['runtime'] = runtime
    return jc_scores


# Input: NetworkX training graph, train_test_split (from mask_test_edges)
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def preferential_attachment_scores(g_train, train_test_split):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack input

    start_time = time.time()
    pa_scores = {}

    # Calculate scores
    pa_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.preferential_attachment(g_train): # (u, v) = node indices, p = Jaccard coefficient
        pa_matrix[u][v] = p
        pa_matrix[v][u] = p # make sure it's symmetric
    pa_matrix = pa_matrix / pa_matrix.max() # Normalize matrix

    runtime = time.time() - start_time
    pa_roc, pa_roc_curve, pa_ap = get_roc_score(test_edges, test_edges_false, pa_matrix)

    pa_scores['test_roc'] = pa_roc
    pa_scores['test_roc_curve'] = pa_roc_curve
    pa_scores['test_ap'] = pa_ap
    pa_scores['runtime'] = runtime
    return pa_scores


# Input: train_test_split (from mask_test_edges)
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def spectral_clustering_scores(train_test_split, random_state=0):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack input

    start_time = time.time()
    sc_scores = {}

    # Perform spectral clustering link prediction
    spectral_emb = spectral_embedding(adj_train, n_components=16, random_state=random_state)
    sc_score_matrix = np.dot(spectral_emb, spectral_emb.T)

    runtime = time.time() - start_time
    sc_test_roc, sc_test_roc_curve, sc_test_ap = get_roc_score(test_edges, test_edges_false, sc_score_matrix, apply_sigmoid=True)
    sc_val_roc, sc_val_roc_curve, sc_val_ap = get_roc_score(val_edges, val_edges_false, sc_score_matrix, apply_sigmoid=True)

    # Record scores
    sc_scores['test_roc'] = sc_test_roc
    sc_scores['test_roc_curve'] = sc_test_roc_curve
    sc_scores['test_ap'] = sc_test_ap

    sc_scores['val_roc'] = sc_val_roc
    sc_scores['val_roc_curve'] = sc_val_roc_curve
    sc_scores['val_ap'] = sc_val_ap

    sc_scores['runtime'] = runtime
    return sc_scores

# Input: NetworkX training graph, train_test_split (from mask_test_edges), n2v hyperparameters
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def node2vec_scores(
    g_train, train_test_split,
    P = 1, # Return hyperparameter
    Q = 1, # In-out hyperparameter
    WINDOW_SIZE = 10, # Context size for optimization
    NUM_WALKS = 10, # Number of walks per source
    WALK_LENGTH = 80, # Length of walk per source
    DIMENSIONS = 128, # Embedding dimension
    DIRECTED = False, # Graph directed/undirected
    WORKERS = 8, # Num. parallel workers
    ITER = 1, # SGD epochs
    edge_score_mode = "edge-emb", # Whether to use bootstrapped edge embeddings + LogReg (like in node2vec paper), 
        # or simple dot-product (like in GAE paper) for edge scoring
    verbose=1,
    ):

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack train-test split

    start_time = time.time()

    # Preprocessing, generate walks
    g_n2v = node2vec.Graph(g_train, DIRECTED, P, Q) # create node2vec graph instance
    g_n2v.preprocess_transition_probs()
    if verbose == 2:
        walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH, verbose=True)
    else:
        walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH, verbose=False)
    walks = [map(str, walk) for walk in walks]

    # Train skip-gram model
    model = Word2Vec(walks, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)

    # Store embeddings mapping
    emb_mappings = model.wv

    # Create node embeddings matrix (rows = nodes, columns = embedding features)
    emb_list = []
    for node_index in range(0, adj_train.shape[0]):
        node_str = str(node_index)
        node_emb = emb_mappings[node_str]
        emb_list.append(node_emb)
    emb_matrix = np.vstack(emb_list)

    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
        # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    if edge_score_mode == "edge-emb":
        
        def get_edge_embeddings(edge_list):
            embs = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_matrix[node1]
                emb2 = emb_matrix[node2]
                edge_emb = np.multiply(emb1, emb2)
                embs.append(edge_emb)
            embs = np.array(embs)
            return embs

        # Train-set edge embeddings
        pos_train_edge_embs = get_edge_embeddings(train_edges)
        neg_train_edge_embs = get_edge_embeddings(train_edges_false)
        train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs = get_edge_embeddings(val_edges)
            neg_val_edge_embs = get_edge_embeddings(val_edges_false)
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

        # Test-set edge embeddings, labels
        pos_test_edge_embs = get_edge_embeddings(test_edges)
        neg_test_edge_embs = get_edge_embeddings(test_edges_false)
        test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

        # Train logistic regression classifier on train-set edge embeddings
        edge_classifier = LogisticRegression(random_state=0)
        edge_classifier.fit(train_edge_embs, train_edge_labels)

        # Predicted edge scores: probability of being of class "1" (real edge)
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

        runtime = time.time() - start_time

        # Calculate scores
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            n2v_val_roc = roc_auc_score(val_edge_labels, val_preds)
            n2v_val_roc_curve = roc_curve(val_edge_labels, val_preds)
            n2v_val_ap = average_precision_score(val_edge_labels, val_preds)
        else:
            n2v_val_roc = None
            n2v_val_roc_curve = None
            n2v_val_ap = None
        
        n2v_test_roc = roc_auc_score(test_edge_labels, test_preds)
        n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
        n2v_test_ap = average_precision_score(test_edge_labels, test_preds)


    # Generate edge scores using simple dot product of node embeddings (like in GAE paper)
    elif edge_score_mode == "dot-product":
        score_matrix = np.dot(emb_matrix, emb_matrix.T)
        runtime = time.time() - start_time

        # Val set scores
        if len(val_edges) > 0:
            n2v_val_roc, n2v_val_roc_curve, n2v_val_ap = get_roc_score(val_edges, val_edges_false, score_matrix, apply_sigmoid=True)
        else:
            n2v_val_roc = None
            n2v_val_roc_curve = None
            n2v_val_ap = None
        
        # Test set scores
        n2v_test_roc, n2v_test_roc_curve, n2v_test_ap = get_roc_score(test_edges, test_edges_false, score_matrix, apply_sigmoid=True)

    else:
        print "Invalid edge_score_mode! Either use edge-emb or dot-product."

    # Record scores
    n2v_scores = {}

    n2v_scores['test_roc'] = n2v_test_roc
    n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    n2v_scores['test_ap'] = n2v_test_ap

    n2v_scores['val_roc'] = n2v_val_roc
    n2v_scores['val_roc_curve'] = n2v_val_roc_curve
    n2v_scores['val_ap'] = n2v_val_ap

    n2v_scores['runtime'] = runtime

    return n2v_scores


# Input: original adj_sparse, train_test_split (from mask_test_edges), features matrix, n2v hyperparameters
# Output: dictionary with ROC AUC, ROC Curve, AP, Runtime
def gae_scores(
    adj_sparse,
    train_test_split,
    features_matrix=None,
    LEARNING_RATE = 0.01,
    EPOCHS = 200,
    HIDDEN1_DIM = 32,
    HIDDEN2_DIM = 16,
    DROPOUT = 0,
    edge_score_mode="dot-product",
    verbose=1
    ):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack train-test split

    start_time = time.time()

    # Train on CPU (hide GPU) due to memory constraints
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    # Convert features from normal matrix --> sparse matrix --> tuple
        # features_tuple contains: (list of matrix coordinates, list of values, matrix dimensions)
    if features_matrix is None:
        x = sp.lil_matrix(np.identity(adj_sparse.shape[0]))
    else:
        x = sp.lil_matrix(features_matrix)
    features_tuple = sparse_to_tuple(x)
    features_shape = features_tuple[2]

    # Get graph attributes (to feed into model)
    num_nodes = adj_sparse.shape[0] # number of nodes in adjacency matrix
    num_features = features_shape[1] # number of features (columsn of features matrix)
    features_nonzero = features_tuple[1].shape[0] # number of non-zero entries in features matrix (or length of values list)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = deepcopy(adj_sparse)
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    # Normalize adjacency matrix
    adj_norm = preprocess_graph(adj_train)

    # Add in diagonals
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    # How much to weigh positive examples (true edges) in cost print_function
      # Want to weigh less-frequent classes higher, so as to prevent model output bias
      # pos_weight = (num. negative samples / (num. positive samples)
    pos_weight = float(adj_sparse.shape[0] * adj_sparse.shape[0] - adj_sparse.sum()) / adj_sparse.sum()

    # normalize (scale) average weighted cost
    norm = adj_sparse.shape[0] * adj_sparse.shape[0] / float((adj_sparse.shape[0] * adj_sparse.shape[0] - adj_sparse.sum()) * 2)

    # Create VAE model
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero,
                       HIDDEN1_DIM, HIDDEN2_DIM)

    opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                           validate_indices=False), [-1]),
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm,
                               learning_rate=LEARNING_RATE)

    cost_val = []
    acc_val = []
    val_roc_score = []

    prev_embs = []

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Train model
    for epoch in range(EPOCHS):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features_tuple, placeholders)
        feed_dict.update({placeholders['dropout']: DROPOUT})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        # Evaluate predictions
        feed_dict.update({placeholders['dropout']: 0})
        gae_emb = sess.run(model.z_mean, feed_dict=feed_dict)

        prev_embs.append(gae_emb)

        gae_score_matrix = np.dot(gae_emb, gae_emb.T)

        # # TODO: remove this (debugging)
        # if not np.isfinite(gae_score_matrix).all():
        #     print 'Found non-finite value in GAE score matrix! Epoch: {}'.format(epoch)
        #     with open('numpy-nan-debugging.pkl', 'wb') as f:
        #         dump_info = {}
        #         dump_info['gae_emb'] = gae_emb
        #         dump_info['epoch'] = epoch
        #         dump_info['gae_score_matrix'] = gae_score_matrix
        #         dump_info['adj_norm'] = adj_norm
        #         dump_info['adj_label'] = adj_label
        #         dump_info['features_tuple'] = features_tuple
        #         # dump_info['feed_dict'] = feed_dict
        #         dump_info['prev_embs'] = prev_embs
        #         pickle.dump(dump_info, f, protocol=2)
        # # END TODO


        roc_curr, roc_curve_curr, ap_curr = get_roc_score(val_edges, val_edges_false, gae_score_matrix, apply_sigmoid=True)
        val_roc_score.append(roc_curr)

        # Print results for this epoch
        if verbose == 2:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
              "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t))

    if verbose == 2:
        print("Optimization Finished!")

    # Print final results
    feed_dict.update({placeholders['dropout']: 0})
    gae_emb = sess.run(model.z_mean, feed_dict=feed_dict)

    # Dot product edge scores (default)
    if edge_score_mode == "dot-product":
        gae_score_matrix = np.dot(gae_emb, gae_emb.T)

        runtime = time.time() - start_time

        # Calculate final scores
        gae_val_roc, gae_val_roc_curve, gae_val_ap = get_roc_score(val_edges, val_edges_false, gae_score_matrix)
        gae_test_roc, gae_test_roc_curve, gae_test_ap = get_roc_score(test_edges, test_edges_false, gae_score_matrix)
    
    # Take bootstrapped edge embeddings (via hadamard product)
    elif edge_score_mode == "edge-emb":
        def get_edge_embeddings(edge_list):
            embs = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = gae_emb[node1]
                emb2 = gae_emb[node2]
                edge_emb = np.multiply(emb1, emb2)
                embs.append(edge_emb)
            embs = np.array(embs)
            return embs

        # Train-set edge embeddings
        pos_train_edge_embs = get_edge_embeddings(train_edges)
        neg_train_edge_embs = get_edge_embeddings(train_edges_false)
        train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs = get_edge_embeddings(val_edges)
            neg_val_edge_embs = get_edge_embeddings(val_edges_false)
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

        # Test-set edge embeddings, labels
        pos_test_edge_embs = get_edge_embeddings(test_edges)
        neg_test_edge_embs = get_edge_embeddings(test_edges_false)
        test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

        # Train logistic regression classifier on train-set edge embeddings
        edge_classifier = LogisticRegression(random_state=0)
        edge_classifier.fit(train_edge_embs, train_edge_labels)

        # Predicted edge scores: probability of being of class "1" (real edge)
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

        runtime = time.time() - start_time

        # Calculate scores
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            gae_val_roc = roc_auc_score(val_edge_labels, val_preds)
            gae_val_roc_curve = roc_curve(val_edge_labels, val_preds)
            gae_val_ap = average_precision_score(val_edge_labels, val_preds)
        else:
            gae_val_roc = None
            gae_val_roc_curve = None
            gae_val_ap = None
        
        gae_test_roc = roc_auc_score(test_edge_labels, test_preds)
        gae_test_roc_curve = roc_curve(test_edge_labels, test_preds)
        gae_test_ap = average_precision_score(test_edge_labels, test_preds)

    # Record scores
    gae_scores = {}

    gae_scores['test_roc'] = gae_test_roc
    gae_scores['test_roc_curve'] = gae_test_roc_curve
    gae_scores['test_ap'] = gae_test_ap

    gae_scores['val_roc'] = gae_val_roc
    gae_scores['val_roc_curve'] = gae_val_roc_curve
    gae_scores['val_ap'] = gae_val_ap

    gae_scores['val_roc_per_epoch'] = val_roc_score
    gae_scores['runtime'] = runtime
    return gae_scores
    


# Input: adjacency matrix (in sparse format), features_matrix (normal format), test_frac, val_frac, verbose
    # Verbose: 0 - print nothing, 1 - print scores, 2 - print scores + GAE training progress
# Returns: Dictionary of results (ROC AUC, ROC Curve, AP, Runtime) for each link prediction method
def calculate_all_scores(adj_sparse, features_matrix=None, \
        test_frac=.3, val_frac=.1, random_state=0, verbose=1, \
        train_test_split_file=None):
    np.random.seed(random_state) # Guarantee consistent train/test split
    tf.set_random_seed(random_state) # Consistent GAE training

    # Prepare LP scores dictionary
    lp_scores = {}

    ### ---------- PREPROCESSING ---------- ###
    train_test_split = None
    try: # If found existing train-test split, use that file
        with open(train_test_split_file, 'rb') as f:
            train_test_split = pickle.load(f)
            print 'Found existing train-test split!'
    except: # Else, generate train-test split on the fly
        print 'Generating train-test split...'
        train_test_split = mask_test_edges(adj_sparse, test_frac=test_frac, val_frac=val_frac)
    
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack tuple
    g_train = nx.from_scipy_sparse_matrix(adj_train) # new graph object with only non-hidden edges

    # Inspect train/test split
    if verbose >= 1:
        print "Total nodes:", adj_sparse.shape[0]
        print "Total edges:", int(adj_sparse.nnz/2) # adj is symmetric, so nnz (num non-zero) = 2*num_edges
        print "Training edges (positive):", len(train_edges)
        print "Training edges (negative):", len(train_edges_false)
        print "Validation edges (positive):", len(val_edges)
        print "Validation edges (negative):", len(val_edges_false)
        print "Test edges (positive):", len(test_edges)
        print "Test edges (negative):", len(test_edges_false)
        print ''
        print "------------------------------------------------------"


    ### ---------- LINK PREDICTION BASELINES ---------- ###
    # Adamic-Adar
    aa_scores = adamic_adar_scores(g_train, train_test_split)
    lp_scores['aa'] = aa_scores
    if verbose >= 1:
        print ''
        print 'Adamic-Adar Test ROC score: ', str(aa_scores['test_roc'])
        print 'Adamic-Adar Test AP score: ', str(aa_scores['test_ap'])

    # Jaccard Coefficient
    jc_scores = jaccard_coefficient_scores(g_train, train_test_split)
    lp_scores['jc'] = jc_scores
    if verbose >= 1:
        print ''
        print 'Jaccard Coefficient Test ROC score: ', str(jc_scores['test_roc'])
        print 'Jaccard Coefficient Test AP score: ', str(jc_scores['test_ap'])

    # Preferential Attachment
    pa_scores = preferential_attachment_scores(g_train, train_test_split)
    lp_scores['pa'] = pa_scores
    if verbose >= 1:
        print ''
        print 'Preferential Attachment Test ROC score: ', str(pa_scores['test_roc'])
        print 'Preferential Attachment Test AP score: ', str(pa_scores['test_ap'])


    ### ---------- SPECTRAL CLUSTERING ---------- ###
    sc_scores = spectral_clustering_scores(train_test_split)
    lp_scores['sc'] = sc_scores
    if verbose >= 1:
        print ''
        print 'Spectral Clustering Validation ROC score: ', str(sc_scores['val_roc'])
        print 'Spectral Clustering Validation AP score: ', str(sc_scores['val_ap'])
        print 'Spectral Clustering Test ROC score: ', str(sc_scores['test_roc'])
        print 'Spectral Clustering Test AP score: ', str(sc_scores['test_ap'])


    ### ---------- NODE2VEC ---------- ###
    # node2vec settings
    # NOTE: When p = q = 1, this is equivalent to DeepWalk
    P = 1 # Return hyperparameter
    Q = 1 # In-out hyperparameter
    WINDOW_SIZE = 10 # Context size for optimization
    NUM_WALKS = 10 # Number of walks per source
    WALK_LENGTH = 80 # Length of walk per source
    DIMENSIONS = 128 # Embedding dimension
    DIRECTED = False # Graph directed/undirected
    WORKERS = 8 # Num. parallel workers
    ITER = 1 # SGD epochs

    # Using bootstrapped edge embeddings + logistic regression
    n2v_edge_emb_scores = node2vec_scores(g_train, train_test_split,
        P, Q, WINDOW_SIZE, NUM_WALKS, WALK_LENGTH, DIMENSIONS, DIRECTED, WORKERS, ITER,
        "edge-emb",
        verbose)
    lp_scores['n2v_edge_emb'] = n2v_edge_emb_scores

    if verbose >= 1:
        print ''
        print 'node2vec (Edge Embeddings) Validation ROC score: ', str(n2v_edge_emb_scores['val_roc'])
        print 'node2vec (Edge Embeddings) Validation AP score: ', str(n2v_edge_emb_scores['val_ap'])
        print 'node2vec (Edge Embeddings) Test ROC score: ', str(n2v_edge_emb_scores['test_roc'])
        print 'node2vec (Edge Embeddings) Test AP score: ', str(n2v_edge_emb_scores['test_ap'])

    # Using dot products to calculate edge scores
    n2v_dot_prod_scores = node2vec_scores(g_train, train_test_split,
        P, Q, WINDOW_SIZE, NUM_WALKS, WALK_LENGTH, DIMENSIONS, DIRECTED, WORKERS, ITER,
        "dot-product",
        verbose)
    lp_scores['n2v_dot_prod'] = n2v_dot_prod_scores

    if verbose >= 1:
        print ''
        print 'node2vec (Dot Product) Validation ROC score: ', str(n2v_dot_prod_scores['val_roc'])
        print 'node2vec (Dot Product) Validation AP score: ', str(n2v_dot_prod_scores['val_ap'])
        print 'node2vec (Dot Product) Test ROC score: ', str(n2v_dot_prod_scores['test_roc'])
        print 'node2vec (Dot Product) Test AP score: ', str(n2v_dot_prod_scores['test_ap'])


    ### ---------- (VARIATIONAL) GRAPH AUTOENCODER ---------- ###
    # GAE hyperparameters
    LEARNING_RATE = 0.01 # Default: 0.01
    EPOCHS = 200
    HIDDEN1_DIM = 32
    HIDDEN2_DIM = 16
    DROPOUT = 0

    # Use dot product
    tf.set_random_seed(random_state) # Consistent GAE training
    gae_results = gae_scores(adj_sparse, train_test_split, features_matrix,
        LEARNING_RATE, EPOCHS, HIDDEN1_DIM, HIDDEN2_DIM, DROPOUT,
        "dot-product",
        verbose)
    lp_scores['gae'] = gae_results

    if verbose >= 1:
        print ''
        print 'GAE (Dot Product) Validation ROC score: ', str(gae_results['val_roc'])
        print 'GAE (Dot Product) Validation AP score: ', str(gae_results['val_ap'])
        print 'GAE (Dot Product) Test ROC score: ', str(gae_results['test_roc'])
        print 'GAE (Dot Product) Test AP score: ', str(gae_results['test_ap'])


    # # Use edge embeddings
    # tf.set_random_seed(random_state) # Consistent GAE training
    # gae_edge_emb_results = gae_scores(adj_sparse, train_test_split, features_matrix,
    #     LEARNING_RATE, EPOCHS, HIDDEN1_DIM, HIDDEN2_DIM, DROPOUT,
    #     "edge-emb",
    #     verbose)
    # lp_scores['gae_edge_emb'] = gae_edge_emb_results

    # if verbose >= 1:
    #     print ''
    #     print 'GAE (Edge Embeddings) Validation ROC score: ', str(gae_edge_emb_results['val_roc'])
    #     print 'GAE (Edge Embeddings) Validation AP score: ', str(gae_edge_emb_results['val_ap'])
    #     print 'GAE (Edge Embeddings) Test ROC score: ', str(gae_edge_emb_results['test_roc'])
    #     print 'GAE (Edge Embeddings) Test AP score: ', str(gae_edge_emb_results['test_ap'])


    ### ---------- RETURN RESULTS ---------- ###
    return lp_scores
