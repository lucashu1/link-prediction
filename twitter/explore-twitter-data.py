import networkx as nx
import os

DATA_DIR = '/Users/lucashu/Downloads/twitter/'

# Get list of twitter data files
twitter_files = os.listdir(DATA_DIR)
edgelists = [filename for filename in twitter_files if '.edges' in filename]

# Prep file to write results
out_f = open('./twitter-ego-summary.txt', 'wb')

# Track ego network with most nodes, edges
max_nodes = 0
max_nodes_file = None
max_edges = 0
max_edges_file = None

# Iterate over edgelist files
for edgelist_file in edgelists:
	# Read edgelist --> graph
	edges_f = open(DATA_DIR+edgelist_file, 'rb')
	g = nx.read_edgelist(edges_f, nodetype=int, create_using=nx.DiGraph())

	print 'Current file: ', edgelist_file
	print 'Number of nodes: ', g.number_of_nodes()
	print 'Number of edges: ', g.number_of_edges()
	print ''

	out_f.write('Current file: ' + edgelist_file + '\n')
	out_f.write('Number of nodes: ' + str(g.number_of_nodes()) + '\n')
	out_f.write('Number of edges: ' + str(g.number_of_edges()) + '\n')
	out_f.write('\n')

	# Update max values
	if g.number_of_nodes() > max_nodes:
		max_nodes_file = edgelist_file
		max_nodes = g.number_of_nodes()

	if g.number_of_edges() > max_edges:
		max_edges_file = edgelist_file
		max_edges = g.number_of_edges()

# Print final results
print 'Most nodes: ', max_nodes_file, ' (', max_nodes, ') nodes'
print 'Most edges: ', max_edges_file, ' (', max_edges, ') edges'

out_f.write('Most nodes: ' + max_nodes_file + ' (' + str(max_nodes) + ') nodes' + '\n')
out_f.write('Most edges: ' + max_edges_file + ' (' + str(max_edges) + ') edges' + '\n')