from gae.layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# Base model superclass
class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


# Graph AutoEncoder model
class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, 
      hidden1_dim=32, hidden2_dim=16, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.build()

    def _build(self):
        # First GCN Layer: (A, X) --> H (hidden layer features)
        self.hidden1 = GraphConvolution(input_dim=self.input_dim,
                                              output_dim=self.hidden1_dim,
                                              adj=self.adj,
                                              # features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        # Second GCN Layer: (A, H) --> Z (mode embeddings)
        self.embeddings = GraphConvolution(input_dim=self.hidden1_dim,
                                           output_dim=self.hidden2_dim,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.hidden1)

        # Z_mean for AE. No noise added (because not a VAE)
        self.z_mean = self.embeddings

        # Inner-Product Decoder: Z (embeddings) --> A (reconstructed adj.)
        self.reconstructions = InnerProductDecoder(input_dim=self.hidden2_dim,
                                      act=lambda x: x,
                                      logging=self.logging)(self.embeddings)


# Graph Variational Auto-Encoder model
class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, 
      hidden1_dim=32, hidden2_dim=16, flatten_output=True, dtype=tf.float32, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.flatten_output=flatten_output
        self.dtype=dtype
        self.build()

    def _build(self):
        # First GCN Layer: (A, X) --> H (hidden layer features)
        self.hidden1 = GraphConvolution(input_dim=self.input_dim,
                                              output_dim=self.hidden1_dim,
                                              adj=self.adj,
                                              # features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              dtype=self.dtype,
                                              logging=self.logging)(self.inputs)

        # Second GCN Layer: (A, H) --> Z_mean (node embeddings)
        self.z_mean = GraphConvolution(input_dim=self.hidden1_dim,
                                       output_dim=self.hidden2_dim,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       dtype=self.dtype,
                                       logging=self.logging)(self.hidden1)

        # Also second GCN Layer: (A, H) --> Z_log_stddev (for VAE noise)
        self.z_log_std = GraphConvolution(input_dim=self.hidden1_dim,
                                          output_dim=self.hidden2_dim,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          dtype=self.dtype,
                                          logging=self.logging)(self.hidden1)

        # Sampling operation: z = z_mean + (random_noise_factor) * z_stddev
        self.z = self.z_mean + tf.random_normal([self.n_samples, self.hidden2_dim], dtype=self.dtype) * tf.exp(self.z_log_std)

        # Inner-Product Decoder: Z (embeddings) --> A (reconstructed adj.)
        self.reconstructions = InnerProductDecoder(input_dim=self.hidden2_dim,
                                      act=lambda x: x,
                                      flatten=self.flatten_output,
                                      logging=self.logging)(self.z)
