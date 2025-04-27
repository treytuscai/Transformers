'''transformer_layers.py
Layers related to transformer neural networks.
YOUR NAMES HERE
CS 444: Deep Learning
'''
import tensorflow as tf

import layers
from tf_util import interleave_cols

class Embedding(layers.Layer):
    '''Embedding layer. Takes a mini-batch of ints and for net_in extracts the weights at the specified indices.'''
    def __init__(self, name, input_dim, embed_dim, prev_layer_or_block=None):
        '''Embedding layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Drop_0). Used for debugging and printing summary of net.
        input_dim. int.
            The number of neurons in the input layer `M`.
        embed_dim. int.
            The number of neurons in the current layer `H`.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Initialize the layer's parameters.
        '''
        pass

    def has_wts(self):
        '''Returns whether the Embedding layer has weights. It does...'''
        return True

    def init_params(self, input_dim, embed_dim):
        '''Initializes the Embedding layer's weights. There should be no bias.

        Parameters:
        -----------
        input_dim: int.
            Number of neurons in the Input layer (`M`).
        embed_dim: int.
            Number of neurons in the current layer (`H`).

        NOTE:
        - Remember to turn off the bias.
        - Use He initialization.
        '''

    def compute_net_input(self, x):
        '''Computes the net input for the current Embedding layer.

        Parameters:
        -----------
        x: tf.constant. tf.int32. shape=(B, T).
            Mini-batch of int indices.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The net_in, which is the weights extracted at the specified indices.

        NOTE:
        - This layer does NOT use lazy initialization.
        - The presence of the time dimension should not affect your code compared to if it were not there.
        '''
        pass

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Embedding layer output({self.layer_name}) shape: {self.output_shape}'


class PositionalEncoding(layers.Layer):
    '''Positional Encoding layer that implements sin/cos position coding.'''
    def __init__(self, name, embed_dim, prev_layer_or_block=None):
        '''PositionalEncoding layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Drop_0). Used for debugging and printing summary of net.
        embed_dim. int.
            The number of neurons in the current layer `H` and in the Embedding layer below.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Print a warning/error if the embedding dimension (H) is not even, since this layer's sin/cos coding requires
        an even split.
        '''
        pass

    def create_position_encoding(self, embed_dim, seq_len):
        '''Creates a positional encoding tensor using the sin/cos scheme for a sequence of length `seq_len` tokens
        for each of the `embed_dim`/H neurons. See notebook for a refresher on the equation.

        Parameters:
        -----------
        embed_dim: int.
            The number of neurons in the Embedding layer (H).
        seq_len: int.
            The length of sequences processed by the transformer.

        Returns:
        --------
        tf.constant. shape=(1, T, H).
            A positional encoding tensor, where the first axis is a singleton dimension to handle the batch dimension,
            T is the sequence length, and H is the number of embedding layer neurons.

        NOTE:
        - The provided `interleave_cols` function should be helpful, as should be tf.expand_dims.
        - To allow TensorFlow track the flow of gradients, you should implement this with 100% TensorFlow and no loops.
        '''
        pass

    def compute_net_input(self, x):
        '''Computes the net input for the current PositionalEncoding layer, which is the sum of the input with the
        position coding tensor.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, T, H).
            Input from the layer beneath in the network.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The net_in, the input with position coding added.

        NOTE: This layer uses lazy initialization. This means that if the position code has not been defined yet,
        we call `create_position_encoding` to create it and set the result to the instance variable.
        '''
        pass

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Positional encoding layer output({self.layer_name}) shape: {self.output_shape}'
