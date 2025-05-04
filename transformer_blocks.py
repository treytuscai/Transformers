'''transformer_blocks.py
Blocks related to transformer neural networks.
YOUR NAMES HERE
CS 444: Deep Learning
'''
import tensorflow as tf

import block
from layers import Dense, Dropout
from transformer_layers import PositionalEncoding
from tf_util import tril


class QueryKeyValueBlock(block.Block):
    '''Block that encapsulates the Dense layers that generate the queries, keys, and values.'''

    def __init__(self, blockname, units, prev_layer_or_block):
        '''QueryKeyValueBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        units. int.
            The number of neurons in each of the Dense layers in the block. All Dense layers have the same number of
            units (H — i.e. embed_dim).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.

        Properties of all layers:
        ---------------------------
        - They are along separate branches. Think about what this means for their previous layer/block reference.
        - He initialization.
        - Layer normalization.
        - Linear/identity activation.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Assemble layers in the block.
        '''
        super().__init__(blockname=blockname, prev_layer_or_block=prev_layer_or_block)

        self.q_layer = Dense(name=f"{blockname}_query",
                             units=units,
                             activation='linear',
                             prev_layer_or_block=prev_layer_or_block,
                             wt_init='he',
                             do_batch_norm=False,
                             do_layer_norm=True)

        self.k_layer = Dense(name=f"{blockname}_key",
                             units=units,
                             activation='linear',
                             prev_layer_or_block=prev_layer_or_block,
                             wt_init='he',
                             do_batch_norm=False,
                             do_layer_norm=True)

        self.v_layer = Dense(name=f"{blockname}_value",
                             units=units,
                             activation='linear',
                             prev_layer_or_block=prev_layer_or_block,
                             wt_init='he',
                             do_batch_norm=False,
                             do_layer_norm=True)

    def __call__(self, query_input, key_input, value_input):
        '''Forward pass through the QKV Block with activations that should represent the input to respective QKV layers.

        Parameters:
        -----------
        query_input: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from a prev layer/block that are the input to the query layer.
        key_input: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from a prev layer/block that are the input to the key layer.
        value_input: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from a prev layer/block that are the input to the value layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            Activations produced by the query layer.
        tf.constant. tf.float32s. shape=(B, T, H).
            Activations produced by the key layer.
        tf.constant. tf.float32s. shape=(B, T, H).
            Activations produced by the value layer.
        '''
        x_q = self.q_layer(query_input)
        x_k = self.k_layer(key_input)
        x_v = self.v_layer(value_input)
        return x_q, x_k, x_v


class AttentionBlock(block.Block):
    '''Block that encapsulates the fundamental attention mechanism.'''

    def __init__(self, blockname, num_heads, units, prev_layer_or_block, dropout_rate=0.1, causal=True):
        '''AttentionBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        num_heads: int.
            Number of attention heads to use in the attention block.
        units: int.
            Number of neurons in the attention block (H — i.e. embed_dim).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        dropout_rate: float.
            The dropout rate (R) to use in the Dropout layer that is in the attention block that is applied to the
            attention values.
        causal: bool.
            Whether to apply a causal mask to remove/mask out the ability for the layer to pay attention to tokens
            that are in the future of the current one in the sequence.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create any instance variables to save any information that will be helpful to access during the forward pass.
        3. Create the Dropout layer.
        4. For efficiency, it is helpful to pre-compute the attention gain and assign it to an instance variable
        (e.g. as self.gain) so that you can use it during the forward pass. You have all the info here that is needed
        to compute the gain.
        '''

        # 1.
        super().__init__(blockname=blockname, prev_layer_or_block=prev_layer_or_block)
        # 2.
        self.H = units
        self.A = num_heads
        self.causal = causal
        self.layers = []
        # 3.
        self.dropout_layer = Dropout(
            name='dropout1',
            rate=dropout_rate,
            prev_layer_or_block=prev_layer_or_block)
        self.layers.append(self.dropout_layer)
        # 4.
        # H may be something else sice its H_qkv but i think thats the same as H_qkv I DONT KNOW WHERE TO GET H_qkv other than in the qkv passed in
        self.atn_gain = tf.sqrt(self.H/self.A)**(-1)

    def __call__(self, queries, keys, values):
        '''Forward pass through the attention block with activations from the query, key, and value layers.

        Parameters:
        -----------
        queries: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the query layer.
        keys: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the keys layer.
        values: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the values layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The attended values.

        NOTE:
        1. Follow the blueprint from class on computing the various phases of attention (i.e. A_1, A_2, A_3, and A_4).
        2. Refer to the notebook for a refresher on the big-picture equations.
        3. You will need to rely on batch matrix multiplication. The code does not differ from regular multiplication,
        but it affects the setup of the shapes.
        4. It is important to keep track of shapes at the various phases. I suggest keeping track of shapes in code
        comments above each line of code you write.
        5. Don't forget that you pre-computed the attention gain.
        6. Don't forget to incorporate the causal mask to implement causal attention (if that option is turned on).
        The function `tril` from tf_util should be very helpful.11
        '''
        self.batchsize = queries.shape[0]
        self.T = queries.shape[1]
        self.H_qkv = queries.shape[2]

        # the notes have us putting it as (B,T,A,H/A) but then asks for (B,A,T,H/A) later so it is a bit convoluted here
        queries = tf.reshape(queries, shape=(
            self.batchsize, self.T, self.A, self.H//self.A))  # (B,T,A,H/A)
        keys = tf.reshape(keys, shape=(
            self.batchsize, self.T, self.A, self.H//self.A))  # (B,T,A,H/A)
        values = tf.reshape(values, shape=(
            self.batchsize, self.T, self.A, self.H//self.A))  # (B,T,A,H/A)
        values_transposed = tf.transpose(
            values, perm=[0, 2, 1, 3])  # (B,A,T,H/A)
        # transpose last two dims of keys
        keys_transposed = tf.transpose(
            keys, perm=[0, 2, 3, 1])  # (B,T,H/A,A)
        queries_transposed = tf.transpose(
            queries, perm=[0, 2, 1, 3])

        # compute match scores
        a1 = tf.matmul(queries_transposed, keys_transposed) * self.atn_gain

        # step 3: apply causal mask
        if self.causal:
            # tril was causing issues so just gonna index with lower triangular matrix
            causal_mask = tf.linalg.band_part(tf.ones((self.T, self.T)), -1, 0)
            # reshape to (1, 1, T, T) for broadcasting over B and A
            causal_mask = tf.reshape(causal_mask, (1, 1, self.T, self.T))
            # replace 0s (future positions) with -1e9, 1s with 0.0
            neg_inf = tf.constant(-1e9, dtype=tf.float32)
            a2 = tf.where(causal_mask == 1, a1, neg_inf)

        else:
            a2 = a1

        # step 4 compute A3
        a3 = tf.nn.softmax(a2, axis=-1)  # (B,A,T,T)
        # step 5 apply dropout
        a4 = self.dropout_layer(a3)  # (B,A,T,T)

        # step 6: unlock values of V
        v_unlock = a4 @ values_transposed  # (B,A,T,H/A)

        # step 7: convert shapes to original
        v_unlock = tf.transpose(v_unlock, perm=[0, 2, 1, 3])  # (B,T,A,H/A)

        v_unlock = tf.reshape(v_unlock, shape=(
            self.batchsize, self.T, self.H))  # (B,T,H)
        return v_unlock


class MultiHeadAttentionBlock(block.Block):
    '''Block that encapsulates MultiHeadAttention and related blocks. Here is a summary of the block:

    QueryKeyValueBlock → MultiHead Attention → Dense → Dropout

    All the layers/subblocks have H (i.e. num_embed) neurons. The Dense layer uses He init and a linear act fun.

    NOTE: The Dense layer in this block (according to the paper) does NOT use layer norm.
    '''

    def __init__(self, blockname, num_heads, units, prev_layer_or_block, dropout_rate=0.1, causal=True):
        '''MultiHeadAttentionBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        num_heads: int.
            Number of attention heads to use in the attention block.
        units: int.
            Number of neurons in the attention block (H — i.e. embed_dim).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        dropout_rate: float.
            The dropout rate (R) to use in the Dropout layer that is in the attention block that is applied to the
            attention values. The dropout rate is the same for the dropout layer in this block and the attention
            subblock.
        causal: bool.
            Whether to apply a causal mask to remove/mask out the ability for the layer to pay attention to tokens
            that are in the future of the current one in the sequence.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create all the layers and blocks.
        '''
        super().__init__(blockname=blockname, prev_layer_or_block=prev_layer_or_block)
        self.layers = []

        # QueryKeyValueBlock
        self.qkv_block = QueryKeyValueBlock(
            blockname=f"{blockname}_qkv",
            units=units,
            prev_layer_or_block=prev_layer_or_block
        )
        self.layers.append(self.qkv_block)

        # AttentionBlock 
        self.attn_block = AttentionBlock(
            blockname=f"{blockname}_attn",
            num_heads=num_heads,
            units=units,
            prev_layer_or_block=self.qkv_block,
            dropout_rate=dropout_rate,
            causal=causal
        )
        self.layers.append(self.attn_block)

        # Dense
        self.out_proj = Dense(
            name=f"{blockname}_out_proj",
            units=units,
            activation='linear',
            prev_layer_or_block=self.attn_block,
            wt_init='he',
            do_batch_norm=False,
            do_layer_norm=False
        )
        self.layers.append(self.out_proj)

        # Dropout
        self.final_dropout = Dropout(
            name=f"{blockname}_dropout",
            rate=dropout_rate,
            prev_layer_or_block=self.out_proj
        )
        self.layers.append(self.final_dropout)

    def __call__(self, x):
        '''Forward pass through the MultiHead Attention Block.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the layer/block below.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The output netActs
        '''
        queries, keys, values = self.qkv_block(x, x, x)
        attn_output = self.attn_block(queries, keys, values)
        projected = self.out_proj(attn_output)
        output = self.final_dropout(projected)
        return output


class MLPBlock(block.Block):
    '''MLP block that tends to follow the attention block. Composed of the following layers:

    Dense → Dense → Dropout

    Implements a bottleneck design: 1st Dense layer has 4x the units and the 2nd Dense layer has 1x.

    1st Dense layer:
    ----------------
    - Uses the gelu activation function, layernorm

    2nd Dense layer:
    ----------------
    - Uses the linear/identity activation function, no layernorm
    '''

    def __init__(self, blockname, units, prev_layer_or_block, exp_factor=4, dropout_rate=0.1):
        '''MLPBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        units: int.
            Number of neurons in the MLP block dense layers (H — i.e. embed_dim).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        exp_factor: int.
            The expansion factor that scales the number of units in the 1st Dense layer. Controls how large the
            bottleneck is in the block.
        dropout_rate: float.
            The dropout rate (R) to use in the Dropout layer.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create all the layers and blocks.
        '''
        pass

    def __call__(self, x):
        '''Forward pass through the MLPBlock with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the layer/block below.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The output netActs
        '''
        pass


class TransformerBlock(block.Block):
    '''The Transformer Block, composed of a single MultiHeadAtention Block followed by a single MLP Block.'''

    def __init__(self, blockname, units, num_heads, prev_layer_or_block, dropout_rate=0.1):
        '''TransformerBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        units: int.
            Number of neurons in the Transformer block (H — i.e. embed_dim).
        num_heads: int.
            Number of attention heads to use in the attention block.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        dropout_rate: float.
            The dropout rate (R) to use throughout the block.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create all the layers and blocks.
        '''
        pass

    def __call__(self, x):
        '''Forward pass through the Transformer block with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the layer/block below.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The output netActs

        NOTE: Don't forget the residual connections that allows the input to skip to the end of each block.
        '''
        pass


class PositionalEncodingBlock(block.Block):
    '''Block that combines PositionalEncoding layer and a Dropout layer in the following order:

    PositionalEncoding → Dropout
    '''

    def __init__(self, blockname, embed_dim, prev_layer_or_block, dropout_rate=0.1):
        '''PositionalEncodingBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        embed_dim: int.
            Number of neurons in the Embedding layer (H — i.e. embed_dim).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        dropout_rate: float.
            The dropout rate (R) to use in the dropout layer1.

        TODO:
        1. Call and pass in relevant information into the superclass constructor.
        2. Create all the layers.
        '''
        pass

    def __call__(self, x):
        '''Forward pass through the block with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, T, H).
            netActs in the mini-batch from the layer/block below.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, T, H).
            The output netActs
        '''
        pass
