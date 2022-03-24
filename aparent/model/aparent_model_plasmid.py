from __future__ import print_function
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, ConvLSTM2D, BatchNormalization
from keras.layers import Concatenate, Reshape
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from keras import backend as K
import keras.losses

import tensorflow as tf

import isolearn.keras as iso

from aparent.losses import *

def load_aparent_model(batch_size, use_sample_weights=False) :
    
    #APARENT parameters
    seq_input_shape = (1, 205, 4)
    lib_input_shape = (13,)
    distal_pas_shape = (1,)
    num_outputs_iso = 1
    num_outputs_cut = 206

    
    #Shared model definition
    layer_1 = Conv2D(96, (1, 8), padding='valid', activation='relu')
    layer_1_pool = MaxPooling2D(pool_size=(1, 2))
    layer_2 = Conv2D(128, (1, 6), padding='valid', activation='relu')
    layer_dense = Dense(256, activation='relu')
    layer_drop = Dropout(0.2)

    def shared_model(seq_input) :
        return layer_drop(
            layer_dense(
                Flatten()(
                    layer_2(
                        layer_1_pool(
                            layer_1(
                                seq_input
                            )
                        )
                    )
                )
            )
        )

    
    #Plasmid model definition

    #Inputs
    seq_input = Input(shape=seq_input_shape)
    lib_input = Input(shape=lib_input_shape)
    plasmid_count = Input(shape=(1,))

    #Outputs
    true_iso = Input(shape=(num_outputs_iso,))
    true_cut = Input(shape=(num_outputs_cut,))

    plasmid_out_shared = Concatenate()([shared_model(seq_input), lib_input])

    plasmid_out_cut = Dense(num_outputs_cut, activation='softmax', kernel_initializer='zeros')(plasmid_out_shared)
    #plasmid_out_iso = Lambda(lambda cl: K.sum(cl[:, 80:80+25], axis=-1))(plasmid_out_cut)
    plasmid_out_iso = Dense(num_outputs_iso, activation='sigmoid', kernel_initializer='zeros')(plasmid_out_shared)

    plasmid_model = Model(
        inputs=[
            seq_input,
            lib_input
        ],
        outputs=[
            plasmid_out_iso,
            plasmid_out_cut
        ]
    )

    
    #Loss model definition
    sigmoid_kl_divergence = get_sigmoid_kl_divergence(batch_size, use_sample_weights=use_sample_weights)
    kl_divergence = get_kl_divergence(batch_size, use_sample_weights=use_sample_weights)
    
    plasmid_loss_iso = Lambda(sigmoid_kl_divergence, output_shape = (1,))([true_iso, plasmid_out_iso, plasmid_count])
    plasmid_loss_cut = Lambda(kl_divergence, output_shape = (1,))([true_cut, plasmid_out_cut, plasmid_count])

    total_loss = Lambda(
        lambda l: 0.5 * l[0] + 0.5 * l[1],
        output_shape = (1,)
    )(
        [
            plasmid_loss_iso,
            plasmid_loss_cut
        ]
    )

    loss_model = Model([
        seq_input,
        lib_input,
        plasmid_count,
        true_iso,
        true_cut
    ], total_loss)

    return [ ('plasmid_iso_cut', plasmid_model), ('loss', loss_model) ]





