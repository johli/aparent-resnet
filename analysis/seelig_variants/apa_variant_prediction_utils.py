import pandas as pd
import scipy
import numpy as np
import scipy.sparse as sp

from scipy.stats import pearsonr

import operator

def append_predictions(seq_df, seq_cuts, variant_df, variant_cuts_var, variant_cuts_ref, pred_df, cuts_pred, proximal_start=49, proximal_end=90, isoform_pseudo_count=1.0) :
    #Join dataframe with prediction table and calculate true cut probabilities

    seq_df['row_index_true'] = np.arange(len(seq_df), dtype=np.int)
    pred_df['row_index_pred'] = np.arange(len(pred_df), dtype=np.int)

    seq_df = seq_df.join(pred_df.set_index('master_seq'), on='master_seq', how='inner').copy().reset_index(drop=True)

    seq_cuts = seq_cuts[np.ravel(seq_df['row_index_true'].values), :]
    cut_true = np.concatenate([np.array(seq_cuts[:, 180 + 20: 180 + 205].todense()), np.array(seq_cuts[:, -1].todense()).reshape(-1, 1)], axis=-1)# - 1
    
    seq_df['proximal_count'] = [np.sum(cut_true[i, proximal_start:proximal_end]) for i in range(len(seq_df))]
    seq_df['total_count'] = [np.sum(cut_true[i, :]) for i in range(len(seq_df))]
    
    seq_df['iso_true'] = (seq_df['proximal_count'] + isoform_pseudo_count) / (seq_df['total_count'] + 2. * isoform_pseudo_count)
    seq_df['logodds_true'] = np.log(seq_df['iso_true'] / (1.0 - seq_df['iso_true']))

    if cuts_pred is not None :
        cut_pred = np.array(cuts_pred[np.ravel(seq_df['row_index_pred'].values), :].todense())
        
        seq_df['iso_pred_from_cuts'] = [np.clip(np.sum(cut_pred[i, proximal_start:proximal_end]), 1e-6, 1. - 1e-6) for i in range(len(seq_df))]
        seq_df['logodds_pred_from_cuts'] = np.log(seq_df['iso_pred_from_cuts'] / (1.0 - seq_df['iso_pred_from_cuts']))

        seq_df['mean_logodds_pred'] = (seq_df['logodds_pred'] + seq_df['logodds_pred_from_cuts']) / 2.0

    #Join variant dataframe with prediction table and calculate true cut probabilities

    variant_df['row_index_true'] = np.arange(len(variant_df), dtype=np.int)

    variant_df = variant_df.join(pred_df.rename(columns={'iso_pred' : 'iso_pred_var', 'logodds_pred' : 'logodds_pred_var', 'row_index_pred' : 'row_index_pred_var'}).set_index('master_seq'), on='master_seq', how='inner').copy().reset_index(drop=True)
    variant_df = variant_df.join(pred_df.rename(columns={'iso_pred' : 'iso_pred_ref', 'logodds_pred' : 'logodds_pred_ref', 'row_index_pred' : 'row_index_pred_ref'}).set_index('master_seq'), on='wt_seq', how='inner').copy().reset_index(drop=True)

    variant_cuts_var = variant_cuts_var[np.ravel(variant_df['row_index_true'].values), :]
    variant_cuts_ref = variant_cuts_ref[np.ravel(variant_df['row_index_true'].values), :]

    cut_true_var = np.concatenate([np.array(variant_cuts_var[:, 180 + 20: 180 + 205].todense()), np.array(variant_cuts_var[:, -1].todense()).reshape(-1, 1)], axis=-1)# - 1
    
    cut_true_ref = np.concatenate([np.array(variant_cuts_ref[:, 180 + 20: 180 + 205].todense()), np.array(variant_cuts_ref[:, -1].todense()).reshape(-1, 1)], axis=-1)# - 1
    
    variant_df['proximal_count_var'] = [np.sum(cut_true_var[i, proximal_start:proximal_end]) for i in range(len(variant_df))]
    variant_df['total_count_var'] = [np.sum(cut_true_var[i, :]) for i in range(len(variant_df))]
    
    variant_df['iso_true_var'] = (variant_df['proximal_count_var'] + isoform_pseudo_count) / (variant_df['total_count_var'] + 2. * isoform_pseudo_count)
    variant_df['logodds_true_var'] = np.log(variant_df['iso_true_var'] / (1.0 - variant_df['iso_true_var']))
    
    variant_df['proximal_count_ref'] = [np.sum(cut_true_ref[i, proximal_start:proximal_end]) for i in range(len(variant_df))]
    variant_df['total_count_ref'] = [np.sum(cut_true_ref[i, :]) for i in range(len(variant_df))]
    
    variant_df['iso_true_ref'] = (variant_df['proximal_count_ref'] + isoform_pseudo_count) / (variant_df['total_count_ref'] + 2. * isoform_pseudo_count)
    variant_df['logodds_true_ref'] = np.log(variant_df['iso_true_ref'] / (1.0 - variant_df['iso_true_ref']))
    
    variant_df['delta_logodds_true'] = variant_df['logodds_true_var'] - variant_df['logodds_true_ref']
    
    variant_df['delta_logodds_pred'] = variant_df['logodds_pred_var'] - variant_df['logodds_pred_ref']
    
    if cuts_pred is not None :
        cut_pred_var = np.array(cuts_pred[np.ravel(variant_df['row_index_pred_var'].values), :].todense())
        cut_pred_ref = np.array(cuts_pred[np.ravel(variant_df['row_index_pred_ref'].values), :].todense())
        
        variant_df['iso_pred_from_cuts_var'] = [np.clip(np.sum(cut_pred_var[i, proximal_start:proximal_end]), 1e-6, 1. - 1e-6) for i in range(len(variant_df))]
        variant_df['iso_pred_from_cuts_ref'] = [np.clip(np.sum(cut_pred_ref[i, proximal_start:proximal_end]), 1e-6, 1. - 1e-6) for i in range(len(variant_df))]

        variant_df['logodds_pred_from_cuts_var'] = np.log(variant_df['iso_pred_from_cuts_var'] / (1.0 - variant_df['iso_pred_from_cuts_var']))
        variant_df['logodds_pred_from_cuts_ref'] = np.log(variant_df['iso_pred_from_cuts_ref'] / (1.0 - variant_df['iso_pred_from_cuts_ref']))
        
        variant_df['delta_logodds_pred_from_cuts'] = variant_df['logodds_pred_from_cuts_var'] - variant_df['logodds_pred_from_cuts_ref']

        variant_df['mean_delta_logodds_pred'] = (variant_df['delta_logodds_pred'] + variant_df['delta_logodds_pred_from_cuts']) / 2.0

    return seq_df, variant_df
