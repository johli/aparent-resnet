import pandas as pd
import scipy
import numpy as np
import scipy.sparse as sp

from scipy.stats import pearsonr

import operator

def append_predictions(seq_df, seq_cuts, variant_df, variant_cuts_var, variant_cuts_ref, pred_df, cuts_pred) :
    #Join dataframe with prediction table and calculate true cut probabilities

    seq_df['row_index_true'] = np.arange(len(seq_df), dtype=np.int)
    pred_df['row_index_pred'] = np.arange(len(pred_df), dtype=np.int)

    seq_df = seq_df.join(pred_df.set_index('master_seq'), on='master_seq', how='inner').copy().reset_index(drop=True)

    seq_cuts = seq_cuts[np.ravel(seq_df['row_index_true'].values), :]
    cut_pred = np.array(cuts_pred[np.ravel(seq_df['row_index_pred'].values), :].todense())
    cut_pred = np.concatenate([np.zeros((cut_pred.shape[0], 1)), cut_pred[:, :184], cut_pred[:, 185].reshape(-1, 1)], axis=-1)

    cut_true = np.concatenate([np.array(seq_cuts[:, 180 + 20: 180 + 205].todense()), np.array(seq_cuts[:, -1].todense()).reshape(-1, 1)], axis=-1)
    #Add small pseudo count to true cuts
    cut_true += 0.0005
    cut_true = cut_true / np.sum(cut_true, axis=-1).reshape(-1, 1)

    seq_df['cut_prob_true'] = [cut_true[i, :] for i in range(len(seq_df))]
    seq_df['cut_prob_pred'] = [cut_pred[i, :] for i in range(len(seq_df))]


    seq_df['iso_pred_from_cuts'] = np.sum(cut_pred[:, 49: 90], axis=-1)
    seq_df['logodds_pred_from_cuts'] = np.log(seq_df['iso_pred_from_cuts'] / (1.0 - seq_df['iso_pred_from_cuts']))

    seq_df['mean_logodds_pred'] = (seq_df['logodds_pred'] + seq_df['logodds_pred_from_cuts']) / 2.0

    #Join variant dataframe with prediction table and calculate true cut probabilities

    variant_df['row_index_true'] = np.arange(len(variant_df), dtype=np.int)

    variant_df = variant_df.join(pred_df.rename(columns={'iso_pred' : 'iso_pred_var', 'logodds_pred' : 'logodds_pred_var', 'row_index_pred' : 'row_index_pred_var'}).set_index('master_seq'), on='master_seq', how='inner').copy().reset_index(drop=True)
    variant_df = variant_df.join(pred_df.rename(columns={'iso_pred' : 'iso_pred_ref', 'logodds_pred' : 'logodds_pred_ref', 'row_index_pred' : 'row_index_pred_ref'}).set_index('master_seq'), on='wt_seq', how='inner').copy().reset_index(drop=True)

    variant_cuts_var = variant_cuts_var[np.ravel(variant_df['row_index_true'].values), :]
    variant_cuts_ref = variant_cuts_ref[np.ravel(variant_df['row_index_true'].values), :]

    cut_true_var = np.concatenate([np.array(variant_cuts_var[:, 180 + 20: 180 + 205].todense()), np.array(variant_cuts_var[:, -1].todense()).reshape(-1, 1)], axis=-1)
    #Add small pseudo count to true cuts
    cut_true_var += 0.0005
    cut_true_var = cut_true_var / np.sum(cut_true_var, axis=-1).reshape(-1, 1)

    cut_true_ref = np.concatenate([np.array(variant_cuts_ref[:, 180 + 20: 180 + 205].todense()), np.array(variant_cuts_ref[:, -1].todense()).reshape(-1, 1)], axis=-1)
    #Add small pseudo count to true cuts
    cut_true_ref += 0.0005
    cut_true_ref = cut_true_ref / np.sum(cut_true_ref, axis=-1).reshape(-1, 1)

    cut_pred_var = np.array(cuts_pred[np.ravel(variant_df['row_index_pred_var'].values), :].todense())
    cut_pred_var = np.concatenate([np.zeros((cut_pred_var.shape[0], 1)), cut_pred_var[:, :184], cut_pred_var[:, 185].reshape(-1, 1)], axis=-1)

    cut_pred_ref = np.array(cuts_pred[np.ravel(variant_df['row_index_pred_ref'].values), :].todense())
    cut_pred_ref = np.concatenate([np.zeros((cut_pred_ref.shape[0], 1)), cut_pred_ref[:, :184], cut_pred_ref[:, 185].reshape(-1, 1)], axis=-1)

    variant_df['cut_prob_true_var'] = [cut_true_var[i, :] for i in range(len(variant_df))]
    variant_df['cut_prob_pred_var'] = [cut_pred_var[i, :] for i in range(len(variant_df))]

    variant_df['cut_prob_true_ref'] = [cut_true_ref[i, :] for i in range(len(variant_df))]
    variant_df['cut_prob_pred_ref'] = [cut_pred_ref[i, :] for i in range(len(variant_df))]


    variant_df['iso_pred_from_cuts_var'] = np.sum(cut_pred_var[:, 49: 90], axis=-1)
    variant_df['iso_pred_from_cuts_ref'] = np.sum(cut_pred_ref[:, 49: 90], axis=-1)
    variant_df['logodds_pred_from_cuts_var'] = np.log(variant_df['iso_pred_from_cuts_var'] / (1.0 - variant_df['iso_pred_from_cuts_var']))
    variant_df['logodds_pred_from_cuts_ref'] = np.log(variant_df['iso_pred_from_cuts_ref'] / (1.0 - variant_df['iso_pred_from_cuts_ref']))

    variant_df['delta_logodds_pred'] = variant_df['logodds_pred_var'] - variant_df['logodds_pred_ref']
    variant_df['delta_logodds_pred_from_cuts'] = variant_df['logodds_pred_from_cuts_var'] - variant_df['logodds_pred_from_cuts_ref']

    variant_df['mean_delta_logodds_pred'] = (variant_df['delta_logodds_pred'] + variant_df['delta_logodds_pred_from_cuts']) / 2.0

    return seq_df, variant_df
