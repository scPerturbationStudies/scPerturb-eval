import os
import torch
import matplotlib.pyplot as plt
from scipy.stats import ranksums, wilcoxon
from scipy.stats import spearmanr, pearsonr, linregress
from scipy.stats import wasserstein_distance_nd
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from functools import reduce
# import ot
import scanpy as sc
import anndata as ad
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects.vectors import DataFrame as RDataFrame
from collections import defaultdict
from itertools import combinations
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from scipy.sparse import issparse, csr_matrix
from utils.common import calculate_cpm

np.random.seed(42)


def spearman_corr(pred_adata, true_adata):
    
    if issparse(pred_adata.X):
        pred_adata.X = pred_adata.X.toarray()
    if issparse(true_adata.X):
        true_adata.X = true_adata.X.toarray()
    pred_mean = np.mean(pred_adata.X, axis = 0)
    true_mean = np.mean(true_adata.X,axis = 0)
    if len(np.unique(pred_mean))==1 or len(np.unique(true_mean))==1:
        pred_mean = np.append(pred_mean, 1)
        true_mean = np.append(true_mean, 1)
    corr = spearmanr(pred_mean, true_mean)[0]
    return corr

def pearson_corr(pred_adata, true_adata):

    if issparse(pred_adata.X):
        pred_adata.X = pred_adata.X.toarray()
    if issparse(true_adata.X):
        true_adata.X = true_adata.X.toarray()
    pred_mean = np.mean(pred_adata.X, axis = 0)
    true_mean = np.mean(true_adata.X, axis = 0)
    if len(np.unique(pred_mean))==1 or len(np.unique(true_mean))==1:
        pred_mean = np.append(pred_mean, 1)
        true_mean = np.append(true_mean, 1)
    corr = pearsonr(pred_mean, true_mean)[0]
    return corr

def r2(pred_adata, true_adata):
    
    if issparse(pred_adata.X):
        pred_adata.X = pred_adata.X.toarray()
    if issparse(true_adata.X):
        true_adata.X = true_adata.X.toarray()
    pred_mean = np.mean(pred_adata.X, axis = 0)
    true_mean = np.mean(true_adata.X,axis = 0)
    if len(np.unique(pred_mean))==1 or len(np.unique(true_mean))==1:
        pred_mean = np.append(pred_mean, 1)
        true_mean = np.append(true_mean, 1)
    slope, intercept, r_value, _, _ = linregress(pred_mean, true_mean)
    r_squared = r_value ** 2
    return r_squared

def r2_skl(pred_adata, true_adata):
    
    pred_mean = np.mean(pred_adata.X, axis = 0)
    true_mean = np.mean(true_adata.X,axis = 0)
    r_squared = r2_score(pred_mean, true_mean)
    return r_squared

def stepwise_r2(ctrl_adata, true_adata, pred_adata):
    ctrl_means = np.asarray(ctrl_adata.X.mean(axis=0)).ravel()
    stim_means = np.asarray(true_adata.X.mean(axis=0)).ravel()
    pred_mean = np.asarray(pred_adata.X.mean(axis=0)).ravel()
    genes = ctrl_adata.var_names
    sorted_idx = np.argsort(ctrl_means) 
    sorted_genes = genes[sorted_idx]
    stim_sorted = stim_means[sorted_idx]
    pred_sorted = pred_mean[sorted_idx]

    start_value = 2
    num_genes = range(start_value, len(sorted_genes) + 1)
    corr_list = []
    r2_list = []
    for i in num_genes:
        if len(np.unique(pred_sorted[:i])) == 1 or len(np.unique(stim_sorted[:i])) == 1:
            r_squared = r2_score(pred_sorted[:i], stim_sorted[:i])
            corr = 0 if r_squared<0 else np.sqrt(r_squared)
        else:
            slope, intercept, r_value, _, _ = linregress(pred_sorted[:i], stim_sorted[:i])
            r_squared = r_value ** 2
            corr = pearsonr(pred_sorted[:i], stim_sorted[:i])[0]
        corr_list.append(corr)
        r2_list.append(r_squared)
    auc_corr = np.trapz(corr_list, x=num_genes) / (num_genes[-1] - num_genes[0])
    auc_r2 = np.trapz(r2_list, x=num_genes) / (num_genes[-1] - num_genes[0])

    return auc_corr, auc_r2


def z_correlation(pred_adata, true_adata, ctrl_adata):

    pred_mean = np.mean(pred_adata.X, axis = 0)
    true_mean = np.mean(true_adata.X,axis = 0)
    ctrl_mean = np.mean(ctrl_adata.X,axis = 0)
    ctrl_std = np.std(ctrl_adata.X, axis=0)

    nonzero_ctrl = ctrl_mean != 0
    pred_mean = pred_mean[nonzero_ctrl]
    true_mean = true_mean[nonzero_ctrl]
    ctrl_mean = ctrl_mean[nonzero_ctrl]
    ctrl_std = ctrl_std[nonzero_ctrl]

    pred_mean = (pred_mean - ctrl_mean)/ctrl_std
    true_mean = (true_mean - ctrl_mean)/ctrl_std

    corr = pearsonr(pred_mean, true_mean)[0]
    return corr
    

def spearman_corr_per_gene(pred_df_ratio, true_df_ratio, remove_nan=True):
    corr = []
    pred_df_ratio = pred_df_ratio.fillna(0)
    true_df_ratio = true_df_ratio.fillna(0)
    for col in range(len(pred_df_ratio.columns)):
        corr.append(spearmanr(pred_df_ratio.iloc[:,col], true_df_ratio.iloc[:,col])[0])

    print(len([y for y in corr if np.isnan(y)]))
    if remove_nan:
        corr = [x for x in corr if x==x]  
     
    return corr


def rigorous_spearman_corr_per_gene(true_adata_list, pred_adata_list, true_s_adata_list):
    n_splits = 10
    test_fraction = 0.5
    corr_dict = defaultdict(list)
    output_dict = {}
    standardized_dict = {}

    for _ in range(n_splits):
        print(f"split {_+1}")
        temp_test = []
        temp_model = defaultdict(list)
        for primary in range(len(true_adata_list)):
            n_test = int(test_fraction * true_adata_list[primary].n_obs)
            test_indices = np.random.choice(true_adata_list[primary].n_obs, size=n_test, replace=False)
            train_indices = np.setdiff1d(np.arange(true_adata_list[primary].n_obs), test_indices)
            train_adata = true_adata_list[primary][train_indices].copy()
            test_adata = true_adata_list[primary][test_indices].copy()

            temp_test.append(np.mean(test_adata.X, axis=0))
            temp_model['true'].append(np.mean(train_adata.X, axis=0))
            for model in pred_adata_list.keys():
                sampled_adata = adata_sample(pred_adata_list[model][primary], n_test)
                temp_model[model].append(np.mean(sampled_adata.X, axis=0))
            sampled_true_all = adata_sample(true_s_adata_list[primary], n_test)
            temp_model['random_all'].append(np.mean(sampled_true_all.X, axis=0))
        for model in temp_model.keys():
            corr_dict[model].append(spearman_corr_per_gene(
                pd.DataFrame(np.stack(temp_model[model], axis=0)),
                pd.DataFrame(np.stack(temp_test, axis=0)),
                remove_nan=False))
    for model in corr_dict.keys():
        output_dict[model] = np.mean(pd.DataFrame(np.stack(corr_dict[model], axis=0)), axis=0)
        output_dict[model] = [x for x in output_dict[model] if x==x]
        if model != 'true':
            pred = pd.DataFrame(np.stack(corr_dict[model], axis=0))
            true = pd.DataFrame(np.stack(corr_dict['true'], axis=0))
            random = pd.DataFrame(np.stack(corr_dict['random_all'], axis=0))
            standardized_dict[model] = (pred - true)
            # standardized_dict[model] = standardized_dict[model].replace([np.inf, -np.inf], np.nan)
            standardized_dict[model] = np.mean(standardized_dict[model], axis=0)
            standardized_dict[model] = [x for x in standardized_dict[model] if x==x]

    return output_dict, standardized_dict

def prediction_error(ctrl_df_ratio, pred_dict, true_df_ratio):
    top_genes = np.mean(ctrl_df_ratio, axis=0).nlargest(1000).index
    pred_top = {}
    true_top = true_df_ratio[top_genes]
    l2_distance = {}
    for model in pred_dict.keys():
        pred_top[model] = pred_dict[model][top_genes]
        l2_distance[model] = np.asarray(np.sqrt(((pred_top[model] - true_top) ** 2).sum(axis=1))).reshape(-1)
    return l2_distance 

def mse(pred_adata, true_adata):
    if issparse(pred_adata.X):
        pred_adata.X = pred_adata.X.toarray()
    if issparse(true_adata.X):
        true_adata.X = true_adata.X.toarray()
    pred_mean = np.mean(pred_adata.X, axis = 0)
    true_mean = np.mean(true_adata.X,axis = 0)
    num_genes = len(pred_mean)
    # mse_value = (((pred_mean - true_mean)**2).sum())/num_genes
    mse_value = (np.square(pred_mean - true_mean).sum()) / num_genes
    return mse_value

def cosine_sim(pred_adata, true_adata):
    if issparse(pred_adata.X):
        pred_adata.X = pred_adata.X.toarray()
    if issparse(true_adata.X):
        true_adata.X = true_adata.X.toarray()
    pred_mean = np.mean(pred_adata.X, axis = 0)
    true_mean = np.mean(true_adata.X,axis = 0)
    pred_reshaped = np.array(pred_mean).reshape(1,-1)
    true_reshaped = np.array(true_mean).reshape(1,-1)

    cosine_sim = cosine_similarity(pred_reshaped, true_reshaped)[0][0]
    return cosine_sim

# def ot_rank_per_gene(pred_adata, true_adata):
#     pred_df = pred_adata.to_df()
#     true_df = true_adata.to_df()
#     genes = pred_df.shape[1]
#     ranks = []

#     for gene in range(genes):
#         pred_list= [pd.DataFrame(pred_df.iloc[:,gene])] * genes
#         true_list = [pd.DataFrame(true_df.iloc[:,n]) for n in range(genes)]
#         distances = wasserstein(pred_list, true_list)
#         sorted_indx = np.argsort(distances)
#         ranks.append(sorted_indx.tolist().index(gene))

#     return ranks


def wasserstein(pred_list, true_list, unused_parameter=True):
    distances = []
    for pred, true in zip(pred_list, true_list):
        pred = pred.toarray() if not isinstance(pred, np.ndarray) else pred
        true = true.toarray() if not isinstance(true, np.ndarray) else true
        wasserstein_dist = wasserstein_distance_nd(pred, true)
        distances.append(wasserstein_dist)

    return distances


# def wasserstein_deprecated(pred_list, true_list, normalization=False):
def sinkhorn_mahalanobis(pred_list, true_list, normalization=False):
    distances = []
    for pred, true in zip(pred_list, true_list):
        if normalization:
            true_expression = np.asarray(true)/np.asarray(true).sum(axis=1, keepdims=True)
            pred_expression = np.asarray(pred)/np.asarray(pred).sum(axis=1, keepdims=True)
        else:
            true_expression = np.asarray(true)
            pred_expression = np.asarray(pred)
        
        # Create weights for the distributions (equal weight per cell)
        weights_true = np.ones(true_expression.shape[0]) / true_expression.shape[0]
        weights_pred = np.ones(pred_expression.shape[0]) / pred_expression.shape[0]

        pca = PCA(n_components=int(min(true_expression.shape[1], true_expression.shape[0]-1)/4))
        X_reduced = pca.fit_transform(true_expression)
        Y_reduced = pca.transform(pred_expression)

        # Step 2: Define the cost matrix (distance between cells)
        # cost_matrix = ot.dist(true_expression, pred_expression, metric="euclidean")
        # cost_matrix = ot.dist(true_expression, pred_expression, metric="Mahalanobis")
        cost_matrix = ot.dist(X_reduced, Y_reduced, metric="Mahalanobis")

        # Step 3: Solve the optimal transport problem
        # optimal_transport_plan = ot.emd(weights_true, weights_pred, cost_matrix)
        # optimal_transport_plan = ot.sinkhorn(weights_true, weights_pred, cost_matrix, reg=1e-2)
        # distance = np.sum(optimal_transport_plan * cost_matrix)
        distance = ot.sinkhorn2(weights_true, weights_pred, cost_matrix, reg=0.1)

        distances.append(distance)

    return distances

def corr_top_k(modality_adata, pred_mean, primary_variable, ood_primary):

    modality_df_ratio = modality_adata.to_df()
    n_values = np.arange(10, modality_df_ratio.shape[0]+1, 10)
    corrs = []

    for i in range(modality_df_ratio.shape[0]):
        corrs.append(spearmanr(modality_df_ratio.iloc[i,:], pred_mean)[0])
    # print(corrs)
    ascending_indices = np.argsort(corrs)
    descending_indices = ascending_indices[::-1]
    primary_lables = (modality_adata.obs[primary_variable] == ood_primary).astype(int)
    # print(primary_lables.value_counts())

    top_k = [
        (primary_lables.iloc[descending_indices[0:n]]).sum() for n in n_values
    ]
    
    return pd.DataFrame({'n': n_values, 'top_n': top_k})

def corr_top_k_per_primary(pred_adata, true_adata, ctrl_adata):
    pred = pred_adata.to_df()
    true = true_adata.to_df()
    ctrl = ctrl_adata.to_df()
    pred_num = pred.shape[0]
    true_num = true.shape[0]
    ctrl_num = ctrl.shape[0]
    n_values = np.arange(10, true_num+ctrl_num+1, 10)
    pred_abs_store = []
    pred_ratio_store = []


    for pred_cell in range(pred_num):
        corrs = []
        for i in range(true_num):
            corrs.append(spearmanr(true.iloc[i,:], pred.iloc[pred_cell,:])[0])
        for i in range(ctrl_num):
            corrs.append(spearmanr(ctrl.iloc[i,:], pred.iloc[pred_cell,:])[0])
        descending_indices = np.argsort(corrs)[::-1]
        labels = (descending_indices < true_num).astype(int)
        temp_abs = [(labels[0:n]).sum() for n in n_values]
        temp_ratio = [(labels[0:n]).sum()/(n) for n in n_values]
        denominator = true_num/(true_num+ctrl_num)
        pred_abs_store.append(temp_abs)
        pred_ratio_store.append([a/denominator for a in temp_ratio])

    df_abs = pd.DataFrame(pred_abs_store)
    df_ratio = pd.DataFrame(pred_ratio_store)
    top_k_mean_abs = df_abs.mean(axis=0).values
    top_k_mean_ratio = df_ratio.mean(axis=0).values

    base = [(true_num*n)/(true_num+ctrl_num) for n in n_values]
    return pd.DataFrame({'n': n_values, 'true_positives': top_k_mean_abs}),\
        pd.DataFrame({'n': n_values, 'true_positives': top_k_mean_ratio}),\
        pd.DataFrame({'n': n_values, 'true_positives': base})


def true_deg_pos(pred_adata, true_adata, ctrl_adata, zeros=False):
    
    n_values = np.arange(10, 501, 10)
    
    pred = pred_adata.to_df()
    true = true_adata.to_df()
    ctrl = ctrl_adata.to_df()
                       
    down_pred, up_pred = get_pvalues(ctrl, pred, zeros)
    down_true, up_true = get_pvalues(ctrl, true, zeros)
    
    # if deg_type == 'down':
    true_labels_down = (down_true < 0.05).astype(int)     
    sorted_indices_down = np.argsort(down_pred)
        
    # elif deg_type == 'up':
    true_labels_up = (up_true < 0.05).astype(int)     
    sorted_indices_up = np.argsort(up_pred)
               
    true_positives_down = [
        # true_labels.iloc[0, sorted_indices.iloc[0, :n].values].sum() for n in n_values
        true_labels_down.iloc[0, sorted_indices_down[0, :n]].sum() for n in n_values
        ]
    true_positives_up = [
        # true_labels.iloc[0, sorted_indices.iloc[0, :n].values].sum() for n in n_values
        true_labels_up.iloc[0, sorted_indices_up[0, :n]].sum() for n in n_values
        ]
    
    return pd.DataFrame({'n': n_values, 'true_positives': true_positives_down}), \
        pd.DataFrame({'n': n_values, 'true_positives': true_positives_up})


def sorted_pvalue_common_genes(pred_adata, true_adata, ctrl_adata, zeros=False):
    
    n_values = np.arange(10, 501, 10)
    
    pred = pred_adata.to_df()
    true = true_adata.to_df()
    ctrl = ctrl_adata.to_df()
                       
    down_pred, up_pred = get_pvalues(ctrl, pred, zeros)
    down_true, up_true = get_pvalues(ctrl, true, zeros)

    sorted_true_down = np.argsort(down_true)
    sorted_pred_down = np.argsort(down_pred)

    sorted_true_up = np.argsort(up_true)
    sorted_pred_up = np.argsort(up_pred)

    sorted_common_down = [
        len(set(sorted_true_down[0, :n]).intersection(sorted_pred_down[0, :n])) for n in n_values
    ]
    sorted_common_up = [
        len(set(sorted_true_up[0, :n]).intersection(sorted_pred_up[0, :n])) for n in n_values
    ]
    
    return pd.DataFrame({'n': n_values, 'true_positives': sorted_common_down}),\
          pd.DataFrame({'n': n_values, 'true_positives': sorted_common_up})


def sorted_pvalue_rank(pred_adata, true_adata, ctrl_adata, zeros=False):
    
    n_values = np.arange(10, 501, 10)
    num_genes = pred_adata.X.shape[1]
    
    pred = pred_adata.to_df()
    true = true_adata.to_df()
    ctrl = ctrl_adata.to_df()
                       
    down_pred, up_pred = get_pvalues(ctrl, pred, zeros)
    down_true, up_true = get_pvalues(ctrl, true, zeros)

    sorted_true_down = list(np.argsort(down_true).flatten())
    sorted_pred_down = list(np.argsort(down_pred).flatten())

    sorted_true_up = list(np.argsort(up_true).flatten())
    sorted_pred_up = list(np.argsort(up_pred).flatten())

    ranked_down = [(sorted_pred_down.index(sorted_true_down[i]))/num_genes for i in range(500)]
    ranked_up = [(sorted_pred_up.index(sorted_true_up[i]))/num_genes for i in range(500)]

    sorted_rank_down = [np.mean(ranked_down[0:n]) for n in n_values]
    sorted_rank_up = [np.mean(ranked_up[0:n]) for n in n_values]

    perfect = np.arange(num_genes)/num_genes
    perfect_model = [np.mean(perfect[0:n]) for n in n_values]
    
    return pd.DataFrame({'n': n_values, 'true_positives': sorted_rank_down}),\
        pd.DataFrame({'n': n_values, 'true_positives': sorted_rank_up}),\
        pd.DataFrame({'n': n_values, 'true_positives': perfect_model})

def combined_deg(pred_adata, true_adata, ctrl_adata, zeros=False):
    
    n_values = np.arange(10, 501, 10)
    
    pred = pred_adata.to_df()
    true = true_adata.to_df()
    ctrl = ctrl_adata.to_df()
                       
    down_pred, _ = get_pvalues(ctrl, pred, zeros, all_conditions=True)
    down_true, _ = get_pvalues(ctrl, true, zeros, all_conditions=True)

    return calculate_combined_deg(down_pred, down_true, n_values)


def combined_deg_seurat(pred_adata, true_adata, ctrl_adata, test_use, find_variable_features=False, scale_factor=10000):

    n_values = np.arange(10, 501, 10)


    total_count = 1e4
    if issparse(pred_adata.X):
        exp_data = np.exp(pred_adata.X.data) - 1
        pred_adata.layers['counts'] = csr_matrix((exp_data, pred_adata.X.indices, pred_adata.X.indptr),
                                                shape=pred_adata.X.shape) / total_count
    else:
        pred_adata.layers['counts'] = (np.exp(pred_adata.X) - 1) / total_count
    # pred_adata.layers['counts'] = (np.exp(np.asarray(pred_adata.X))-1)/total_count
    
    combined_true = ad.concat([true_adata, ctrl_adata], label="batch", keys=["stimulated", "ctrl"])
    combined_pred = ad.concat([pred_adata, ctrl_adata], label="batch", keys=["stimulated", "ctrl"])
    combined_pred.obs_names_make_unique()
    down_true = seurat_get_down_pvalues(combined_true, "batch", "stimulated", "ctrl", test_use, find_variable_features, scale_factor)
    down_pred = seurat_get_down_pvalues(combined_pred, "batch", "stimulated", "ctrl", test_use, find_variable_features, scale_factor)

    return calculate_combined_deg(down_pred, down_true, n_values)


def calculate_combined_deg(down_pred, down_true, n_values):
    sorted_true = np.argsort(down_true)
    sorted_pred = np.argsort(down_pred)

    sorted_combined_common = [
        len(set(sorted_true[0, :int(n/2)]).intersection(sorted_pred[0, :int(n/2)])) +\
            len(set(sorted_true[0, -int(n/2):]).intersection(sorted_pred[0, -int(n/2):])) for n in n_values
    ]
    
    return pd.DataFrame({'n': n_values, 'true_positives': sorted_combined_common})


def combined_deg_basemodel(adata, primary_variable, modality_variable, ood_modality, zeros=False):
    n_values = np.arange(10, 501, 10)
    primaries = adata.obs[primary_variable].unique()

    pvalue_dict = get_pvalue_dict(adata, primary_variable, modality_variable, ood_modality, zeros)
    basemodel_deg_mean = {}
    basemodel_deg_min = {}

    for ood in primaries:
        non_ood_pvalues = [pvalue_dict[a] for a in primaries if a != ood]
        aggregated_not_ood_pvalues = reduce(lambda a, b: a.add(b), non_ood_pvalues)
        aggregated_not_ood_pvalues = aggregated_not_ood_pvalues/(len(primaries)-1)
        basemodel_deg_mean[ood] = calculate_combined_deg(aggregated_not_ood_pvalues, pvalue_dict[ood], n_values)
        min_not_ood_pvalues = pd.DataFrame(pd.concat(non_ood_pvalues).min(axis=0)).transpose()
        basemodel_deg_min[ood] = calculate_combined_deg(min_not_ood_pvalues, pvalue_dict[ood], n_values)

    return basemodel_deg_mean, basemodel_deg_min

def get_pvalue_dict(adata, primary_variable, modality_variable, ood_modality, zeros=False):
    primaries = adata.obs[primary_variable].unique()
    
    ctrl = adata[adata.obs[modality_variable] != ood_modality]
    stim = adata[adata.obs[modality_variable] == ood_modality]

    pvalue_dict = {}

    for ood in primaries:
        pvalue_dict[ood], _ = get_pvalues(ctrl[ctrl.obs[primary_variable]==ood].to_df(), 
                                       stim[stim.obs[primary_variable]==ood].to_df(), zeros,True)
    return pvalue_dict


def get_pvalue_dict_seurat(adata, primary_variable, modality_variable, ood_modality, control_key):
    primaries = adata.obs[primary_variable].unique()
    pvalue_dict = {}

    for ood in primaries:
        pvalue_dict[ood] = seurat_get_down_pvalues(adata[adata.obs[primary_variable]==ood].copy(), 
            modality_variable, ood_modality, control_key, "MAST")
    
    return pvalue_dict

    
def add_basemodel(pvalue_dict, ood_primary):
    n_values = np.arange(10, 501, 10)
    primaries = pvalue_dict.keys()

    non_ood_pvalues = [pvalue_dict[a] for a in primaries if a != ood_primary]
    aggregated_not_ood_pvalues = reduce(lambda a, b: a.add(b), non_ood_pvalues)
    aggregated_not_ood_pvalues = aggregated_not_ood_pvalues/(len(primaries)-1)
    basemodel_deg_mean = calculate_combined_deg(aggregated_not_ood_pvalues, pvalue_dict[ood_primary], n_values)
    min_not_ood_pvalues = pd.DataFrame(pd.concat(non_ood_pvalues).min(axis=0)).transpose()
    basemodel_deg_min = calculate_combined_deg(min_not_ood_pvalues, pvalue_dict[ood_primary], n_values)

    return basemodel_deg_mean, basemodel_deg_min
    

def get_pvalues(ctrls, preds, zeros, all_conditions=False):
    pvalues_down = []
    pvalues_up = []
    
    for gene in ctrls.columns:
            
        condition_1_expression = ctrls[gene] # ctrl
        condition_2_expression = preds[gene] # stimulated
    
        non_zero_condition_1_expression = condition_1_expression[condition_1_expression != 0]
        non_zero_condition_2_expression = condition_2_expression[condition_2_expression != 0]
        
        if all_conditions:
            if zeros:
                p_down = ranksums(condition_2_expression, condition_1_expression, alternative= 'less').pvalue
                p_up = 1 - p_down
            else:
                if len(non_zero_condition_1_expression) > 0 and len(non_zero_condition_2_expression) > 0:
                    p_down = ranksums(non_zero_condition_2_expression, non_zero_condition_1_expression, alternative= 'less').pvalue
                    p_up = 1 - p_down
                elif len(non_zero_condition_1_expression) == 0 and len(non_zero_condition_2_expression) == 0:
                    p_down = 0.5
                    p_up = 1 - p_down
                elif len(non_zero_condition_1_expression) > 0:
                    _, p_value = wilcoxon(np.array(non_zero_condition_1_expression) - 0)
                    p_down = p_value/2
                    p_up = 1 - p_down
                else:
                    _, p_value = wilcoxon(np.array(non_zero_condition_2_expression) - 0)
                    p_up = p_value/2
                    p_down = 1 - p_up
                    
        else:    
        
            if len(non_zero_condition_1_expression) > 0 and len(non_zero_condition_2_expression) > 0:
                # Wilcoxon rank-sum test
                if zeros:
                    p_down = ranksums(condition_2_expression, condition_1_expression, alternative= 'less').pvalue
                    # p_up = ranksums(condition_2_expression, condition_1_expression, alternative= 'greater').pvalue
                    p_up = 1 - p_down
                else:
                    p_down = ranksums(non_zero_condition_2_expression, non_zero_condition_1_expression, alternative= 'less').pvalue
                    # p_up = ranksums(non_zero_condition_2_expression, non_zero_condition_1_expression, alternative= 'greater').pvalue
                    p_up = 1 - p_down
                

            else:
                p_down = 2.0
                p_up = 2.0
           
        pvalues_down.append(p_down)
        pvalues_up.append(p_up)
    
    results_down = pd.DataFrame({
        'p_value': pvalues_down
    })
    
    results_up = pd.DataFrame({
        'p_value': pvalues_up
    })
    
    final_down = results_down.transpose()
    final_down.columns = ctrls.columns
    final_up = results_up.transpose()
    final_up.columns = ctrls.columns
    
    return final_down, final_up


def process_subtract_control(adata, primary_variable, modality_variable, ood_modality):
    modality_adata = adata[adata.obs[modality_variable] == ood_modality].copy()
    ood_primaries = adata.obs[primary_variable].unique()
    control_mean_dict = {}
    control_mean_dict_nonzero = {}
    for ood in ood_primaries:
        temp_ratio_df_ratio = adata[(adata.obs[modality_variable] != ood_modality) & 
                                                 (adata.obs[primary_variable] == ood)].to_df()
        control_mean_dict[ood] = np.mean(temp_ratio_df_ratio, axis=0).values
        control_mean_dict_nonzero[ood] = np.nan_to_num(np.mean(temp_ratio_df_ratio.replace(0, np.nan), axis=0).values)
    for row in range(modality_adata.X.shape[0]):
        modality_adata.X[row,:] = modality_adata.X[row,:] - control_mean_dict[modality_adata.obs[primary_variable].iloc[row]]
    return modality_adata, control_mean_dict, control_mean_dict_nonzero

def subtract_control(control_mean_dict, per_gene_list,ood_primaries):
    for idx, ood in enumerate(ood_primaries):
        per_gene_list[idx] = per_gene_list[idx] - control_mean_dict[ood]
    return per_gene_list


def k_nearest_neighbors(pred_adata, true_adata, ctrl_adata, model, ood_primary, pca_components=10, n_neighbors=10):

    control_df = ctrl_adata.to_df()
    stim_df = true_adata.to_df()
    pred_df = pred_adata.to_df()

    control_label = "control"
    stim_label = "stimulation"
    pred_label = "prediction"

    control_df["type"] = control_label
    stim_df["type"] = stim_label
    pred_df["type"] = pred_label

    combined = pd.concat([control_df, stim_df, pred_df], ignore_index=True)

    labels = combined["type"]
    gene_data = combined.drop(columns=["type"])

    # scaler = StandardScaler()
    # gene_data = scaler.fit_transform(gene_data)

    pca = PCA(n_components=pca_components) 
    pca_data = pca.fit_transform(gene_data)

    pred_pca = pca_data[labels == "prediction"]

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(pca_data)
    _, indices = nbrs.kneighbors(pred_pca)

    ratios = []
    for neighbors in indices:
        neighbor_types = labels.iloc[neighbors]
        control_ratio = (neighbor_types == control_label).sum()
        stimulation_ratio = (neighbor_types == stim_label).sum()
        prediction_ratio = (neighbor_types == pred_label).sum()
        ratios.append({
            "control_ratio": control_ratio,
            "stimulation_ratio": stimulation_ratio,
            "prediction_ratio": prediction_ratio
        })

    ratios_df = pd.DataFrame(ratios)
    ratios_df = ratios_df/n_neighbors

    prediction_ratio_threshold = len(pred_df)/(len(pred_df) + len(stim_df))
    # ratios_df['score'] = (2*ratios_df["stimulation_ratio"] - ratios_df["control_ratio"])/(ratios_df["prediction_ratio"])
    # ratios_df['score'] = ratios_df["stimulation_ratio"] - ratios_df["control_ratio"]
    # ratios_df['score'] = ratios_df["stimulation_ratio"] + ratios_df["stimulation_ratio"]/ratios_df["prediction_ratio"] - ratios_df["control_ratio"]
    # ratios_df['score'] = 2*ratios_df["stimulation_ratio"] - ratios_df["control_ratio"] - 2*np.abs(ratios_df["prediction_ratio"]-prediction_ratio_threshold)
    ratios_df['score'] = 2*ratios_df["stimulation_ratio"] - ratios_df["control_ratio"] - np.abs(ratios_df["prediction_ratio"]-prediction_ratio_threshold)
    
    print(f'knn-score: {model} - {ood_primary}')
    print(ratios_df)

    return ratios_df['score'].mean()


def seurat_deg(adata, grouping_variable, ident_1, ident_2, test_use, find_variable_features=False, scale_factor=10000, latent_vars=None):

    seurat_obj = create_seurat_object(adata.copy())
    pandas2ri.activate()
    # meta_data = seurat_obj.slots["meta.data"]
    ro.globalenv['seurat_obj'] = seurat_obj
    print(r(f'summary(seurat_obj@meta.data${grouping_variable})'))
    
    r(f"Idents(seurat_obj) <- seurat_obj@meta.data${grouping_variable}")
    print(r("table(Idents(seurat_obj))"))

    r(f"""seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = {scale_factor})""")
    if find_variable_features:
        r("""seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 4000)""")
    # r("""seurat_obj <- ScaleData(seurat_obj)""")

    # de_genes = r("""FindMarkers(object = seurat_obj,ident.1 = "stimulated",ident.2 = "ctrl",test.use = "MAST",min.pct = 0.1,logfc.threshold = 0.25)""")

    r("""
    safe_FindMarkers <- function(obj, ident1, ident2, test_use, latent_vars=NULL) {
      res <- tryCatch(
        {
          if (is.null(latent_vars)) {
            FindMarkers(
              object=obj,
              ident.1=ident1,
              ident.2=ident2,
              test.use=test_use,
              verbose=FALSE
            )
          } else {
            FindMarkers(
              object=obj,
              ident.1=ident1,
              ident.2=ident2,
              test.use=test_use,
              verbose=FALSE,
              latent.vars=latent_vars
            )
          }
        },
        error=function(e) NULL
      )

      # If NULL → return empty standardized DF
      if (is.null(res)) {
        return(data.frame(
          p_val=numeric(0),
          avg_log2FC=numeric(0),
          pct.1=numeric(0),
          pct.2=numeric(0),
          p_val_adj=numeric(0)
        ))
      }
      return(res)
    }
    """)

    safe_FindMarkers = ro.r["safe_FindMarkers"]
    seurat_obj_r = ro.r["seurat_obj"]

    if latent_vars is None:
        r_res = safe_FindMarkers(seurat_obj_r, ident_1, ident_2, test_use)
        print("using safe_FindMarkers (no latent vars)")
    else:
        print(f"using safe_FindMarkers with latent variable(s): {latent_vars}")
        r_res = safe_FindMarkers(seurat_obj_r, ident_1, ident_2, test_use, latent_vars)

    if isinstance(r_res, RDataFrame):
        de_genes_df = pandas2ri.rpy2py(r_res)
    else:
        de_genes_df = r_res  # Use as is

    # if latent_vars is None:
    #     de_genes = r(f"FindMarkers(object = seurat_obj,ident.1 = '{ident_1}',ident.2 = '{ident_2}',test.use = '{test_use}', verbose = FALSE)")
    # else:
    #     print(f'with covariate of {latent_vars}')
    #     de_genes = r(f"FindMarkers(object = seurat_obj,ident.1 = '{ident_1}',ident.2 = '{ident_2}',test.use = '{test_use}', verbose = FALSE, latent.vars = '{latent_vars}')")

    ## de_genes = r(f"suppressWarnings(suppressMessages({FindMarkers(object = seurat_obj, ident.1 = '{ident_1}', ident.2 = '{ident_2}', test.use = '{test_use}', verbose = FALSE)}))")

    # if isinstance(de_genes, RDataFrame):
    #     de_genes_df = pandas2ri.rpy2py(de_genes)
    # else:
    #     de_genes_df = de_genes  # Use as is
    
    return de_genes_df

def seurat_get_down_pvalues(adata, grouping_variable, ident_1, ident_2, test_use, find_variable_features=False, scale_factor=10000, latent_vars=None):

    deg_df = seurat_deg(adata, grouping_variable, ident_1, ident_2, test_use, find_variable_features, scale_factor, latent_vars)
    gene_names = adata.var_names
    df_names = deg_df.index
    pvalues = []
    for gene in gene_names:
        if gene in df_names:
            p_val = deg_df.loc[gene,'p_val']
            logFC = deg_df.loc[gene,'avg_log2FC']
            if logFC>0:
                p_val = 1-p_val
        else:
            p_val = 0.5
        pvalues.append(p_val)

    return pd.DataFrame(np.asarray(pvalues).reshape((1,-1)))


def seurat_get_down_up_genes(adata, grouping_variable, ident_1, ident_2, test_use, threshold=0.05, find_variable_features=False, scale_factor=10000, latent_vars=None, logfc_threshold=0.25):

    print(f'adata shape: {adata.X.shape}')
    deg_df = seurat_deg(adata, grouping_variable, ident_1, ident_2, test_use, find_variable_features, scale_factor, latent_vars)
    gene_names = adata.var_names
    down_genes = []
    up_genes = []
    if deg_df is not None:
        df_names = deg_df.index
        for gene in gene_names:
            if gene in df_names and deg_df.loc[gene,'p_val_adj'] < threshold:
            # if gene in df_names and deg_df.loc[gene,'p_val'] < threshold:
                # p_val = deg_df.loc[gene,'p_val_adj']
                logFC = deg_df.loc[gene,'avg_log2FC']
                if logFC<0:
                # if logFC < -1*logfc_threshold:
                    down_genes.append(gene)
                else:
                # elif logFC > logfc_threshold:
                    up_genes.append(gene)

    return down_genes, up_genes


def seurat_clustering(adata, resolution, all_adata, all_seurat=None, find_variable_features=True, scale=True):

    pandas2ri.activate()
    numpy2ri.activate()
    if all_seurat is None:
        print('Create Seurat object for all_adata...')
        all_seurat = create_seurat_object(all_adata.copy())
    else:
        print('Using precomputed all_seurat object.')
    # all_seurat = create_seurat_object(all_adata.copy())
    print(f'create seurat object of combined')
    seurat_obj = create_seurat_object(adata.copy())
    # ro.globalenv['all_seurat'] = all_seurat
    ro.globalenv['all_seurat'] = ro.r("""
    function(obj) {
        unserialize(serialize(obj, NULL))
    }
    """)(all_seurat)
    ro.globalenv['seurat_obj'] = seurat_obj

    print(f'normalize adatas')
    r("""
    DefaultAssay(all_seurat) <- 'RNA'
    DefaultAssay(seurat_obj) <- 'RNA'
    all_seurat <- NormalizeData(all_seurat)
    seurat_obj <- NormalizeData(seurat_obj)
    """)

    print(f'scale adatas')
    if scale:
        if issparse(all_adata.X):
            all_adata.X = all_adata.X.toarray()
        gene_means = all_adata.X.mean(axis=0)  
        gene_sds = all_adata.X.std(axis=0, ddof=1)
        ro.globalenv['gene_means'] = gene_means
        ro.globalenv['gene_sds'] = gene_sds
        ro.globalenv['gene_names'] = list(all_adata.var_names)

        r("""
        all_seurat <- ScaleData(all_seurat)
        seurat_obj <- ScaleData(seurat_obj)
        names(gene_means) <- gene_names
        names(gene_sds) <- gene_names
        seurat_data <- LayerData(seurat_obj, assay = "RNA", layer = "data")
        scaled_values <- sweep(seurat_data, 1, gene_means[rownames(seurat_data)], "-")
        scaled_values <- sweep(scaled_values, 1, gene_sds[rownames(seurat_data)], "/")
        ld <- LayerData(seurat_obj, assay = "RNA", layer = "scale.data")
        common_genes <- intersect(rownames(ld), rownames(scaled_values))
        common_cells <- intersect(colnames(ld), colnames(scaled_values))

        cat('genes(ld) = ', length(rownames(ld)), '\n')
        cat('cells(ld) = ', length(colnames(ld)), '\n')
        cat('genes(scaled_values) = ', length(rownames(scaled_values)), '\n')
        cat('cells(scaled_values) = ', length(colnames(scaled_values)), '\n')
        cat('length(common_genes) between ld and scaled_value = ', length(common_genes), '\n')
        cat('length(common_cells) between ld and scaled_value = ', length(common_cells), '\n')

        scaled_aligned <- scaled_values[common_genes, common_cells, drop = FALSE]
        scaled_aligned <- scaled_aligned[rownames(ld), colnames(ld), drop = FALSE]
        ld[,] <- as.matrix(scaled_aligned)
        LayerData(seurat_obj, assay = "RNA", layer = "scale.data") <- ld
        """)
    # if scale:
    #     r("seurat_obj <- ScaleData(seurat_obj)")
    else:
        r("all_seurat <- ScaleData(all_seurat, do.scale = FALSE)")
        r("seurat_obj <- ScaleData(seurat_obj, do.scale = FALSE)")
    if find_variable_features:
        r("all_seurat <- FindVariableFeatures(all_seurat, nfeatures = 4000)")
        r("seurat_obj <- FindVariableFeatures(seurat_obj, nfeatures = 4000)")
    r("""
    common_genes <- intersect(rownames(all_seurat), rownames(seurat_obj))
    all_seurat <- subset(all_seurat, features = common_genes)
    seurat_obj <- subset(seurat_obj, features = common_genes)
    """)
    print(f'Run PCA')
    if find_variable_features:
        r("all_seurat <- RunPCA(all_seurat)")
    else:
        r("all_features <- rownames(all_seurat)")
        r("all_seurat <- Seurat::RunPCA(all_seurat, features = all_features)")
    # Project into all_adata's PCA space
    r("""
    common_genes <- intersect(rownames(LayerData(seurat_obj, assay = "RNA", layer = "scale.data")),
                            rownames(Loadings(all_seurat[["pca"]])))
    cat('common genes after RunPCA on all_seurat = ', length(common_genes), '\n')
    seurat_mat <- LayerData(seurat_obj, assay = "RNA", layer = "scale.data")[common_genes, , drop=FALSE]
    loadings <- Loadings(all_seurat[["pca"]])[common_genes, , drop=FALSE]

    # Project into PCA space
    pcs <- t(seurat_mat) %*% loadings
    colnames(pcs) <- colnames(Embeddings(all_seurat, "pca"))
    rownames(pcs) <- colnames(seurat_obj)

    seurat_obj[["pca"]] <- CreateDimReducObject(
        embeddings = pcs,
        key = "PC_",
        loadings = loadings,
        assay = "RNA"
    )
    """)
    # common_gene_set = set(r("common_genes"))
    # with open(f"common_seurat.pkl", "wb") as log_file:
    #     pickle.dump(common_gene_set, log_file)
    # r("all_features <- rownames(seurat_obj)")
    # r("seurat_obj <- Seurat::RunPCA(seurat_obj, features = all_features)")
    # r("seurat_obj <- RunPCA(seurat_obj)")
    r(f"num_dim <- min(10, ncol(seurat_obj[['pca']]@cell.embeddings))")
    r(f"seurat_obj <- FindNeighbors(seurat_obj, dims = 1:num_dim)")
    r(f"seurat_obj <- FindClusters(seurat_obj, resolution = {resolution})")

    clusters = np.array(r("Idents(seurat_obj)"))
    print(f'end of seurat clustering')
    return clusters

def seurat_clustering_python(adata, resolution, all_adata, find_variable_features=True, scale=True):
    adata = adata.copy()
    all_adata = all_adata.copy()

    # --- Variable features ---
    if find_variable_features:
        # sc.pp.highly_variable_genes(all_adata, n_top_genes=4000, subset=False, flavor="seurat")
        sc.pp.highly_variable_genes(all_adata, layer='counts', n_top_genes=4000, subset=False, flavor="seurat_v3")
        # sc.pp.highly_variable_genes(adata, n_top_genes=4000, subset=False, flavor="seurat")

        # common_hvgs = adata.var_names[adata.var["highly_variable"]].intersection(
        #     all_adata.var_names[all_adata.var["highly_variable"]]
        # )
        hvgs = all_adata.var_names[all_adata.var["highly_variable"]]
        adata = adata[:, hvgs].copy()
        all_adata = all_adata[:, hvgs].copy()
        # with open(f"common_scanpy.pkl", "wb") as log_file:
        #     pickle.dump(hvgs, log_file)
    
    if scale:
        scaler = StandardScaler(with_mean=True, with_std=True)

        # Convert sparse matrices if needed
        allX = all_adata.X.toarray() if issparse(all_adata.X) else all_adata.X
        Xq = adata.X.toarray() if issparse(adata.X) else adata.X

        # Fit on all_adata and transform both
        all_adata.X = scaler.fit_transform(allX)
        adata.X = scaler.transform(Xq)
    else:
        sc.pp.scale(all_adata, zero_center=False)
        sc.pp.scale(adata, zero_center=False)
    

    # --- PCA on reference (all_adata) ---
    sc.tl.pca(all_adata, svd_solver='arpack')

    # Project query (adata) into reference PCA space
    loadings = all_adata.varm['PCs']  # genes × PCs
    adata.obsm['X_pca'] = adata.X @ loadings  # cells × PCs

    # --- Neighbors & clustering ---
    # sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=15, n_pcs=40)
    sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=20, n_pcs=10)
    sc.tl.louvain(adata, resolution=resolution)

    clusters = adata.obs['louvain'].to_numpy()
    return clusters


def create_seurat_object(adata):
    counts = adata.layers['counts'].copy()

    if issparse(counts):
        counts = counts.toarray()

    pandas2ri.activate()
    Seurat = importr("Seurat")
    SeuratObject = importr("SeuratObject")

    expr_matrix = pd.DataFrame(counts.T, index=adata.var_names, columns=adata.obs_names)
    meta_data = adata.obs.copy()

    r_expr_matrix = pandas2ri.py2rpy(expr_matrix)
    r_meta_data = pandas2ri.py2rpy(meta_data)

    seurat_obj = SeuratObject.CreateSeuratObject(counts=r_expr_matrix, meta_data=r_meta_data)

    return seurat_obj


def find_neighbors(adata, cell_idx, k=5):

    data = np.array(adata.X)
    knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')  
    knn.fit(data)
    distances, neighbors = knn.kneighbors(data[cell_idx].reshape(1, -1))
    neighbor_idx = neighbors[0][1:]
    # for cell in neighbor_idx:
    #     print(adata.obs.iloc[cell]['cell_type'])
    return neighbor_idx


def kmeans_clustering(adata, n_clusters=2, n_pcs=50, scale=True):

    data = adata.to_df().values

    if scale:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    pca = PCA(n_components=min(n_pcs, data.shape[1]))
    data_pca = pca.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data_pca)
    return cluster_labels


def mixing_index(pred_adata, true_adata, n_clusters=2, n_pcs=50):

    combined_adata = ad.concat([pred_adata, true_adata], label="batch", keys=["pred", "true"])
    combined_adata.obs_names_make_unique()
    batch_labels = combined_adata.obs['batch'].values
    cluster_labels = kmeans_clustering(combined_adata, n_clusters, n_pcs)

    pred_correct_clustered = 0
    num_pred = pred_adata.X.shape[0]
    num_true = true_adata.X.shape[0]
    expected_proportion = num_pred/num_true

    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        batch_counts = pd.Series(batch_labels[cluster_indices]).value_counts()
        N1 = batch_counts.get('pred', 0)
        N2 = batch_counts.get('true', 0)
        expected_pred = N2 * expected_proportion
        pred_correct_clustered += min(N1, expected_pred)
    
    average_mixing_index = pred_correct_clustered / num_pred

    return average_mixing_index


def mixing_index_seurat_old(pred_adata, true_adata, all_adata, train_adata, data_params, ood_primary, ood_modality, resolution=2.0, do_layer_process=True, find_variable_features=True):

    if do_layer_process:
        total_count = 1e4
        if issparse(pred_adata.X):
            exp_data = np.exp(pred_adata.X.data) - 1
            pred_adata.layers['counts'] = csr_matrix((exp_data, pred_adata.X.indices, pred_adata.X.indptr),
                                                    shape=pred_adata.X.shape) / total_count
        else:
            pred_adata.layers['counts'] = (np.exp(pred_adata.X) - 1) / total_count
        # pred_adata.layers['counts'] = (np.exp(np.asarray(pred_adata.X))-1)/total_count

    all_adata_minus_stim = \
        all_adata[~((all_adata.obs[data_params['modality_variable']]==ood_modality) & 
        (all_adata.obs[data_params['primary_variable']]==ood_primary))].copy()
    if train_adata:
        all_adata_minus_test = ad.concat([all_adata_minus_stim, train_adata])
        all_adata_minus_test.obs_names_make_unique()
    else:
        all_adata_minus_test = all_adata_minus_stim.copy()
    # combined_adata = ad.concat([pred_adata, true_adata], label="batch", keys=["pred", "true"])
    # combined_adata = ad.concat([pred_adata, true_adata, all_adata], label="batch", keys=["pred", "true", "all_adata"])
    combined_adata = ad.concat([pred_adata, true_adata, all_adata_minus_test], label="batch", keys=["pred", "true", "all_adata"])
    combined_adata.obs_names_make_unique()
    cluster_labels = seurat_clustering(combined_adata, resolution, all_adata=all_adata, find_variable_features=find_variable_features, scale=True)
    # cluster_labels = seurat_clustering_python(combined_adata, resolution, all_adata=all_adata, find_variable_features=find_variable_features, scale=True)
    cluster_uniques = np.unique(cluster_labels)
    print(f'number of clusters: {len(cluster_uniques)}')
    batch_labels = combined_adata.obs['batch'].values

    pred_correct_clustered = 0
    num_pred = pred_adata.X.shape[0]
    num_true = true_adata.X.shape[0]
    expected_proportion = num_pred/num_true

    for cluster in cluster_uniques:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        batch_counts = pd.Series(batch_labels[cluster_indices]).value_counts()
        N1 = batch_counts.get('pred', 0)
        N2 = batch_counts.get('true', 0)
        expected_pred = N2 * expected_proportion
        pred_correct_clustered += min(N1, expected_pred)
    
    average_mixing_index = pred_correct_clustered / num_pred

    return average_mixing_index


def mixing_index_seurat(pred_adata, true_adata, all_adata, all_seurat=None, resolution=2.0, do_layer_process=True, find_variable_features=True):
    if do_layer_process:
        total_count = 1e4
        if issparse(pred_adata.X):
            exp_data = np.exp(pred_adata.X.data) - 1
            pred_adata.layers['counts'] = csr_matrix((exp_data, pred_adata.X.indices, pred_adata.X.indptr),
                                                    shape=pred_adata.X.shape) / total_count
        else:
            pred_adata.layers['counts'] = (np.exp(pred_adata.X) - 1) / total_count
        # pred_adata.layers['counts'] = (np.exp(np.asarray(pred_adata.X))-1)/total_count

    print(f'combine adatas:')
    combined_adata = ad.concat([pred_adata, true_adata, all_adata], label="batch", keys=["pred", "true", "all_adata"])
    print(f'unique indices:')
    combined_adata.obs_names_make_unique()
    print(f'start of seurat clustering')
    cluster_labels = seurat_clustering(combined_adata, resolution, all_adata=all_adata, all_seurat=all_seurat, find_variable_features=find_variable_features, scale=True)
    # cluster_labels = seurat_clustering_python(combined_adata, resolution, all_adata=all_adata, find_variable_features=find_variable_features, scale=True)
    cluster_uniques = np.unique(cluster_labels)
    print(f'number of clusters: {len(cluster_uniques)}')
    batch_labels = combined_adata.obs['batch'].values

    pred_correct_clustered = 0
    num_pred = pred_adata.X.shape[0]
    num_true = true_adata.X.shape[0]
    expected_proportion = num_pred/num_true

    for cluster in cluster_uniques:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        batch_counts = pd.Series(batch_labels[cluster_indices]).value_counts()
        N1 = batch_counts.get('pred', 0)
        N2 = batch_counts.get('true', 0)
        expected_pred = N2 * expected_proportion
        pred_correct_clustered += min(N1, expected_pred)
    
    average_mixing_index = pred_correct_clustered / num_pred

    return average_mixing_index


def mixing_index_kmeans_topdeg(pred_adata, true_adata, true_logfc_df, n_clusters, num_top_deg):

    if len(true_logfc_df) < num_top_deg:
        print(f"the number of significant genes is less than threshold: {num_top_deg}")
        return 0
    genes = true_logfc_df.index
    top_deg = genes[0:num_top_deg]
    pred_adata = pred_adata[:, pred_adata.var_names.isin(top_deg)].copy()
    true_adata = true_adata[:, true_adata.var_names.isin(top_deg)].copy()

    return mixing_index(pred_adata, true_adata, n_clusters)

def mixing_index_seurat_topdeg(pred_adata, true_adata, true_logfc_df, num_top_deg, do_layer_process, all_adata):

    if len(true_logfc_df) < num_top_deg:
        print(f"the number of significant genes is less than threshold: {num_top_deg}")
        return 0
    genes = true_logfc_df.index
    top_deg = genes[0:num_top_deg]
    pred_adata = pred_adata[:, pred_adata.var_names.isin(top_deg)].copy()
    true_adata = true_adata[:, true_adata.var_names.isin(top_deg)].copy()

    return mixing_index_seurat(pred_adata, true_adata, all_adata=all_adata, do_layer_process=do_layer_process)


def logfc_correlation(pred_adata, ctrl_adata, true_logfc_df, ident_1, ident_2, do_layer_process=True):

    if do_layer_process:
        total_count = 1e4
        if issparse(pred_adata.X):
            exp_data = np.exp(pred_adata.X.data) - 1
            pred_adata.layers['counts'] = csr_matrix((exp_data, pred_adata.X.indices, pred_adata.X.indptr),
                                                    shape=pred_adata.X.shape) / total_count
        else:
            pred_adata.layers['counts'] = (np.exp(pred_adata.X) - 1) / total_count
        # pred_adata.layers['counts'] = (np.exp(np.asarray(pred_adata.X))-1)/total_count

    combined_pred = ad.concat([pred_adata, ctrl_adata], label="batch", keys=[ident_1, ident_2])
    combined_pred.obs_names_make_unique()
    pred_logfc_df = seurat_deg(combined_pred, "batch", ident_1, ident_2, test_use='MAST', find_variable_features=False)

    all_genes = ctrl_adata.var_names
    true_logfc = []
    pred_logfc = []

    for gene in all_genes:
        if gene in true_logfc_df.index and gene in pred_logfc_df.index:
            true_logfc.append(true_logfc_df.loc[gene, 'avg_log2FC'])
            pred_logfc.append(pred_logfc_df.loc[gene, 'avg_log2FC'])
        elif gene in true_logfc_df.index:
            true_logfc.append(true_logfc_df.loc[gene, 'avg_log2FC'])
            pred_logfc.append(0)
        elif gene in pred_logfc_df.index:
            true_logfc.append(0)
            pred_logfc.append(pred_logfc_df.loc[gene, 'avg_log2FC'])
        else:
            true_logfc.append(0)
            pred_logfc.append(0)

    return pearsonr(true_logfc, pred_logfc)[0]

def logfc_dict(pred_adata, ctrl_adata, true_logfc_df, ident_1, ident_2, do_layer_process=True):

    if do_layer_process:
        total_count = 1e4
        if issparse(pred_adata.X):
            exp_data = np.exp(pred_adata.X.data) - 1
            pred_adata.layers['counts'] = csr_matrix((exp_data, pred_adata.X.indices, pred_adata.X.indptr),
                                                    shape=pred_adata.X.shape) / total_count
        else:
            pred_adata.layers['counts'] = (np.exp(pred_adata.X) - 1) / total_count

    combined_pred = ad.concat([pred_adata, ctrl_adata], label="batch", keys=[ident_1, ident_2])
    combined_pred.obs_names_make_unique()
    pred_logfc_df = seurat_deg(combined_pred, "batch", ident_1, ident_2, test_use='MAST', find_variable_features=False)

    all_genes = ctrl_adata.var_names
    true_logfc_dict = {}
    pred_logfc_dict = {}

    for gene in all_genes:
        if gene in true_logfc_df.index and gene in pred_logfc_df.index:
            true_logfc_dict[gene] = true_logfc_df.loc[gene, 'avg_log2FC']
            pred_logfc_dict[gene] = pred_logfc_df.loc[gene, 'avg_log2FC']
        elif gene in true_logfc_df.index:
            true_logfc_dict[gene] = true_logfc_df.loc[gene, 'avg_log2FC']
            pred_logfc_dict[gene] = 0
        elif gene in pred_logfc_df.index:
            true_logfc_dict[gene] = 0
            pred_logfc_dict[gene] = pred_logfc_df.loc[gene, 'avg_log2FC']
        else:
            true_logfc_dict[gene] = 0
            pred_logfc_dict[gene] = 0

    return true_logfc_dict, pred_logfc_dict


def compute_knn_loss(adata1, adata2, k=125):
# def compute_knn_loss(adata1, adata2):

    X = adata1.X.toarray() if hasattr(adata1.X, "toarray") else np.array(adata1.X)
    Y = adata2.X.toarray() if hasattr(adata2.X, "toarray") else np.array(adata2.X)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    # k = int(np.ceil(np.log10(len(X)))) * 10

    # Compute pairwise distances separately
    dist_X_X = torch.cdist(X, X, p=2)  # (n_X, n_X)
    dist_X_Y = torch.cdist(X, Y, p=2)  # (n_X, n_Y)
    dist_Y_X = dist_X_Y.transpose(0, 1)  # (n_Y, n_X)
    dist_Y_Y = torch.cdist(Y, Y, p=2)  # (n_Y, n_Y)

    # Get k-nearest neighbors (excluding self for same-group comparisons)
    knn_X_X = dist_X_X.topk(k=k+1, largest=False).indices[:, 1:]  # (n_X, k) - Exclude self
    knn_X_Y = dist_X_Y.topk(k=k, largest=False).indices           # (n_X, k)
    knn_Y_Y = dist_Y_Y.topk(k=k+1, largest=False).indices[:, 1:]  # (n_Y, k) - Exclude self
    knn_Y_X = dist_Y_X.topk(k=k, largest=False).indices           # (n_Y, k)

    # Compute distances to k-nearest neighbors
    D_X_X = torch.gather(dist_X_X, 1, knn_X_X).mean(dim=1)  # (n_X,)
    D_X_Y = torch.gather(dist_X_Y, 1, knn_X_Y).mean(dim=1)  # (n_X,)

    D_Y_Y = torch.gather(dist_Y_Y, 1, knn_Y_Y).mean(dim=1)  # (n_Y,)
    D_Y_X = torch.gather(dist_Y_X, 1, knn_Y_X).mean(dim=1)  # (n_Y,)

    # Compute squared loss for each point
    loss_X = (D_X_X - D_X_Y).pow(2)
    loss_Y = (D_Y_Y - D_Y_X).pow(2)

    # return (loss_X.mean() + loss_Y.mean()) / 2
    return (torch.sqrt(loss_X.mean()) + torch.sqrt(loss_Y.mean())) / 2


def compute_Edistance(adata1, adata2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X1 = torch.tensor(adata1.X.toarray() if hasattr(adata1.X, "toarray") else adata1.X,
                      dtype=torch.float32, device=device)
    X2 = torch.tensor(adata2.X.toarray() if hasattr(adata2.X, "toarray") else adata2.X,
                      dtype=torch.float32, device=device)

    n1, n2 = len(X1), len(X2)

    # Pairwise distances
    d12 = torch.cdist(X1, X2, p=2).mean()
    d11 = torch.cdist(X1, X1, p=2)
    d22 = torch.cdist(X2, X2, p=2)

    # Subtract diagonals to avoid self-distance
    d11_mean = d11.sum() / (n1 * (n1 - 1)) if n1 > 1 else 0
    d22_mean = d22.sum() / (n2 * (n2 - 1)) if n2 > 1 else 0

    # E-distance formula
    edist = 2 * d12 - d11_mean - d22_mean
    return edist.item()

def rbf_kernel(X, Y=None, gamma=None):
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    pairwise_sq_dists = cdist(X, Y, metric='sqeuclidean')  # Squared Euclidean distance
    return np.exp(-gamma * pairwise_sq_dists)

def rbf_kernel_torch(X, Y=None, gamma=None):
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    # Compute squared Euclidean distance efficiently
    X_norm = (X ** 2).sum(1).view(-1, 1)
    Y_norm = (Y ** 2).sum(1).view(1, -1)
    dists = X_norm + Y_norm - 2.0 * X @ Y.T
    return torch.exp(-gamma * dists)

def knn_mask(kernel_matrix, X, Y, k, exclude_self):
    # Find k-nearest neighbors of each point in X w.r.t. Y
    nbrs = NearestNeighbors(n_neighbors=k + int(exclude_self), metric='euclidean').fit(Y)
    distances, indices = nbrs.kneighbors(X)

    mask = np.zeros_like(kernel_matrix, dtype=bool)
    for i, idx in enumerate(indices):
        if exclude_self and X is Y:
            idx = idx[idx != i]  # remove self
        mask[i, idx] = True

    return kernel_matrix[mask]

def knn_mask_from_kernel(kernel_matrix, k, exclude_self):
    mask = np.zeros_like(kernel_matrix, dtype=bool)
    for i in range(kernel_matrix.shape[0]):
        # sort by RBF similarity (descending)
        idx = np.argsort(-kernel_matrix[i])  
        if exclude_self:
            idx = idx[idx != i]
        idx = idx[:k]
        mask[i, idx] = True
    return kernel_matrix[mask]

def knn_mask_from_kernel_torch(kernel_matrix, k, exclude_self, largest=True):
    # Sort and take top-k indices
    vals, idx = torch.topk(kernel_matrix, k + (1 if exclude_self else 0), dim=1, largest=largest)

    if exclude_self:
        # Drop self index from each row
        mask = torch.zeros_like(kernel_matrix, dtype=torch.bool)
        row_idx = torch.arange(kernel_matrix.size(0)).unsqueeze(1)
        mask[row_idx, idx[:, 1:]] = True
    else:
        mask = torch.zeros_like(kernel_matrix, dtype=torch.bool)
        row_idx = torch.arange(kernel_matrix.size(0)).unsqueeze(1)
        mask[row_idx, idx] = True

    return kernel_matrix[mask]

def compute_local_mmd(adata1, adata2, gamma=None, k=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X1 = torch.tensor(adata1.X.toarray() if hasattr(adata1.X, "toarray") else adata1.X,
                      dtype=torch.float32, device=device)
    X2 = torch.tensor(adata2.X.toarray() if hasattr(adata2.X, "toarray") else adata2.X,
                      dtype=torch.float32, device=device)
    K11 = rbf_kernel_torch(X1, X1, gamma)
    K22 = rbf_kernel_torch(X2, X2, gamma)
    K12 = rbf_kernel_torch(X1, X2, gamma)
    K21 = K12.T

    # Apply kNN mask and compute mean over selected elements
    mean_K11 = knn_mask_from_kernel_torch(K11, k, exclude_self=True).mean()
    mean_K22 = knn_mask_from_kernel_torch(K22, k, exclude_self=True).mean()
    mean_K12 = knn_mask_from_kernel_torch(K12, k, exclude_self=False).mean()
    mean_K21 = knn_mask_from_kernel_torch(K21, k, exclude_self=False).mean()
    # print("K11 (masked) mean:", np.mean(mean_K11))
    # print("K22 (masked) mean:", np.mean(mean_K22))
    # print("K12 (masked) mean:", np.mean(mean_K12))
    # print("K21 (masked) mean:", np.mean(mean_K21))
    # print("Number of nonzero entries in K11:", np.count_nonzero(mean_K11))

    mmd_sq = mean_K11 + mean_K22 - (mean_K12 + mean_K21)
    # return 0 if mmd_sq<0 else np.sqrt(mmd_sq)
    # return mmd_sq.item()
    return torch.sqrt(mmd_sq).item() if mmd_sq>=0 else -torch.sqrt(-mmd_sq).item()

def compute_local_Edistance(adata1, adata2, k):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X1 = torch.tensor(adata1.X.toarray() if hasattr(adata1.X, "toarray") else adata1.X,
                      dtype=torch.float32, device=device)
    X2 = torch.tensor(adata2.X.toarray() if hasattr(adata2.X, "toarray") else adata2.X,
                      dtype=torch.float32, device=device)
    K11 = -torch.cdist(X1, X1, p=2)
    K22 = -torch.cdist(X2, X2, p=2)
    K12 = -torch.cdist(X1, X2, p=2)
    K21 = K12.T

    # Apply kNN mask and compute mean over selected elements
    mean_K11 = knn_mask_from_kernel_torch(K11, k, exclude_self=True).mean()
    mean_K22 = knn_mask_from_kernel_torch(K22, k, exclude_self=True).mean()
    mean_K12 = knn_mask_from_kernel_torch(K12, k, exclude_self=False).mean()
    mean_K21 = knn_mask_from_kernel_torch(K21, k, exclude_self=False).mean()
    # print("K11 (masked) mean:", np.mean(mean_K11))
    # print("K22 (masked) mean:", np.mean(mean_K22))
    # print("K12 (masked) mean:", np.mean(mean_K12))
    # print("K21 (masked) mean:", np.mean(mean_K21))

    dist_sq = mean_K11 + mean_K22 - (mean_K12 + mean_K21)
    return dist_sq.item()

def compute_mmd(adata1, adata2, gamma=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # X1 = adata1.X.toarray() if hasattr(adata1.X, "toarray") else np.array(adata1.X)
    # X2 = adata2.X.toarray() if hasattr(adata2.X, "toarray") else np.array(adata2.X)
    X1 = torch.tensor(adata1.X.toarray() if hasattr(adata1.X, "toarray") else adata1.X,
                      dtype=torch.float32, device=device)
    X2 = torch.tensor(adata2.X.toarray() if hasattr(adata2.X, "toarray") else adata2.X,
                      dtype=torch.float32, device=device)

    n1 = len(X1)
    n2 = len(X2)

    # K11 = rbf_kernel(X1, X1, gamma)
    # K22 = rbf_kernel(X2, X2, gamma)
    # K12 = rbf_kernel(X1, X2, gamma)
    K11 = rbf_kernel_torch(X1, X1, gamma)
    K22 = rbf_kernel_torch(X2, X2, gamma)
    K12 = rbf_kernel_torch(X1, X2, gamma)

    K11 = exclude_self(K11)
    K22 = exclude_self(K22)

    # Compute MMD squared
    # K11_mean = (K11.sum() - n1)/(n1 * (n1 - 1)) # exclude self
    # K22_mean = (K22.sum() - n2)/(n2 * (n2 - 1)) # exclude self 
    # K12_mean = K12.mean()
    mmd_sq = K11.mean() + K22.mean() - 2 * K12.mean()
    # mmd_sq = K11_mean + K22_mean - 2 * K12_mean
    # return np.sqrt(mmd_sq)  # Return MMD (not squared)
    return torch.sqrt(mmd_sq).item() if mmd_sq>=0 else -torch.sqrt(-mmd_sq).item()

def exclude_self(X):
    mask = torch.ones_like(X, dtype=torch.bool)
    idx = torch.arange(X.size(0)).unsqueeze(1)
    mask[idx, idx] = False
    return X[mask]

def compute_mmd_topdeg(pred_adata, true_adata, true_logfc_df, num_top_deg):

    if len(true_logfc_df) < num_top_deg:
        print(f"the number of significant genes is less than threshold: {num_top_deg}")
        return 0
    genes = true_logfc_df.index
    top_deg = genes[0:num_top_deg]
    pred_adata = pred_adata[:, pred_adata.var_names.isin(top_deg)].copy()
    true_adata = true_adata[:, true_adata.var_names.isin(top_deg)].copy()

    return compute_mmd(pred_adata, true_adata)

def rigorous_mmd_topdeg(true_adata, pred_dict, true_s_adata, primary_variable, true_logfc_df, num_top_deg):

    n_splits = 10
    test_fraction = 0.5
    metric_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)
    n_clusters = int(np.log1p(n_test))
    # n_clusters = 2

    for _ in range(n_splits):
        print(f"start of repeat {_ + 1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        train_adata = true_adata[train_indices].copy()
        test_adata = true_adata[test_indices].copy()

        metric_dict['true'].append(compute_mmd_topdeg(train_adata.copy(), test_adata.copy(), true_logfc_df, num_top_deg=num_top_deg))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            metric_dict[model].append(compute_mmd_topdeg(sampled_adata, test_adata.copy(), true_logfc_df, num_top_deg=num_top_deg))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        metric_dict['random_all'].append(compute_mmd_topdeg(sampled_true_all, test_adata.copy(), true_logfc_df, num_top_deg=num_top_deg))

        true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
        mmd_best = 1000000
        for p in true_s_dict.keys():
            sampled_adata = adata_sample(true_s_dict[p], n_test)
            mmd = compute_mmd_topdeg(sampled_adata, test_adata.copy(), true_logfc_df, num_top_deg=num_top_deg)
            if mmd < mmd_best:
                mmd_best = mmd
        metric_dict['best'].append(mmd_best)

    # for key in metric_dict:
    #     metric_dict[key] = list(-np.log10(np.array(metric_dict[key])))

    return metric_dict

def rigorous_mixing_index_seurat(true_adata, pred_dict, true_s_adata, data_params, ood_primary, ood_modality, all_adata, include_best=False):

    all_seurat = create_seurat_object(all_adata.copy())
    n_splits = 10
    # n_splits = 5
    test_fraction = 0.5
    mixing_index_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    for _ in range(n_splits):
        print(f'start of repeat {_+1}')
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        train_adata = true_adata[train_indices].copy()
        test_adata = true_adata[test_indices].copy()

        print(f'model true, repeat {_+1}:')
        # mixing_index_dict['true'].append(mixing_index_seurat(train_adata.copy(), test_adata.copy(), all_adata.copy(), None, data_params, ood_primary, ood_modality, do_layer_process=False))
        mixing_index_dict['true'].append(mixing_index_seurat(train_adata.copy(), test_adata.copy(), all_adata.copy(), all_seurat=all_seurat, do_layer_process=False))

        for model in pred_dict.keys():
            print(f'model {model}, repeat {_+1}:')
            sampled_adata = adata_sample(pred_dict[model], n_test)
            if model == 'noperturb':
                # mixing_index_dict[model].append(mixing_index_seurat(sampled_adata, test_adata.copy(), all_adata.copy(), train_adata, data_params, ood_primary, ood_modality, do_layer_process=False))
                mixing_index_dict[model].append(mixing_index_seurat(sampled_adata, test_adata.copy(), all_adata.copy(), all_seurat=all_seurat, do_layer_process=False))
            else:
                # mixing_index_dict[model].append(mixing_index_seurat(sampled_adata, test_adata.copy(), all_adata.copy(), train_adata, data_params, ood_primary, ood_modality))
                mixing_index_dict[model].append(mixing_index_seurat(sampled_adata, test_adata.copy(), all_adata.copy(), all_seurat=all_seurat))
        
        print(f'model random_all, repeat {_+1}:')
        sampled_true_all = adata_sample(true_s_adata, n_test)
        # mixing_index_dict['random_all'].append(mixing_index_seurat(sampled_true_all, test_adata.copy(), all_adata.copy(), train_adata, data_params, ood_primary, ood_modality, do_layer_process=False))
        mixing_index_dict['random_all'].append(mixing_index_seurat(sampled_true_all, test_adata.copy(), all_adata.copy(), all_seurat=all_seurat, do_layer_process=False))

        if include_best:
            true_s_dict = split_adata_by_primary(true_s_adata, data_params['primary_variable'])
            mixing_best = 0
            for p in true_s_dict.keys():
                sampled_adata = adata_sample(true_s_dict[p], n_test)
                # mixing = mixing_index_seurat(sampled_adata, test_adata.copy(), all_adata.copy(), train_adata, data_params, ood_primary, ood_modality, do_layer_process=False)
                mixing = mixing_index_seurat(sampled_adata, test_adata.copy(), all_adata.copy(), all_seurat=all_seurat, do_layer_process=False)
                if mixing > mixing_best:
                    mixing_best = mixing
            mixing_index_dict['best'].append(mixing_best)

    return mixing_index_dict


def rigorous_mixing_index_seurat_topdeg(true_adata, pred_dict, true_s_adata, primary_variable, true_logfc_df, num_top_deg, all_adata):

    n_splits = 10
    test_fraction = 0.5
    mixing_index_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    for _ in range(n_splits):
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        train_adata = true_adata[train_indices].copy()
        test_adata = true_adata[test_indices].copy()

        mixing_index_dict['true'].append(mixing_index_seurat_topdeg(train_adata.copy(), test_adata.copy(), true_logfc_df, num_top_deg, all_adata=all_adata.copy(), do_layer_process=False))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            if model == 'noperturb':
                mixing_index_dict[model].append(mixing_index_seurat_topdeg(sampled_adata, test_adata.copy(), true_logfc_df, num_top_deg, all_adata=all_adata.copy(), do_layer_process=False))
            else:
                mixing_index_dict[model].append(mixing_index_seurat_topdeg(sampled_adata, test_adata.copy(), true_logfc_df, num_top_deg, all_adata=all_adata.copy(), do_layer_process=True))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        mixing_index_dict['random_all'].append(mixing_index_seurat_topdeg(sampled_true_all, test_adata.copy(), true_logfc_df, num_top_deg, all_adata=all_adata.copy(), do_layer_process=False))

        true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
        mixing_best = 0
        for p in true_s_dict.keys():
            sampled_adata = adata_sample(true_s_dict[p], n_test)
            mixing = mixing_index_seurat_topdeg(sampled_adata, test_adata.copy(), true_logfc_df, num_top_deg, all_adata=all_adata.copy(), do_layer_process=False)
            if mixing > mixing_best:
                mixing_best = mixing
        mixing_index_dict['best'].append(mixing_best)

    return mixing_index_dict


def rigorous_mixing_index_kmeans(true_adata, pred_dict, true_s_adata, primary_variable, include_best=False):

    n_splits = 10
    test_fraction = 0.5
    mixing_index_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)
    n_clusters = int(np.log1p(n_test))
    # n_clusters = 2

    for _ in range(n_splits):
        print(f"start of repeat {_ + 1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        train_adata = true_adata[train_indices].copy()
        test_adata = true_adata[test_indices].copy()

        mixing_index_dict['true'].append(mixing_index(train_adata.copy(), test_adata.copy(), n_clusters=n_clusters))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            mixing_index_dict[model].append(mixing_index(sampled_adata, test_adata.copy(), n_clusters))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        mixing_index_dict['random_all'].append(mixing_index(sampled_true_all, test_adata.copy(), n_clusters))

        if include_best:
            true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
            mixing_best = 0
            for p in true_s_dict.keys():
                sampled_adata = adata_sample(true_s_dict[p], n_test)
                mixing = mixing_index(sampled_adata, test_adata.copy(), n_clusters)
                if mixing > mixing_best:
                    mixing_best = mixing
            mixing_index_dict['best'].append(mixing_best)

    return mixing_index_dict


def rigorous_mixing_index_kmeans_topdeg(true_adata, pred_dict, true_s_adata, primary_variable, true_logfc_df, num_top_deg):

    n_splits = 10
    test_fraction = 0.5
    mixing_index_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)
    n_clusters = int(np.log1p(n_test))
    # n_clusters = 2

    for _ in range(n_splits):
        print(f"start of repeat {_ + 1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        train_adata = true_adata[train_indices].copy()
        test_adata = true_adata[test_indices].copy()

        mixing_index_dict['true'].append(mixing_index_kmeans_topdeg(train_adata.copy(), test_adata.copy(), true_logfc_df, n_clusters=n_clusters, num_top_deg=num_top_deg))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            mixing_index_dict[model].append(mixing_index_kmeans_topdeg(sampled_adata, test_adata.copy(), true_logfc_df, n_clusters=n_clusters, num_top_deg=num_top_deg))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        mixing_index_dict['random_all'].append(mixing_index_kmeans_topdeg(sampled_true_all, test_adata.copy(), true_logfc_df, n_clusters=n_clusters, num_top_deg=num_top_deg))

        true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
        mixing_best = 0
        for p in true_s_dict.keys():
            sampled_adata = adata_sample(true_s_dict[p], n_test)
            mixing = mixing_index_kmeans_topdeg(sampled_adata, test_adata.copy(), true_logfc_df, n_clusters=n_clusters, num_top_deg=num_top_deg)
            if mixing > mixing_best:
                mixing_best = mixing
        mixing_index_dict['best'].append(mixing_best)

    return mixing_index_dict

def rigorous_spearman(true_adata, pred_dict, true_s_adata, primary_variable, include_best=False):
    n_splits = 10
    test_fraction = 0.5
    correlation_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    for _ in range(n_splits):
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        train_adata = true_adata[train_indices].copy()
        test_adata = true_adata[test_indices].copy()

        correlation_dict['true'].append(spearman_corr(train_adata, test_adata))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            correlation_dict[model].append(spearman_corr(sampled_adata, test_adata))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        correlation_dict['random_all'].append(spearman_corr(sampled_true_all, test_adata))

        if include_best:
            true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
            corr_best = 0
            for p in true_s_dict.keys():
                sampled_adata = adata_sample(true_s_dict[p], n_test)
                corr = spearman_corr(sampled_adata, test_adata)
                if corr > corr_best:
                    corr_best = corr
            correlation_dict['best'].append(corr_best)
    
    return correlation_dict


def spearman_corr_topdeg(pred_adata, true_adata, true_logfc_df, num_top_deg):
    
    if len(true_logfc_df) < num_top_deg:
        print(f"the number of significant genes is less than threshold: {num_top_deg}")
        return 0
    genes = true_logfc_df.index
    top_deg = genes[0:num_top_deg]
    pred_adata = pred_adata[:, pred_adata.var_names.isin(top_deg)].copy()
    true_adata = true_adata[:, true_adata.var_names.isin(top_deg)].copy()
    pred_mean = np.mean(pred_adata.X, axis = 0)
    true_mean = np.mean(true_adata.X,axis = 0)
    corr = spearmanr(pred_mean, true_mean)[0]
    return corr

def rigorous_spearman_topdeg(true_adata, pred_dict, true_s_adata, primary_variable, true_logfc_df, num_top_deg):
    n_splits = 10
    test_fraction = 0.5
    correlation_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    for _ in range(n_splits):
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        train_adata = true_adata[train_indices].copy()
        test_adata = true_adata[test_indices].copy()

        correlation_dict['true'].append(spearman_corr_topdeg(train_adata, test_adata, true_logfc_df, num_top_deg))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            correlation_dict[model].append(spearman_corr_topdeg(sampled_adata, test_adata, true_logfc_df, num_top_deg))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        correlation_dict['random_all'].append(spearman_corr_topdeg(sampled_true_all, test_adata, true_logfc_df, num_top_deg))

        true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
        corr_best = 0
        for p in true_s_dict.keys():
            sampled_adata = adata_sample(true_s_dict[p], n_test)
            corr = spearman_corr_topdeg(sampled_adata, test_adata, true_logfc_df, num_top_deg)
            if corr > corr_best:
                corr_best = corr
        correlation_dict['best'].append(corr_best)
    
    return correlation_dict


def rigorous_pearson_r2(true_adata, pred_dict, true_s_adata, primary_variable, include_best=False):
    n_splits = 10
    test_fraction = 0.5
    correlation_dict = defaultdict(list)
    r2_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    for _ in range(n_splits):
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        train_adata = true_adata[train_indices].copy()
        test_adata = true_adata[test_indices].copy()

        correlation_dict['true'].append(pearson_corr(train_adata, test_adata))
        r2_dict['true'].append(r2(train_adata, test_adata))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            # print(f'shape of sampled: {sampled_adata.X.shape}')
            correlation_dict[model].append(pearson_corr(sampled_adata, test_adata))
            r2_dict[model].append(r2(sampled_adata, test_adata))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        correlation_dict['random_all'].append(pearson_corr(sampled_true_all, test_adata))
        r2_dict['random_all'].append(r2(sampled_true_all, test_adata))

        if include_best:
            true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
            corr_best = 0
            r2_best = 0
            for p in true_s_dict.keys():
                sampled_adata = adata_sample(true_s_dict[p], n_test)
                corr = pearson_corr(sampled_adata, test_adata)
                r2_value = r2(sampled_adata, test_adata)
                if corr > corr_best:
                    corr_best = corr
                if r2_value > r2_best:
                    r2_best = r2_value
            correlation_dict['best'].append(corr_best)
            r2_dict['best'].append(r2_best)

    return correlation_dict, r2_dict

def rigorous_r2_skl(true_adata, pred_dict, true_s_adata, primary_variable, include_best=False):
    n_splits = 10
    test_fraction = 0.5
    r2_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    for _ in range(n_splits):
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        train_adata = true_adata[train_indices].copy()
        test_adata = true_adata[test_indices].copy()

        r2_dict['true'].append(r2_skl(train_adata, test_adata))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            r2_dict[model].append(r2_skl(sampled_adata, test_adata))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        r2_dict['random_all'].append(r2_skl(sampled_true_all, test_adata))

        if include_best:
            true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
            r2_best = 0
            for p in true_s_dict.keys():
                sampled_adata = adata_sample(true_s_dict[p], n_test)
                r2_value = r2_skl(sampled_adata, test_adata)
                if r2_value > r2_best:
                    r2_best = r2_value
            r2_dict['best'].append(r2_best)

    return r2_dict

def rigorous_mse(true_adata, pred_dict, true_s_adata, primary_variable, include_best=False):
    n_splits = 10
    test_fraction = 0.5
    metric_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    for _ in range(n_splits):
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        train_adata = true_adata[train_indices].copy()
        test_adata = true_adata[test_indices].copy()

        metric_dict['true'].append(mse(train_adata, test_adata))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            metric_dict[model].append(mse(sampled_adata, test_adata))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        metric_dict['random_all'].append(mse(sampled_true_all, test_adata))

        if include_best:
            true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
            metric_best = 10000000
            for p in true_s_dict.keys():
                sampled_adata = adata_sample(true_s_dict[p], n_test)
                metirc = mse(sampled_adata, test_adata)
                if metric < metric_best:
                    metric_best = metric
            metric_dict['best'].append(metric_best)

    return metric_dict

def rigorous_cosine_sim(true_adata, pred_dict, true_s_adata, primary_variable, include_best=False):
    n_splits = 10
    test_fraction = 0.5
    metric_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    for _ in range(n_splits):
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        train_adata = true_adata[train_indices].copy()
        test_adata = true_adata[test_indices].copy()

        metric_dict['true'].append(cosine_sim(train_adata, test_adata))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            metric_dict[model].append(cosine_sim(sampled_adata, test_adata))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        metric_dict['random_all'].append(cosine_sim(sampled_true_all, test_adata))

        if include_best:
            true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
            metric_best = 0
            for p in true_s_dict.keys():
                sampled_adata = adata_sample(true_s_dict[p], n_test)
                metirc = cosine_sim(sampled_adata, test_adata)
                if metric > metric_best:
                    metric_best = metric
            metric_dict['best'].append(metric_best)

    return metric_dict


def rigorous_stepwise_r2(true_adata, pred_dict, true_s_adata, ctrl_adata, primary_variable):
    n_splits = 10
    test_fraction = 0.5
    corr_dict = defaultdict(list)
    r2_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    for _ in range(n_splits):
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        train_adata = true_adata[train_indices].copy()
        test_adata = true_adata[test_indices].copy()

        
        output = stepwise_r2(ctrl_adata, test_adata, train_adata)
        corr_dict['true'].append(output[0])
        r2_dict['true'].append(output[1])

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            output = stepwise_r2(ctrl_adata, test_adata, sampled_adata)
            corr_dict[model].append(output[0])
            r2_dict[model].append(output[1])
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        output = stepwise_r2(ctrl_adata, test_adata, sampled_true_all)
        corr_dict['random_all'].append(output[0])
        r2_dict['random_all'].append(output[1])

        ctrl_means = np.asarray(ctrl_adata.X.mean(axis=0)).ravel()
        stim_means = np.asarray(test_adata.X.mean(axis=0)).ravel()
        genes = ctrl_adata.var_names
        sorted_idx = np.argsort(ctrl_means) 
        sorted_genes = genes[sorted_idx]
        ctrl_sorted = ctrl_means[sorted_idx]
        stim_sorted = stim_means[sorted_idx]
        outlier_baseline = stim_sorted.copy()
        outlier_baseline[:4000] = ctrl_sorted[:4000]
        start_value = 2
        num_genes = range(start_value, len(sorted_genes) + 1)
        corr_list = []
        r2_list = []
        for i in num_genes:
            if len(np.unique(outlier_baseline[:i])) == 1 or len(np.unique(stim_sorted[:i])) == 1:
                r_squared = r2_score(outlier_baseline[:i], stim_sorted[:i])
                corr = 0 if r_squared<0 else np.sqrt(r_squared)
            else:
                slope, intercept, r_value, _, _ = linregress(outlier_baseline[:i], stim_sorted[:i])
                r_squared = r_value ** 2
                corr = pearsonr(outlier_baseline[:i], stim_sorted[:i])[0]
            corr_list.append(corr)
            r2_list.append(r_squared)
        auc_corr = np.trapz(corr_list, x=num_genes) / (num_genes[-1] - num_genes[0])
        auc_r2 = np.trapz(r2_list, x=num_genes) / (num_genes[-1] - num_genes[0])
        corr_dict['outlier_1000_baseline'].append(auc_corr)
        r2_dict['outlier_1000_baseline'].append(auc_r2)

    return corr_dict, r2_dict

def rigorous_wasserstein(true_adata, pred_dict, true_s_adata, primary_variable, include_best=False):
    n_splits = 10
    test_fraction = 0.5
    metric_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    for _ in range(n_splits):
        print(f"start of repeat {_ + 1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        test_adata = true_adata[test_indices].copy()
        train_adata = true_adata[train_indices].copy()

        metric_dict['true'].extend(wasserstein([train_adata.X], [test_adata.X], True))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            metric_dict[model].extend(wasserstein([sampled_adata.X], [test_adata.X], True))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        metric_dict['random_all'].extend(wasserstein([sampled_true_all.X], [test_adata.X], True))

        if include_best:
            true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
            metric_best = 1000000
            for p in true_s_dict.keys():
                sampled_adata = adata_sample(true_s_dict[p], n_test)
                metric = wasserstein([sampled_adata.X], [test_adata.X], True)
                if metric[0] < metric_best:
                    metric_best = metric[0]
            metric_dict['best'].append(metric_best)

    # for key in metric_dict:
    #     metric_dict[key] = list(-np.log10(np.array(metric_dict[key]))) 

    return metric_dict

def rigorous_sinkhorn_mahalanobis(true_adata, pred_dict, true_s_adata, primary_variable, include_best=False):
    n_splits = 10
    test_fraction = 0.5
    metric_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    for _ in range(n_splits):
        print(f"start of repeat {_ + 1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        test_adata = true_adata[test_indices].copy()
        train_adata = true_adata[train_indices].copy()

        metric_dict['true'].extend(sinkhorn_mahalanobis([train_adata.X], [test_adata.X], True))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            metric_dict[model].extend(sinkhorn_mahalanobis([sampled_adata.X], [test_adata.X], True))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        metric_dict['random_all'].extend(sinkhorn_mahalanobis([sampled_true_all.X], [test_adata.X], True))

        if include_best:
            true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
            metric_best = 1000000
            for p in true_s_dict.keys():
                sampled_adata = adata_sample(true_s_dict[p], n_test)
                metric = sinkhorn_mahalanobis([sampled_adata.X], [test_adata.X], True)
                if metric[0] < metric_best:
                    metric_best = metric[0]
            metric_dict['best'].append(metric_best)

    return metric_dict

def wasserstein_topdeg(pred_adata, true_adata, true_logfc_df, num_top_deg):

    if len(true_logfc_df) < num_top_deg:
        print(f"the number of significant genes is less than threshold: {num_top_deg}")
        return 0
    genes = true_logfc_df.index
    top_deg = genes[0:num_top_deg]
    pred_adata = pred_adata[:, pred_adata.var_names.isin(top_deg)].copy()
    true_adata = true_adata[:, true_adata.var_names.isin(top_deg)].copy()

    return wasserstein([pred_adata.X], [true_adata.X], True)[0]

def rigorous_wasserstein_topdeg(true_adata, pred_dict, true_s_adata, primary_variable, true_logfc_df, num_top_deg):

    n_splits = 10
    test_fraction = 0.5
    metric_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)
    n_clusters = int(np.log1p(n_test))
    # n_clusters = 2

    for _ in range(n_splits):
        print(f"start of repeat {_ + 1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        train_adata = true_adata[train_indices].copy()
        test_adata = true_adata[test_indices].copy()

        metric_dict['true'].append(wasserstein_topdeg(train_adata.copy(), test_adata.copy(), true_logfc_df, num_top_deg=num_top_deg))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            metric_dict[model].append(wasserstein_topdeg(sampled_adata, test_adata.copy(), true_logfc_df, num_top_deg=num_top_deg))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        metric_dict['random_all'].append(wasserstein_topdeg(sampled_true_all, test_adata.copy(), true_logfc_df, num_top_deg=num_top_deg))

        true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
        best = 1000000
        for p in true_s_dict.keys():
            sampled_adata = adata_sample(true_s_dict[p], n_test)
            wass = wasserstein_topdeg(sampled_adata, test_adata.copy(), true_logfc_df, num_top_deg=num_top_deg)
            if wass < best:
                best = wass
        metric_dict['best'].append(best)

    # for key in metric_dict:
    #     metric_dict[key] = list(-np.log10(np.array(metric_dict[key])))

    return metric_dict


def rigorous_mmd(true_adata, pred_dict, true_s_adata, primary_variable, include_best=False):
    n_splits = 10
    test_fraction = 0.5
    metric_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    for _ in range(n_splits):
        print(f"start of repeat {_ + 1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        test_adata = true_adata[test_indices].copy()
        train_adata = true_adata[train_indices].copy()

        metric_dict['true'].append(compute_mmd(train_adata, test_adata))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            metric_dict[model].append(compute_mmd(sampled_adata, test_adata))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        metric_dict['random_all'].append(compute_mmd(sampled_true_all, test_adata))

        if include_best:
            true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
            metric_best = 1000000
            for p in true_s_dict.keys():
                sampled_adata = adata_sample(true_s_dict[p], n_test)
                metric = compute_mmd(sampled_adata, test_adata)
                if metric < metric_best:
                    metric_best = metric
            metric_dict['best'].append(metric_best)

    # for key in metric_dict:
    #      metric_dict[key] = list(-np.log10(np.array(metric_dict[key])))
        
    return metric_dict
    

def rigorous_local_mmd(true_adata, pred_dict, true_s_adata, primary_variable, k, include_best=False):
    print(f'number of k: {k}')
    n_splits = 10
    test_fraction = 0.5
    metric_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    for _ in range(n_splits):
        print(f"start of repeat {_ + 1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        test_adata = true_adata[test_indices].copy()
        train_adata = true_adata[train_indices].copy()

        metric_dict['true'].append(compute_local_mmd(train_adata, test_adata, k=k))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            metric_dict[model].append(compute_local_mmd(sampled_adata, test_adata, k=k))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        metric_dict['random_all'].append(compute_local_mmd(sampled_true_all, test_adata, k=k))

        if include_best:
            true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
            metric_best = 1000000
            for p in true_s_dict.keys():
                sampled_adata = adata_sample(true_s_dict[p], n_test)
                metric = compute_local_mmd(sampled_adata, test_adata, k=k)
                if metric < metric_best:
                    metric_best = metric
            metric_dict['best'].append(metric_best)
        
    return metric_dict

def rigorous_Edistance(true_adata, pred_dict, true_s_adata, primary_variable, include_best=False):
    n_splits = 10
    test_fraction = 0.5
    metric_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    for _ in range(n_splits):
        print(f"start of repeat {_ + 1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        test_adata = true_adata[test_indices].copy()
        train_adata = true_adata[train_indices].copy()

        metric_dict['true'].append(compute_Edistance(train_adata, test_adata))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            metric_dict[model].append(compute_Edistance(sampled_adata, test_adata))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        metric_dict['random_all'].append(compute_Edistance(sampled_true_all, test_adata))

        if include_best:
            true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
            metric_best = 1000000
            for p in true_s_dict.keys():
                sampled_adata = adata_sample(true_s_dict[p], n_test)
                metric = compute_Edistance(sampled_adata, test_adata)
                if metric < metric_best:
                    metric_best = metric
            metric_dict['best'].append(metric_best)
        
    return metric_dict

def rigorous_local_Edistance(true_adata, pred_dict, true_s_adata, primary_variable, k, include_best=False):
    print(f'number of k: {k}')
    n_splits = 10
    test_fraction = 0.5
    metric_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    for _ in range(n_splits):
        print(f"start of repeat {_ + 1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        test_adata = true_adata[test_indices].copy()
        train_adata = true_adata[train_indices].copy()

        metric_dict['true'].append(compute_local_Edistance(train_adata, test_adata, k=k))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            metric_dict[model].append(compute_local_Edistance(sampled_adata, test_adata, k=k))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        metric_dict['random_all'].append(compute_local_Edistance(sampled_true_all, test_adata, k=k))

        if include_best:
            true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
            metric_best = 1000000
            for p in true_s_dict.keys():
                sampled_adata = adata_sample(true_s_dict[p], n_test)
                metric = compute_local_Edistance(sampled_adata, test_adata, k=k)
                if metric < metric_best:
                    metric_best = metric
            metric_dict['best'].append(metric_best)
        
    return metric_dict

def rigorous_knn_loss(true_adata, pred_dict, true_s_adata, primary_variable, include_best=False):
    n_splits = 10
    test_fraction = 0.5
    metric_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    for _ in range(n_splits):
        print(f"start of repeat {_ + 1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        test_adata = true_adata[test_indices].copy()
        train_adata = true_adata[train_indices].copy()

        metric_dict['true'].append(compute_knn_loss(train_adata, test_adata))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            metric_dict[model].append(compute_knn_loss(sampled_adata, test_adata))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        metric_dict['random_all'].append(compute_knn_loss(sampled_true_all, test_adata))

        if include_best:
            true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
            metric_best = 1000000
            for p in true_s_dict.keys():
                sampled_adata = adata_sample(true_s_dict[p], n_test)
                metric = compute_knn_loss(sampled_adata, test_adata)
                if metric < metric_best:
                    metric_best = metric
            metric_dict['best'].append(metric_best)
        
    return metric_dict


def rigorous_deg_f1(true_adata, ctrl_adata, pred_dict, true_s_adata, primary_variable, ood_primary, include_best=False):
    # n_splits = 10
    n_splits = 5
    test_fraction = 0.5
    f1_down_dict = defaultdict(list)
    f1_up_dict = defaultdict(list)
    precision_down_dict = defaultdict(list)
    precision_up_dict = defaultdict(list)
    recall_down_dict = defaultdict(list)
    recall_up_dict = defaultdict(list)
    typeI_down_dict = defaultdict(list)
    typeI_up_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    temp_folder = os.path.join('temp', 'deg_files')
    os.makedirs(temp_folder, exist_ok=True)

    log_file = open(f"{temp_folder}/log_{ood_primary}.txt", "a")
    
    for i in range(n_splits):
        print(f"start of repeat: {i+1}")
        # log_file.write("####################################  ")
        # log_file.write(f" repeat: {i+1}\n")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        test_adata = true_adata[test_indices].copy()
        train_adata = true_adata[train_indices].copy()

        ctrl_sample = adata_sample(ctrl_adata, n_test)

        combined_test = ad.concat([test_adata, ctrl_sample], label="batch", keys=["stimulated", "ctrl"])
        combined_test.obs_names_make_unique()
        real_degs = seurat_get_down_up_genes(combined_test, "batch", "stimulated", "ctrl", "MAST")
        print(f"real down degs: {len(real_degs[0])}")
        print(f"real up degs: {len(real_degs[1])}")
        # log_file.write(f"real down degs: {len(real_degs[0])}\n")
        # log_file.write(f"real up degs: {len(real_degs[1])}\n")

        print(f"save model: true-{i+1}")
        process_for_f1(
            'true', real_degs, train_adata, ctrl_sample, f1_down_dict, f1_up_dict, precision_down_dict, precision_up_dict, 
            recall_down_dict, recall_up_dict, typeI_down_dict, typeI_up_dict, log_file=log_file)
    
        # total_count = 1e6
        total_count = 1e4
        for model in pred_dict.keys():
            pred_adata = pred_dict[model]
            pred_adata_sample = adata_sample(pred_adata, n_test)
            # pred_adata_sample.X = np.where(pred_adata_sample.X < 0, 0, pred_adata_sample.X)
            if model != 'noperturb':
                if issparse(pred_adata_sample.X):
                    exp_data = np.exp(pred_adata_sample.X.data) - 1
                    pred_adata_sample.layers['counts'] = csr_matrix((exp_data, pred_adata_sample.X.indices, pred_adata_sample.X.indptr),
                                                            shape=pred_adata_sample.X.shape) / total_count
                else:
                    pred_adata_sample.layers['counts'] = (np.exp(pred_adata_sample.X) - 1) / total_count
                # pred_adata_sample.layers['counts'] = (np.exp(np.asarray(pred_adata_sample.X))-1)/total_count
            print(f"save model: {model}-{i+1}")
            process_for_f1(
                model, real_degs, pred_adata_sample, ctrl_sample, f1_down_dict, f1_up_dict, precision_down_dict, precision_up_dict, 
                recall_down_dict, recall_up_dict, typeI_down_dict, typeI_up_dict, log_file=log_file)
    
        sampled_true_all = adata_sample(true_s_adata, n_test)
        print(f"save model: random_all-{i+1}")
        process_for_f1(
                'random_all', real_degs, sampled_true_all, ctrl_sample, f1_down_dict, f1_up_dict, precision_down_dict, precision_up_dict, 
                recall_down_dict, recall_up_dict, typeI_down_dict, typeI_up_dict, log_file=log_file)
    
        if include_best:
            true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
            f1_down_best = 0
            f1_up_best = 0
            precision_down_best = 0
            precision_up_best = 0
            recall_down_best = 0
            recall_up_best = 0
            typeI_down_best = 1
            typeI_up_best = 1
            for p in true_s_dict.keys():
                alt_adata = true_s_dict[p]
                alt_adata_sample = adata_sample(alt_adata, n_test)
                f1_down, precision_down, recall_down, typeI_down, f1_up, precision_up, recall_up, typeI_up = \
                    process_for_f1(
                        'best', real_degs, alt_adata_sample, ctrl_sample, f1_down_dict, f1_up_dict, precision_down_dict, precision_up_dict, 
                        recall_down_dict, recall_up_dict, typeI_down_dict, typeI_up_dict, append=False, log_file=log_file)
                if f1_down > f1_down_best:
                    f1_down_best = f1_down
                if f1_up > f1_up_best:
                    f1_up_best = f1_up
                if precision_down > precision_down_best:
                    precision_down_best = precision_down
                if precision_up > precision_up_best:
                    precision_up_best = precision_up
                if recall_down > recall_down_best:
                    recall_down_best = recall_down
                if recall_up > recall_up_best:
                    recall_up_best = recall_up
                if typeI_down < typeI_down_best:
                    typeI_down_best = typeI_down
                if typeI_up < typeI_up_best:
                    typeI_up_best = typeI_up

            print(f"save model: best-{i+1}")
            f1_down_dict['best'].append(f1_down_best)
            f1_up_dict['best'].append(f1_up_best)
            precision_down_dict['best'].append(precision_down_best)
            precision_up_dict['best'].append(precision_up_best)
            recall_down_dict['best'].append(recall_down_best)
            recall_up_dict['best'].append(recall_up_best)
            typeI_down_dict['best'].append(typeI_down_best)
            typeI_up_dict['best'].append(typeI_up_best)

    log_file.close()

    return f1_down_dict, f1_up_dict, precision_down_dict, precision_up_dict, recall_down_dict, recall_up_dict, typeI_down_dict, typeI_up_dict


def rigorous_store_deg_df(true_adata, ctrl_adata, pred_dict, true_s_adata):
    # n_splits = 10
    n_splits = 5
    test_fraction = 0.5
    deg_df_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    for i in range(n_splits):
        print(f"start of repeat: {i+1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        test_adata = true_adata[test_indices].copy()
        train_adata = true_adata[train_indices].copy()

        ctrl_sample = adata_sample(ctrl_adata, n_test)

        combined_test = ad.concat([test_adata, ctrl_sample], label="batch", keys=["stimulated", "ctrl"])
        combined_test.obs_names_make_unique()
        deg_df = seurat_deg(combined_test, "batch", "stimulated", "ctrl", test_use="MAST")
        deg_df_dict['real'].append(deg_df)
        print(f"save model: true-{i+1}")
        combined_train = ad.concat([train_adata, ctrl_sample], label="batch", keys=["stimulated", "ctrl"])
        combined_train.obs_names_make_unique()
        deg_df = seurat_deg(combined_train, "batch", "stimulated", "ctrl", test_use="MAST")
        deg_df_dict['true'].append(deg_df)

        total_count = 1e4
        for model in pred_dict.keys():
            pred_adata = pred_dict[model]
            pred_adata_sample = adata_sample(pred_adata, n_test)
            if model != 'noperturb':
                if issparse(pred_adata_sample.X):
                    exp_data = np.exp(pred_adata_sample.X.data) - 1
                    pred_adata_sample.layers['counts'] = csr_matrix((exp_data, pred_adata_sample.X.indices, pred_adata_sample.X.indptr),
                                                            shape=pred_adata_sample.X.shape) / total_count
                else:
                    pred_adata_sample.layers['counts'] = (np.exp(pred_adata_sample.X) - 1) / total_count
                # pred_adata_sample.layers['counts'] = (np.exp(np.asarray(pred_adata_sample.X))-1)/total_count
            print(f"save model: {model}-{i+1}")
            combined_pred = ad.concat([pred_adata_sample, ctrl_sample], label="batch", keys=["stimulated", "ctrl"])
            combined_pred.obs_names_make_unique()
            deg_df = seurat_deg(combined_pred, "batch", "stimulated", "ctrl", test_use="MAST")
            deg_df_dict[model].append(deg_df)
    
        sampled_true_all = adata_sample(true_s_adata, n_test)
        print(f"save model: random_all-{i+1}")
        combined_pred = ad.concat([sampled_true_all, ctrl_sample], label="batch", keys=["stimulated", "ctrl"])
        combined_pred.obs_names_make_unique()
        deg_df = seurat_deg(combined_pred, "batch", "stimulated", "ctrl", test_use="MAST")
        deg_df_dict['random_all'].append(deg_df)
    
    return deg_df_dict


def rigorous_logfc_correlation(true_adata, pred_dict, ctrl_adata, true_s_adata, primary_variable, include_best=False):

    n_splits = 10
    test_fraction = 0.5
    metric_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)

    for _ in range(n_splits):
        print(f"start of repeat {_+1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        train_adata = true_adata[train_indices].copy()
        test_adata = true_adata[test_indices].copy()

        ctrl_sample = adata_sample(ctrl_adata, n_test)
        ## check using all ctrl adata
        # ctrl_sample = ctrl_adata.copy()
        ctrl_true_adata = ad.concat([test_adata, ctrl_sample], label="batch", keys=["stimulated", "ctrl"])
        ctrl_true_adata.obs_names_make_unique()

        true_logfc_df = seurat_deg(
            ctrl_true_adata, "batch", "stimulated", "ctrl", test_use='MAST', find_variable_features=False)

        metric_dict['true'].append(logfc_correlation(train_adata.copy(), ctrl_sample.copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=False))

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            if model == 'noperturb':
                metric_dict[model].append(logfc_correlation(sampled_adata, ctrl_sample.copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=False))
            else:
                metric_dict[model].append(logfc_correlation(sampled_adata, ctrl_sample.copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=True))
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        metric_dict['random_all'].append(logfc_correlation(sampled_true_all, ctrl_sample.copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=False))

        if include_best:
            true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
            metric_best = 0
            for p in true_s_dict.keys():
                sampled_adata = adata_sample(true_s_dict[p], n_test)
                metric = logfc_correlation(sampled_adata, ctrl_sample.copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=False)
                if metric > metric_best:
                    metric_best = metric
            metric_dict['best'].append(metric_best)

    return metric_dict

def rigorous_logfc_dict(true_adata, pred_dict, ctrl_adata, true_s_adata, primary_variable):

    n_splits = 10
    test_fraction = 0.5
    metric_dict = defaultdict(dict)
    n_test = int(test_fraction * true_adata.n_obs)

    for repeat in range(1, n_splits+1):
        print(f"start of repeat {repeat}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        train_adata = true_adata[train_indices].copy()
        test_adata = true_adata[test_indices].copy()

        ctrl_sample = adata_sample(ctrl_adata, n_test)
        ## check using all ctrl adata
        # ctrl_sample = ctrl_adata.copy()
        ctrl_true_adata = ad.concat([test_adata, ctrl_sample], label="batch", keys=["stimulated", "ctrl"])
        ctrl_true_adata.obs_names_make_unique()

        true_logfc_df = seurat_deg(
            ctrl_true_adata, "batch", "stimulated", "ctrl", test_use='MAST', find_variable_features=False)

        metric_dict['true'][repeat] = logfc_dict(train_adata.copy(), ctrl_sample.copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=False)

        for model in pred_dict.keys():
            sampled_adata = adata_sample(pred_dict[model], n_test)
            if model == 'noperturb':
                metric_dict[model][repeat] = logfc_dict(sampled_adata, ctrl_sample.copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=False)
            else:
                metric_dict[model][repeat] = logfc_dict(sampled_adata, ctrl_sample.copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=True)
        
        sampled_true_all = adata_sample(true_s_adata, n_test)
        metric_dict['random_all'][repeat] = logfc_dict(sampled_true_all, ctrl_sample.copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=False)

    return metric_dict


def noise_effect_spearman(true_adata, true_s_adata, n_repeats=3, percent_interval=10, method='sampling'):
    test_fraction = 0.5
    spearman_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)
    percents = np.arange(0,101,percent_interval)

    for repeat in range(n_repeats):
        print(f"start of repeat {repeat+1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        test_adata = true_adata[test_indices].copy()
        train_adata = true_adata[train_indices].copy()
        
        gene_sets = split_genes_to_groups(true_adata.n_vars, len(percents)-1)
        gene_sets.insert(0, np.array([]).astype(int))
        corr_list = []

        if method == 'random_all':
            if issparse(true_s_adata.X):
                true_s_adata.X = true_s_adata.X.toarray()
            if issparse(true_s_adata.layers['counts']):
                true_s_adata.layers['counts'] = true_s_adata.layers['counts'].toarray()
        for p in percents:
            print(f"repeat: {repeat+1}, percent: {p}")
            p_temp = 0 if p==0 else percent_interval
            if mode == 'nested':
                train_adata = add_noise(train_adata, true_s_adata, gene_sets[int(p/percent_interval)], method, p=_temp)
                metric = spearman_corr(train_adata, test_adata)
            else:
                noised_train_adata = add_noise(train_adata.copy(), true_s_adata, None, method, p=p)
                metric = spearman_corr(noised_train_adata, test_adata)
            corr_list.append(metric)    
        spearman_dict[f"repeat {repeat+1}"] = pd.DataFrame({'n': percents, 'true_positives': corr_list})
    return spearman_dict

def noise_effect_wasserstein(true_adata, true_s_adata, n_repeats=3, percent_interval=10, method='random_all', mode='nested'):
    test_fraction = 0.5
    metric_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)
    percents = np.arange(0,101,percent_interval)

    for repeat in range(n_repeats):
        print(f"start of repeat {repeat+1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        test_adata = true_adata[test_indices].copy()
        train_adata = true_adata[train_indices].copy()
        
        gene_sets = split_genes_to_groups(true_adata.n_vars, len(percents)-1)
        gene_sets.insert(0, np.array([]).astype(int))
        metric_list = []

        if method == 'random_all':
            if issparse(true_s_adata.X):
                true_s_adata.X = true_s_adata.X.toarray()
            if issparse(true_s_adata.layers['counts']):
                true_s_adata.layers['counts'] = true_s_adata.layers['counts'].toarray()
        for p in percents:
            print(f"repeat: {repeat+1}, percent: {p}")
            p_temp = 0 if p==0 else percent_interval
            if mode == 'nested':
                train_adata = add_noise(train_adata, true_s_adata, gene_sets[int(p/percent_interval)], method, p=p_temp)
                metric = wasserstein([train_adata.X], [test_adata.X], True)[0]
                # metric = -np.log10(metric)
            else:
                noised_train_adata = add_noise(train_adata.copy(), true_s_adata, None, method, p=p)
                metric = wasserstein([noised_train_adata.X], [test_adata.X], True)[0]
                # metric = -np.log10(metric)
            metric_list.append(metric)
        metric_dict[f"repeat {repeat+1}"] = pd.DataFrame({'n': percents, 'true_positives': metric_list})

    return metric_dict

def noise_effect_mmd(true_adata, true_s_adata, n_repeats=3, percent_interval=10, method='random_all', mode='nested'):
    test_fraction = 0.5
    metric_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)
    percents = np.arange(0,101,percent_interval)

    for repeat in range(n_repeats):
        print(f"start of repeat {repeat+1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        test_adata = true_adata[test_indices].copy()
        train_adata = true_adata[train_indices].copy()
        
        gene_sets = split_genes_to_groups(true_adata.n_vars, len(percents)-1)
        gene_sets.insert(0, np.array([]).astype(int))
        metric_list = []

        if method == 'random_all':
            if issparse(true_s_adata.X):
                true_s_adata.X = true_s_adata.X.toarray()
            if issparse(true_s_adata.layers['counts']):
                true_s_adata.layers['counts'] = true_s_adata.layers['counts'].toarray()
        for p in percents:
            print(f"repeat: {repeat+1}, percent: {p}")
            p_temp = 0 if p==0 else percent_interval
            if mode == 'nested':
                train_adata = add_noise(train_adata, true_s_adata, gene_sets[int(p/percent_interval)], method, p=p_temp)
                # metric = -np.log10(compute_mmd(train_adata, test_adata))
                metric = compute_mmd(train_adata, test_adata)
            else:
                noised_train_adata = add_noise(train_adata.copy(), true_s_adata, None, method, p=p)
                # metric = -np.log10(compute_mmd(noised_train_adata, test_adata))
                metric = compute_mmd(noised_train_adata, test_adata)
            metric_list.append(metric)
        metric_dict[f"repeat {repeat+1}"] = pd.DataFrame({'n': percents, 'true_positives': metric_list})
    
    return metric_dict

def noise_effect_local_mmd(true_adata, true_s_adata, n_repeats=3, percent_interval=10, method='sampling', mode='nested', k=None):
    test_fraction = 0.5
    metric_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)
    percents = np.arange(0,101,percent_interval)

    for repeat in range(n_repeats):
        print(f"start of repeat {repeat+1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        test_adata = true_adata[test_indices].copy()
        train_adata = true_adata[train_indices].copy()
        
        gene_sets = split_genes_to_groups(true_adata.n_vars, len(percents)-1)
        gene_sets.insert(0, np.array([]).astype(int))
        metric_list = []

        if method == 'random_all':
            if issparse(true_s_adata.X):
                true_s_adata.X = true_s_adata.X.toarray()
            if issparse(true_s_adata.layers['counts']):
                true_s_adata.layers['counts'] = true_s_adata.layers['counts'].toarray()
        for p in percents:
            print(f"repeat: {repeat+1}, percent: {p}")
            p_temp = 0 if p==0 else percent_interval
            if mode == 'nested':
                train_adata = add_noise(train_adata, true_s_adata, gene_sets[int(p/percent_interval)], method, p=p_temp)
                metric = compute_local_mmd(train_adata, test_adata, k=k)
            else:
                noised_train_adata = add_noise(train_adata.copy(), true_s_adata, None, method, p=p)
                metric = compute_local_mmd(noised_train_adata, test_adata, k=k)
            metric_list.append(metric)
        metric_dict[f"repeat {repeat+1}"] = pd.DataFrame({'n': percents, 'true_positives': metric_list})
    
    return metric_dict   

def noise_effect_Edistance(true_adata, true_s_adata, n_repeats=3, percent_interval=10, method='random_all', mode='nested'):
    test_fraction = 0.5
    metric_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)
    percents = np.arange(0,101,percent_interval)

    for repeat in range(n_repeats):
        print(f"start of repeat {repeat+1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        test_adata = true_adata[test_indices].copy()
        train_adata = true_adata[train_indices].copy()
        
        gene_sets = split_genes_to_groups(true_adata.n_vars, len(percents)-1)
        gene_sets.insert(0, np.array([]).astype(int))
        metric_list = []

        if method == 'random_all':
            if issparse(true_s_adata.X):
                true_s_adata.X = true_s_adata.X.toarray()
            if issparse(true_s_adata.layers['counts']):
                true_s_adata.layers['counts'] = true_s_adata.layers['counts'].toarray()
        for p in percents:
            print(f"repeat: {repeat+1}, percent: {p}")
            p_temp = 0 if p==0 else percent_interval
            if mode == 'nested':
                train_adata = add_noise(train_adata, true_s_adata, gene_sets[int(p/percent_interval)], method, p=p_temp)
                metric = compute_Edistance(train_adata, test_adata)
            else:
                noised_train_adata = add_noise(train_adata.copy(), true_s_adata, None, method, p=p)
                metric = compute_Edistance(noised_train_adata, test_adata)
            metric_list.append(metric)
        metric_dict[f"repeat {repeat+1}"] = pd.DataFrame({'n': percents, 'true_positives': metric_list})
    return metric_dict

def noise_effect_local_Edistance(true_adata, true_s_adata, n_repeats=3, percent_interval=10, method='sampling', mode='nested', k=None):
    test_fraction = 0.5
    metric_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)
    percents = np.arange(0,101,percent_interval)

    for repeat in range(n_repeats):
        print(f"start of repeat {repeat+1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        test_adata = true_adata[test_indices].copy()
        train_adata = true_adata[train_indices].copy()
        
        gene_sets = split_genes_to_groups(true_adata.n_vars, len(percents)-1)
        gene_sets.insert(0, np.array([]).astype(int))
        metric_list = []

        if method == 'random_all':
            if issparse(true_s_adata.X):
                true_s_adata.X = true_s_adata.X.toarray()
            if issparse(true_s_adata.layers['counts']):
                true_s_adata.layers['counts'] = true_s_adata.layers['counts'].toarray()
        for p in percents:
            print(f"repeat: {repeat+1}, percent: {p}")
            p_temp = 0 if p==0 else percent_interval
            if mode == 'nested':
                train_adata = add_noise(train_adata, true_s_adata, gene_sets[int(p/percent_interval)], method, p=p_temp)
                metric = compute_local_Edistance(train_adata, test_adata, k=k)
            else:
                noised_train_adata = add_noise(train_adata.copy(), true_s_adata, None, method, p=p)
                metric = compute_local_Edistance(noised_train_adata, test_adata, k=k)
            metric_list.append(metric)
        metric_dict[f"repeat {repeat+1}"] = pd.DataFrame({'n': percents, 'true_positives': metric_list})
    
    return metric_dict

def noise_effect_knn_loss(true_adata, true_s_adata, n_repeats=3, percent_interval=10, method='sampling', mode='nested', k=30):
    test_fraction = 0.5
    metric_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)
    percents = np.arange(0,101,percent_interval)

    for repeat in range(n_repeats):
        print(f"start of repeat {repeat+1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        test_adata = true_adata[test_indices].copy()
        train_adata = true_adata[train_indices].copy()
        
        gene_sets = split_genes_to_groups(true_adata.n_vars, len(percents)-1)
        gene_sets.insert(0, np.array([]).astype(int))
        metric_list = []

        if method == 'random_all':
            if issparse(true_s_adata.X):
                true_s_adata.X = true_s_adata.X.toarray()
            if issparse(true_s_adata.layers['counts']):
                true_s_adata.layers['counts'] = true_s_adata.layers['counts'].toarray()
        for p in percents:
            print(f"repeat: {repeat+1}, percent: {p}")
            p_temp = 0 if p==0 else percent_interval
            if mode == 'nested':
                train_adata = add_noise(train_adata, true_s_adata, gene_sets[int(p/percent_interval)], method, p=p_temp)
                metric = compute_knn_loss(train_adata, test_adata, k=k)
            else:
                noised_train_adata = add_noise(train_adata.copy(), true_s_adata, None, method, p=p)
                metric = compute_knn_loss(noised_train_adata, test_adata, k=k)
            metric_list.append(metric)
        metric_dict[f"repeat {repeat+1}"] = pd.DataFrame({'n': percents, 'true_positives': metric_list})
    return metric_dict


def noise_effect_mixing_index(true_adata, true_s_adata, all_adata, n_repeats=3, percent_interval=10, method='random_all', mode='nested'):
    all_seurat = create_seurat_object(all_adata.copy())
    test_fraction = 0.5
    mixing_index_dict = defaultdict(list)
    n_test = int(test_fraction * true_adata.n_obs)
    percents = np.arange(0,101,percent_interval)

    for repeat in range(n_repeats):
        print(f"start of repeat {repeat+1}")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        test_adata = true_adata[test_indices].copy()
        train_adata = true_adata[train_indices].copy()
        
        gene_sets = split_genes_to_groups(true_adata.n_vars, len(percents)-1)
        gene_sets.insert(0, np.array([]).astype(int))
        mixing_list = []

        if method == 'random_all':
            if issparse(true_s_adata.X):
                true_s_adata.X = true_s_adata.X.toarray()
            if issparse(true_s_adata.layers['counts']):
                true_s_adata.layers['counts'] = true_s_adata.layers['counts'].toarray()
        for p in percents:
            print(f"repeat: {repeat+1}, percent: {p}")
            p_temp = 0 if p==0 else percent_interval
            if mode == 'nested':
                train_adata = add_noise(train_adata, true_s_adata, gene_sets[int(p/percent_interval)], method, p=p_temp)
                metric = mixing_index_seurat(train_adata, test_adata, all_adata=all_adata.copy(), all_seurat=all_seurat, do_layer_process=False)
            else:
                noised_train_adata = add_noise(train_adata.copy(), true_s_adata, None, method, p=p)
                metric = mixing_index_seurat(noised_train_adata, test_adata, all_adata=all_adata.copy(), all_seurat=all_seurat, do_layer_process=False)
            mixing_list.append(metric)
        mixing_index_dict[f"repeat {repeat+1}"] = pd.DataFrame({'n': percents, 'true_positives': mixing_list})
    return mixing_index_dict

def noise_effect_f1(true_adata, ctrl_adata, true_s_adata, ood_primary, n_repeats=3, percent_interval=10, method='sampling'):
    test_fraction = 0.5
    f1_down_dict = defaultdict(list)
    f1_up_dict = defaultdict(list)
    precision_down_dict = defaultdict(list)
    precision_up_dict = defaultdict(list)
    recall_down_dict = defaultdict(list)
    recall_up_dict = defaultdict(list)
    dicts = [f1_down_dict, f1_up_dict, precision_down_dict, precision_up_dict, recall_down_dict, recall_up_dict]
    n_test = int(test_fraction * true_adata.n_obs)
    percents = np.arange(0,101,percent_interval)

    log_file = open(f"log_{ood_primary}.txt", "a")
    
    for repeat in range(n_repeats):
        print(f"start of repeat: {repeat+1}")
        log_file.write("####################################  ")
        log_file.write(f" repeat: {repeat+1}\n")
        test_indices = np.random.choice(true_adata.n_obs, size=n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(true_adata.n_obs), test_indices)
        
        test_adata = true_adata[test_indices].copy()
        train_adata = true_adata[train_indices].copy()

        ctrl_sample = adata_sample(ctrl_adata, n_test)

        combined_test = ad.concat([test_adata, ctrl_sample], label="batch", keys=["stimulated", "ctrl"])
        real_degs = seurat_get_down_up_genes(combined_test, "batch", "stimulated", "ctrl", "MAST")
        print(f"real down degs: {len(real_degs[0])}")
        print(f"real up degs: {len(real_degs[1])}")
        log_file.write(f"real down degs: {len(real_degs[0])}\n")
        log_file.write(f"real up degs: {len(real_degs[1])}\n")

        gene_sets = split_genes_to_groups(true_adata.n_vars, len(percents)-1)
        gene_sets.insert(0, np.array([]).astype(int))

        model_name = f"repeat {repeat+1}"
        if method == 'random_all':
            if issparse(true_s_adata.X):
                true_s_adata.X = true_s_adata.X.toarray()
            if issparse(true_s_adata.layers['counts']):
                true_s_adata.layers['counts'] = true_s_adata.layers['counts'].toarray()
        for p in percents:
            print(f"repeat: {repeat+1}, percent: {p}")
            p_temp = 0 if p==0 else percent_interval
            if mode == 'nested':
                train_adata = add_noise(train_adata, true_s_adata, gene_sets[int(p/percent_interval)], method, p=p_temp)
                process_for_f1(
                    model_name, real_degs, train_adata, ctrl_sample, f1_down_dict, f1_up_dict, precision_down_dict, \
                    precision_up_dict, recall_down_dict, recall_up_dict, append=True, log_file=log_file)
            else:
                noised_train_adata = add_noise(train_adata.copy(), true_s_adata, None, method, p=p)
                process_for_f1(
                    model_name, real_degs, noised_train_adata, ctrl_sample, f1_down_dict, f1_up_dict, precision_down_dict, \
                    precision_up_dict, recall_down_dict, recall_up_dict, append=True, log_file=log_file)
        for dict_ in dicts:
            dict_[model_name] = convert_list2DF(percents, dict_[model_name])

    log_file.close()

    return f1_down_dict, f1_up_dict, precision_down_dict, precision_up_dict, recall_down_dict, recall_up_dict

    
def get_correlated_gene_pairs(adata, correlation_method='pearson', significance_threshold = 0.05):
    gene_var = np.var(adata.X, axis=0)
    adata = adata[:, gene_var > 0].copy()
    # print(np.isnan(adata.X).sum(), np.isinf(adata.X).sum())  # Should print (0,0)
    # print((adata.X > 100).sum())
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=500)  
    print("calculating hvg done!")
    adata_filtered = adata[:, adata.var["highly_variable"]].copy()
    gene_expression = pd.DataFrame(adata_filtered.X.toarray() if hasattr(adata_filtered.X, "toarray") else adata_filtered.X, 
                               columns=adata_filtered.var_names, index=adata_filtered.obs_names)
    gene_pairs = list(combinations(gene_expression.columns, 2))
    correlation_results = []
    for gene1, gene2 in gene_pairs:
        if correlation_method == "pearson":
            r, p = pearsonr(gene_expression[gene1], gene_expression[gene2])
        else:
            r, p = spearmanr(gene_expression[gene1], gene_expression[gene2])
        correlation_results.append((gene1, gene2, r, p))
    correlation_df = pd.DataFrame(correlation_results, columns=["Gene1", "Gene2", "Correlation", "p-value"])
    correlation_df["p-adjusted"] = multipletests(correlation_df["p-value"], method="fdr_bh")[1]
    significant_correlations = correlation_df[correlation_df["p-adjusted"] < significance_threshold]
    positives_df = significant_correlations[significant_correlations["Correlation"] > 0].copy()
    negatives_df = significant_correlations[significant_correlations["Correlation"] < 0].copy()
    positive_correlated_gene_pair = [frozenset({positives_df.loc[i,"Gene1"], positives_df.loc[i, "Gene2"]}) for i in positives_df.index]
    negative_correlated_gene_pair = [frozenset({negatives_df.loc[i,"Gene1"], negatives_df.loc[i, "Gene2"]}) for i in negatives_df.index]
    return positive_correlated_gene_pair, negative_correlated_gene_pair


def gg_corr_f1(test_adata, train_adata, primary, real_gg_dict, correlation_method = 'pearson', significance_threshold = 0.05):
    if primary not in real_gg_dict:
        print(f"entered in if loop, {primary} not in dict")
        real_gg_dict[primary] = get_correlated_gene_pairs(test_adata, correlation_method, significance_threshold)
    pred_gene_pairs = get_correlated_gene_pairs(train_adata, correlation_method, significance_threshold)
    f1, precision, recall = calculate_f1(real_gg_dict[primary], pred_gene_pairs)
    print(f"f1: {f1}, precision: {precision}, recall: {recall}")
    return f1


def rigorous_subset_spearman(true_adata, ctrl_adata, pred_dict, true_s_adata, primary_variable, true_logfc_df):
    non_significant, trivial, non_trivial = define_trivial(true_adata, ctrl_adata, true_logfc_df)
    print(f"# non_significant: {len(non_significant)}\n# trivial: {len(trivial)}\n# non_trivial: {len(non_trivial)}")
    
    pred_subset = {pred: adata[:, non_trivial].copy() for pred, adata in pred_dict.items()}
    non_trivial_dict = rigorous_spearman(true_adata[:,non_trivial].copy(), pred_subset, true_s_adata[:,non_trivial].copy(), primary_variable)
    pred_subset = {pred: adata[:, trivial].copy() for pred, adata in pred_dict.items()}
    trivial_dict = rigorous_spearman(true_adata[:,trivial].copy(), pred_subset, true_s_adata[:,trivial].copy(), primary_variable)
    pred_subset = {pred: adata[:, non_significant].copy() for pred, adata in pred_dict.items()}
    non_significant_dict = rigorous_spearman(true_adata[:,non_significant].copy(), pred_subset, true_s_adata[:,non_significant].copy(), primary_variable)

    return non_trivial_dict, trivial_dict, non_significant_dict, [len(non_trivial), len(trivial), len(non_significant)]

def rigorous_subset_logfc_correlation(true_adata, ctrl_adata, pred_dict, true_s_adata, primary_variable, true_logfc_df):
    non_significant, trivial, non_trivial = define_trivial(true_adata, ctrl_adata, true_logfc_df)
    print(f"# non_significant: {len(non_significant)}\n# trivial: {len(trivial)}\n# non_trivial: {len(non_trivial)}")

    non_trivial_dict, trivial_dict, non_significant_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    metric_dict = rigorous_logfc_dict(true_adata, pred_dict, ctrl_adata, true_s_adata, primary_variable)

    for model, values in metric_dict.items():
        repeats = values.keys()
        for repeat in repeats:
            list1 = [metric_dict[model][repeat][0][g] for g in non_trivial]
            list2 = [metric_dict[model][repeat][1][g] for g in non_trivial]
            non_trivial_dict[model].append(pearsonr(list1, list2)[0])
            list1 = [metric_dict[model][repeat][0][g] for g in trivial]
            list2 = [metric_dict[model][repeat][1][g] for g in trivial]
            trivial_dict[model].append(pearsonr(list1, list2)[0])
            list1 = [metric_dict[model][repeat][0][g] for g in non_significant]
            list2 = [metric_dict[model][repeat][1][g] for g in non_significant]
            non_significant_dict[model].append(pearsonr(list1, list2)[0])

    # pred_subset = {pred: adata[:, non_trivial].copy() for pred, adata in pred_dict.items()}
    # non_trivial_dict = rigorous_logfc_correlation(true_adata[:,non_trivial].copy(), pred_subset, ctrl_adata[:,non_trivial].copy(), true_s_adata[:,non_trivial].copy(), primary_variable)
    # pred_subset = {pred: adata[:, trivial].copy() for pred, adata in pred_dict.items()}
    # trivial_dict = rigorous_logfc_correlation(true_adata[:,trivial].copy(), pred_subset, ctrl_adata[:,trivial].copy(), true_s_adata[:,trivial].copy(), primary_variable)
    # pred_subset = {pred: adata[:, non_significant].copy() for pred, adata in pred_dict.items()}
    # non_significant_dict = rigorous_logfc_correlation(true_adata[:,non_significant].copy(), pred_subset, ctrl_adata[:,non_significant].copy(), true_s_adata[:,non_significant].copy(), primary_variable)

    return non_trivial_dict, trivial_dict, non_significant_dict, [len(non_trivial), len(trivial), len(non_significant)]
    # return non_trivial_dict, [len(non_trivial), len(trivial), len(non_significant)]

def rigorous_subset_pearson_r2(true_adata, ctrl_adata, pred_dict, true_s_adata, primary_variable, true_logfc_df, original_true=None):
    if original_true is None:
        original_true = true_adata.copy()
    non_significant, trivial, non_trivial = define_trivial(original_true, ctrl_adata, true_logfc_df)
    print(f"# non_significant: {len(non_significant)}\n# trivial: {len(trivial)}\n# non_trivial: {len(non_trivial)}")
    
    pred_subset = {pred: adata[:, non_trivial].copy() for pred, adata in pred_dict.items()}
    non_trivial_dict = rigorous_pearson_r2(true_adata[:,non_trivial].copy(), pred_subset, true_s_adata[:,non_trivial].copy(), primary_variable)
    pred_subset = {pred: adata[:, trivial].copy() for pred, adata in pred_dict.items()}
    trivial_dict = rigorous_pearson_r2(true_adata[:,trivial].copy(), pred_subset, true_s_adata[:,trivial].copy(), primary_variable)
    pred_subset = {pred: adata[:, non_significant].copy() for pred, adata in pred_dict.items()}
    non_significant_dict = rigorous_pearson_r2(true_adata[:,non_significant].copy(), pred_subset, true_s_adata[:,non_significant].copy(), primary_variable)

    return non_trivial_dict, trivial_dict, non_significant_dict, [len(non_trivial), len(trivial), len(non_significant)]

def rigorous_subset_mmd(true_adata, ctrl_adata, pred_dict, true_s_adata, primary_variable, true_logfc_df):
    non_significant, trivial, non_trivial = define_trivial(true_adata, ctrl_adata, true_logfc_df)
    print(f"# non_significant: {len(non_significant)}\n# trivial: {len(trivial)}\n# non_trivial: {len(non_trivial)}")
    
    pred_subset = {pred: adata[:, non_trivial].copy() for pred, adata in pred_dict.items()}
    non_trivial_dict = rigorous_mmd(true_adata[:,non_trivial].copy(), pred_subset, true_s_adata[:,non_trivial].copy(), primary_variable)
    pred_subset = {pred: adata[:, trivial].copy() for pred, adata in pred_dict.items()}
    trivial_dict = rigorous_mmd(true_adata[:,trivial].copy(), pred_subset, true_s_adata[:,trivial].copy(), primary_variable)
    pred_subset = {pred: adata[:, non_significant].copy() for pred, adata in pred_dict.items()}
    non_significant_dict = rigorous_mmd(true_adata[:,non_significant].copy(), pred_subset, true_s_adata[:,non_significant].copy(), primary_variable)

    return non_trivial_dict, trivial_dict, non_significant_dict, [len(non_trivial), len(trivial), len(non_significant)]

def rigorous_subset_local_mmd(true_adata, ctrl_adata, pred_dict, true_s_adata, primary_variable, true_logfc_df, k=None):
    non_significant, trivial, non_trivial = define_trivial(true_adata, ctrl_adata, true_logfc_df)
    print(f"# non_significant: {len(non_significant)}\n# trivial: {len(trivial)}\n# non_trivial: {len(non_trivial)}")
    
    pred_subset = {pred: adata[:, non_trivial].copy() for pred, adata in pred_dict.items()}
    non_trivial_dict = rigorous_local_mmd(true_adata[:,non_trivial].copy(), pred_subset, true_s_adata[:,non_trivial].copy(), primary_variable, k=k)
    pred_subset = {pred: adata[:, trivial].copy() for pred, adata in pred_dict.items()}
    trivial_dict = rigorous_local_mmd(true_adata[:,trivial].copy(), pred_subset, true_s_adata[:,trivial].copy(), primary_variable, k=k)
    pred_subset = {pred: adata[:, non_significant].copy() for pred, adata in pred_dict.items()}
    non_significant_dict = rigorous_local_mmd(true_adata[:,non_significant].copy(), pred_subset, true_s_adata[:,non_significant].copy(), primary_variable, k=k)

    return non_trivial_dict, trivial_dict, non_significant_dict, [len(non_trivial), len(trivial), len(non_significant)]

def rigorous_subset_local_Edistance(true_adata, ctrl_adata, pred_dict, true_s_adata, primary_variable, true_logfc_df, k=None):
    non_significant, trivial, non_trivial = define_trivial(true_adata, ctrl_adata, true_logfc_df)
    print(f"# non_significant: {len(non_significant)}\n# trivial: {len(trivial)}\n# non_trivial: {len(non_trivial)}")
    
    pred_subset = {pred: adata[:, non_trivial].copy() for pred, adata in pred_dict.items()}
    non_trivial_dict = rigorous_local_Edistance(true_adata[:,non_trivial].copy(), pred_subset, true_s_adata[:,non_trivial].copy(), primary_variable, k=k)
    pred_subset = {pred: adata[:, trivial].copy() for pred, adata in pred_dict.items()}
    trivial_dict = rigorous_local_Edistance(true_adata[:,trivial].copy(), pred_subset, true_s_adata[:,trivial].copy(), primary_variable, k=k)
    pred_subset = {pred: adata[:, non_significant].copy() for pred, adata in pred_dict.items()}
    non_significant_dict = rigorous_local_Edistance(true_adata[:,non_significant].copy(), pred_subset, true_s_adata[:,non_significant].copy(), primary_variable, k=k)

    return non_trivial_dict, trivial_dict, non_significant_dict, [len(non_trivial), len(trivial), len(non_significant)]

def rigorous_subset_Edistance(true_adata, ctrl_adata, pred_dict, true_s_adata, primary_variable, true_logfc_df):
    non_significant, trivial, non_trivial = define_trivial(true_adata, ctrl_adata, true_logfc_df)
    print(f"# non_significant: {len(non_significant)}\n# trivial: {len(trivial)}\n# non_trivial: {len(non_trivial)}")
    
    pred_subset = {pred: adata[:, non_trivial].copy() for pred, adata in pred_dict.items()}
    non_trivial_dict = rigorous_Edistance(true_adata[:,non_trivial].copy(), pred_subset, true_s_adata[:,non_trivial].copy(), primary_variable)
    pred_subset = {pred: adata[:, trivial].copy() for pred, adata in pred_dict.items()}
    trivial_dict = rigorous_Edistance(true_adata[:,trivial].copy(), pred_subset, true_s_adata[:,trivial].copy(), primary_variable)
    pred_subset = {pred: adata[:, non_significant].copy() for pred, adata in pred_dict.items()}
    non_significant_dict = rigorous_Edistance(true_adata[:,non_significant].copy(), pred_subset, true_s_adata[:,non_significant].copy(), primary_variable)

    return non_trivial_dict, trivial_dict, non_significant_dict, [len(non_trivial), len(trivial), len(non_significant)]

def rigorous_subset_knn_loss(true_adata, ctrl_adata, pred_dict, true_s_adata, primary_variable, true_logfc_df):
    non_significant, trivial, non_trivial = define_trivial(true_adata, ctrl_adata, true_logfc_df)
    print(f"# non_significant: {len(non_significant)}\n# trivial: {len(trivial)}\n# non_trivial: {len(non_trivial)}")
    
    pred_subset = {pred: adata[:, non_trivial].copy() for pred, adata in pred_dict.items()}
    non_trivial_dict = rigorous_knn_loss(true_adata[:,non_trivial].copy(), pred_subset, true_s_adata[:,non_trivial].copy(), primary_variable)
    pred_subset = {pred: adata[:, trivial].copy() for pred, adata in pred_dict.items()}
    trivial_dict = rigorous_knn_loss(true_adata[:,trivial].copy(), pred_subset, true_s_adata[:,trivial].copy(), primary_variable)
    pred_subset = {pred: adata[:, non_significant].copy() for pred, adata in pred_dict.items()}
    non_significant_dict = rigorous_knn_loss(true_adata[:,non_significant].copy(), pred_subset, true_s_adata[:,non_significant].copy(), primary_variable)

    return non_trivial_dict, trivial_dict, non_significant_dict, [len(non_trivial), len(trivial), len(non_significant)]

def rigorous_subset_wasserstein(true_adata, ctrl_adata, pred_dict, true_s_adata, primary_variable, true_logfc_df):
    non_significant, trivial, non_trivial = define_trivial(true_adata, ctrl_adata, true_logfc_df)
    print(f"# non_significant: {len(non_significant)}\n# trivial: {len(trivial)}\n# non_trivial: {len(non_trivial)}")
    
    pred_subset = {pred: adata[:, non_trivial].copy() for pred, adata in pred_dict.items()}
    non_trivial_dict = rigorous_wasserstein(true_adata[:,non_trivial].copy(), pred_subset, true_s_adata[:,non_trivial].copy(), primary_variable)
    pred_subset = {pred: adata[:, trivial].copy() for pred, adata in pred_dict.items()}
    trivial_dict = rigorous_wasserstein(true_adata[:,trivial].copy(), pred_subset, true_s_adata[:,trivial].copy(), primary_variable)
    pred_subset = {pred: adata[:, non_significant].copy() for pred, adata in pred_dict.items()}
    non_significant_dict = rigorous_wasserstein(true_adata[:,non_significant].copy(), pred_subset, true_s_adata[:,non_significant].copy(), primary_variable)

    return non_trivial_dict, trivial_dict, non_significant_dict, [len(non_trivial), len(trivial), len(non_significant)]

def rigorous_subset_sinkhorn_mahalanobis(true_adata, ctrl_adata, pred_dict, true_s_adata, primary_variable, true_logfc_df):
    non_significant, trivial, non_trivial = define_trivial(true_adata, ctrl_adata, true_logfc_df)
    print(f"# non_significant: {len(non_significant)}\n# trivial: {len(trivial)}\n# non_trivial: {len(non_trivial)}")
    
    pred_subset = {pred: adata[:, non_trivial].copy() for pred, adata in pred_dict.items()}
    non_trivial_dict = rigorous_sinkhorn_mahalanobis(true_adata[:,non_trivial].copy(), pred_subset, true_s_adata[:,non_trivial].copy(), primary_variable)
    # pred_subset = {pred: adata[:, trivial].copy() for pred, adata in pred_dict.items()}
    # trivial_dict = rigorous_sinkhorn_mahalanobis(true_adata[:,trivial].copy(), pred_subset, true_s_adata[:,trivial].copy(), primary_variable)
    pred_subset = {pred: adata[:, non_significant].copy() for pred, adata in pred_dict.items()}
    non_significant_dict = rigorous_sinkhorn_mahalanobis(true_adata[:,non_significant].copy(), pred_subset, true_s_adata[:,non_significant].copy(), primary_variable)

    # return non_trivial_dict, trivial_dict, non_significant_dict, [len(non_trivial), len(trivial), len(non_significant)]
    return non_trivial_dict, non_significant_dict, [len(non_trivial), len(trivial), len(non_significant)]

def rigorous_subset_mixing_index_kmeans(true_adata, ctrl_adata, pred_dict, true_s_adata, primary_variable, true_logfc_df):
    non_significant, trivial, non_trivial = define_trivial(true_adata, ctrl_adata, true_logfc_df)
    print(f"# non_significant: {len(non_significant)}\n# trivial: {len(trivial)}\n# non_trivial: {len(non_trivial)}")
    
    pred_subset = {pred: adata[:, non_trivial].copy() for pred, adata in pred_dict.items()}
    non_trivial_dict = rigorous_mixing_index_kmeans(true_adata[:,non_trivial].copy(), pred_subset, true_s_adata[:,non_trivial].copy(), primary_variable)
    pred_subset = {pred: adata[:, trivial].copy() for pred, adata in pred_dict.items()}
    trivial_dict = rigorous_mixing_index_kmeans(true_adata[:,trivial].copy(), pred_subset, true_s_adata[:,trivial].copy(), primary_variable)
    pred_subset = {pred: adata[:, non_significant].copy() for pred, adata in pred_dict.items()}
    non_significant_dict = rigorous_mixing_index_kmeans(true_adata[:,non_significant].copy(), pred_subset, true_s_adata[:,non_significant].copy(), primary_variable)

    return non_trivial_dict, trivial_dict, non_significant_dict, [len(non_trivial), len(trivial), len(non_significant)]

def rigorous_subset_mixing_index_seurat(true_adata, ctrl_adata, pred_dict, true_s_adata, data_params, ood_primary, ood_modality, true_logfc_df, all_adata):
    non_significant, trivial, non_trivial = define_trivial(true_adata, ctrl_adata, true_logfc_df)
    print(f"# non_significant: {len(non_significant)}\n# trivial: {len(trivial)}\n# non_trivial: {len(non_trivial)}")
    
    if len(trivial) < 15:
        trivial_dict = {pred:None for pred in pred_dict.keys()}
        trivial_dict.update({'true':None, 'random_all':None})
        print( f"Too few trivial genes ({len(trivial)}) for PCA-based evaluation. "
        "Skipping mixing-index computation.")
    else:
        pred_subset = {pred: adata[:, trivial].copy() for pred, adata in pred_dict.items()}
        trivial_dict = rigorous_mixing_index_seurat(true_adata[:,trivial].copy(), pred_subset, true_s_adata[:,trivial].copy(), data_params, ood_primary, ood_modality, all_adata=all_adata[:,trivial].copy())
    if len(non_trivial) < 15:
        non_trivial_dict = {pred:None for pred in pred_dict.keys()}
        non_trivial_dict.update({'true':None, 'random_all':None})
        print( f"Too few non-trivial genes ({len(non_trivial)}) for PCA-based evaluation. "
        "Skipping mixing-index computation.")
    else:
        pred_subset = {pred: adata[:, non_trivial].copy() for pred, adata in pred_dict.items()}
        non_trivial_dict = rigorous_mixing_index_seurat(true_adata[:,non_trivial].copy(), pred_subset, true_s_adata[:,non_trivial].copy(), data_params, ood_primary, ood_modality, all_adata=all_adata[:,non_trivial].copy())
    pred_subset = {pred: adata[:, non_significant].copy() for pred, adata in pred_dict.items()}
    non_significant_dict = rigorous_mixing_index_seurat(true_adata[:,non_significant].copy(), pred_subset, true_s_adata[:,non_significant].copy(), data_params, ood_primary, ood_modality, all_adata=all_adata[:,non_significant].copy())

    # return non_trivial_dict, non_significant_dict, [len(non_trivial), len(non_significant)]
    return non_trivial_dict, trivial_dict, non_significant_dict, [len(non_trivial), len(trivial), len(non_significant)]

def rigorous_subset_deg_f1(true_adata, ctrl_adata, pred_dict, true_s_adata, primary_variable, true_logfc_df, ood_primary):
    non_significant, trivial, non_trivial = define_trivial(true_adata, ctrl_adata, true_logfc_df)
    down_up_dict = define_trivial_up_down(true_adata, ctrl_adata, true_logfc_df)
    print(f"# non_significant: {len(non_significant)}\n# trivial: {len(trivial)}\n# non_trivial: {len(non_trivial)}")
    
    pred_subset = {pred: adata[:, non_trivial].copy() for pred, adata in pred_dict.items()}
    non_trivial_dicts = rigorous_deg_f1(
        true_adata[:,non_trivial].copy(), ctrl_adata[:,non_trivial].copy(), pred_subset, true_s_adata[:,non_trivial].copy(), primary_variable, ood_primary)
    pred_subset = {pred: adata[:, trivial].copy() for pred, adata in pred_dict.items()}
    trivial_dicts = rigorous_deg_f1(
        true_adata[:,trivial].copy(), ctrl_adata[:,trivial].copy(), pred_subset, true_s_adata[:,trivial].copy(), primary_variable, ood_primary)
    # pred_subset = {pred: adata[:, non_significant].copy() for pred, adata in pred_dict.items()}
    # non_significant_dicts = rigorous_deg_f1(
    #     true_adata[:,non_significant].copy(), ctrl_adata[:,non_significant].copy(), pred_subset, true_s_adata[:,non_significant].copy(), primary_variable, ood_primary)

    return non_trivial_dicts, trivial_dicts, [len(non_trivial), len(trivial), len(non_significant)], down_up_dict

def semi_rigorous_mixing_index_seurat(test_adata, pred_dict, true_s_adata, primary_variable, all_adata, train_adata=None, include_best=False, resolution=2.0):

    mixing_index_dict = defaultdict(list)
    all_seurat = create_seurat_object(all_adata.copy())

    if train_adata is not None:
        n_test_balanced = int((test_adata.n_obs + train_adata.n_obs)/2)
        test_adata = adata_sample(test_adata, n_test_balanced)
        true_s_adata_without_train = true_s_adata[~true_s_adata.obs.index.isin(train_adata.obs.index)].copy()
        train_adata = adata_sample(train_adata, n_test_balanced)
        print(f'model true:')
        mixing_index_dict['true'].append(mixing_index_seurat(train_adata.copy(), test_adata.copy(), all_adata=all_adata.copy(), all_seurat=all_seurat, do_layer_process=False, resolution=resolution))
    else:
        n_test_balanced = int(test_adata.n_obs/2)
        chosen_index = np.random.choice(test_adata.n_obs, size=1)
        train_adata = test_adata[chosen_index].copy()
        train_adata = adata_sample(train_adata, n_test_balanced)
        test_adata = adata_sample(test_adata, n_test_balanced)
        # mixing_index_dict['true'].append(0)
        print(f'model true:')
        mixing_index_dict['true'].append(mixing_index_seurat(train_adata.copy(), test_adata.copy(), all_adata=all_adata.copy(), all_seurat=all_seurat, do_layer_process=False, resolution=resolution))
        true_s_adata_without_train = true_s_adata.copy()

    for model in pred_dict.keys():
        sampled_adata = adata_sample(pred_dict[model], n_test_balanced)
        print(f'model {model}:')
        if model == 'noperturb':
            mixing_index_dict[model].append(mixing_index_seurat(sampled_adata, test_adata.copy(), all_adata=all_adata.copy(), all_seurat=all_seurat, do_layer_process=False, resolution=resolution))
        else:
            mixing_index_dict[model].append(mixing_index_seurat(sampled_adata, test_adata.copy(), all_adata=all_adata.copy(), all_seurat=all_seurat, resolution=resolution))
    
    sampled_true_all = adata_sample(true_s_adata, n_test_balanced)
    print(f'model random_all + seen_true:')
    mixing_index_dict['random_all+seen_true'].append(mixing_index_seurat(sampled_true_all, test_adata.copy(), all_adata=all_adata.copy(), all_seurat=all_seurat, do_layer_process=False, resolution=resolution))

    sampled_true_all_without_train = adata_sample(true_s_adata_without_train, n_test_balanced)
    print(f'model random_all:')
    mixing_index_dict['random_all'].append(mixing_index_seurat(sampled_true_all_without_train, test_adata.copy(), all_adata=all_adata.copy(), all_seurat=all_seurat, do_layer_process=False, resolution=resolution))

    if include_best:
        true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
        mixing_best = 0
        for p in true_s_dict.keys():
            sampled_adata = adata_sample(true_s_dict[p], n_test_balanced)
            print(f'model best:')
            mixing = mixing_index_seurat(sampled_adata, test_adata.copy(), all_adata=all_adata.copy(), all_seurat=all_seurat, do_layer_process=False, resolution=resolution)
            if mixing > mixing_best:
                mixing_best = mixing
        mixing_index_dict['best'].append((mixing_best))

    return mixing_index_dict

def semi_rigorous_logfc_correlation(test_adata, pred_dict, ctrl_adata, true_s_adata, primary_variable, train_adata=None, include_best=False):

    metric_dict = defaultdict(list)

    if train_adata is not None:
        n_test_balanced = int((test_adata.n_obs + train_adata.n_obs)/2)
        true_s_adata_without_train = true_s_adata[~true_s_adata.obs.index.isin(train_adata.obs.index)].copy()
    else:
        n_test_balanced = int(test_adata.n_obs/2)
        chosen_index = np.random.choice(test_adata.n_obs, size=1)
        train_adata = test_adata[chosen_index].copy()
        true_s_adata_without_train = true_s_adata.copy()

    test_adata = adata_sample(test_adata, n_test_balanced)
    train_adata = adata_sample(train_adata, n_test_balanced)
    ctrl_adata = adata_sample(ctrl_adata, n_test_balanced)
    ctrl_true_adata = ad.concat([test_adata, ctrl_adata], label="batch", keys=["stimulated", "ctrl"])
    ctrl_true_adata.obs_names_make_unique()
    true_logfc_df = seurat_deg(
        ctrl_true_adata, "batch", "stimulated", "ctrl", test_use='MAST', find_variable_features=False)
    metric_dict['true'].append(logfc_correlation(
        train_adata.copy(), ctrl_adata.copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=False))

    for model in pred_dict.keys():
        sampled_adata = adata_sample(pred_dict[model], n_test_balanced)
        if model == 'noperturb':
            metric_dict[model].append(logfc_correlation(
                sampled_adata.copy(), ctrl_adata.copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=False))
        else:
            metric_dict[model].append(logfc_correlation(
                sampled_adata.copy(), ctrl_adata.copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=True))
    
    sampled_true_all = adata_sample(true_s_adata, n_test_balanced)
    metric_dict['random_all+seen_true'].append(logfc_correlation(
        sampled_true_all.copy(), ctrl_adata.copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=False))

    sampled_true_all_without_train = adata_sample(true_s_adata_without_train, n_test_balanced)
    metric_dict['random_all'].append(logfc_correlation(
        sampled_true_all_without_train.copy(), ctrl_adata.copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=False))

    if include_best:
        true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
        corr_best = 0
        for p in true_s_dict.keys():
            sampled_adata = adata_sample(true_s_dict[p], n_test_balanced)
            corr = logfc_correlation(
                sampled_adata.copy(), ctrl_adata.copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=False)
            if corr > corr_best:
                corr_best = corr
        metric_dict['best'].append(corr_best)

    return metric_dict


def semi_rigorous_nontrivial_logfc_correlation(test_adata, pred_dict, ctrl_adata, ctrl_true_adata, true_s_adata, data_params, ood_modality, train_adata=None, include_best=False):

    metric_dict = defaultdict(list)

    if train_adata is not None:
        n_test_balanced = int((test_adata.n_obs + train_adata.n_obs)/2)
        true_s_adata_without_train = true_s_adata[~true_s_adata.obs.index.isin(train_adata.obs.index)].copy()
    else:
        n_test_balanced = int(test_adata.n_obs/2)
        chosen_index = np.random.choice(test_adata.n_obs, size=1)
        train_adata = test_adata[chosen_index].copy()
        true_s_adata_without_train = true_s_adata.copy()

    true_logfc_df = seurat_deg(
        ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=False)
    trua_all = ctrl_true_adata[ctrl_true_adata.obs[data_params['modality_variable']]==ood_modality].copy()
    non_significant, trivial, non_trivial = define_trivial(true_all, ctrl_adata, true_logfc_df)
    print(f"# non_significant: {len(non_significant)}\n# trivial: {len(trivial)}\n# non_trivial: {len(non_trivial)}")

    test_adata = adata_sample(test_adata, n_test_balanced)
    train_adata = adata_sample(train_adata, n_test_balanced)
    ctrl_adata = adata_sample(ctrl_adata, n_test_balanced)
    # ctrl_true_adata = ad.concat([test_adata, ctrl_adata], label="batch", keys=["stimulated", "ctrl"])
    # ctrl_true_adata.obs_names_make_unique()
    # true_logfc_df = seurat_deg(
    #     ctrl_true_adata, "batch", "stimulated", "ctrl", test_use='MAST', find_variable_features=False)
    metric_dict['true'].append(logfc_correlation(
        train_adata[:,non_trivial].copy(), ctrl_adata[:,non_trivial].copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=False))

    for model in pred_dict.keys():
        sampled_adata = adata_sample(pred_dict[model], n_test_balanced)
        if model == 'noperturb':
            metric_dict[model].append(logfc_correlation(
                sampled_adata[:,non_trivial].copy(), ctrl_adata[:,non_trivial].copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=False))
        else:
            metric_dict[model].append(logfc_correlation(
                sampled_adata[:,non_trivial].copy(), ctrl_adata[:,non_trivial].copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=True))
    
    sampled_true_all = adata_sample(true_s_adata, n_test_balanced)
    metric_dict['random_all+seen_true'].append(logfc_correlation(
        sampled_true_all[:,non_trivial].copy(), ctrl_adata[:,non_trivial].copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=False))

    sampled_true_all_without_train = adata_sample(true_s_adata_without_train, n_test_balanced)
    metric_dict['random_all'].append(logfc_correlation(
        sampled_true_all_without_train[:,non_trivial].copy(), ctrl_adata[:,non_trivial].copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=False))

    if include_best:
        true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
        corr_best = 0
        for p in true_s_dict.keys():
            sampled_adata = adata_sample(true_s_dict[p], n_test_balanced)
            corr = logfc_correlation(
                sampled_adata[:,non_trivial].copy(), ctrl_adata[:,non_trivial].copy(), true_logfc_df, 'stimulated', 'ctrl', do_layer_process=False)
            if corr > corr_best:
                corr_best = corr
        metric_dict['best'].append(corr_best)

    return metric_dict

def semi_rigorous_nontrivial_r2(test_adata, pred_dict, true_s_adata, ctrl_true_adata, data_params, ood_modality, train_adata=None, include_best=False):

    metric_dict = defaultdict(list)

    if train_adata is not None:
        n_test_balanced = int((test_adata.n_obs + train_adata.n_obs)/2)
        true_s_adata_without_train = true_s_adata[~true_s_adata.obs.index.isin(train_adata.obs.index)].copy()
    else:
        n_test_balanced = int(test_adata.n_obs/2)
        chosen_index = np.random.choice(test_adata.n_obs, size=1)
        train_adata = test_adata[chosen_index].copy()
        true_s_adata_without_train = true_s_adata.copy()

    true_logfc_df = seurat_deg(
        ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=False)
    trua_all = ctrl_true_adata[ctrl_true_adata.obs[data_params['modality_variable']]==ood_modality].copy()
    ctrl_adata = ctrl_true_adata[ctrl_true_adata.obs[data_params['modality_variable']]!=ood_modality].copy()
    non_significant, trivial, non_trivial = define_trivial(true_adata, ctrl_adata, true_logfc_df)
    print(f"# non_significant: {len(non_significant)}\n# trivial: {len(trivial)}\n# non_trivial: {len(non_trivial)}")

    test_adata = adata_sample(test_adata, n_test_balanced)
    train_adata = adata_sample(train_adata, n_test_balanced)
    ctrl_adata = adata_sample(ctrl_adata, n_test_balanced)
    # ctrl_true_adata = ad.concat([test_adata, ctrl_adata], label="batch", keys=["stimulated", "ctrl"])
    # ctrl_true_adata.obs_names_make_unique()
    # true_logfc_df = seurat_deg(
    #     ctrl_true_adata, "batch", "stimulated", "ctrl", test_use='MAST', find_variable_features=False)
    metric_dict['true'].append(r2(train_adata[:,non_trivial].copy(), test_adata[:,non_trivial].copy()))

    for model in pred_dict.keys():
        sampled_adata = adata_sample(pred_dict[model], n_test_balanced)
        metric_dict[model].append(r2(sampled_adata[:,non_trivial].copy(), test_adata[:,non_trivial].copy()))
    
    sampled_true_all = adata_sample(true_s_adata, n_test_balanced)
    metric_dict['random_all+seen_true'].append(r2(sampled_true_all[:,non_trivial].copy(), test_adata[:,non_trivial].copy()))

    sampled_true_all_without_train = adata_sample(true_s_adata_without_train, n_test_balanced)
    metric_dict['random_all'].append(r2(sampled_true_all_without_train[:,non_trivial].copy(), test_adata[:,non_trivial].copy()))

    if include_best:
        true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
        corr_best = 0
        for p in true_s_dict.keys():
            sampled_adata = adata_sample(true_s_dict[p], n_test_balanced)
            corr = r2(sampled_adata[:,non_trivial].copy(), test_adata[:,non_trivial].copy())
            if corr > corr_best:
                corr_best = corr
        metric_dict['best'].append(corr_best)

    return metric_dict

def semi_rigorous_pearson_r2(test_adata, pred_dict, true_s_adata, primary_variable, train_adata=None, include_best=False):

    correlation_dict = defaultdict(list)
    r2_dict = defaultdict(list)

    if train_adata is not None:
        n_test_balanced = int((test_adata.n_obs + train_adata.n_obs)/2)
        test_adata = adata_sample(test_adata, n_test_balanced)
        true_s_adata_without_train = true_s_adata[~true_s_adata.obs.index.isin(train_adata.obs.index)].copy()
        train_adata = adata_sample(train_adata, n_test_balanced)
        correlation_dict['true'].append(pearson_corr(train_adata.copy(), test_adata.copy()))
        r2_dict['true'].append(r2(train_adata.copy(), test_adata.copy()))
    else:
        n_test_balanced = int(test_adata.n_obs/2)
        chosen_index = np.random.choice(test_adata.n_obs, size=1)
        train_adata = test_adata[chosen_index].copy()
        train_adata = adata_sample(train_adata, n_test_balanced)
        # correlation_dict['true'].append(0)
        # r2_dict['true'].append(0)
        test_adata = adata_sample(test_adata, n_test_balanced)
        correlation_dict['true'].append(pearson_corr(train_adata.copy(), test_adata.copy()))
        r2_dict['true'].append(r2(train_adata.copy(), test_adata.copy()))
        true_s_adata_without_train = true_s_adata.copy()

    for model in pred_dict.keys():
        sampled_adata = adata_sample(pred_dict[model], n_test_balanced)
        correlation_dict[model].append(pearson_corr(sampled_adata, test_adata.copy()))
        r2_dict[model].append(r2(sampled_adata, test_adata.copy()))
    
    sampled_true_all = adata_sample(true_s_adata, n_test_balanced)
    correlation_dict['random_all+seen_true'].append(pearson_corr(sampled_true_all, test_adata.copy()))
    r2_dict['random_all+seen_true'].append(r2(sampled_true_all, test_adata.copy()))

    sampled_true_all_without_train = adata_sample(true_s_adata_without_train, n_test_balanced)
    correlation_dict['random_all'].append(pearson_corr(sampled_true_all_without_train, test_adata.copy()))
    r2_dict['random_all'].append(r2(sampled_true_all_without_train, test_adata.copy()))

    if include_best:
        true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
        correlation_best = 0
        r2_best = 0
        for p in true_s_dict.keys():
            sampled_adata = adata_sample(true_s_dict[p], n_test_balanced)
            correlation = pearson_corr(sampled_adata, test_adata.copy())
            r2_ = r2(sampled_adata, test_adata.copy())
            if correlation > correlation_best:
                correlation_best = correlation
            if r2_ > r2_best:
                r2_best = r2_
        correlation_dict['best'].append((correlation_best))
        r2_dict['best'].append((r2_best))

    return correlation_dict, r2_dict


def semi_rigorous_stepwise_r2(test_adata, pred_dict, true_s_adata, ctrl_adata, primary_variable, train_adata=None):

    corr_dict = defaultdict(list)
    r2_dict = defaultdict(list)

    if train_adata is not None:
        n_test_balanced = int((test_adata.n_obs + train_adata.n_obs)/2)
        test_adata = adata_sample(test_adata, n_test_balanced)
        true_s_adata_without_train = true_s_adata[~true_s_adata.obs.index.isin(train_adata.obs.index)].copy()
        train_adata = adata_sample(train_adata, n_test_balanced)
        output = stepwise_r2(ctrl_adata, test_adata.copy(), train_adata.copy())
        corr_dict['true'].append(output[0])
        r2_dict['true'].append(output[1])
    else:
        n_test_balanced = int(test_adata.n_obs/2)
        chosen_index = np.random.choice(test_adata.n_obs, size=1)
        train_adata = test_adata[chosen_index].copy()
        train_adata = adata_sample(train_adata, n_test_balanced)
        # metric_dict['true'].append(0)
        test_adata = adata_sample(test_adata, n_test_balanced)
        output = stepwise_r2(ctrl_adata, test_adata.copy(), train_adata.copy())
        corr_dict['true'].append(output[0])
        r2_dict['true'].append(output[1])
        true_s_adata_without_train = true_s_adata.copy()

    for model in pred_dict.keys():
        sampled_adata = adata_sample(pred_dict[model], n_test_balanced)
        output = stepwise_r2(ctrl_adata, test_adata.copy(), sampled_adata)
        corr_dict[model].append(output[0])
        r2_dict[model].append(output[1])
    
    sampled_true_all = adata_sample(true_s_adata, n_test_balanced)
    output = stepwise_r2(ctrl_adata, test_adata.copy(), sampled_true_all)
    corr_dict['random_all+seen_true'].append(output[0])
    r2_dict['random_all+seen_true'].append(output[1])

    sampled_true_all_without_train = adata_sample(true_s_adata_without_train, n_test_balanced)
    output = stepwise_r2(ctrl_adata, test_adata.copy(), sampled_true_all_without_train)
    corr_dict['random_all'].append(output[0])
    r2_dict['random_all'].append(output[1])

    return corr_dict, r2_dict

def semi_rigorous_mmd(test_adata, pred_dict, true_s_adata, primary_variable, train_adata=None, include_best=False):

    metric_dict = defaultdict(list)

    if train_adata is not None:
        n_test_balanced = int((test_adata.n_obs + train_adata.n_obs)/2)
        test_adata = adata_sample(test_adata, n_test_balanced)
        true_s_adata_without_train = true_s_adata[~true_s_adata.obs.index.isin(train_adata.obs.index)].copy()
        train_adata = adata_sample(train_adata, n_test_balanced)
        metric_dict['true'].append(compute_mmd(train_adata, test_adata))
    else:
        # metric_dict['true'].append(0)
        n_test_balanced = int(test_adata.n_obs/2)
        chosen_index = np.random.choice(test_adata.n_obs, size=1)
        train_adata = test_adata[chosen_index].copy()
        train_adata = adata_sample(train_adata, n_test_balanced)
        test_adata = adata_sample(test_adata, n_test_balanced)
        metric_dict['true'].append(compute_mmd(train_adata, test_adata))
        true_s_adata_without_train = true_s_adata.copy()

    for model in pred_dict.keys():
        sampled_adata = adata_sample(pred_dict[model], n_test_balanced)
        metric_dict[model].append(compute_mmd(sampled_adata, test_adata))
    
    sampled_true_all = adata_sample(true_s_adata, n_test_balanced)
    metric_dict['random_all+seen_true'].append(compute_mmd(sampled_true_all, test_adata))

    sampled_true_all_without_train = adata_sample(true_s_adata_without_train, n_test_balanced)
    metric_dict['random_all'].append(compute_mmd(sampled_true_all_without_train, test_adata.copy()))

    if include_best:
        true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
        metric_best = 1000000
        for p in true_s_dict.keys():
            sampled_adata = adata_sample(true_s_dict[p], n_test_balanced)
            metric = compute_mmd(sampled_adata, test_adata)
            if metric < metric_best:
                metric_best = metric
        metric_dict['best'].append(metric_best)

    # for key in metric_dict:
    #     metric_dict[key] = list(-np.log10(np.array(metric_dict[key])))
        
    return metric_dict

def semi_rigorous_Edistance(test_adata, pred_dict, true_s_adata, primary_variable, train_adata=None, include_best=False):

    metric_dict = defaultdict(list)

    if train_adata is not None:
        n_test_balanced = int((test_adata.n_obs + train_adata.n_obs)/2)
        test_adata = adata_sample(test_adata, n_test_balanced)
        true_s_adata_without_train = true_s_adata[~true_s_adata.obs.index.isin(train_adata.obs.index)].copy()
        train_adata = adata_sample(train_adata, n_test_balanced)
        metric_dict['true'].append(compute_Edistance(train_adata, test_adata))
    else:
        # metric_dict['true'].append(0)
        n_test_balanced = int(test_adata.n_obs/2)
        chosen_index = np.random.choice(test_adata.n_obs, size=1)
        train_adata = test_adata[chosen_index].copy()
        train_adata = adata_sample(train_adata, n_test_balanced)
        test_adata = adata_sample(test_adata, n_test_balanced)
        metric_dict['true'].append(compute_Edistance(train_adata, test_adata))
        true_s_adata_without_train = true_s_adata.copy()

    for model in pred_dict.keys():
        sampled_adata = adata_sample(pred_dict[model], n_test_balanced)
        metric_dict[model].append(compute_Edistance(sampled_adata, test_adata))
    
    sampled_true_all = adata_sample(true_s_adata, n_test_balanced)
    metric_dict['random_all+seen_true'].append(compute_Edistance(sampled_true_all, test_adata))

    sampled_true_all_without_train = adata_sample(true_s_adata_without_train, n_test_balanced)
    metric_dict['random_all'].append(compute_Edistance(sampled_true_all_without_train, test_adata.copy()))

    if include_best:
        true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
        metric_best = 1000000
        for p in true_s_dict.keys():
            sampled_adata = adata_sample(true_s_dict[p], n_test_balanced)
            metric = compute_Edistance(sampled_adata, test_adata)
            if metric < metric_best:
                metric_best = metric
        metric_dict['best'].append(metric_best)

    # for key in metric_dict:
    #     metric_dict[key] = list(-np.log10(np.array(metric_dict[key])))
        
    return metric_dict

def semi_rigorous_local_mmd(test_adata, pred_dict, true_s_adata, primary_variable, k, train_adata=None, include_best=False):

    metric_dict = defaultdict(list)

    if train_adata is not None:
        n_test_balanced = int((test_adata.n_obs + train_adata.n_obs)/2)
        test_adata = adata_sample(test_adata, n_test_balanced)
        true_s_adata_without_train = true_s_adata[~true_s_adata.obs.index.isin(train_adata.obs.index)].copy()
        train_adata = adata_sample(train_adata, n_test_balanced)
        metric_dict['true'].append(compute_local_mmd(train_adata, test_adata, k=k))
    else:
        # metric_dict['true'].append(0)
        n_test_balanced = int(test_adata.n_obs/2)
        chosen_index = np.random.choice(test_adata.n_obs, size=1)
        train_adata = test_adata[chosen_index].copy()
        train_adata = adata_sample(train_adata, n_test_balanced)
        test_adata = adata_sample(test_adata, n_test_balanced)
        metric_dict['true'].append(compute_local_mmd(train_adata, test_adata, k=k))
        true_s_adata_without_train = true_s_adata.copy()

    for model in pred_dict.keys():
        sampled_adata = adata_sample(pred_dict[model], n_test_balanced)
        metric_dict[model].append(compute_local_mmd(sampled_adata, test_adata, k=k))
    
    sampled_true_all = adata_sample(true_s_adata, n_test_balanced)
    metric_dict['random_all+seen_true'].append(compute_local_mmd(sampled_true_all, test_adata, k=k))

    sampled_true_all_without_train = adata_sample(true_s_adata_without_train, n_test_balanced)
    metric_dict['random_all'].append(compute_local_mmd(sampled_true_all_without_train, test_adata, k=k))

    if include_best:
        true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
        metric_best = 1000000
        for p in true_s_dict.keys():
            sampled_adata = adata_sample(true_s_dict[p], n_test_balanced)
            metric = compute_local_mmd(sampled_adata, test_adata, k=k)
            if metric < metric_best:
                metric_best = metric
        metric_dict['best'].append(metric_best)
        
    return metric_dict

def semi_rigorous_local_Edistance(test_adata, pred_dict, true_s_adata, primary_variable, k, train_adata=None, include_best=False):

    metric_dict = defaultdict(list)

    if train_adata is not None:
        n_test_balanced = int((test_adata.n_obs + train_adata.n_obs)/2)
        test_adata = adata_sample(test_adata, n_test_balanced)
        true_s_adata_without_train = true_s_adata[~true_s_adata.obs.index.isin(train_adata.obs.index)].copy()
        train_adata = adata_sample(train_adata, n_test_balanced)
        metric_dict['true'].append(compute_local_Edistance(train_adata, test_adata, k=k))
    else:
        # metric_dict['true'].append(0)
        n_test_balanced = int(test_adata.n_obs/2)
        chosen_index = np.random.choice(test_adata.n_obs, size=1)
        train_adata = test_adata[chosen_index].copy()
        train_adata = adata_sample(train_adata, n_test_balanced)
        test_adata = adata_sample(test_adata, n_test_balanced)
        metric_dict['true'].append(compute_local_Edistance(train_adata, test_adata, k=k))
        true_s_adata_without_train = true_s_adata.copy()

    for model in pred_dict.keys():
        sampled_adata = adata_sample(pred_dict[model], n_test_balanced)
        metric_dict[model].append(compute_local_Edistance(sampled_adata, test_adata, k=k))
    
    sampled_true_all = adata_sample(true_s_adata, n_test_balanced)
    metric_dict['random_all+seen_true'].append(compute_local_Edistance(sampled_true_all, test_adata, k=k))

    sampled_true_all_without_train = adata_sample(true_s_adata_without_train, n_test_balanced)
    metric_dict['random_all'].append(compute_local_Edistance(sampled_true_all_without_train, test_adata, k=k))

    if include_best:
        true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
        metric_best = 1000000
        for p in true_s_dict.keys():
            sampled_adata = adata_sample(true_s_dict[p], n_test_balanced)
            metric = compute_local_Edistance(sampled_adata, test_adata, k=k)
            if metric < metric_best:
                metric_best = metric
        metric_dict['best'].append(metric_best)
        
    return metric_dict

def semi_rigorous_knn_loss(test_adata, pred_dict, true_s_adata, primary_variable, train_adata=None, include_best=False):

    metric_dict = defaultdict(list)

    if train_adata is not None:
        n_test_balanced = int((test_adata.n_obs + train_adata.n_obs)/2)
        test_adata = adata_sample(test_adata, n_test_balanced)
        true_s_adata_without_train = true_s_adata[~true_s_adata.obs.index.isin(train_adata.obs.index)].copy()
        train_adata = adata_sample(train_adata, n_test_balanced)
        metric_dict['true'].append(compute_knn_loss(train_adata, test_adata))
    else:
        # metric_dict['true'].append(0)
        n_test_balanced = int(test_adata.n_obs/2)
        chosen_index = np.random.choice(test_adata.n_obs, size=1)
        train_adata = test_adata[chosen_index].copy()
        train_adata = adata_sample(train_adata, n_test_balanced)
        test_adata = adata_sample(test_adata, n_test_balanced)
        metric_dict['true'].append(compute_knn_loss(train_adata, test_adata))
        true_s_adata_without_train = true_s_adata.copy()

    for model in pred_dict.keys():
        sampled_adata = adata_sample(pred_dict[model], n_test_balanced)
        metric_dict[model].append(compute_knn_loss(sampled_adata, test_adata))
    
    sampled_true_all = adata_sample(true_s_adata, n_test_balanced)
    metric_dict['random_all+seen_true'].append(compute_knn_loss(sampled_true_all, test_adata))

    sampled_true_all_without_train = adata_sample(true_s_adata_without_train, n_test_balanced)
    metric_dict['random_all'].append(compute_knn_loss(sampled_true_all_without_train, test_adata))

    if include_best:
        true_s_dict = split_adata_by_primary(true_s_adata, primary_variable)
        metric_best = 1000000
        for p in true_s_dict.keys():
            sampled_adata = adata_sample(true_s_dict[p], n_test_balanced)
            metric = compute_knn_loss(sampled_adata, test_adata)
            if metric < metric_best:
                metric_best = metric
        metric_dict['best'].append(metric_best)
        
    return metric_dict

def split_genes_to_groups(num_genes, num_groups):
    genes = np.arange(num_genes)
    np.random.shuffle(genes)
    split_sets = np.array_split(genes, num_groups)
    return split_sets

def add_noise(train_adata, source_adata, gene_set=None, method='sampling', p=None):
    if issparse(train_adata.X):
        train_adata.X = train_adata.X.toarray()
    if issparse(train_adata.layers['counts']):
        train_adata.layers['counts'] = train_adata.layers['counts'].toarray()
    num_cells, num_genes = train_adata.X.shape
    if method == 'random_all':
        total_values = num_cells * num_genes
        num_replacements = int((p / 100) * total_values)
        rand_indices = np.random.choice(total_values, num_replacements, replace=False)
        row_indices, col_indices = np.divmod(rand_indices, num_genes)
        source_row_indices = np.random.randint(source_adata.X.shape[0], size=num_replacements)
        train_adata.X[row_indices, col_indices] = source_adata.X[source_row_indices, col_indices].copy()
        train_adata.layers['counts'][row_indices, col_indices] = source_adata.layers['counts'][source_row_indices, col_indices].copy()
        ## added this line 
        calculate_cpm(train_adata, do_sparse=False)
        return train_adata
    elif method == 'shuffle' and gene_set is None:
        num_shuffled_genes = int((p / 100) * num_genes)
        rand_indices = np.random.choice(num_genes, num_shuffled_genes, replace=False)
        for gene in rand_indices:
            values = train_adata.X[:, gene].copy()
            counts = train_adata.layers['counts'][:, gene].copy()
            shuffle_idx = np.random.permutation(num_cells)
            train_adata.X[:, gene] = values[shuffle_idx]
            train_adata.layers['counts'][:, gene] = counts[shuffle_idx]
        ## added this line
        calculate_cpm(train_adata, do_sparse=False)    
        return train_adata
    all_gene_names = train_adata.var_names
    if gene_set is None:
        raise ValueError('ERROR: gene set is not given!')
    gene_set_names = all_gene_names[gene_set]
    for gene in gene_set_names:
        gene_idx_train = train_adata.var_names.get_loc(gene)
        gene_idx_source = source_adata.var_names.get_loc(gene)
        if method == 'shuffle':
            values = train_adata.X[:, gene_idx_train].copy()
            counts = train_adata.layers['counts'][:, gene_idx_train].copy()
            shuffle_idx = np.random.permutation(num_cells)
            train_adata.X[:, gene_idx_train] = values[shuffle_idx]
            train_adata.layers['counts'][:, gene_idx_train] = counts[shuffle_idx]
        elif method == 'sampling' and gene in source_adata.var_names:
            sampled_cell_indices = np.random.choice(source_adata.n_obs, num_cells, replace=True)
            sampled_values_X = source_adata.X[sampled_cell_indices, gene_idx_source].flatten()
            sampled_values_counts = source_adata.layers['counts'][sampled_cell_indices, gene_idx_source].flatten()
            train_adata.X[:, gene_idx_train] = sampled_values_X
            train_adata.layers['counts'][:, gene_idx_train] = sampled_values_counts
        else:
            raise ValueError('ERROR: the method for noise adding is not valid!')
    ## added this line
    calculate_cpm(train_adata, do_sparse=False)
    return train_adata

def convert_list2DF(col1_values, col2_values, col1='n', col2='true_positives'):
    return pd.DataFrame({col1: col1_values, col2: col2_values})

def process_for_f1(
    model_name, real_degs, pred_adata, ctrl_adata, f1_down_dict, f1_up_dict, precision_down_dict, precision_up_dict, 
    recall_down_dict, recall_up_dict, typeI_down_dict=None, typeI_up_dict=None, append=True, log_file=None):
    combined_pred = ad.concat([pred_adata, ctrl_adata], label="batch", keys=["stimulated", "ctrl"])
    combined_pred.obs_names_make_unique()
    pred_degs = seurat_get_down_up_genes(combined_pred, "batch", "stimulated", "ctrl", "MAST")

    if log_file is not None:
        log_file.write(f"{model_name} down degs: {len(pred_degs[0])}\n")
        log_file.write(f"{model_name} up degs: {len(pred_degs[1])}\n")
    
    f1_down, precision_down, recall_down = calculate_f1(real_degs[0], pred_degs[0])
    f1_up, precision_up, recall_up = calculate_f1(real_degs[1], pred_degs[1])
    typeI_down = calculate_typeI(real_degs[0], pred_degs[0], pred_adata.var_names)
    typeI_up = calculate_typeI(real_degs[1], pred_degs[1], pred_adata.var_names)
    if append:
        f1_down_dict[model_name].append(f1_down)
        f1_up_dict[model_name].append(f1_up)
        precision_down_dict[model_name].append(precision_down)
        precision_up_dict[model_name].append(precision_up)
        recall_down_dict[model_name].append(recall_down)
        recall_up_dict[model_name].append(recall_up)
        if typeI_down_dict is not None:
            typeI_down_dict[model_name].append(typeI_down)
            typeI_up_dict[model_name].append(typeI_up)
    if typeI_down_dict is None:
        return f1_down, precision_down, recall_down, f1_up, precision_up, recall_up
    else:
        return f1_down, precision_down, recall_down, typeI_down, f1_up, precision_up, recall_up, typeI_up



def adata_sample(adata, size):
    replace = False
    if adata.n_obs < size:
        replace = True
    chosen_indices = np.random.choice(adata.n_obs, size=size, replace=replace)
    chosen_adata = adata[chosen_indices].copy()

    return chosen_adata

def split_adata_by_primary(adata, primary_variable):
    primaries = adata.obs[primary_variable].unique()
    adata_dict = {}
    for p in primaries:
        adata_dict[p] = adata[adata.obs[primary_variable] == p].copy()

    return adata_dict

def calculate_f1(real_degs, pred_degs):
    if not isinstance(real_degs, tuple):
        real_degs = [real_degs]
        pred_degs = [pred_degs]

    tp_total = 0
    fp_total = 0
    fn_total = 0
    for real_deg, pred_deg in zip(real_degs, pred_degs):
        tp = len(set(real_deg).intersection(set(pred_deg)))
        fp = len(set(pred_deg) - set(real_deg))
        fn = len(set(real_deg) - set(pred_deg))
        tp_total += tp
        fp_total += fp
        fn_total += fn
    if (tp_total + 0.5 * (fp_total + fn_total)) > 0:
        f1 = tp_total/(tp_total + 0.5 * (fp_total + fn_total))
    else:
        f1 = 0

    if (tp_total + fp_total) > 0:
        precision = tp_total / (tp_total + fp_total)
    else:
        precision = 0

    if (tp_total + fn_total) > 0:
        recall = tp_total / (tp_total + fn_total)
    else:
        recall = 0
        
    return f1, precision, recall


def calculate_typeI(real_degs, pred_degs, gene_names):
    fp = len(set(pred_degs) - set(real_degs))
    all_negatives = len(set(gene_names) - set(real_degs))
    if all_negatives > 0:
        return fp/all_negatives
    else:
        print('no negatives were found')
        return 0


def define_trivial(true_adata, ctrl_adata, true_logfc_df):
    all_genes = true_adata.var_names
    n_true = true_adata.X.shape[0]
    n_ctrl = ctrl_adata.X.shape[0]
    print(f'n_true={n_true}, n_ctrl={n_ctrl}')

    ctrl_nonzero_ratio = np.sum(ctrl_adata.X!=0)/(ctrl_adata.X.shape[0]*ctrl_adata.X.shape[1])
    stim_nonzero_ratio = np.sum(true_adata.X!=0)/(true_adata.X.shape[0]*true_adata.X.shape[1])

    non_significant = [gene for gene in all_genes if (gene not in true_logfc_df.index or true_logfc_df.loc[gene,'p_val'] > 0.05)]
    # trivial = [gene for gene in true_logfc_df.index if (gene not in non_significant and (np.all(true_adata[:,gene].X == 0) or np.all(ctrl_adata[:,gene].X == 0)))]
    # trivial = [gene for gene in true_logfc_df.index if (gene not in non_significant and (np.sum(true_adata[:,gene].X != 0)<=1 or np.sum(ctrl_adata[:,gene].X != 0)<=1))]
    trivial = [gene for gene in all_genes if (gene not in non_significant and (
        np.all(true_adata[:,gene].X == 0) or np.all(ctrl_adata[:,gene].X == 0) or
        (np.sum(true_adata[:,gene].X != 0)/np.sum(ctrl_adata[:,gene].X != 0)>(10*n_true/n_ctrl) and np.sum(ctrl_adata[:,gene].X != 0)/n_ctrl < ctrl_nonzero_ratio) or 
        (np.sum(ctrl_adata[:,gene].X != 0)/np.sum(true_adata[:,gene].X != 0)>(10*n_ctrl/n_true) and np.sum(true_adata[:,gene].X != 0)/n_true < stim_nonzero_ratio)))]
    non_trivial = [gene for gene in all_genes if (gene not in trivial and gene not in non_significant)]

    return non_significant, trivial, non_trivial

def define_trivial_up_down(true_adata, ctrl_adata, true_logfc_df):
    non_significant, trivial, non_trivial = define_trivial(true_adata, ctrl_adata, true_logfc_df)

    trivial_down = [gene for gene in trivial if true_logfc_df.loc[gene,'avg_log2FC']<0]
    trivial_up = [gene for gene in trivial if true_logfc_df.loc[gene,'avg_log2FC']>0]
    non_trivial_down = [gene for gene in non_trivial if true_logfc_df.loc[gene, 'avg_log2FC']<0]
    non_trivial_up = [gene for gene in non_trivial if true_logfc_df.loc[gene, 'avg_log2FC']>0]

    return {'non_significant':len(non_significant), 
        'trivial':{'down':trivial_down, 'up':trivial_up}, 'non_trivial':{'down':non_trivial_down, 'up':non_trivial_up}}



def compute_delta_over_std(data_dict):
    deltas = []
    n0_values = []
    for df in data_dict.values():
        if not isinstance(df, pd.DataFrame):
            raise ValueError("All values in data_dict must be pandas DataFrames")
        val_n0 = df.loc[df['n'] == 0, 'true_positives'].values[0]
        val_n100 = df.loc[df['n'] == 100, 'true_positives'].values[0]
        deltas.append(val_n100 - val_n0)
        n0_values.append(val_n0)
    avg_delta = np.mean(deltas)
    std_n0 = np.std(n0_values, ddof=1)  
    if std_n0 == 0:
        return np.nan
    return avg_delta / (4*std_n0)
    # return avg_delta - std_n0**2
    # return avg_delta - std_n0
    # return avg_delta

def sensitivity2shuffling_rank_sum(data_dict):
    n0_values = []
    n100_values = []
    num_n100_more_than_n0 = 0
    for df in data_dict.values():
        if not isinstance(df, pd.DataFrame):
            raise ValueError("All values in data_dict must be pandas DataFrames")
        val_n0 = df.loc[df['n'] == 0, 'true_positives'].values[0]
        val_n100 = df.loc[df['n'] == 100, 'true_positives'].values[0]
        n0_values.append(val_n0)
        n100_values.append(val_n100)
    n0_values = np.array(n0_values)
    n100_values = np.array(n100_values)
    for n100 in n100_values:
        num_n100_more_than_n0 += np.sum(n100 > n0_values)
    return num_n100_more_than_n0 / (len(n0_values)*len(n100_values))

def scipy_ranksums(data_dict):
    n0_values = []
    n100_values = []
    for df in data_dict.values():
        if not isinstance(df, pd.DataFrame):
            raise ValueError("All values in data_dict must be pandas DataFrames")
        val_n0 = df.loc[df['n'] == 0, 'true_positives'].values[0]
        val_n100 = df.loc[df['n'] == 100, 'true_positives'].values[0]
        n0_values.append(val_n0)
        n100_values.append(val_n100)
    ranksum = ranksums(n100_values, n0_values, alternative='greater')
    return ranksum


def loess(x, y, frac=0.2):
    smoothed = sm.nonparametric.lowess(y, x, frac=frac)  
    x_smooth, y_smooth = smoothed[:,0], smoothed[:,1]
    return x_smooth, y_smooth
