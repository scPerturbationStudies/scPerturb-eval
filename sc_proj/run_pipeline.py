import numpy as np
import os
import scanpy as sc
import configparser
import shutil
import pandas as pd
import pickle
import anndata as ad
from collections import defaultdict
from scipy.sparse import issparse, csr_matrix
from models.AE import train_AE, predict_AE
from models.VAE_KNN import train_VAE_KNN, predict_VAE_KNN
from models.VAE_KNN_DEG import train_VAE_KNN_DEG, predict_VAE_KNN_DEG
from models.VAE_KNN_logFC import train_VAE_KNN_logFC, predict_VAE_KNN_logFC
from models.VAE import train_VAE, predict_VAE, generate_from_VAE
from models.cpa import train_CPA, predict_CPA
from models.scPRAM import train_scPRAM, predict_scPRAM
from models.clip_ae import train_CLIP_AE, predict_CLIP_AE
from models.clip_ae_res import train_CLIP_AE_RES, predict_CLIP_AE_RES
from models.noperturb import predict_noperturb
from models.random_all import predict_random_all
from models.baseline_logfc import predict_baseline_logfc
from models.baseline_logfc_modified import predict_baseline_logfc_modified
from models.baseline_logfc_all import predict_baseline_logfc_all
# from models.scGPT import train_scGPT
from utils.build_params import ae, cae, cvae_knn, cvae_knn_deg, cvae_deg, cvae_knn_logfc, clip_cae, cpa, scPRAM, scGPT,\
    noperturb, random_all, baseline_logfc, baseline_logfc_all, baseline_self_logfc, baseline_logfc_modified,\
    build_data_params, vae
from utils.data_handler import prepare_train_data
from utils.common import calculate_cpm, get_impute_data
from utils.assessment import true_deg_pos, spearman_corr, spearman_corr_per_gene, prediction_error, wasserstein, corr_top_k,\
    process_subtract_control, sorted_pvalue_common_genes, sorted_pvalue_rank, combined_deg, combined_deg_basemodel,\
    corr_top_k_per_primary, subtract_control, k_nearest_neighbors, mixing_index, seurat_deg, mixing_index_seurat,\
    combined_deg_seurat, z_correlation
import utils.assessment as assess
import utils.plots as plots
from utils.plots import correlation_barplot, true_pos_lineplot, draw_boxplot, top_k_plot, pca_plot, lineplot_with_errorbar


def train(models, datasets, model_config_path, data_config_path, save_folder, setting, pid_percentage):
        config_model = configparser.ConfigParser()            
        config_data = configparser.ConfigParser()
        
        for data in datasets:
            config_data_path = os.path.join(data_config_path, f'{data}.ini')
            config_data.read(config_data_path)
            data_params = build_data_params(config_data)

            ######################## create and save imputated adata ################################################
            # adata = get_impute_data(
            #     data, data_params['primary_variable'], data_params['modality_variable'], data_params["data"], 'scRecover')
            
            # if adata is None:
            #     raise ValueError('ERROR: the imputated data cannot be found or obtained!')

            #########################################################################################################

            adata = sc.read(data_params["data"])
                       
            for model in models:
                
                config_model_path = os.path.join(model_config_path, f'{model}.ini')
                config_model.read(config_model_path)
                
                build_config_func = globals()[model]
                hyperparams = build_config_func(config_model)
                
                if 'train_func' in hyperparams:
                    train_func = globals().get(hyperparams['train_func'])
                else:
                    raise ValueError("ERROR: train_func must be specified in the model config!")

                path_to_save = os.path.join(data_params['base_path'], 'train_predict', data, model, save_folder)                
                os.makedirs(path_to_save, exist_ok=True)

                # Copy the config files to the save directory
                shutil.copy(config_data_path, path_to_save)
                shutil.copy(config_model_path, path_to_save)
               
                for ood_primary in data_params["ood_primaries"]:
                    for ood_modality in data_params["ood_modality"]:
                        print(f"training {ood_primary}-{ood_modality}")
                    
                        cond_model_path = os.path.join(path_to_save, 'models', f"{ood_primary}-{ood_modality}")
                        os.makedirs(cond_model_path, exist_ok=True)
                        
                        train_adata = prepare_train_data(adata, data_params['primary_variable'], data_params['modality_variable'], data_params['control_key'],  
                                                                            ood_primary, ood_modality, cond_model_path, setting, pid_percentage)
                        
                        # TRAINING
                        train_func(train_adata, hyperparams, data_params, setting, ood_primary, path_to_save=cond_model_path)
                
            
def predict(models, datasets, model_config_path, data_config_path, save_folder, setting, pid_percentage):
    
    config_model = configparser.ConfigParser()            
    config_data = configparser.ConfigParser()
    
    for data in datasets:
        config_data_path = os.path.join(data_config_path, f'{data}.ini')
        config_data.read(config_data_path)
        data_params = build_data_params(config_data)

        ######################## create and save imputated adata ################################################
        # adata = get_impute_data(
        #     data, data_params['primary_variable'], data_params['modality_variable'], data_params["data"], 'scRecover')
        
        # if adata is None:
        #     raise ValueError('ERROR: the imputated data cannot be found or obtained!')

        #########################################################################################################

        adata = sc.read(data_params["data"])       
        for model in models:
            
            config_model_path = os.path.join(model_config_path, f'{model}.ini')
            config_model.read(config_model_path)
            build_config_func = globals()[model]
            hyperparams = build_config_func(config_model)
                            
            predict_func = globals().get(hyperparams['predict_func'])

            path_to_save = os.path.join(data_params['base_path'], 'train_predict', data, model, save_folder)                
            os.makedirs(path_to_save, exist_ok=True)
        
            for ood_primary in data_params["ood_primaries"]:
                for ood_modality in data_params["ood_modality"]:
                    print(f"predicting {ood_primary}-{ood_modality}")
                     # LOAD MODEL
                    cond_model_path = os.path.join(path_to_save, 'models', f"{ood_primary}-{ood_modality}")
                    os.makedirs(cond_model_path, exist_ok=True)
                    
                    train_adata = prepare_train_data(adata, data_params['primary_variable'], data_params['modality_variable'], data_params['control_key'], 
                                                                        ood_primary, ood_modality, cond_model_path, setting, pid_percentage)
                    
                    predict_func(train_adata, hyperparams, data_params, setting, ood_primary, ood_modality, path_to_save=cond_model_path)



def evaluate(models, datasets, data_config_path, base_path, setting, evaluation_metrics, save_folder, pid_percentage):
      
    config_data = configparser.ConfigParser()
    
    for evaluation_metric in evaluation_metrics:

        for data in datasets:
            config_data_path = os.path.join(data_config_path, f'{data}.ini')
            config_data.read(config_data_path)
            data_params = build_data_params(config_data)
            path_to_save = os.path.join(base_path, 'evaluation', data, save_folder, evaluation_metric)
            os.makedirs(path_to_save, exist_ok=True)

            ######################## create and save imputated adata ################################################
            # adata = get_impute_data(
            #     data, data_params['primary_variable'], data_params['modality_variable'], data_params["data"], 'scRecover')
            
            # if adata is None:
            #     raise ValueError('ERROR: the imputated data cannot be found or obtained!')

            #########################################################################################################
            
            adata = sc.read(data_params["data"]) 

            if not issparse(adata.X):
                adata = calculate_cpm(adata, do_sparse=False)
            else:
                adata = calculate_cpm(adata, do_sparse=True)
            #########################################################
            # to be deleted -> evaluation on low dimension
            # gene_subset = np.load(data_params['gene_subset'])
            # adata = adata[:, gene_subset].copy()
            #########################################################  
            
            for ood_modality in data_params['ood_modality']:

                ood_modality_adata = adata[adata.obs[data_params['modality_variable']].isin([ood_modality, data_params['control_key']])].copy()
                
                all_corr = []
                logfc_corr = []
                mixing_seurat = []
                results_down = {}
                results_up = {}
                per_gene_true = []
                per_gene_true_adata = []
                per_gene_true_adata_delta = []
                per_gene_pred = defaultdict(list)
                per_gene_pred_adata = defaultdict(list)
                per_gene_pred_adata_delta = defaultdict(list)
                per_gene_except_ood_adata = []
                per_gene_except_ood_adata_delta = []
                per_gene_true_nonzero = []
                per_gene_pred_nonzero = defaultdict(list)
                distribution_true = []
                distribution_pred = defaultdict(list)
                results_gsea = {}
                modality_adata = ood_modality_adata[ood_modality_adata.obs[data_params['modality_variable']] == ood_modality].copy()
                pca_dict = {}
                knn_dict = defaultdict(list)
                mixing_dict = {}
                pvalue_dict_seurat = None
        
                # modality_adata_normalized, control_mean_dict, control_mean_dict_nonzero = process_subtract_control(
                #     ood_modality_adata, data_params['primary_variable'], data_params['modality_variable'], ood_modality)
                        
                if evaluation_metric == 'combined_deg_basemodel':
                    results_down['mean'], results_down['min'] = combined_deg_basemodel(ood_modality_adata, data_params['primary_variable'], 
                                                            data_params['modality_variable'], ood_modality, zeros=True)
                    true_pos_lineplot(results_down['mean'], '', 'basemodel-mean', data, path_to_save, 
                                        "Number of Top True p-values (up+down)", "Number of common genes with top predicted p-values (up+down)")
                    true_pos_lineplot(results_down['min'], '', 'basemodel-min', data, path_to_save, 
                                        "Number of Top True p-values (up+down)", "Number of common genes with top predicted p-values (up+down)")
                    continue
                elif evaluation_metric == 'combined_deg_basemodel_nonzero':
                    results_down['mean'], results_down['min'] = combined_deg_basemodel(ood_modality_adata, data_params['primary_variable'], 
                                                            data_params['modality_variable'], ood_modality, zeros=False)
                    true_pos_lineplot(results_down['mean'], '', 'basemodel-mean-nonzero', data, path_to_save, 
                                        "Number of Top True p-values (up+down)", "Number of common genes with top predicted p-values (up+down)")
                    true_pos_lineplot(results_down['min'], '', 'basemodel-min-nonzero', data, path_to_save, 
                                        "Number of Top True p-values (up+down)", "Number of common genes with top predicted p-values (up+down)")
                    continue
                
                for ood_primary in data_params["ood_primaries"]:
                    ################## delete later #####################
                    # ['B', 'CD14 Mono', 'CD16 Mono', 'CD4 T', 'CD8 T', 'DC', 'NK', 'T']
                    # if ood_primary not in ['CD4 T', 'B', 'DC']:
                    #     continue
                    # if ood_primary != 'NK':
                    #     continue
                    ################## delete later #####################
        
                    true_adata =  ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] == ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                    ctrl_adata =  ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] == ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == data_params['control_key'])].copy()
                    #ctrl_true_adata = ood_modality_adata[ood_modality_adata.obs[data_params['primary_variable']] == ood_primary].copy()
                    ctrl_true_adata = ood_modality_adata[
                        (ood_modality_adata.obs[data_params['primary_variable']] == ood_primary) & 
                        (ood_modality_adata.obs[data_params['modality_variable']].isin([data_params['control_key'], ood_modality]))].copy()

                    true_except_ood = ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] != ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()

                    ood_indices_test = None

                    if setting == 'pid':
                        final_folder = os.path.join(
                            '../outputs/dependencies', f'pid_test_indices/{data}/{int(float(pid_percentage)*100)}/{save_folder}/{ood_primary}-{ood_modality}')
                        ood_indices_test = np.load(os.path.join(final_folder,'test_indices.npy'), allow_pickle=True)
                        print(f"number of ood_indices_test: {len(ood_indices_test)}")
                        print(f"number of true_except_ood: {true_except_ood.X.shape[0]}")
                        true_except_ood = ad.concat([true_except_ood, true_adata[~true_adata.obs.index.isin(ood_indices_test)].copy()])
                        print(f"number of true_except_ood after adding seen data: {true_except_ood.X.shape[0]}")
                        true_adata =  true_adata[ood_indices_test].copy()
                        ##### revise this statement. should ctrl_true_adata be filtered for pid settings or not, specially for logfc_correlation???????
                        # ctrl_true_adata = ad.concat([ctrl_adata, true_adata])
        
                    true_logfc_df = None
        
                    per_gene_true.append(np.mean(true_adata.X, axis=0))
                    per_gene_true_adata.append(true_adata)
                    per_gene_except_ood_adata.append(true_except_ood)
                    if evaluation_metric == 'rigorous_spearman_corr_per_gene_delta':
                        X = ctrl_adata.X.toarray() if issparse(ctrl_adata.X) else ctrl_adata.X
                        c = np.mean(X, axis=0)
                        true_except_ood_copy = true_except_ood.copy()
                        if issparse(true_except_ood_copy.X):
                            true_except_ood_copy.X = true_except_ood_copy.X.toarray()
                        true_except_ood_copy.X  = true_except_ood_copy.X - c

                        true_copy = true_adata.copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        true_copy.X = true_copy.X - c
                        per_gene_true_adata_delta.append(true_copy)
                        per_gene_except_ood_adata_delta.append(true_except_ood_copy)
                        
                    if evaluation_metric == 'spearman_corr_per_gene_nonzero' or evaluation_metric == 'normalized_corr_per_gene_nonzero':
                        per_gene_true_nonzero.append(np.nan_to_num(np.mean(true_adata.to_df().replace(0, np.NaN), axis=0).values))
        
                    distribution_true.append(true_adata.X)
        
                    pred_dict = {}

                    for model in models:

                        # pred_path = os.path.join(base_path, 'train_predict', data, model, setting)                            
                        pred_path = os.path.join(base_path, 'train_predict', data, model, save_folder)                            
                        cond_pred_path = os.path.join(pred_path, 'models', f"{ood_primary}-{ood_modality}")
                        
                        pred_adata = sc.read(os.path.join(cond_pred_path, 'pred_adata.h5ad'))

                        #########################################################
                        # to be deleted -> evaluation on low dimension
                        # if pred_adata.n_vars > len(gene_subset):
                        #     pred_adata = pred_adata[:, gene_subset].copy()
                        #########################################################
        
                        pred_dict[model] = pred_adata.copy()
                        
                        per_gene_pred[model].append(np.mean(pred_adata.X, axis=0))
                        per_gene_pred_adata[model].append(pred_adata)
                        if evaluation_metric == 'rigorous_spearman_corr_per_gene_delta':
                            X = ctrl_adata.X.toarray() if issparse(ctrl_adata.X) else ctrl_adata.X
                            c = np.mean(X, axis=0)
                            pred_adata_copy = pred_adata.copy()
                            if issparse(pred_adata_copy.X):
                                pred_adata_copy.X = pred_adata_copy.X.toarray()
                            pred_adata_copy.X  = pred_adata_copy.X - c

                            per_gene_pred_adata_delta[model].append(pred_adata_copy)
        
                        if evaluation_metric == 'spearman_corr_per_gene_nonzero' or evaluation_metric == 'normalized_corr_per_gene_nonzero':
                            per_gene_pred_nonzero[model].append(np.nan_to_num(np.mean(pred_adata.to_df().replace(0, np.NaN), axis=0).values))
        
                        elif evaluation_metric == 'wasserstein_distance':
                            distribution_pred[model].append(pred_adata.X)
        
                        elif evaluation_metric == 'pca':
                            pca_dict[model] = pred_adata.copy()
        
                        elif evaluation_metric == 'knn':
                            knn_dict[model].append(k_nearest_neighbors(
                                pred_adata, true_adata, ctrl_adata, model, ood_primary, pca_components=10, n_neighbors=int(len(pred_adata.X)/2)))
                                # pred_adata, true_adata, ctrl_adata, model, ood_primary, pca_components=2, n_neighbors=int(len(pred_adata.X)/2)))
                                # pred_adata, true_adata, ctrl_adata, model, ood_primary, pca_components=10, n_neighbors=100))
        
                        elif evaluation_metric == 'spearman_corr':                    
                            all_corr.append(spearman_corr(pred_adata, true_adata))
        
                        elif evaluation_metric == 'z_correlation':
                            all_corr.append(z_correlation(pred_adata, true_adata, ctrl_adata))
        
                        elif evaluation_metric == 'logfc_correlation':
                            if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=False)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                            corr = assess.logfc_correlation(pred_adata, ctrl_adata, true_logfc_df,ood_modality, data_params['control_key'])
                            logfc_corr.append(corr)
                        
                        elif evaluation_metric == 'deg_true_positives_nonzero':
                            results_down[model], results_up[model] = true_deg_pos(pred_adata, true_adata, ctrl_adata)
        
                        elif evaluation_metric == 'deg_true_positives_all':
                            results_down[model], results_up[model] = true_deg_pos(pred_adata, true_adata, ctrl_adata, True)
                        
                        elif evaluation_metric == 'sorted_pvalue_common_genes':
                            results_down[model], results_up[model] = sorted_pvalue_common_genes(pred_adata, true_adata, ctrl_adata, True)
        
                        elif evaluation_metric == 'sorted_pvalue_rank':
                            results_down[model], results_up[model], perfect_rank_sorted = sorted_pvalue_rank(pred_adata, true_adata, ctrl_adata, True)
        
                        elif evaluation_metric == 'combined_deg':
                            results_down[model] = combined_deg(pred_adata, true_adata, ctrl_adata, zeros=True)
        
                        elif evaluation_metric == 'combined_deg_nonzero':
                            results_down[model] = combined_deg(pred_adata, true_adata, ctrl_adata, zeros=False)
        
                        elif evaluation_metric == 'combined_deg_seurat':
                            results_down[model] = combined_deg_seurat(pred_adata, true_adata, ctrl_adata, "MAST")
        
                        elif evaluation_metric == 'corr_top_k':
                            results_gsea[model] = corr_top_k(modality_adata, np.mean(pred_adata.X, axis=0), data_params['primary_variable'], ood_primary)
        
                        elif evaluation_metric == 'normalized_corr_top_k':
                            results_gsea[model] = corr_top_k(
                                modality_adata_normalized, np.mean(pred_adata.X, axis=0)-control_mean_dict[ood_primary], 
                                data_params['primary_variable'], ood_primary)
                        
                        elif evaluation_metric == 'corr_top_k_per_primary':
                            results_down[model], results_up[model], base_ratio = corr_top_k_per_primary(pred_adata, true_adata, ctrl_adata)
        
                        elif evaluation_metric == 'mixing_index':
                            mixing_list = []
                            for k in range(1,11):
                                mixing_list.append(mixing_index(pred_adata.copy(), true_adata.copy(), n_clusters=k))
                            mixing_dict[model] = pd.DataFrame({'n': np.arange(1,11), 'true_positives': mixing_list})
        
                        elif evaluation_metric == 'mixing_index_seurat':
                            mixing_seurat.append(mixing_index_seurat(pred_adata.copy(), true_adata.copy(), all_data=ood_modality_adata.copy()))
        
                        elif evaluation_metric == 'mixing_index_seurat_topdeg':
                            if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=True)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                            mixing_seurat.append(assess.mixing_index_seurat_topdeg(pred_adata.copy(), true_adata.copy(), true_logfc_df, num_top_deg=100))

                        elif evaluation_metric == 'mixing_index_kmeans_topdeg':
                            if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=True)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                            mixing_dict.append(assess.mixing_index_kmeans_topdeg(pred_adata.copy(), true_adata.copy(), true_logfc_df, n_clusters=2, num_top_deg=100))
        
                    if evaluation_metric == 'pca':
                        pca_plot(ctrl_adata.copy(), true_adata.copy(), pca_dict, data_params['modality_variable'], ood_primary, data, path_to_save)
        
                    # elif evaluation_metric == 'ot_rank_per_gene':
                    #     draw_boxplot(distribution_pred, data, path_to_save, f'{ood_primary}-distance-rank-per-gene')
                                        
                    elif evaluation_metric.startswith('deg_true_positives') :
                        true_pos_lineplot(results_down, ood_primary, 'down', data, path_to_save, 
                                        "Number of Top Predicted p-values", "Number of True Positives")
                        true_pos_lineplot(results_up, ood_primary, 'up', data, path_to_save,
                                        "Number of Top Predicted p-values", "Number of True Positives")
                        
                    elif evaluation_metric == 'sorted_pvalue_common_genes':
                        true_pos_lineplot(results_down, ood_primary, 'down', data, path_to_save, 
                                        "Number of Top True p-values", "Number of common genes with top predicted p-values")
                        true_pos_lineplot(results_up, ood_primary, 'up', data, path_to_save,
                                        "Number of Top True p-values", "Number of common genes with top predicted p-values")
                        
                    elif evaluation_metric == 'sorted_pvalue_rank':
                        results_down['perfect_model'] = perfect_rank_sorted
                        results_up['perfect_model'] = perfect_rank_sorted
                        true_pos_lineplot(results_down, ood_primary, 'down', data, path_to_save, 
                                        "Number of Top True p-values", "mean rank of top genes of true in predicted p-values")
                        true_pos_lineplot(results_up, ood_primary, 'up', data, path_to_save,
                                        "Number of Top True p-values", "mean rank of top genes of true in predicted p-values")
                        
                    elif evaluation_metric == 'combined_deg':
                        pvalue_dict = assess.get_pvalue_dict(ood_modality_adata, data_params['primary_variable'], data_params['modality_variable'], ood_modality, zeros=True)
                        results_down['basemodel_mean'], results_down['basemodel_min'] = assess.add_basemodel(pvalue_dict, ood_primary)
                        true_pos_lineplot(results_down, ood_primary, evaluation_metric, data, path_to_save, 
                                        "Number of Top True p-values (up+down)", "Number of common genes with top predicted p-values (up+down)")
                        
                    elif evaluation_metric == 'combined_deg_nonzero':
                        pvalue_dict = assess.get_pvalue_dict(ood_modality_adata, data_params['primary_variable'], data_params['modality_variable'], ood_modality, zeros=False)
                        results_down['basemodel_mean'], results_down['basemodel_min'] = assess.add_basemodel(pvalue_dict, ood_primary)
                        true_pos_lineplot(results_down, ood_primary, evaluation_metric, data, path_to_save, 
                                        "Number of Top True p-values (up+down)", "Number of common genes with top predicted p-values (up+down)")
        
                    elif evaluation_metric == 'combined_deg_seurat':
                        if pvalue_dict_seurat is None:
                            pvalue_dict_seurat = assess.get_pvalue_dict_seurat(ood_modality_adata, data_params['primary_variable'],  data_params['modality_variable'], ood_modality, data_params['control_key'])
                        results_down['basemodel_mean'], results_down['basemodel_min'] = assess.add_basemodel(pvalue_dict_seurat, ood_primary)
                        true_pos_lineplot(results_down, ood_primary, evaluation_metric, data, path_to_save, 
                                        "Number of Top True p-values (up+down)", "Number of common genes with top predicted p-values (up+down)")
                    
                    
                    elif evaluation_metric.endswith('corr_top_k'):
                        top_k_plot(results_gsea, models, ood_primary, data, path_to_save)
        
                    elif evaluation_metric == 'corr_top_k_per_primary':
                        results_down['random base'] = base_ratio
                        true_pos_lineplot(results_down, ood_primary, 'top-k-per-primary-abs', data, path_to_save, 
                                        "Number of Top predicted correlation", "number of top correlations related to stim cells (vs ctrl cells)")
                        true_pos_lineplot(results_up, ood_primary, 'top-k-per-primary-ratio', data, path_to_save, 
                                        "Number of Top predicted correlation", "odds of top correlations related to stim cells (vs ctrl cells)")
        
                    elif evaluation_metric == 'mixing_index':
                        true_pos_lineplot(mixing_dict, ood_primary, 'mixing-index', data, path_to_save, 
                                        "number of clusters", "weighted mixing index")
        
                    elif evaluation_metric == 'rigorous_mixing_index_seurat':
                        max_size = 15000
                        if ood_modality_adata.n_obs > max_size: 
                            chosen_indices = np.random.choice(ood_modality_adata.n_obs, size=max_size, replace=False)
                            all_adata = ood_modality_adata[chosen_indices].copy()
                        else:
                            all_adata = ood_modality_adata.copy()
                        mixing_index_dict = assess.rigorous_mixing_index_seurat(true_adata, pred_dict, true_except_ood, data_params, ood_primary, ood_modality, all_adata=all_adata)
                        draw_boxplot(
                            mixing_index_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}")
                        with open(f'{path_to_save}/mixing_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(mixing_index_dict, f)

                    elif evaluation_metric == 'semi_rigorous_mixing_index_seurat':
                        resolution = 2.0
                        max_size = 15000
                        if ood_modality_adata.n_obs > max_size: 
                            chosen_indices = np.random.choice(ood_modality_adata.n_obs, size=max_size, replace=False)
                            all_adata = ood_modality_adata[chosen_indices].copy()
                        else:
                            all_adata = ood_modality_adata.copy()
                        if ood_indices_test is not None:
                            train_adata =  ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] == ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                            train_adata = train_adata[~train_adata.obs.index.isin(ood_indices_test)].copy()
                        else:
                            train_adata = None
                        mixing_index_dict = assess.semi_rigorous_mixing_index_seurat(
                            true_adata, pred_dict, true_except_ood, data_params['primary_variable'], all_adata=all_adata, train_adata=train_adata, resolution=resolution)
                        with open(f'{path_to_save}/mixing_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(mixing_index_dict, f)
                        draw_boxplot(
                            mixing_index_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}", title=f'resolution = {resolution}')

                    elif evaluation_metric == 'rigorous_mixing_index_seurat_topdeg':
                        num_top_deg=10
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=True)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        mixing_index_dict = assess.rigorous_mixing_index_seurat_topdeg(true_adata, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df, num_top_deg=num_top_deg)
                        final_path_to_save = f"{path_to_save}_{num_top_deg}"
                        os.makedirs(final_path_to_save, exist_ok=True)
                        draw_boxplot(
                            # mixing_index_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}", f"setting {setting}, top {num_top_deg} genes")
                            mixing_index_dict, data, final_path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}", f"setting {setting}, top {num_top_deg} genes")
                        # with open(f'{path_to_save}/mixing_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                        with open(f'{final_path_to_save}/mixing_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(mixing_index_dict, f)

                    elif evaluation_metric == 'rigorous_mixing_index_kmeans':
                        mixing_index_dict = assess.rigorous_mixing_index_kmeans(true_adata, pred_dict, true_except_ood, data_params['primary_variable'])
                        draw_boxplot(
                            mixing_index_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}")
                        with open(f'{path_to_save}/mixing_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(mixing_index_dict, f)

                    elif evaluation_metric == 'rigorous_mixing_index_kmeans_topdeg':
                        num_top_deg=10
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=True)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        mixing_index_dict = assess.rigorous_mixing_index_kmeans_topdeg(true_adata, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df, num_top_deg=num_top_deg)
                        final_path_to_save = f"{path_to_save}_{num_top_deg}"
                        os.makedirs(final_path_to_save, exist_ok=True)
                        draw_boxplot(
                            mixing_index_dict, data, final_path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}", f"setting {setting}, top {num_top_deg} genes")
                        with open(f'{final_path_to_save}/mixing_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(mixing_index_dict, f)

                    elif evaluation_metric == 'mixing_kmeans_vs_topgenes':
                        all_results = {}  # {method_name: {num_top_genes: [list of 10 values]}}
                        num_top_genes_list = []
                        t = 3
                        while 2**t <= 2048:
                            num_top_genes_list.append(2**t)
                            t += 1
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=False)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        mixing_index_dict = assess.rigorous_mixing_index_kmeans(true_adata, pred_dict, true_except_ood, data_params['primary_variable'])
                        for method, scores in mixing_index_dict.items():
                            if method not in all_results:
                                all_results[method] = {}
                            all_results[method][ood_modality_adata.n_vars] = scores
                        for num_top_genes in num_top_genes_list:
                            print(f"Running for num_top_genes = {num_top_genes}")
                            mixing_index_dict = assess.rigorous_mixing_index_kmeans_topdeg(true_adata, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df, num_top_deg=num_top_genes)
                            # Aggregate results
                            for method, scores in mixing_index_dict.items():
                                if method not in all_results:
                                    all_results[method] = {}
                                all_results[method][num_top_genes] = scores
                        with open(f'{path_to_save}/mixing_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(all_results, f)
                        lineplot_with_errorbar(all_results, ood_primary, data, path_to_save, xlabel='Number of Top Genes', ylabel='Mixing Index', 
                            filename=f"{evaluation_metric}_{ood_primary}_{ood_modality}", title=f'Mixing Index vs Number of Top Genes, setting {setting} - {ood_primary}')

                    elif evaluation_metric == 'mmd_vs_topdeg':
                        all_results = {}  # {method_name: {num_top_genes: [list of 10 values]}}
                        num_top_genes_list = []
                        t = 1
                        while 2**t <= 2048:
                            num_top_genes_list.append(2**t)
                            t += 1
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=False)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        metric_dict = assess.rigorous_mmd(true_adata, pred_dict, true_except_ood, data_params['primary_variable'])
                        for method, scores in metric_dict.items():
                            if method not in all_results:
                                all_results[method] = {}
                            all_results[method][ood_modality_adata.n_vars] = scores
                        for num_top_genes in num_top_genes_list:
                            print(f"Running for num_top_genes = {num_top_genes}")
                            metric_dict = assess.rigorous_mmd_topdeg(true_adata, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df, num_top_deg=num_top_genes)
                            # Aggregate results
                            for method, scores in metric_dict.items():
                                if method not in all_results:
                                    all_results[method] = {}
                                all_results[method][num_top_genes] = scores
                        with open(f'{path_to_save}/mmd_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(all_results, f)
                        lineplot_with_errorbar(all_results, ood_primary, data, path_to_save, xlabel='Number of Top Genes', ylabel='MMD (-log10)', 
                            filename=f"{evaluation_metric}_{ood_primary}_{ood_modality}", title=f'MMD vs Number of Top Genes, setting {setting} - {ood_primary}')
                    
                    elif evaluation_metric == 'spearman_vs_topdeg':
                        all_results = {}  # {method_name: {num_top_genes: [list of 10 values]}}
                        num_top_genes_list = []
                        t = 1
                        while 2**t <= 2048:
                            num_top_genes_list.append(2**t)
                            t += 1
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=False)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        metric_dict = assess.rigorous_spearman(true_adata, pred_dict, true_except_ood, data_params['primary_variable'])
                        for method, scores in metric_dict.items():
                            if method not in all_results:
                                all_results[method] = {}
                            all_results[method][ood_modality_adata.n_vars] = scores
                        for num_top_genes in num_top_genes_list:
                            print(f"Running for num_top_genes = {num_top_genes}")
                            metric_dict = assess.rigorous_spearman_topdeg(true_adata, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df, num_top_deg=num_top_genes)
                            # Aggregate results
                            for method, scores in metric_dict.items():
                                if method not in all_results:
                                    all_results[method] = {}
                                all_results[method][num_top_genes] = scores
                        with open(f'{path_to_save}/spearman_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(all_results, f)
                        lineplot_with_errorbar(all_results, ood_primary, data, path_to_save, xlabel='Number of Top Genes', ylabel='Spearman Correlation', 
                            filename=f"{evaluation_metric}_{ood_primary}_{ood_modality}", title=f'Spearman vs Number of Top Genes, setting {setting} - {ood_primary}')

                    elif evaluation_metric == 'norm_spearman_vs_topdeg':
                        means = np.mean(ood_modality_adata.X, axis=0)
                        adata_copy = ood_modality_adata.copy()
                        adata_copy.X = adata_copy.X - means   
                        true_except_ood = adata_copy[(adata_copy.obs[data_params['primary_variable']] != ood_primary) & 
                                                (adata_copy.obs[data_params['modality_variable']] == ood_modality)].copy()
                        
                        true_adata.X = true_adata.X - means
                        for p in pred_dict.keys():
                            (pred_dict[p]).X = (pred_dict[p]).X - means

                        all_results = {}  # {method_name: {num_top_genes: [list of 10 values]}}
                        num_top_genes_list = []
                        t = 3
                        while 2**t <= 2048:
                            num_top_genes_list.append(2**t)
                            t += 1
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=False)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        metric_dict = assess.rigorous_spearman(true_adata, pred_dict, true_except_ood, data_params['primary_variable'])
                        for method, scores in metric_dict.items():
                            if method not in all_results:
                                all_results[method] = {}
                            all_results[method][ood_modality_adata.n_vars] = scores
                        for num_top_genes in num_top_genes_list:
                            print(f"Running for num_top_genes = {num_top_genes}")
                            metric_dict = assess.rigorous_spearman_topdeg(true_adata, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df, num_top_deg=num_top_genes)
                            # Aggregate results
                            for method, scores in metric_dict.items():
                                if method not in all_results:
                                    all_results[method] = {}
                                all_results[method][num_top_genes] = scores
                        with open(f'{path_to_save}/spearman_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(all_results, f)
                        lineplot_with_errorbar(all_results, ood_primary, data, path_to_save, xlabel='Number of Top Genes', ylabel='Normalized Spearman Correlation', 
                            filename=f"{evaluation_metric}_{ood_primary}_{ood_modality}", title=f'Norm Spearman vs Number of Top Genes, setting {setting} - {ood_primary}')


                    elif evaluation_metric == 'wasserstein_vs_topdeg':
                        all_results = {}  # {method_name: {num_top_genes: [list of 10 values]}}
                        num_top_genes_list = []
                        t = 1
                        while 2**t <= 2048:
                            num_top_genes_list.append(2**t)
                            t += 1
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=False)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        metric_dict = assess.rigorous_wasserstein(true_adata, pred_dict, true_except_ood, data_params['primary_variable'], include_best=True)
                        for method, scores in metric_dict.items():
                            if method not in all_results:
                                all_results[method] = {}
                            all_results[method][ood_modality_adata.n_vars] = scores
                        for num_top_genes in num_top_genes_list:
                            print(f"Running for num_top_genes = {num_top_genes}")
                            metric_dict = assess.rigorous_wasserstein_topdeg(true_adata, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df, num_top_deg=num_top_genes)
                            # Aggregate results
                            for method, scores in metric_dict.items():
                                if method not in all_results:
                                    all_results[method] = {}
                                all_results[method][num_top_genes] = scores
                        with open(f'{path_to_save}/wass_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(all_results, f)
                        lineplot_with_errorbar(all_results, ood_primary, data, path_to_save, xlabel='Number of Top Genes', ylabel='Wasserstein (-log10)', 
                            filename=f"{evaluation_metric}_{ood_primary}_{ood_modality}", title=f'Wasserstein vs Number of Top Genes, setting {setting} - {ood_primary}')
   
        
                    elif evaluation_metric == 'mixing_kmeans_vs_random_genes':
                        all_results_top = {}  # {method_name: {num_top_genes: [list of 10 values]}}
                        all_results_random = {}  
                        num_top_genes_list = []
                        t = 1
                        while 2**t <= 2048:
                            num_top_genes_list.append(2**t)
                            t += 1
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=False)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        df_shuffled = true_logfc_df.sample(frac=1)
                        temp = assess.rigorous_mixing_index_kmeans(true_adata, pred_dict, true_except_ood, data_params['primary_variable'])
                        for method, scores in temp.items():
                            if method not in all_results_top:
                                all_results_top[method] = {}
                            if method not in all_results_random:
                                all_results_random[method] = {}
                            all_results_top[method][ood_modality_adata.n_vars] = scores
                            all_results_random[method][ood_modality_adata.n_vars] = scores
                        for num_top_genes in num_top_genes_list:
                            print(f"Running for num_top_genes = {num_top_genes}")
                            mixing_index_dict_top = assess.rigorous_mixing_index_kmeans_topdeg(true_adata, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df, num_top_deg=num_top_genes)
                            mixing_index_dict_random = assess.rigorous_mixing_index_kmeans_topdeg(true_adata, pred_dict, true_except_ood, data_params['primary_variable'], df_shuffled, num_top_deg=num_top_genes)
                            # Aggregate results
                            for method, scores in mixing_index_dict_top.items():
                                if method not in all_results_top:
                                    all_results_top[method] = {}
                                all_results_top[method][num_top_genes] = scores
                            for method, scores in mixing_index_dict_random.items():
                                if method not in all_results_random:
                                    all_results_random[method] = {}
                                all_results_random[method][num_top_genes] = scores
                        final_path = os.path.join(path_to_save, ood_primary)
                        os.makedirs(final_path, exist_ok=True)
                        with open(f'{final_path}/mixing_dict_top_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(all_results_top, f)
                        with open(f'{final_path}/mixing_dict_random_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(all_results_random, f)
                        for method in all_results_top.keys():
                            combined_result = {f"{method}_top": all_results_top[method], f"{method}_random": all_results_random[method]}
                            lineplot_with_errorbar(combined_result, ood_primary, data, final_path, xlabel='Number of selected Genes', ylabel='Mixing Index', 
                                filename=f"{evaluation_metric}_{ood_primary}_{method}", title=f'Mixing Index vs Number of Selected Genes, setting {setting} - {ood_primary} - {method}', y_min=0, y_max=1)
        
                    
                    elif evaluation_metric == 'rigorous_spearman':
                        true_except_ood_copy = true_except_ood.copy()
                        if issparse(true_except_ood_copy.X):
                            true_except_ood_copy.X = true_except_ood_copy.X.toarray()
                        true_copy = true_adata.copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        pred_dict_copy = pred_dict.copy()
                        for p in pred_dict_copy.keys():
                            if issparse(pred_dict_copy[p].X):
                                pred_dict_copy[p].X = pred_dict_copy[p].X.toarray()
                        correlation_dict = assess.rigorous_spearman(true_copy, pred_dict_copy, true_except_ood_copy, data_params['primary_variable'])
                        draw_boxplot(
                            correlation_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}")
                        with open(f'{path_to_save}/corr_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(correlation_dict, f)
            

                    elif evaluation_metric == 'rigorous_pearson_r2':
                        true_except_ood_norm = true_except_ood.copy()
                        if issparse(true_except_ood_norm.X):
                            true_except_ood_norm.X = true_except_ood_norm.X.toarray()
                        true_copy = true_adata.copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        pred_dict_copy = pred_dict.copy()
                        for p in pred_dict_copy.keys():
                            if issparse(pred_dict_copy[p].X):
                                pred_dict_copy[p].X = pred_dict_copy[p].X.toarray()

                        correlation_dict, r2_dict = assess.rigorous_pearson_r2(true_copy, pred_dict_copy, true_except_ood_norm, data_params['primary_variable'])
                        with open(f'{path_to_save}/corr_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(correlation_dict, f)
                        with open(f'{path_to_save}/r2_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(r2_dict, f)
                        draw_boxplot(
                            correlation_dict, data, path_to_save, f"pearson_corr_{ood_primary}_{ood_modality}")
                        draw_boxplot(
                            r2_dict, data, path_to_save, f"r2_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'rigorous_mse':
                        metric_dict = assess.rigorous_mse(true_adata.copy(), pred_dict, true_except_ood.copy(), data_params['primary_variable'])
                        with open(f'{path_to_save}/mse_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        draw_boxplot(
                            metric_dict, data, path_to_save, f"mse_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'rigorous_normalized_mse':
                        X = ood_modality_adata.X.toarray() if issparse(ood_modality_adata.X) else ood_modality_adata.X
                        c = np.mean(X, axis=0)
                        c_std = np.std(X, axis=0)
                        nonzero_all = c != 0
                        c = c[nonzero_all]
                        c_std = c_std[nonzero_all]
                        print(f'number of nonzero genes: {len(c)}')

                        true_except_ood = ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] != ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                        
                        true_except_ood_norm = true_except_ood[:, nonzero_all].copy()
                        if issparse(true_except_ood_norm.X):
                            true_except_ood_norm.X = true_except_ood_norm.X.toarray()
                        true_except_ood_norm.X  = (true_except_ood_norm.X - c) / c_std

                        true_copy = true_adata[:, nonzero_all].copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        true_copy.X = (true_copy.X - c) / c_std

                        pred_dict_copy = pred_dict.copy()
                        for p in pred_dict_copy.keys():
                            pred_dict_copy[p] = pred_dict_copy[p][:, nonzero_all].copy()
                            if issparse(pred_dict_copy[p].X):
                                pred_dict_copy[p].X = pred_dict_copy[p].X.toarray()
                            pred_dict_copy[p].X = (pred_dict_copy[p].X - c) / c_std
                            
                        metric_dict = assess.rigorous_mse(true_copy, pred_dict_copy, true_except_ood_norm, data_params['primary_variable'])
                        with open(f'{path_to_save}/mse_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        draw_boxplot(
                            metric_dict, data, path_to_save, f"mse_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'rigorous_cosine_sim':
                        metric_dict = assess.rigorous_cosine_sim(true_adata.copy(), pred_dict, true_except_ood.copy(), data_params['primary_variable'])
                        with open(f'{path_to_save}/cosine_sim_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        draw_boxplot(
                            metric_dict, data, path_to_save, f"cosine_sim_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'rigorous_stepwise_r2':
                        corr_dict, r2_dict = assess.rigorous_stepwise_r2(true_adata, pred_dict, true_except_ood, ctrl_adata, data_params['primary_variable'])
                        with open(f'{path_to_save}/auc_corr_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(corr_dict, f)
                        with open(f'{path_to_save}/auc_r2_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(r2_dict, f)
                        draw_boxplot(
                            corr_dict, data, path_to_save, f"auc_corr_{ood_primary}_{ood_modality}")
                        draw_boxplot(
                            r2_dict, data, path_to_save, f"auc_r2_{ood_primary}_{ood_modality}")
        
                    elif evaluation_metric == 'rigorous_normalized_spearman':
                        c = np.mean(ctrl_adata.X, axis=0)
                        c_std = np.std(ctrl_adata.X, axis=0)
                        nonzero_ctrl = c != 0
                        c = c[nonzero_ctrl]
                        c_std = c_std[nonzero_ctrl]

                        true_except_ood = ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] != ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                        
                        true_except_ood_norm = true_except_ood[:, nonzero_ctrl].copy()
                        true_except_ood_norm.X  = (true_except_ood_norm.X - c) / c_std
                        true_copy = true_adata[:, nonzero_ctrl].copy()
                        true_copy.X = (true_copy.X - c) / c_std

                        for p in pred_dict.keys():
                            pred_dict[p] = pred_dict[p][:, nonzero_ctrl].copy()
                            pred_dict[p].X = (pred_dict[p].X - c) / c_std
                            
                        correlation_dict = assess.rigorous_spearman(true_copy, pred_dict, true_except_ood_norm, data_params['primary_variable'])
                        draw_boxplot(
                            correlation_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}")
                        with open(f'{path_to_save}/normalized_corr_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(correlation_dict, f)

                    elif evaluation_metric == 'rigorous_normalized_pearson_r2':
                        X = ctrl_adata.X.toarray() if issparse(ctrl_adata.X) else ctrl_adata.X
                        c = np.mean(X, axis=0)
                        c_std = np.std(X, axis=0)
                        nonzero_ctrl = c != 0
                        c = c[nonzero_ctrl]
                        c_std = c_std[nonzero_ctrl]

                        true_except_ood = ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] != ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                        
                        true_except_ood_norm = true_except_ood[:, nonzero_ctrl].copy()
                        if issparse(true_except_ood_norm.X):
                            true_except_ood_norm.X = true_except_ood_norm.X.toarray()
                        true_except_ood_norm.X  = (true_except_ood_norm.X - c) / c_std

                        true_copy = true_adata[:, nonzero_ctrl].copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        true_copy.X = (true_copy.X - c) / c_std

                        pred_dict_copy = pred_dict.copy()
                        for p in pred_dict_copy.keys():
                            pred_dict_copy[p] = pred_dict_copy[p][:, nonzero_ctrl].copy()
                            if issparse(pred_dict_copy[p].X):
                                pred_dict_copy[p].X = pred_dict_copy[p].X.toarray()
                            pred_dict_copy[p].X = (pred_dict_copy[p].X - c) / c_std
                            
                        correlation_dict, r2_dict = assess.rigorous_pearson_r2(true_copy, pred_dict_copy, true_except_ood_norm, data_params['primary_variable'])
                        with open(f'{path_to_save}/corr_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(correlation_dict, f)
                        with open(f'{path_to_save}/r2_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(r2_dict, f)
                        draw_boxplot(
                            correlation_dict, data, path_to_save, f"pearson_corr_{ood_primary}_{ood_modality}")
                        draw_boxplot(
                            r2_dict, data, path_to_save, f"r2_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'rigorous_normalized_r2_all':
                        X = ood_modality_adata.X.toarray() if issparse(ood_modality_adata.X) else ood_modality_adata.X
                        c = np.mean(X, axis=0)
                        c_std = np.std(X, axis=0)
                        nonzero_all = c != 0
                        c = c[nonzero_all]
                        c_std = c_std[nonzero_all]
                        print(f'number of nonzero genes: {len(c)}')

                        true_except_ood = ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] != ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                        
                        true_except_ood_norm = true_except_ood[:, nonzero_all].copy()
                        if issparse(true_except_ood_norm.X):
                            true_except_ood_norm.X = true_except_ood_norm.X.toarray()
                        true_except_ood_norm.X  = (true_except_ood_norm.X - c) / c_std

                        true_copy = true_adata[:, nonzero_all].copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        true_copy.X = (true_copy.X - c) / c_std

                        pred_dict_copy = pred_dict.copy()
                        for p in pred_dict_copy.keys():
                            pred_dict_copy[p] = pred_dict_copy[p][:, nonzero_all].copy()
                            if issparse(pred_dict_copy[p].X):
                                pred_dict_copy[p].X = pred_dict_copy[p].X.toarray()
                            pred_dict_copy[p].X = (pred_dict_copy[p].X - c) / c_std
                            
                        correlation_dict, r2_dict = assess.rigorous_pearson_r2(true_copy, pred_dict_copy, true_except_ood_norm, data_params['primary_variable'])
                        with open(f'{path_to_save}/corr_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(correlation_dict, f)
                        with open(f'{path_to_save}/r2_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(r2_dict, f)
                        draw_boxplot(
                            correlation_dict, data, path_to_save, f"pearson_corr_{ood_primary}_{ood_modality}")
                        draw_boxplot(
                            r2_dict, data, path_to_save, f"r2_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'rigorous_normalized_r2_skl':
                        X = ood_modality_adata.X.toarray() if issparse(ood_modality_adata.X) else ood_modality_adata.X
                        c = np.mean(X, axis=0)
                        c_std = np.std(X, axis=0)
                        nonzero_all = c != 0
                        c = c[nonzero_all]
                        c_std = c_std[nonzero_all]
                        print(f'number of nonzero genes: {len(c)}')

                        true_except_ood = ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] != ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                        
                        true_except_ood_norm = true_except_ood[:, nonzero_all].copy()
                        if issparse(true_except_ood_norm.X):
                            true_except_ood_norm.X = true_except_ood_norm.X.toarray()
                        true_except_ood_norm.X  = (true_except_ood_norm.X - c) / c_std

                        true_copy = true_adata[:, nonzero_all].copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        true_copy.X = (true_copy.X - c) / c_std

                        pred_dict_copy = pred_dict.copy()
                        for p in pred_dict_copy.keys():
                            pred_dict_copy[p] = pred_dict_copy[p][:, nonzero_all].copy()
                            if issparse(pred_dict_copy[p].X):
                                pred_dict_copy[p].X = pred_dict_copy[p].X.toarray()
                            pred_dict_copy[p].X = (pred_dict_copy[p].X - c) / c_std
                            
                        r2_dict = assess.rigorous_r2_skl(true_copy, pred_dict_copy, true_except_ood_norm, data_params['primary_variable'])
                        with open(f'{path_to_save}/r2_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(r2_dict, f)
                        draw_boxplot(
                            r2_dict, data, path_to_save, f"r2_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'rigorous_normalized_r2_brbic':
                        modality_adata = ood_modality_adata[ood_modality_adata.obs[data_params['modality_variable']] == ood_modality].copy()
                        primaries_modality = modality_adata.obs[data_params['primary_variable']].unique()
                        mean_list = []
                        for p in primaries_modality:
                            adata_primary = modality_adata[modality_adata.obs[data_params['primary_variable']] == p].copy()
                            X = adata_primary.X.toarray() if issparse(adata_primary.X) else adata_primary.X
                            mean_list.append(np.mean(X, axis=0))
                        mean_stack = np.vstack(mean_list)
                        c = np.mean(mean_stack, axis=0).reshape(1,-1)
                        print(f'shape of means: {c.shape}')
                        
                        # X = modality_adata.X.toarray() if issparse(modality_adata.X) else modality_adata.X
                        # c = np.mean(X, axis=0)
                        # c_std = np.std(X, axis=0)
                        # nonzero_all = c != 0
                        # c = c[nonzero_all]
                        # c_std = c_std[nonzero_all]
                        # print(f'number of nonzero genes: {len(c)}')

                        true_except_ood = ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] != ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                        
                        # true_except_ood_norm = true_except_ood[:, nonzero_all].copy()
                        true_except_ood_norm = true_except_ood.copy()
                        if issparse(true_except_ood_norm.X):
                            true_except_ood_norm.X = true_except_ood_norm.X.toarray()
                        # true_except_ood_norm.X  = (true_except_ood_norm.X - c) / c_std
                        true_except_ood_norm.X  = true_except_ood_norm.X - c

                        # true_copy = true_adata[:, nonzero_all].copy()
                        true_copy = true_adata.copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        # true_copy.X = (true_copy.X - c) / c_std
                        true_copy.X = true_copy.X - c

                        pred_dict_copy = pred_dict.copy()
                        for p in pred_dict_copy.keys():
                            # pred_dict_copy[p] = pred_dict_copy[p][:, nonzero_all].copy()
                            if issparse(pred_dict_copy[p].X):
                                pred_dict_copy[p].X = pred_dict_copy[p].X.toarray()
                            # pred_dict_copy[p].X = (pred_dict_copy[p].X - c) / c_std
                            pred_dict_copy[p].X = pred_dict_copy[p].X - c
                            
                        correlation_dict, r2_dict = assess.rigorous_pearson_r2(true_copy, pred_dict_copy, true_except_ood_norm, data_params['primary_variable'])
                        with open(f'{path_to_save}/corr_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(correlation_dict, f)
                        with open(f'{path_to_save}/r2_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(r2_dict, f)
                        draw_boxplot(
                            correlation_dict, data, path_to_save, f"pearson_corr_{ood_primary}_{ood_modality}")
                        draw_boxplot(
                            r2_dict, data, path_to_save, f"r2_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'rigorous_pearson_delta':
                        X = ctrl_adata.X.toarray() if issparse(ctrl_adata.X) else ctrl_adata.X
                        c = np.mean(X, axis=0)

                        true_except_ood = ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] != ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                        true_except_ood_norm = true_except_ood.copy()
                        if issparse(true_except_ood_norm.X):
                            true_except_ood_norm.X = true_except_ood_norm.X.toarray()
                        true_except_ood_norm.X  = true_except_ood_norm.X - c

                        true_copy = true_adata.copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        true_copy.X = true_copy.X - c

                        pred_dict_copy = pred_dict.copy()
                        for p in pred_dict_copy.keys():
                            if issparse(pred_dict_copy[p].X):
                                pred_dict_copy[p].X = pred_dict_copy[p].X.toarray()
                            pred_dict_copy[p].X = pred_dict_copy[p].X - c
                            
                        correlation_dict, r2_dict = assess.rigorous_pearson_r2(true_copy, pred_dict_copy, true_except_ood_norm, data_params['primary_variable'])
                        with open(f'{path_to_save}/corr_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(correlation_dict, f)
                        with open(f'{path_to_save}/r2_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(r2_dict, f)
                        draw_boxplot(
                            correlation_dict, data, path_to_save, f"pearson_corr_{ood_primary}_{ood_modality}")
                        draw_boxplot(
                            r2_dict, data, path_to_save, f"r2_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'rigorous_spearman_delta':
                        X = ctrl_adata.X.toarray() if issparse(ctrl_adata.X) else ctrl_adata.X
                        c = np.mean(X, axis=0)

                        true_except_ood = ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] != ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                        true_except_ood_norm = true_except_ood.copy()
                        if issparse(true_except_ood_norm.X):
                            true_except_ood_norm.X = true_except_ood_norm.X.toarray()
                        true_except_ood_norm.X  = true_except_ood_norm.X - c

                        true_copy = true_adata.copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        true_copy.X = true_copy.X - c

                        pred_dict_copy = pred_dict.copy()
                        for p in pred_dict_copy.keys():
                            if issparse(pred_dict_copy[p].X):
                                pred_dict_copy[p].X = pred_dict_copy[p].X.toarray()
                            pred_dict_copy[p].X = pred_dict_copy[p].X - c
                            
                        correlation_dict = assess.rigorous_spearman(true_copy, pred_dict_copy, true_except_ood_norm, data_params['primary_variable'])
                        with open(f'{path_to_save}/corr_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(correlation_dict, f)
                        draw_boxplot(
                            correlation_dict, data, path_to_save, f"spearman_corr_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'semi_rigorous_pearson_r2':
                        true_except_ood_norm = true_except_ood.copy()
                        if issparse(true_except_ood_norm.X):
                            true_except_ood_norm.X = true_except_ood_norm.X.toarray()

                        true_copy = true_adata.copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()

                        pred_dict_copy = pred_dict.copy()
                        for p in pred_dict_copy.keys():
                            if issparse(pred_dict_copy[p].X):
                                pred_dict_copy[p].X = pred_dict_copy[p].X.toarray()
                            
                        if ood_indices_test is not None:
                            train_adata =  ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] == ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                            train_adata = train_adata[~train_adata.obs.index.isin(ood_indices_test)].copy()
                            train_copy = train_adata.copy()
                            if issparse(train_copy.X):
                                train_copy.X = train_copy.X.toarray()
                        else:
                            train_adata = None
                            train_copy = None

                        correlation_dict, r2_dict = assess.semi_rigorous_pearson_r2(true_copy, pred_dict_copy, true_except_ood_norm, data_params['primary_variable'], train_adata=train_copy)
                        with open(f'{path_to_save}/corr_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(correlation_dict, f)
                        with open(f'{path_to_save}/r2_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(r2_dict, f)
                        draw_boxplot(
                            correlation_dict, data, path_to_save, f"normalized_pearson_corr_{ood_primary}_{ood_modality}")
                        draw_boxplot(
                            r2_dict, data, path_to_save, f"normalized_r2_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'semi_rigorous_normalized_pearson_r2':
                        X = ctrl_adata.X.toarray() if issparse(ctrl_adata.X) else ctrl_adata.X
                        c = np.mean(X, axis=0)
                        c_std = np.std(X, axis=0)
                        nonzero_ctrl = c != 0
                        c = c[nonzero_ctrl]
                        c_std = c_std[nonzero_ctrl]

                        true_except_ood_norm = true_except_ood[:, nonzero_ctrl].copy()
                        if issparse(true_except_ood_norm.X):
                            true_except_ood_norm.X = true_except_ood_norm.X.toarray()
                        true_except_ood_norm.X  = (true_except_ood_norm.X - c) / c_std

                        true_copy = true_adata[:, nonzero_ctrl].copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        true_copy.X = (true_copy.X - c) / c_std

                        pred_dict_copy = pred_dict.copy()
                        for p in pred_dict_copy.keys():
                            pred_dict_copy[p] = pred_dict_copy[p][:, nonzero_ctrl].copy()
                            if issparse(pred_dict_copy[p].X):
                                pred_dict_copy[p].X = pred_dict_copy[p].X.toarray()
                            pred_dict_copy[p].X = (pred_dict_copy[p].X - c) / c_std
                            
                        if ood_indices_test is not None:
                            train_adata =  ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] == ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                            train_adata = train_adata[~train_adata.obs.index.isin(ood_indices_test)].copy()
                            train_copy = train_adata[:, nonzero_ctrl].copy()
                            if issparse(train_copy.X):
                                train_copy.X = train_copy.X.toarray()
                            train_copy.X = (train_copy.X - c) / c_std
                        else:
                            train_adata = None
                            train_copy = None

                        correlation_dict, r2_dict = assess.semi_rigorous_pearson_r2(true_copy, pred_dict_copy, true_except_ood_norm, data_params['primary_variable'], train_adata=train_copy)
                        with open(f'{path_to_save}/corr_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(correlation_dict, f)
                        with open(f'{path_to_save}/r2_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(r2_dict, f)
                        draw_boxplot(
                            correlation_dict, data, path_to_save, f"normalized_pearson_corr_{ood_primary}_{ood_modality}")
                        draw_boxplot(
                            r2_dict, data, path_to_save, f"normalized_r2_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'semi_rigorous_normalized_r2_all':
                        X = ood_modality_adata.X.toarray() if issparse(ood_modality_adata.X) else ood_modality_adata.X
                        c = np.mean(X, axis=0)
                        c_std = np.std(X, axis=0)
                        nonzero_ctrl = c != 0
                        c = c[nonzero_ctrl]
                        c_std = c_std[nonzero_ctrl]

                        true_except_ood_norm = true_except_ood[:, nonzero_ctrl].copy()
                        if issparse(true_except_ood_norm.X):
                            true_except_ood_norm.X = true_except_ood_norm.X.toarray()
                        true_except_ood_norm.X  = (true_except_ood_norm.X - c) / c_std

                        true_copy = true_adata[:, nonzero_ctrl].copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        true_copy.X = (true_copy.X - c) / c_std

                        pred_dict_copy = pred_dict.copy()
                        for p in pred_dict_copy.keys():
                            pred_dict_copy[p] = pred_dict_copy[p][:, nonzero_ctrl].copy()
                            if issparse(pred_dict_copy[p].X):
                                pred_dict_copy[p].X = pred_dict_copy[p].X.toarray()
                            pred_dict_copy[p].X = (pred_dict_copy[p].X - c) / c_std
                            
                        if ood_indices_test is not None:
                            train_adata =  ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] == ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                            train_adata = train_adata[~train_adata.obs.index.isin(ood_indices_test)].copy()
                            train_copy = train_adata[:, nonzero_ctrl].copy()
                            if issparse(train_copy.X):
                                train_copy.X = train_copy.X.toarray()
                            train_copy.X = (train_copy.X - c) / c_std
                        else:
                            train_adata = None
                            train_copy = None

                        correlation_dict, r2_dict = assess.semi_rigorous_pearson_r2(true_copy, pred_dict_copy, true_except_ood_norm, data_params['primary_variable'], train_adata=train_copy)
                        with open(f'{path_to_save}/corr_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(correlation_dict, f)
                        with open(f'{path_to_save}/r2_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(r2_dict, f)
                        draw_boxplot(
                            correlation_dict, data, path_to_save, f"normalized_pearson_corr_{ood_primary}_{ood_modality}")
                        draw_boxplot(
                            r2_dict, data, path_to_save, f"normalized_r2_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'semi_rigorous_normalized_r2_brbic':
                        modality_adata = ood_modality_adata[ood_modality_adata.obs[data_params['modality_variable']] == ood_modality].copy()
                        primaries_modality = modality_adata.obs[data_params['primary_variable']].unique()
                        mean_list = []
                        for p in primaries_modality:
                            adata_primary = modality_adata[modality_adata.obs[data_params['primary_variable']] == p].copy()
                            X = adata_primary.X.toarray() if issparse(adata_primary.X) else adata_primary.X
                            mean_list.append(np.mean(X, axis=0))
                        mean_stack = np.vstack(mean_list)
                        c = np.mean(mean_stack, axis=0).reshape(1,-1)
                        print(f'shape of means: {c.shape}')
                        
                        # X = modality_adata.X.toarray() if issparse(modality_adata.X) else modality_adata.X
                        # c = np.mean(X, axis=0)
                        # c_std = np.std(X, axis=0)
                        # nonzero_ctrl = c != 0
                        # c = c[nonzero_ctrl]
                        # c_std = c_std[nonzero_ctrl]

                        # print(f'number of selected genes: {len(c)}')

                        # true_except_ood_norm = true_except_ood[:, nonzero_ctrl].copy()
                        true_except_ood_norm = true_except_ood.copy()
                        if issparse(true_except_ood_norm.X):
                            true_except_ood_norm.X = true_except_ood_norm.X.toarray()
                        # true_except_ood_norm.X  = (true_except_ood_norm.X - c) / c_std
                        true_except_ood_norm.X  = true_except_ood_norm.X - c

                        # true_copy = true_adata[:, nonzero_ctrl].copy()
                        true_copy = true_adata.copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        # true_copy.X = (true_copy.X - c) / c_std
                        true_copy.X = true_copy.X - c

                        pred_dict_copy = pred_dict.copy()
                        for p in pred_dict_copy.keys():
                            # pred_dict_copy[p] = pred_dict_copy[p][:, nonzero_ctrl].copy()
                            if issparse(pred_dict_copy[p].X):
                                pred_dict_copy[p].X = pred_dict_copy[p].X.toarray()
                            # pred_dict_copy[p].X = (pred_dict_copy[p].X - c) / c_std
                            pred_dict_copy[p].X = pred_dict_copy[p].X - c
                            
                        if ood_indices_test is not None:
                            train_adata =  ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] == ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                            train_adata = train_adata[~train_adata.obs.index.isin(ood_indices_test)].copy()
                            # train_copy = train_adata[:, nonzero_ctrl].copy()
                            train_copy = train_adata.copy()
                            if issparse(train_copy.X):
                                train_copy.X = train_copy.X.toarray()
                            # train_copy.X = (train_copy.X - c) / c_std
                            train_copy.X = train_copy.X - c
                        else:
                            train_adata = None
                            train_copy = None

                        correlation_dict, r2_dict = assess.semi_rigorous_pearson_r2(true_copy, pred_dict_copy, true_except_ood_norm, data_params['primary_variable'], train_adata=train_copy)
                        with open(f'{path_to_save}/corr_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(correlation_dict, f)
                        with open(f'{path_to_save}/r2_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(r2_dict, f)
                        draw_boxplot(
                            correlation_dict, data, path_to_save, f"normalized_pearson_corr_{ood_primary}_{ood_modality}")
                        draw_boxplot(
                            r2_dict, data, path_to_save, f"normalized_r2_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'semi_rigorous_pearson_delta':
                        X = ctrl_adata.X.toarray() if issparse(ctrl_adata.X) else ctrl_adata.X
                        c = np.mean(X, axis=0)

                        true_except_ood_norm = true_except_ood.copy()
                        if issparse(true_except_ood_norm.X):
                            true_except_ood_norm.X = true_except_ood_norm.X.toarray()
                        true_except_ood_norm.X  = true_except_ood_norm.X - c

                        true_copy = true_adata.copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        true_copy.X = true_copy.X - c

                        pred_dict_copy = pred_dict.copy()
                        for p in pred_dict_copy.keys():
                            if issparse(pred_dict_copy[p].X):
                                pred_dict_copy[p].X = pred_dict_copy[p].X.toarray()
                            pred_dict_copy[p].X = pred_dict_copy[p].X - c
                            
                        if ood_indices_test is not None:
                            train_adata =  ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] == ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                            train_adata = train_adata[~train_adata.obs.index.isin(ood_indices_test)].copy()
                            train_copy = train_adata.copy()
                            if issparse(train_copy.X):
                                train_copy.X = train_copy.X.toarray()
                            train_copy.X = train_copy.X - c
                        else:
                            train_adata = None
                            train_copy = None

                        correlation_dict, r2_dict = assess.semi_rigorous_pearson_r2(true_copy, pred_dict_copy, true_except_ood_norm, data_params['primary_variable'], train_adata=train_copy)
                        with open(f'{path_to_save}/corr_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(correlation_dict, f)
                        with open(f'{path_to_save}/r2_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(r2_dict, f)
                        draw_boxplot(
                            correlation_dict, data, path_to_save, f"normalized_pearson_corr_{ood_primary}_{ood_modality}")
                        draw_boxplot(
                            r2_dict, data, path_to_save, f"normalized_r2_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'semi_rigorous_nontrivial_normalized_r2':
                        c = np.mean(ctrl_adata.X, axis=0)
                        c_std = np.std(ctrl_adata.X, axis=0)
                        nonzero_ctrl = c != 0
                        c = c[nonzero_ctrl]
                        c_std = c_std[nonzero_ctrl]

                        true_except_ood_norm = true_except_ood[:, nonzero_ctrl].copy()
                        true_except_ood_norm.X  = (true_except_ood_norm.X - c) / c_std
                        true_copy = true_adata[:, nonzero_ctrl].copy()
                        true_copy.X = (true_copy.X - c) / c_std
                        if ood_indices_test is not None:
                            train_adata =  ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] == ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                            train_adata = train_adata[~train_adata.obs.index.isin(ood_indices_test)].copy()
                            train_copy = train_adata[:, nonzero_ctrl].copy()
                            train_copy.X = (train_copy.X - c) / c_std
                        else:
                            train_adata = None
                            train_copy = None

                        for p in pred_dict.keys():
                            pred_dict[p] = pred_dict[p][:, nonzero_ctrl].copy()
                            pred_dict[p].X = (pred_dict[p].X - c) / c_std
                            
                        r2_dict = assess.semi_rigorous_nontrivial_r2(true_copy, pred_dict, true_except_ood_norm, ctrl_true_adata, data_params, ood_modality, train_adata=train_copy)
                        with open(f'{path_to_save}/r2_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(r2_dict, f)
                        draw_boxplot(
                            r2_dict, data, path_to_save, f"normalized_r2_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'semi_rigorous_stepwise_r2':
                        if ood_indices_test is not None:
                            train_adata =  ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] == ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                            train_adata = train_adata[~train_adata.obs.index.isin(ood_indices_test)].copy()
                        else:
                            train_adata = None
                        corr_dict, r2_dict= assess.semi_rigorous_stepwise_r2(true_adata, pred_dict, true_except_ood, ctrl_adata, data_params['primary_variable'], train_adata=train_adata)
                        with open(f'{path_to_save}/auc_corr_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(corr_dict, f)
                        with open(f'{path_to_save}/auc_r2_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(r2_dict, f)
                        draw_boxplot(
                            corr_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}")
                        draw_boxplot(
                            r2_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'rigorous_wasserstein':
                        metric_dict = assess.rigorous_wasserstein(true_adata, pred_dict, true_except_ood, data_params['primary_variable'])
                        draw_boxplot(
                            metric_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}")
                        with open(f'{path_to_save}/wasserstein_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
        
                    elif evaluation_metric == 'rigorous_mmd':
                        metric_dict = assess.rigorous_mmd(true_adata, pred_dict, true_except_ood, data_params['primary_variable'])
                        with open(f'{path_to_save}/mmd_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        draw_boxplot(
                            metric_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'rigorous_Edistance':
                        metric_dict = assess.rigorous_Edistance(true_adata, pred_dict, true_except_ood, data_params['primary_variable'])
                        with open(f'{path_to_save}/Edistance_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        draw_boxplot(
                            metric_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}")
                            
                    elif evaluation_metric == 'rigorous_local_mmd':
                        k_file = os.path.join(f'../outputs/dependencies', data, ood_modality, 'k_local_mmd.pkl')
                        with open(k_file, 'rb') as f:
                            k_dict = pickle.load(f)
                        k_min = min(30,int(true_adata.shape[0]/2)-1)
                        optimal_k = int(k_dict[ood_primary]) if int(k_dict[ood_primary]) > k_min else k_min
                        print(f'k={optimal_k}')
                        metric_dict = assess.rigorous_local_mmd(true_adata, pred_dict, true_except_ood, data_params['primary_variable'], k=optimal_k)
                        with open(f'{path_to_save}/local_mmd_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        draw_boxplot(
                            metric_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'rigorous_local_Edistance':
                        k_file = os.path.join(f'../outputs/dependencies', data, ood_modality, 'k_local_Edistance.pkl')
                        with open(k_file, 'rb') as f:
                            k_dict = pickle.load(f)
                        k_min = min(30,int(true_adata.shape[0]/2)-1)
                        optimal_k = int(k_dict[ood_primary]) if int(k_dict[ood_primary]) > k_min else k_min
                        print(f'k={optimal_k}')
                        metric_dict = assess.rigorous_local_Edistance(true_adata, pred_dict, true_except_ood, data_params['primary_variable'], k=optimal_k)
                        with open(f'{path_to_save}/local_Edistance_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        draw_boxplot(
                            metric_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'semi_rigorous_mmd':
                        if ood_indices_test is not None:
                            train_adata =  ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] == ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                            train_adata = train_adata[~train_adata.obs.index.isin(ood_indices_test)].copy()
                        else:
                            train_adata = None
                        metric_dict = assess.semi_rigorous_mmd(true_adata, pred_dict, true_except_ood, data_params['primary_variable'], train_adata=train_adata)
                        with open(f'{path_to_save}/mmd_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        draw_boxplot(
                            metric_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'semi_rigorous_Edistance':
                        if ood_indices_test is not None:
                            train_adata =  ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] == ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                            train_adata = train_adata[~train_adata.obs.index.isin(ood_indices_test)].copy()
                        else:
                            train_adata = None
                        metric_dict = assess.semi_rigorous_Edistance(true_adata, pred_dict, true_except_ood, data_params['primary_variable'], train_adata=train_adata)
                        with open(f'{path_to_save}/Edistance_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        draw_boxplot(
                            metric_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}")
                    
                    elif evaluation_metric == 'semi_rigorous_local_mmd':
                        k_file = os.path.join(f'../outputs/dependencies', data, ood_modality, 'k_local_mmd.pkl')
                        with open(k_file, 'rb') as f:
                            k_dict = pickle.load(f)
                        k_min = min(30,int(true_adata.shape[0]/2)-1)
                        optimal_k = int(k_dict[ood_primary]) if int(k_dict[ood_primary]) > k_min else k_min
                        print(f'k={optimal_k}')
                        if ood_indices_test is not None:
                            train_adata =  ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] == ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                            train_adata = train_adata[~train_adata.obs.index.isin(ood_indices_test)].copy()
                        else:
                            train_adata = None
                        metric_dict = assess.semi_rigorous_local_mmd(true_adata, pred_dict, true_except_ood, data_params['primary_variable'], k=optimal_k, train_adata=train_adata)
                        with open(f'{path_to_save}/local_mmd_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        draw_boxplot(
                            metric_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}")
                        print('end of semi rigorous local mmd')

                    elif evaluation_metric == 'semi_rigorous_local_Edistance':
                        k_file = os.path.join(f'../outputs/dependencies', data, ood_modality, 'k_local_Edistance.pkl')
                        with open(k_file, 'rb') as f:
                            k_dict = pickle.load(f)
                        k_min = min(30,int(true_adata.shape[0]/2)-1)
                        optimal_k = int(k_dict[ood_primary]) if int(k_dict[ood_primary]) > k_min else k_min
                        print(f'k={optimal_k}')
                        if ood_indices_test is not None:
                            train_adata =  ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] == ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                            train_adata = train_adata[~train_adata.obs.index.isin(ood_indices_test)].copy()
                        else:
                            train_adata = None
                        metric_dict = assess.semi_rigorous_local_Edistance(true_adata, pred_dict, true_except_ood, data_params['primary_variable'], k=optimal_k, train_adata=train_adata)
                        with open(f'{path_to_save}/local_Edistance_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        draw_boxplot(
                            metric_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}")
                        print('end of semi rigorous local Edistance')

                    elif evaluation_metric == 'semi_rigorous_knn_loss':
                        if ood_indices_test is not None:
                            train_adata =  ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] == ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                            train_adata = train_adata[~train_adata.obs.index.isin(ood_indices_test)].copy()
                        else:
                            train_adata = None
                        metric_dict = assess.semi_rigorous_knn_loss(true_adata, pred_dict, true_except_ood, data_params['primary_variable'], train_adata=train_adata)
                        with open(f'{path_to_save}/knn_loss_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        draw_boxplot(
                            metric_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}")
                        print('end of semi rigorous knn_loss')
        
                    elif evaluation_metric == 'rigorous_store_deg_df':
                        deg_df_dict = assess.rigorous_store_deg_df(true_adata, ctrl_adata, pred_dict, true_except_ood)
                        os.makedirs(f"{path_to_save}/{ood_modality}/{ood_primary}", exist_ok=True)
                        with open(f'{path_to_save}/{ood_modality}/{ood_primary}/deg_df.pkl', 'wb') as f:
                            pickle.dump(deg_df_dict, f)
                    
                    elif evaluation_metric == 'rigorous_deg_f1':
                        f1_down_dict, f1_up_dict, precision_down_dict, precision_up_dict, recall_down_dict, recall_up_dict, typeI_down_dict, typeI_up_dict = \
                            assess.rigorous_deg_f1(true_adata, ctrl_adata, pred_dict, true_except_ood, data_params['primary_variable'], ood_primary)
                        os.makedirs(f"{path_to_save}/f1", exist_ok=True)
                        os.makedirs(f"{path_to_save}/precision", exist_ok=True)
                        os.makedirs(f"{path_to_save}/recall", exist_ok=True)
                        os.makedirs(f"{path_to_save}/typeI", exist_ok=True)
        
                        with open(f'{path_to_save}/f1/f1_down_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(f1_down_dict, f)        
                        with open(f'{path_to_save}/f1/f1_up_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(f1_up_dict, f)   
                        draw_boxplot(
                            f1_down_dict, data, f"{path_to_save}/f1", f"down_{ood_primary}_{ood_modality}", title="f1 score")
                        draw_boxplot(
                            f1_up_dict, data, f"{path_to_save}/f1", f"up_{ood_primary}_{ood_modality}", title="f1 score")
        
                        with open(f'{path_to_save}/precision/precision_down_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(precision_down_dict, f)        
                        with open(f'{path_to_save}/precision/precision_up_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(precision_up_dict, f)    
                        draw_boxplot(
                            precision_down_dict, data, f"{path_to_save}/precision", f"down_{ood_primary}_{ood_modality}", title="precision")
                        draw_boxplot(
                            precision_up_dict, data, f"{path_to_save}/precision", f"up_{ood_primary}_{ood_modality}", title="precision")
        
                        with open(f'{path_to_save}/recall/recall_down_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(recall_down_dict, f)        
                        with open(f'{path_to_save}/recall/recall_up_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(recall_up_dict, f)   
                        draw_boxplot(
                            recall_down_dict, data, f"{path_to_save}/recall", f"down_{ood_primary}_{ood_modality}", title="recall")
                        draw_boxplot(
                            recall_up_dict, data, f"{path_to_save}/recall", f"up_{ood_primary}_{ood_modality}", title="recall")

                        with open(f'{path_to_save}/typeI/typeI_down_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(typeI_down_dict, f)        
                        with open(f'{path_to_save}/typeI/typeI_up_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(typeI_up_dict, f)     
                        draw_boxplot(
                            typeI_down_dict, data, f"{path_to_save}/typeI", f"down_{ood_primary}_{ood_modality}", title="typeI error")
                        draw_boxplot(
                            typeI_up_dict, data, f"{path_to_save}/typeI", f"up_{ood_primary}_{ood_modality}", title="typeI error")

                    elif evaluation_metric == 'rigorous_logfc_correlation':
                        metric_dict = assess.rigorous_logfc_correlation(true_adata, pred_dict, ctrl_adata, true_except_ood, data_params['primary_variable'])
                        with open(f'{path_to_save}/logfc_correlation_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)                    
                        draw_boxplot(
                            metric_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}", title=f"setting {setting}")

                    elif evaluation_metric == 'semi_rigorous_logfc_correlation':
                        if ood_indices_test is not None:
                            train_adata =  ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] == ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                            train_adata = train_adata[~train_adata.obs.index.isin(ood_indices_test)].copy()
                        else:
                            train_adata = None
                        metric_dict = assess.semi_rigorous_logfc_correlation(
                            true_adata, pred_dict, ctrl_adata, true_except_ood, data_params['primary_variable'], train_adata)
                        with open(f'{path_to_save}/logfc_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        draw_boxplot(
                            metric_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}")

                    elif evaluation_metric == 'semi_rigorous_nontrivial_logfc_correlation':
                        if ood_indices_test is not None:
                            train_adata =  ood_modality_adata[(ood_modality_adata.obs[data_params['primary_variable']] == ood_primary) & 
                                                (ood_modality_adata.obs[data_params['modality_variable']] == ood_modality)].copy()
                            train_adata = train_adata[~train_adata.obs.index.isin(ood_indices_test)].copy()
                        else:
                            train_adata = None
                        metric_dict = assess.semi_rigorous_nontrivial_logfc_correlation(
                            true_adata, pred_dict, ctrl_adata, ctrl_true_adata, true_except_ood, data_params, ood_modality, train_adata)
                        with open(f'{path_to_save}/nontrivial_logfc_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        draw_boxplot(
                            metric_dict, data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}")
        
                    elif evaluation_metric == "noise_effect_mixing_index":
                        max_size = 15000
                        if ood_modality_adata.n_obs > max_size: 
                            chosen_indices = np.random.choice(ood_modality_adata.n_obs, size=max_size, replace=False)
                            all_adata = ood_modality_adata[chosen_indices].copy()
                        else:
                            all_adata = ood_modality_adata.copy()
                        mixing_index_dict = assess.noise_effect_mixing_index(true_adata, true_except_ood, all_adata=all_adata, n_repeats=5, percent_interval=2, method='random_all', mode='not_nested')
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/mixing_noise_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(mixing_index_dict, f)
                        true_pos_lineplot(mixing_index_dict, ood_primary, 'mixing-index', data, path_to_save_modality, 
                                        "noise percentage", "mixing_index, noise by random-all sampling", legend_title='', y_min=0, y_max=1)
        
                    elif evaluation_metric == "noise_effect_mixing_index_shuffle":
                        max_size = 15000
                        if ood_modality_adata.n_obs > max_size: 
                            chosen_indices = np.random.choice(ood_modality_adata.n_obs, size=max_size, replace=False)
                            all_adata = ood_modality_adata[chosen_indices].copy()
                        else:
                            all_adata = ood_modality_adata.copy()
                        mixing_index_dict = assess.noise_effect_mixing_index(true_adata, true_except_ood, all_adata=all_adata, n_repeats=5, percent_interval=2, method='shuffle')
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/mixing_noise_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(mixing_index_dict, f)
                        # with open(f'{path_to_save_modality}/mixing_noise_{ood_primary}.pkl', 'rb') as f:
                        #     mixing_index_dict = pickle.load(f)
                        true_pos_lineplot(mixing_index_dict, ood_primary, 'mixing-index', data, path_to_save_modality, 
                                        "noise percentage", "mixing index", title=f"{data[0].upper()+data[1:]} - {ood_primary}\nshuffling noise", 
                                        legend_title='', y_min=0, y_max=1)

                    elif evaluation_metric == "noise_effect_mixing_index_shuffle_not_nested":
                        max_size = 15000
                        if ood_modality_adata.n_obs > max_size: 
                            chosen_indices = np.random.choice(ood_modality_adata.n_obs, size=max_size, replace=False)
                            all_adata = ood_modality_adata[chosen_indices].copy()
                        else:
                            all_adata = ood_modality_adata.copy()
                        mixing_index_dict = assess.noise_effect_mixing_index(true_adata, true_except_ood, all_adata=all_adata, n_repeats=5, percent_interval=2, method='shuffle', mode='not_nested')
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/mixing_noise_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(mixing_index_dict, f)
                        # with open(f'{path_to_save_modality}/mixing_noise_{ood_primary}.pkl', 'rb') as f:
                        #     mixing_index_dict = pickle.load(f)
                        true_pos_lineplot(mixing_index_dict, ood_primary, 'mixing-index', data, path_to_save_modality, 
                                        "noise percentage", "mixing index", title=f"{data[0].upper()+data[1:]} - {ood_primary}\nshuffling noise, not nested", 
                                        legend_title='', y_min=0, y_max=1)

                    elif evaluation_metric == "noise_effect_mmd_shuffle":
                        metric_dict = assess.noise_effect_mmd(true_adata, true_except_ood, n_repeats=5, percent_interval=2, method='shuffle')
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/mmd_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        true_pos_lineplot(metric_dict, ood_primary, 'mmd', data, path_to_save_modality, 
                                        "noise percentage", "mmd, noise by shuffling", legend_title='')

                    elif evaluation_metric == "noise_effect_local_mmd":
                        k_file = os.path.join(f'../outputs/dependencies', data, ood_modality, 'k_local_mmd.pkl')
                        with open(k_file, 'rb') as f:
                            k_dict = pickle.load(f)
                        k_min = min(30,int(true_adata.shape[0]/2)-1)
                        optimal_k = int(k_dict[ood_primary]) if int(k_dict[ood_primary]) > k_min else k_min
                        print(f'k={optimal_k}')
                        metric_dict = assess.noise_effect_local_mmd(true_adata, true_except_ood, n_repeats=5, percent_interval=2, method='random_all', mode='not_nested', k=optimal_k)
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/local_mmd_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        true_pos_lineplot(metric_dict, ood_primary, 'local_mmd', data, path_to_save_modality, 
                                        "noise percentage", f"local mmd, noise by random_all sampling, k={optimal_k}", legend_title='')

                    elif evaluation_metric == "noise_effect_local_Edistance":
                        k_file = os.path.join(f'../outputs/dependencies', data, ood_modality, 'k_local_Edistance.pkl')
                        with open(k_file, 'rb') as f:
                            k_dict = pickle.load(f)
                        k_min = min(30,int(true_adata.shape[0]/2)-1)
                        optimal_k = int(k_dict[ood_primary]) if int(k_dict[ood_primary]) > k_min else k_min
                        print(f'k={optimal_k}')
                        metric_dict = assess.noise_effect_local_Edistance(true_adata, true_except_ood, n_repeats=5, percent_interval=2, method='random_all', mode='not_nested', k=optimal_k)
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/local_Edistance_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        true_pos_lineplot(metric_dict, ood_primary, 'local_Edistance', data, path_to_save_modality, 
                                        "noise percentage", f"local Edistance, noise by random_all sampling, k={optimal_k}", legend_title='')

                    elif evaluation_metric == "noise_effect_local_mmd_shuffle":
                        k_file = os.path.join(f'../outputs/dependencies', data, ood_modality, 'k_local_mmd.pkl')
                        with open(k_file, 'rb') as f:
                            k_dict = pickle.load(f)
                        k_min = min(30,int(true_adata.shape[0]/2)-1)
                        optimal_k = int(k_dict[ood_primary]) if int(k_dict[ood_primary]) > k_min else k_min
                        print(f'k={optimal_k}')
                        metric_dict = assess.noise_effect_local_mmd(true_adata, true_except_ood, n_repeats=5, percent_interval=2, method='shuffle', k=optimal_k)
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/local_mmd_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        true_pos_lineplot(metric_dict, ood_primary, 'local_mmd', data, path_to_save_modality, 
                                        "noise percentage", f"local mmd, noise by shuffling, k={optimal_k}", legend_title='')

                    elif evaluation_metric == "noise_effect_local_Edistance_shuffle":
                        k_file = os.path.join(f'../outputs/dependencies', data, ood_modality, 'k_local_Edistance.pkl')
                        with open(k_file, 'rb') as f:
                            k_dict = pickle.load(f)
                        k_min = min(30,int(true_adata.shape[0]/2)-1)
                        optimal_k = int(k_dict[ood_primary]) if int(k_dict[ood_primary]) > k_min else k_min
                        print(f'k={optimal_k}')
                        metric_dict = assess.noise_effect_local_Edistance(true_adata, true_except_ood, n_repeats=5, percent_interval=2, method='shuffle', k=optimal_k)
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/local_Edistance_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        # with open(f'{path_to_save_modality}/local_Edistance_{ood_primary}.pkl', 'rb') as f:
                        #     metric_dict = pickle.load(f)
                        true_pos_lineplot(metric_dict, ood_primary, 'local_Edistanced', data, path_to_save_modality, 
                                        "noise percentage", f"local E-distance", 
                                        title=f"{data[0].upper()+data[1:]} - {ood_primary}, k={optimal_k}\nshuffling noise", legend_title='')
                    
                    elif evaluation_metric == "noise_effect_local_Edistance_shuffle_not_nested":
                        k_file = os.path.join(f'../outputs/dependencies', data, ood_modality, 'k_local_Edistance.pkl')
                        with open(k_file, 'rb') as f:
                            k_dict = pickle.load(f)
                        k_min = min(30,int(true_adata.shape[0]/2)-1)
                        optimal_k = int(k_dict[ood_primary]) if int(k_dict[ood_primary]) > k_min else k_min
                        print(f'k={optimal_k}')
                        metric_dict = assess.noise_effect_local_Edistance(true_adata, true_except_ood, n_repeats=5, percent_interval=2, method='shuffle', mode='not_nested', k=optimal_k)
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/local_Edistance_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        # with open(f'{path_to_save_modality}/local_Edistance_{ood_primary}.pkl', 'rb') as f:
                        #     metric_dict = pickle.load(f)
                        true_pos_lineplot(metric_dict, ood_primary, 'local_Edistanced', data, path_to_save_modality, 
                                        "noise percentage", f"local E-distance", 
                                        title=f"local E-distance vs shuffling noise, not nested\n{data[0].upper()+data[1:]} - {ood_primary}, k={optimal_k}", legend_title='')

                    elif evaluation_metric == "noise_effect_Edistance_shuffle":
                        metric_dict = assess.noise_effect_Edistance(true_adata, true_except_ood, n_repeats=5, percent_interval=2, method='shuffle')
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/Edistance_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        # with open(f'{path_to_save_modality}/Edistance_{ood_primary}.pkl', 'rb') as f:
                        #     metric_dict = pickle.load(f)
                        true_pos_lineplot(metric_dict, ood_primary, 'E-distance', data, path_to_save_modality, 
                                        "noise percentage", "E-distance", 
                                        title=f"{data[0].upper()+data[1:]} - {ood_primary}\nshuffling noise", legend_title='')
                    
                    elif evaluation_metric == "noise_effect_Edistance_shuffle_not_nested":
                        metric_dict = assess.noise_effect_Edistance(true_adata, true_except_ood, n_repeats=5, percent_interval=2, method='shuffle', mode='not_nested')
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/Edistance_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        # with open(f'{path_to_save_modality}/Edistance_{ood_primary}.pkl', 'rb') as f:
                        #     metric_dict = pickle.load(f)
                        true_pos_lineplot(metric_dict, ood_primary, 'E-distance', data, path_to_save_modality, 
                                        "noise percentage", "E-distance", 
                                        title=f"E-distance vs shuffling noise, not nested\n{data[0].upper()+data[1:]} - {ood_primary}", legend_title='')

                    elif evaluation_metric == "noise_effect_knn_loss_shuffle":
                        metric_dict = assess.noise_effect_knn_loss(true_adata, true_except_ood, n_repeats=5, percent_interval=2, method='shuffle')
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/knn_loss_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        true_pos_lineplot(metric_dict, ood_primary, 'knn_loss', data, path_to_save_modality, 
                                        "noise percentage", "knn loss, noise by shuffling", legend_title='')
                    
                    elif evaluation_metric == "noise_effect_mmd_shuffle_not_nested":
                        metric_dict = assess.noise_effect_mmd(true_adata, true_except_ood, n_repeats=10, percent_interval=2, method='shuffle', mode='not-nested')
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/mmd_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        true_pos_lineplot(metric_dict, ood_primary, 'mmd', data, path_to_save_modality, 
                                        "noise percentage", "mmd, noise by shuffling, not nested", legend_title='')

                    elif evaluation_metric == "noise_effect_wasserstein_shuffle":
                        metric_dict = assess.noise_effect_wasserstein(true_adata, true_except_ood, n_repeats=5, percent_interval=2, method='shuffle')
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/wasserstein_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        # with open(f'{path_to_save_modality}/wasserstein_{ood_primary}.pkl', 'rb') as f:
                        #     metric_dict = pickle.load(f)
                        true_pos_lineplot(metric_dict, ood_primary, 'wasserstein', data, path_to_save_modality, 
                                        "noise percentage", "Wasserstein distance",
                                        title=f"{data[0].upper()+data[1:]} - {ood_primary}\nshuffling noise", legend_title='')
                    
                    elif evaluation_metric == "noise_effect_wasserstein_shuffle_not_nested":
                        metric_dict = assess.noise_effect_wasserstein(true_adata, true_except_ood, n_repeats=5, percent_interval=2, method='shuffle', mode='not_nested')
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/wasserstein_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        # with open(f'{path_to_save_modality}/wasserstein_{ood_primary}.pkl', 'rb') as f:
                        #     metric_dict = pickle.load(f)
                        true_pos_lineplot(metric_dict, ood_primary, 'wasserstein', data, path_to_save_modality, 
                                        "noise percentage", "Wasserstein distance",
                                        title=f"Wasserstein vs shuffling noise, not nested\n{data[0].upper()+data[1:]} - {ood_primary}", legend_title='')
        
                    elif evaluation_metric == "noise_effect_spearman":
                        correlation_dict = assess.noise_effect_spearman(true_adata, true_except_ood, n_repeats=10, percent_interval=1, method='random_all', mode='not_nested')
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/spearman_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(correlation_dict, f)
                        true_pos_lineplot(correlation_dict, ood_primary, 'spearman correlation', data, path_to_save_modality, 
                                        "noise percentage", "spearman correlation", legend_title='', y_min=0, y_max=1)
        
                    elif evaluation_metric == 'noise_effect_normalized_spearman':
                        means = np.mean(ood_modality_adata.X, axis=0)
                        adata_copy = ood_modality_adata.copy()
                        adata_copy.X = adata_copy.X - means   
                        true_except_ood = adata_copy[(adata_copy.obs[data_params['primary_variable']] != ood_primary) & 
                                                (adata_copy.obs[data_params['modality_variable']] == ood_modality)].copy()
                        true_adata.X = true_adata.X - means
                            
                        correlation_dict = assess.noise_effect_spearman(true_adata, true_except_ood, n_repeats=10, percent_interval=1, method='random_all', mode='not_nested')
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/normalized_spearman_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(correlation_dict, f)
                        true_pos_lineplot(correlation_dict, ood_primary, 'normalized spearman correlation', data, path_to_save_modality, 
                                        "noise percentage", "normalized spearman correlation", legend_title='', y_min=0, y_max=1)
        
                    elif evaluation_metric == "noise_effect_wasserstein":
                        metric_dict = assess.noise_effect_wasserstein(true_adata, true_except_ood, n_repeats=5, percent_interval=2, method='random_all', mode='not_nested')
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/wasserstein_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        true_pos_lineplot(metric_dict, ood_primary, 'wasserstein distance', data, path_to_save_modality, 
                                        "noise percentage", "wasserstein, noise by random-all sampling", legend_title='')
        
                    elif evaluation_metric == "noise_effect_mmd":
                        metric_dict = assess.noise_effect_mmd(true_adata, true_except_ood, n_repeats=5, percent_interval=2, method='random_all', mode='not_nested')
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/mmd_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        true_pos_lineplot(metric_dict, ood_primary, 'MMD distance', data, path_to_save_modality, 
                                        "noise percentage", "MMD distance", legend_title='')

                    elif evaluation_metric == "noise_effect_Edistance":
                        metric_dict = assess.noise_effect_Edistance(true_adata, true_except_ood, n_repeats=5, percent_interval=2, method='random_all', mode='not_nested')
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        with open(f'{path_to_save_modality}/Edistance_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(metric_dict, f)
                        true_pos_lineplot(metric_dict, ood_primary, 'E distance', data, path_to_save_modality, 
                                        "noise percentage", "E distance", legend_title='')
        
                    elif evaluation_metric == 'noise_effect_f1':
                        f1_down_dict, f1_up_dict, precision_down_dict, precision_up_dict, recall_down_dict, recall_up_dict = \
                            assess.noise_effect_f1(true_adata, ctrl_adata, true_except_ood, ood_primary, n_repeats=5, percent_interval=1, method='random_all', mode='not_nested')
                        path_to_save_modality = os.path.join(path_to_save, ood_modality)
                        os.makedirs(path_to_save_modality, exist_ok=True)
                        os.makedirs(f"{path_to_save_modality}/f1", exist_ok=True)
                        os.makedirs(f"{path_to_save_modality}/precision", exist_ok=True)
                        os.makedirs(f"{path_to_save_modality}/recall", exist_ok=True)
        
                        with open(f'{path_to_save_modality}/f1/f1_down_dict_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(f1_down_dict, f)        
                        with open(f'{path_to_save_modality}/f1/f1_up_dict_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(f1_up_dict, f)   
                        true_pos_lineplot(f1_down_dict, ood_primary, 'f1 down', data, f"{path_to_save_modality}/f1", 
                                        "noise percentage", "f1 down", legend_title='', y_min=0, y_max=1)
                        true_pos_lineplot(f1_up_dict, ood_primary, 'f1 up', data, f"{path_to_save_modality}/f1", 
                                        "noise percentage", "f1 up", legend_title='', y_min=0, y_max=1)
        
                        with open(f'{path_to_save_modality}/precision/precision_down_dict_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(precision_down_dict, f)        
                        with open(f'{path_to_save_modality}/precision/precision_up_dict_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(precision_up_dict, f)    
                        true_pos_lineplot(precision_down_dict, ood_primary, 'precision down', data, f"{path_to_save_modality}/precision", 
                                        "noise percentage", "precision down", legend_title='', y_min=0, y_max=1)
                        true_pos_lineplot(precision_up_dict, ood_primary, 'precision up', data, f"{path_to_save_modality}/precision", 
                                        "noise percentage", "precision up", legend_title='', y_min=0, y_max=1)
        
                        with open(f'{path_to_save_modality}/recall/recall_down_dict_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(recall_down_dict, f)        
                        with open(f'{path_to_save_modality}/recall/recall_up_dict_{ood_primary}.pkl', 'wb') as f:
                            pickle.dump(recall_up_dict, f)
                        true_pos_lineplot(recall_down_dict, ood_primary, 'recall down', data, f"{path_to_save_modality}/recall", 
                                        "noise percentage", "recall down", legend_title='', y_min=0, y_max=1)
                        true_pos_lineplot(recall_up_dict, ood_primary, 'recall up', data, f"{path_to_save_modality}/recall", 
                                        "noise percentage", "recall up", legend_title='', y_min=0, y_max=1)
                    
                    elif evaluation_metric == 'rigorous_subset_spearman':
                        if true_logfc_df is None:
                            true_logfc_df = seurat_deg(
                                ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=True)
                            true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        non_trivial_dict, trivial_dict, non_significant_dict, num_genes_list = \
                            assess.rigorous_subset_spearman(true_adata, ctrl_adata, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df)
                        with open(f'{path_to_save}/trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(trivial_dict, f)
                        with open(f'{path_to_save}/non_trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_trivial_dict, f)
                        with open(f'{path_to_save}/non_significant_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_significant_dict, f)
                        plots.draw_boxplot_multiple(
                            [non_trivial_dict, trivial_dict, non_significant_dict], data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}",\
                                [f"{num_genes_list[0]} non-trivial genes", f"{num_genes_list[1]} trivial-genes", f"{num_genes_list[2]} non-significant genes (significance threshold: 0.05)"])
                        
                    elif evaluation_metric == 'rigorous_subset_logfc_correlation':
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=False)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        true_copy = true_adata.copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        ctrl_copy = ctrl_adata.copy()
                        if issparse(ctrl_copy.X):
                            ctrl_copy.X = ctrl_copy.X.toarray()
                        non_trivial_dict, trivial_dict, non_significant_dict, num_genes_list = \
                            assess.rigorous_subset_logfc_correlation(true_copy, ctrl_copy, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df)
                        # non_trivial_dict,num_genes_list = \
                        #     assess.rigorous_subset_logfc_correlation(true_adata, ctrl_adata, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df)
                        with open(f'{path_to_save}/trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(trivial_dict, f)
                        with open(f'{path_to_save}/non_trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_trivial_dict, f)
                        with open(f'{path_to_save}/non_significant_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_significant_dict, f)
                        with open(f'{path_to_save}/num_genes_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump({'non_trivial':num_genes_list[0], 'trivial':num_genes_list[1], 'non_significant':num_genes_list[2]}, f)
                        plots.draw_boxplot_multiple(
                            [non_trivial_dict, trivial_dict, non_significant_dict], data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}",\
                                [f"{num_genes_list[0]} non-trivial genes", f"{num_genes_list[1]} trivial-genes", f"{num_genes_list[2]} non-significant genes (significance threshold: 0.05)"])
                        # plots.draw_boxplot_multiple(
                        #     [non_trivial_dict], data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}",\
                        #         [f"{num_genes_list[0]} non-trivial genes"])

                    elif evaluation_metric == 'rigorous_subset_pearson_r2':
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=True)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        true_copy = true_adata.copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        ctrl_copy = ctrl_adata.copy()
                        if issparse(ctrl_copy.X):
                            ctrl_copy.X = ctrl_copy.X.toarray()
                        non_trivial_dict, trivial_dict, non_significant_dict, num_genes_list = \
                            assess.rigorous_subset_pearson_r2(true_copy, ctrl_copy, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df)
                        with open(f'{path_to_save}/trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(trivial_dict, f)
                        with open(f'{path_to_save}/non_trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_trivial_dict, f)
                        with open(f'{path_to_save}/non_significant_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_significant_dict, f)
                        with open(f'{path_to_save}/num_genes_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump({'non_trivial':num_genes_list[0], 'trivial':num_genes_list[1], 'non_significant':num_genes_list[2]}, f)
                        plots.draw_boxplot_multiple(
                            [non_trivial_dict[0], trivial_dict[0], non_significant_dict[0]], data, path_to_save, f"pearson_{ood_primary}_{ood_modality}",\
                                [f"{num_genes_list[0]} non-trivial genes", f"{num_genes_list[1]} trivial-genes", f"{num_genes_list[2]} non-significant genes (significance threshold: 0.05)"])
                        plots.draw_boxplot_multiple(
                            [non_trivial_dict[1], trivial_dict[1], non_significant_dict[1]], data, path_to_save, f"r2_{ood_primary}_{ood_modality}",\
                                [f"{num_genes_list[0]} non-trivial genes", f"{num_genes_list[1]} trivial-genes", f"{num_genes_list[2]} non-significant genes (significance threshold: 0.05)"])

                    elif evaluation_metric == 'rigorous_subset_pearson_delta':
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=True)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        X = ctrl_adata.X.toarray() if issparse(ctrl_adata.X) else ctrl_adata.X
                        c = np.mean(X, axis=0)

                        true_except_ood_norm = true_except_ood.copy()
                        if issparse(true_except_ood_norm.X):
                            true_except_ood_norm.X = true_except_ood_norm.X.toarray()
                        true_except_ood_norm.X  = true_except_ood_norm.X - c

                        true_copy = true_adata.copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        original_true = true_copy.copy()
                        true_copy.X = true_copy.X - c

                        ctrl_copy = ctrl_adata.copy()
                        if issparse(ctrl_copy.X):
                            ctrl_copy.X = ctrl_copy.X.toarray()

                        pred_dict_copy = pred_dict.copy()
                        for p in pred_dict_copy.keys():
                            if issparse(pred_dict_copy[p].X):
                                pred_dict_copy[p].X = pred_dict_copy[p].X.toarray()
                            pred_dict_copy[p].X = pred_dict_copy[p].X - c
                        
                        non_trivial_dict, trivial_dict, non_significant_dict, num_genes_list = \
                            assess.rigorous_subset_pearson_r2(true_copy, ctrl_copy, pred_dict_copy, true_except_ood_norm, data_params['primary_variable'], true_logfc_df, original_true=original_true)
                        with open(f'{path_to_save}/trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(trivial_dict, f)
                        with open(f'{path_to_save}/non_trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_trivial_dict, f)
                        with open(f'{path_to_save}/non_significant_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_significant_dict, f)
                        with open(f'{path_to_save}/num_genes_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump({'non_trivial':num_genes_list[0], 'trivial':num_genes_list[1], 'non_significant':num_genes_list[2]}, f)
                        plots.draw_boxplot_multiple(
                            [non_trivial_dict[0], trivial_dict[0], non_significant_dict[0]], data, path_to_save, f"pearson_delta_{ood_primary}_{ood_modality}",\
                                [f"{num_genes_list[0]} non-trivial genes", f"{num_genes_list[1]} trivial-genes", f"{num_genes_list[2]} non-significant genes (significance threshold: 0.05)"])
                        plots.draw_boxplot_multiple(
                            [non_trivial_dict[1], trivial_dict[1], non_significant_dict[1]], data, path_to_save, f"r2_{ood_primary}_{ood_modality}",\
                                [f"{num_genes_list[0]} non-trivial genes", f"{num_genes_list[1]} trivial-genes", f"{num_genes_list[2]} non-significant genes (significance threshold: 0.05)"])

                    elif evaluation_metric == 'rigorous_subset_mmd':
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=True)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        true_copy = true_adata.copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        ctrl_copy = ctrl_adata.copy()
                        if issparse(ctrl_copy.X):
                            ctrl_copy.X = ctrl_copy.X.toarray()
                        non_trivial_dict, trivial_dict, non_significant_dict, num_genes_list = \
                            assess.rigorous_subset_mmd(true_copy, ctrl_copy, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df)
                        with open(f'{path_to_save}/trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(trivial_dict, f)
                        with open(f'{path_to_save}/non_trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_trivial_dict, f)
                        with open(f'{path_to_save}/non_significant_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_significant_dict, f)
                        with open(f'{path_to_save}/num_genes_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump({'non_trivial':num_genes_list[0], 'trivial':num_genes_list[1], 'non_significant':num_genes_list[2]}, f)
                        subtitles = [f"{num_genes_list[0]} non-trivial genes", f"{num_genes_list[1]} trivial-genes", f"{num_genes_list[2]} non-significant genes (significance threshold: 0.05)"]
                        plots.draw_boxplot_multiple(
                            [non_trivial_dict, trivial_dict, non_significant_dict], data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}",\
                                subtitles)

                    elif evaluation_metric == 'rigorous_subset_Edistance':
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=True)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        true_copy = true_adata.copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        ctrl_copy = ctrl_adata.copy()
                        if issparse(ctrl_copy.X):
                            ctrl_copy.X = ctrl_copy.X.toarray()
                        non_trivial_dict, trivial_dict, non_significant_dict, num_genes_list = \
                            assess.rigorous_subset_Edistance(true_copy, ctrl_copy, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df)
                        with open(f'{path_to_save}/trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(trivial_dict, f)
                        with open(f'{path_to_save}/non_trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_trivial_dict, f)
                        with open(f'{path_to_save}/non_significant_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_significant_dict, f)
                        with open(f'{path_to_save}/num_genes_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump({'non_trivial':num_genes_list[0], 'trivial':num_genes_list[1], 'non_significant':num_genes_list[2]}, f)
                        subtitles = [f"{num_genes_list[0]} non-trivial genes", f"{num_genes_list[1]} trivial-genes", f"{num_genes_list[2]} non-significant genes (significance threshold: 0.05)"]
                        plots.draw_boxplot_multiple(
                            [non_trivial_dict, trivial_dict, non_significant_dict], data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}",\
                                subtitles)

                    elif evaluation_metric == 'rigorous_subset_knn_loss':
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=True)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        non_trivial_dict, trivial_dict, non_significant_dict, num_genes_list = \
                            assess.rigorous_subset_knn_loss(true_adata, ctrl_adata, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df)
                        with open(f'{path_to_save}/trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(trivial_dict, f)
                        with open(f'{path_to_save}/non_trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_trivial_dict, f)
                        with open(f'{path_to_save}/non_significant_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_significant_dict, f)
                        subtitles = [f"{num_genes_list[0]} non-trivial genes", f"{num_genes_list[1]} trivial-genes", f"{num_genes_list[2]} non-significant genes (significance threshold: 0.05)"]
                        plots.draw_boxplot_multiple(
                            [non_trivial_dict, trivial_dict, non_significant_dict], data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}",\
                                subtitles)

                    elif evaluation_metric == 'rigorous_subset_wasserstein':
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=True)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        true_copy = true_adata.copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        ctrl_copy = ctrl_adata.copy()
                        if issparse(ctrl_copy.X):
                            ctrl_copy.X = ctrl_copy.X.toarray()
                        non_trivial_dict, trivial_dict, non_significant_dict, num_genes_list = \
                            assess.rigorous_subset_wasserstein(true_copy, ctrl_copy, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df)
                        with open(f'{path_to_save}/trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(trivial_dict, f)
                        with open(f'{path_to_save}/non_trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_trivial_dict, f)
                        with open(f'{path_to_save}/non_significant_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_significant_dict, f)
                        with open(f'{path_to_save}/num_genes_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump({'non_trivial':num_genes_list[0], 'trivial':num_genes_list[1], 'non_significant':num_genes_list[2]}, f)
                        plots.draw_boxplot_multiple(
                            [non_trivial_dict, trivial_dict, non_significant_dict], data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}",\
                                [f"{num_genes_list[0]} non-trivial genes", f"{num_genes_list[1]} trivial-genes", f"{num_genes_list[2]} non-significant genes (significance threshold: 0.05)"])
                    
                    elif evaluation_metric == 'rigorous_subset_sinkhorn_mahalanobis':
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=True)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        # non_trivial_dict, trivial_dict, non_significant_dict, num_genes_list = \
                        non_trivial_dict, non_significant_dict, num_genes_list = \
                            assess.rigorous_subset_sinkhorn_mahalanobis(true_adata, ctrl_adata, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df)
                        # with open(f'{path_to_save}/trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                        #     pickle.dump(trivial_dict, f)
                        with open(f'{path_to_save}/non_trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_trivial_dict, f)
                        with open(f'{path_to_save}/non_significant_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_significant_dict, f)
                        # plots.draw_boxplot_multiple(
                        #     [non_trivial_dict, trivial_dict, non_significant_dict], data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}",\
                        #         [f"{num_genes_list[0]} non-trivial genes", f"{num_genes_list[1]} trivial-genes", f"{num_genes_list[2]} non-significant genes (significance threshold: 0.05)"])
                        plots.draw_boxplot_multiple(
                            [non_trivial_dict, non_significant_dict], data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}",\
                                [f"{num_genes_list[0]} non-trivial genes", f"{num_genes_list[2]} non-significant genes (significance threshold: 0.05)"])

                    elif evaluation_metric == 'rigorous_subset_mixing_index_kmeans':
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=True)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        non_trivial_dict, trivial_dict, non_significant_dict, num_genes_list = \
                            assess.rigorous_subset_mixing_index_kmeans(true_adata, ctrl_adata, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df)
                        with open(f'{path_to_save}/trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(trivial_dict, f)
                        with open(f'{path_to_save}/non_trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_trivial_dict, f)
                        with open(f'{path_to_save}/non_significant_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_significant_dict, f)
                        plots.draw_boxplot_multiple(
                            [non_trivial_dict, trivial_dict, non_significant_dict], data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}",\
                                [f"{num_genes_list[0]} non-trivial genes", f"{num_genes_list[1]} trivial-genes", f"{num_genes_list[2]} non-significant genes (significance threshold: 0.05)"])

                    elif evaluation_metric == 'rigorous_subset_mixing_index_seurat':
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=True)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        max_size = 15000
                        if ood_modality_adata.n_obs > max_size: 
                            chosen_indices = np.random.choice(ood_modality_adata.n_obs, size=max_size, replace=False)
                            all_adata = ood_modality_adata[chosen_indices].copy()
                        else:
                            all_adata = ood_modality_adata.copy()
                        true_copy = true_adata.copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        ctrl_copy = ctrl_adata.copy()
                        if issparse(ctrl_copy.X):
                            ctrl_copy.X = ctrl_copy.X.toarray()
                        non_trivial_dict, trivial_dict, non_significant_dict, num_genes_list = \
                            assess.rigorous_subset_mixing_index_seurat(true_copy, ctrl_copy, pred_dict, true_except_ood, data_params, ood_primary, ood_modality, true_logfc_df, all_adata=all_adata)
                        with open(f'{path_to_save}/non_trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_trivial_dict, f)
                        with open(f'{path_to_save}/trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(trivial_dict, f)
                        with open(f'{path_to_save}/non_significant_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_significant_dict, f)
                        with open(f'{path_to_save}/num_genes_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump({'non_trivial':num_genes_list[0], 'trivial':num_genes_list[1], 'non_significant':num_genes_list[2]}, f)
                        plots.draw_boxplot_multiple(
                            [non_trivial_dict, trivial_dict, non_significant_dict], data, path_to_save, f"{evaluation_metric}_{ood_primary}_{ood_modality}",\
                                [f"{num_genes_list[0]} non-trivial genes", f"{num_genes_list[1]} trivial-genes", f"{num_genes_list[2]} non-significant genes (significance threshold: 0.05)"])

                    elif evaluation_metric == 'rigorous_subset_deg_f1':
                        if true_logfc_df is None:
                                true_logfc_df = seurat_deg(
                                    ctrl_true_adata, data_params['modality_variable'], ood_modality, data_params['control_key'], test_use='MAST', find_variable_features=True)
                                true_logfc_df = true_logfc_df.sort_values(by="p_val_adj", ascending=True)
                        true_copy = true_adata.copy()
                        if issparse(true_copy.X):
                            true_copy.X = true_copy.X.toarray()
                        ctrl_copy = ctrl_adata.copy()
                        if issparse(ctrl_copy.X):
                            ctrl_copy.X = ctrl_copy.X.toarray()
                        non_trivial_dict, trivial_dict, num_genes_list, down_up_dict = \
                            assess.rigorous_subset_deg_f1(true_copy, ctrl_copy, pred_dict, true_except_ood, data_params['primary_variable'], true_logfc_df, ood_primary)
                        with open(f'{path_to_save}/trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(trivial_dict, f)
                        with open(f'{path_to_save}/non_trivial_dict_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(non_trivial_dict, f)
                        with open(f'{path_to_save}/num_genes_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump({'non_trivial':num_genes_list[0], 'trivial':num_genes_list[1], 'non_significant':num_genes_list[2]}, f)
                        with open(f'{path_to_save}/down_up_{ood_primary}_{ood_modality}.pkl', 'wb') as f:
                            pickle.dump(down_up_dict, f)
                        os.makedirs(f"{path_to_save}/f1", exist_ok=True)
                        os.makedirs(f"{path_to_save}/precision", exist_ok=True)
                        os.makedirs(f"{path_to_save}/recall", exist_ok=True)
                        subtitles = [f"{num_genes_list[0]} non-trivial genes", f"{num_genes_list[1]} trivial-genes"]
                        folders = ["f1", "f1", "precision", "precision", "recall", "recall"]
                        modes = ["down", "up", "down", "up", "down", "up"]
                        for i in range(6):
                            plots.draw_boxplot_multiple(
                                [non_trivial_dict[i], trivial_dict[i]], data, f"{path_to_save}/{folders[i]}", f"{modes[i]}_{ood_primary}_{ood_modality}",\
                                    subtitles, title=f"{folders[i]}")
                        

                if evaluation_metric == 'spearman_corr':  
                    correlation_barplot(all_corr, models, data_params["ood_primaries"], data, path_to_save, 'cell types', 'spearman correlation')  
        
                elif evaluation_metric == 'z_correlation':
                    correlation_barplot(all_corr, models, data_params["ood_primaries"], data, path_to_save, 'cell types', 'z-score correlation') 
        
                elif evaluation_metric == 'logfc_correlation':
                    correlation_barplot(logfc_corr, models, data_params["ood_primaries"], data, path_to_save, 'cell types', 'logFC correlation') 
        
                elif evaluation_metric.startswith('mixing_index_seurat'):
                    print(mixing_seurat)
                    correlation_barplot(mixing_seurat, models, data_params["ood_primaries"], \
                        data, path_to_save, 'cell types', f'{evaluation_metric}', log_scale=False)
                    correlation_barplot(mixing_seurat, models, data_params["ood_primaries"], \
                        f"{data}_log", path_to_save, 'cell types', f'{evaluation_metric}', log_scale=True)
        
                elif evaluation_metric.startswith('spearman_corr_per_gene'):
                    per_gene_corrs = {}
                    if evaluation_metric == 'spearman_corr_per_gene':
                        for model in models:
                            per_gene_corrs[model] = spearman_corr_per_gene(
                                pd.DataFrame(np.stack(per_gene_pred[model], axis=0)),
                                pd.DataFrame(np.stack(per_gene_true, axis=0)))

                    elif evaluation_metric == 'spearman_corr_per_gene_nonzero':
                        for model in models:
                            per_gene_corrs[model] = spearman_corr_per_gene(
                                pd.DataFrame(np.stack(per_gene_pred_nonzero[model], axis=0)),
                                pd.DataFrame(np.stack(per_gene_true_nonzero, axis=0)))
                    draw_boxplot(per_gene_corrs, data, path_to_save, 'Spearman across genes')

                elif evaluation_metric == 'rigorous_spearman_corr_per_gene':
                    per_gene_corrs = {}
                    metric_dict, standardized_dict = assess.rigorous_spearman_corr_per_gene(per_gene_true_adata, per_gene_pred_adata, per_gene_except_ood_adata)
                    with open(f'{path_to_save}/rigorous_per_gene_{ood_modality}.pkl', 'wb') as f:
                        pickle.dump(metric_dict, f)
                    with open(f'{path_to_save}/standardized_per_gene_{ood_modality}.pkl', 'wb') as f:
                        pickle.dump(standardized_dict, f)
                    draw_boxplot(
                        metric_dict, data, path_to_save, f"{evaluation_metric}_{ood_modality}")
                    draw_boxplot(
                        standardized_dict, data, path_to_save, f"standardized_{evaluation_metric}_{ood_modality}")

                elif evaluation_metric == 'rigorous_spearman_corr_per_gene_delta':
                    per_gene_corrs = {}
                    metric_dict, standardized_dict = assess.rigorous_spearman_corr_per_gene(per_gene_true_adata_delta, per_gene_pred_adata_delta, per_gene_except_ood_adata_delta)
                    with open(f'{path_to_save}/rigorous_per_gene_{ood_modality}.pkl', 'wb') as f:
                        pickle.dump(metric_dict, f)
                    with open(f'{path_to_save}/standardized_per_gene_{ood_modality}.pkl', 'wb') as f:
                        pickle.dump(standardized_dict, f)
                    draw_boxplot(
                        metric_dict, data, path_to_save, f"{evaluation_metric}_{ood_modality}")
                    draw_boxplot(
                        standardized_dict, data, path_to_save, f"standardized_{evaluation_metric}_{ood_modality}")
                    
        
                elif evaluation_metric.startswith('normalized_corr_per_gene'):
                    per_gene_corrs = {}
                    if evaluation_metric == 'normalized_corr_per_gene':
                        per_gene_true = subtract_control(control_mean_dict, per_gene_true, data_params['ood_primaries'])
                        for model in models:
                            per_gene_pred[model] = subtract_control(control_mean_dict, per_gene_pred[model], data_params['ood_primaries'])
                            per_gene_corrs[model] = spearman_corr_per_gene(
                                pd.DataFrame(np.stack(per_gene_pred[model], axis=0)),
                                pd.DataFrame(np.stack(per_gene_true, axis=0)))

                    elif evaluation_metric == 'normalized_corr_per_gene_nonzero':
                        per_gene_true_nonzero = subtract_control(control_mean_dict_nonzero, per_gene_true_nonzero, data_params['ood_primaries'])
                        for model in models:
                            per_gene_pred_nonzero[model] = subtract_control(control_mean_dict_nonzero, per_gene_pred_nonzero[model], data_params['ood_primaries'])
                            per_gene_corrs[model] = spearman_corr_per_gene(
                                pd.DataFrame(np.stack(per_gene_pred_nonzero[model], axis=0)),
                                pd.DataFrame(np.stack(per_gene_true_nonzero, axis=0)))
                    draw_boxplot(per_gene_corrs, data, path_to_save, evaluation_metric)
        
                elif evaluation_metric == 'l2_distance':
                    all_ctrl_adata =  ood_modality_adata[ood_modality_adata.obs[data_params['modality_variable']] == data_params['control_key']].copy()
                    true_df = pd.DataFrame(np.stack(per_gene_true, axis=0))
                    pred_dict = {}
                    for model in models:
                        pred_dict[model] = pd.DataFrame(np.stack(per_gene_pred[model], axis=0))
                        
                    distance_dict = prediction_error(pd.DataFrame(all_ctrl_adata.X), pred_dict, true_df)
                    draw_boxplot(distance_dict, data, path_to_save, 'l2_distance')
        
                elif evaluation_metric == 'wasserstein_distance':
                    w_distance = {}
                    for model in models:
                        w_distance[model] = wasserstein(distribution_pred[model], distribution_true, True)
        
                    draw_boxplot(w_distance, data, path_to_save, 'wasserstein_distance')
        
                elif evaluation_metric == 'knn':
                    draw_boxplot(knn_dict, data, path_to_save, 'knn')

        