import cpa
import os
import torch
from utils.common import calculate_cpm
import scanpy as sc
import pandas as pd

def train_CPA(train_adata, hyperparams, data_params, setting, ood_primary, path_to_save='.'):
    
    device = True if torch.cuda.is_available() else False
          
    train_adata.obs['dose'] = train_adata.obs[data_params['modality_variable']].astype(str).apply(lambda x: '+'.join(['1.0' for _ in x.split('+')])).values
    
    cpa.CPA.setup_anndata(train_adata,
                          perturbation_key=data_params['modality_variable'],
                          control_group=data_params['control_key'],
                          dosage_key='dose',
                          categorical_covariate_keys=[data_params['primary_variable']],
                          is_count_data=True,
                          max_comb_len=1,
                         )

    model_params = {
        "n_latent": 64,
        "recon_loss": "nb",
        "doser_type": "linear",
        "n_hidden_encoder": 128,
        "n_layers_encoder": 2,
        "n_hidden_decoder": 512,
        "n_layers_decoder": 2,
        "use_batch_norm_encoder": True,
        "use_layer_norm_encoder": False,
        "use_batch_norm_decoder": False,
        "use_layer_norm_decoder": True,
        "dropout_rate_encoder": 0.0,
        "dropout_rate_decoder": 0.1,
        "variational": False,
        "seed": 6977,
    }

    trainer_params = {
        "n_epochs_kl_warmup": None,
        "n_epochs_pretrain_ae": 30,
        "n_epochs_adv_warmup": 50,
        "n_epochs_mixup_warmup": 0,
        "mixup_alpha": 0.0,
        "adv_steps": None,
        "n_hidden_adv": 64,
        "n_layers_adv": 3,
        "use_batch_norm_adv": True,
        "use_layer_norm_adv": False,
        "dropout_rate_adv": 0.3,
        "reg_adv": 20.0,
        "pen_adv": 5.0,
        "lr": 0.0003,
        "wd": 4e-07,
        "adv_lr": 0.0003,
        "adv_wd": 4e-07,
        "adv_loss": "cce",
        "doser_lr": 0.0003,
        "doser_wd": 4e-07,
        "do_clip_grad": True,
        "gradient_clip_value": 1.0,
        "step_size_lr": 10,
    }
    
    model = cpa.CPA(train_adata, 
                    **model_params,
                    )

    model.train(max_epochs=2000,
                use_gpu=device, 
                batch_size=512,
                plan_kwargs=trainer_params,
                early_stopping_patience=5,
                check_val_every_n_epoch=5,
                save_path=path_to_save
                )
    
    
def predict_CPA(train_adata, hyperparams, data_params, setting, ood_primary, ood_modality, path_to_save):  
    
    device = True if torch.cuda.is_available() else False
    # ood_modality = data_params['ood_modality']

    train_adata.obs['dose'] = train_adata.obs[data_params['modality_variable']].astype(str).apply(lambda x: '+'.join(['1.0' for _ in x.split('+')])).values
    
    #if 'cov_cond' not in train_adata.obs:
    #    train_adata.obs['cov_cond'] = pd.Categorical([None] * train_adata.shape[0])
    

    train_adata.obs['cov_cond'] = pd.Categorical([None] * train_adata.shape[0])
    train_adata.obs['cov_cond'] = train_adata.obs['cov_cond'].cat.add_categories(f'{ood_primary}_{ood_modality}')
    
    
    # train_adata.obs.loc[(train_adata.obs[data_params['primary_variable']] == ood_primary) & 
    #             (train_adata.obs[data_params['modality_variable']] != ood_modality), 'cov_cond'] = f'{ood_primary}_{ood_modality}'
    train_adata.obs.loc[(train_adata.obs[data_params['primary_variable']] == ood_primary) & 
                (train_adata.obs[data_params['modality_variable']] == data_params['control_key']), 'cov_cond'] = f'{ood_primary}_{ood_modality}'
    
    # train_adata.obs.loc[(train_adata.obs[data_params['primary_variable']] == ood_primary) & 
    #             (train_adata.obs[data_params['modality_variable']] != ood_modality), data_params['modality_variable']] = ood_modality
    train_adata.obs.loc[(train_adata.obs[data_params['primary_variable']] == ood_primary) & 
                (train_adata.obs[data_params['modality_variable']] == data_params['control_key']), data_params['modality_variable']] = ood_modality
    
    model = cpa.CPA.load(dir_path=path_to_save,
                          adata=train_adata,
                          use_gpu=device)  
    
    # test_adata = train_adata[(train_adata.obs[data_params['primary_variable']] == ood_primary) & (train_adata.obs[data_params['modality_variable']] == ood_modality)].copy() 
    test_adata = train_adata[train_adata.obs['cov_cond']==f'{ood_primary}_{ood_modality}'].copy() 

    print(f'test_adata: {test_adata.X.shape}')

    model.predict(test_adata, batch_size=2048)
    
    test_adata.X = test_adata.obsm['CPA_pred'].copy()
    test_adata.layers['counts'] = test_adata.obsm['CPA_pred'].copy()
    pred_adata = test_adata.copy()

    ######### test cpm
    pred_adata = calculate_cpm(pred_adata)
    # sc.pp.log1p(pred_adata)

    pred_adata.obs[data_params['modality_variable']] = 'cpa_pred' 
    
    print("Prediction complete.")
    
    pred_file_path = os.path.join(path_to_save, 'pred_adata.h5ad')
    # Save the predicted AnnData object
    pred_adata.write(pred_file_path)