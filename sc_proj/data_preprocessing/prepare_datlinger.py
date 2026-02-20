import scanpy as sc
import pandas as pd
import os
import scipy
import numpy as np

def preprocess_data(adata, min_counts=500, min_genes=750, mt_frac_threshold=0.2, min_cells_gene_filter=100, n_HVG=5000):
    # Filtering cells based on counts and genes
    print('Total number of cells: {:d}'.format(adata.n_obs))
    sc.pp.filter_cells(adata, min_counts=min_counts)
    print('Number of cells after min count filter: {:d}'.format(adata.n_obs))
    sc.pp.filter_cells(adata, min_genes=min_genes)

    # Filtering genes based on cell count
    sc.pp.filter_genes(adata, min_cells=min_cells_gene_filter)
    print('Number of genes after cell filter: {:d}'.format(adata.n_vars))

    # Save original counts in the counts layer
    adata.layers["counts"] = adata.X.copy()

    # Normalize and preprocess
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Convert to a dense matrix
    adata.X = scipy.sparse.csr_matrix(adata.X)

    # Identify highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_HVG, subset=True)
    print('\nNumber of highly variable genes: {:d}'.format(np.sum(adata.var['highly_variable'])))

    return adata


adata_path = '../../datasets/datlinger/GSM5151370_PD213_scifi_2_CRISPR-TCR_77300_MeOH-cells.h5ad'
adata = sc.read(adata_path)
preprocess_data(adata, min_counts= 100, min_genes = 5)

feature_path = '../../datasets/datlinger/GSM5151370_PD213_scifi_2_CRISPR-TCR_77300_MeOH-cells.csv.gz'
features = pd.read_csv(feature_path)
   
adata.obs['gRNA_ID'] = pd.NA
adata.obs['TCR_status'] = pd.NA
adata.obs['cell_line'] = pd.NA
for plate_well in features['plate_well'].unique(): 
    mask = adata.obs['plate_well'] == plate_well
    if mask.sum() > 0:
        adata.obs.loc[mask, 'gRNA_ID'] = features.loc[features['plate_well'] == plate_well, 'gRNA_ID'].values[0]
        adata.obs.loc[mask, 'cell_line'] = features.loc[features['plate_well'] == plate_well, 'cell_line'].values[0]
        adata.obs.loc[mask, 'TCR_status'] = features.loc[features['plate_well'] == plate_well, 'TCR_status'].values[0]

adata.obs['target'] = adata.obs['gRNA_ID'].str.split('_').str[0]
control_wells = ['CTRL00018', 'CTRL00022','CTRL00080', 'CTRL00087', 'CTRL00096', 'CTRL00196', 'CTRL00275','CTRL00320']
    
adata.obs.loc[adata.obs['target'].isin(control_wells), 'target'] = 'control'

adata.X = adata.layers['counts']

output_path = '../../datasets/datlinger'
adata.write(os.path.join(output_path, 'processed_Datlinger2021.h5ad'))