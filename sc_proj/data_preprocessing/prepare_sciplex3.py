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

    # Identify highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_HVG, subset=True)
    print('\nNumber of highly variable genes: {:d}'.format(np.sum(adata.var['highly_variable'])))

    adata.X = adata.layers['counts'].copy()
    adata.X = scipy.sparse.csr_matrix(adata.X)

    return adata

adata = sc.read("../../datasets/srivatsan/SrivatsanTrapnell2020_sciplex3.h5ad")
adata = adata[adata.obs['dose_value'].isin([0, 1000])].copy()

filtered_data = preprocess_data(adata, min_counts= 100, min_genes = 5)
filtered_data.var.index = filtered_data.var['ensembl_id']
filtered_data = filtered_data[~filtered_data.obs["cell_line"].isna()].copy()


output_path = '../../datasets/srivatsan/'
filtered_data.write(os.path.join(output_path, 'processed_SciPlex3.h5ad'))