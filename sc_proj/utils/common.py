import numpy as np
import pandas as pd
import scanpy as sc
import torch
import os
import scipy.sparse as sp
from scipy.sparse import issparse
import utils.assessment as assess
#from rpy2.robjects import r, pandas2ri, numpy2ri
#from rpy2.robjects.packages import importr
#import rpy2.robjects as ro
#from rpy2.robjects.vectors import DataFrame as RDataFrame


def calculate_cpm(adata, total_count=1e4, do_sparse=False):
    
    if "counts" not in adata.layers:
        raise ValueError("adata.layers['counts'] must contain raw counts")

    counts = adata.layers['counts']

    if issparse(counts):
        total_reads = np.array(counts.sum(axis=1)).flatten()
        total_reads[total_reads == 0] = 1  # avoid divide by zero
        # Scale rows to CPM
        scaling_factors = total_count / total_reads
        cpm = counts.multiply(scaling_factors[:, None])  # stays sparse
    else:
        total_reads = np.sum(counts, axis=1)    
        cpm = (counts / total_reads[:, np.newaxis]) * total_count


    if sp.isspmatrix_coo(cpm):
        cpm = cpm.tocsr()
    if do_sparse and not issparse(cpm):
        cpm = sp.csr_matrix(cpm)
    elif not do_sparse and issparse(cpm):
        cpm = cpm.toarray()
    adata.X = cpm
    sc.pp.log1p(adata)
    
    return adata

def load_model(model, path_to_save, validation=True, strict=True):
    
    if validation:
        model_path = os.path.join(path_to_save, 'best_model.pth')
    else:
        model_path = os.path.join(path_to_save, 'final_model.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))['model_state_dict'], strict=strict)
    else:
        raise ValueError(f"Model path does not exist: {model_path}")
        
    return model


def seurat_deg(adata, grouping_variable, ident_1, ident_2, test_use, find_variable_features=False, scale_factor=10000, latent_vars=None):
    de_genes_df = assess.seurat_deg(
        adata, grouping_variable, ident_1, ident_2, test_use, find_variable_features, scale_factor, latent_vars)
    return de_genes_df


# def create_seurat_object(adata):
#     adata.X = adata.layers['counts'].copy()

#     if issparse(adata.X):
#         adata.X = adata.X.toarray()

#     pandas2ri.activate()
#     Seurat = importr("Seurat")
#     SeuratObject = importr("SeuratObject")

#     expr_matrix = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)
#     meta_data = adata.obs.copy()

#     r_expr_matrix = pandas2ri.py2rpy(expr_matrix)
#     r_meta_data = pandas2ri.py2rpy(meta_data)

#     seurat_obj = SeuratObject.CreateSeuratObject(counts=r_expr_matrix, meta_data=r_meta_data)

#     return seurat_obj

# def seurat_deg(adata, grouping_variable, ident_1, ident_2, test_use, find_variable_features=False, scale_factor=10000, latent_vars=None):

#     seurat_obj = create_seurat_object(adata.copy())
#     pandas2ri.activate()
#     ro.globalenv['seurat_obj'] = seurat_obj
#     print(r(f'summary(seurat_obj@meta.data${grouping_variable})'))
    
#     r(f"Idents(seurat_obj) <- seurat_obj@meta.data${grouping_variable}")
#     print(r("table(Idents(seurat_obj))"))

#     r(f"""seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = {scale_factor})""")
#     if find_variable_features:
#         r("""seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 4000)""")

#     if latent_vars is None:
#         de_genes = r(f"FindMarkers(object = seurat_obj,ident.1 = '{ident_1}',ident.2 = '{ident_2}',test.use = '{test_use}', verbose = FALSE)")
#     else:
#         print(f'with covariate of {latent_vars}')
#         de_genes = r(f"FindMarkers(object = seurat_obj,ident.1 = '{ident_1}',ident.2 = '{ident_2}',test.use = '{test_use}', verbose = FALSE, latent.vars = '{latent_vars}')")

#     if isinstance(de_genes, RDataFrame):
#         de_genes_df = pandas2ri.rpy2py(de_genes)
#     else:
#         de_genes_df = de_genes  # Use as is

#     return de_genes_df