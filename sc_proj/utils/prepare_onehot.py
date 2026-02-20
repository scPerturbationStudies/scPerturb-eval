import numpy as np
import os
import pickle



def get_onehot_dict(adata, onehot_params):
    category_to_index = {}
    onehot_dict = {}

    # Create a mapping of categories to indices for each column
    for column in onehot_params:
        unique_values = adata.obs[column].unique().tolist()
        categories = [f'{value}' for value in unique_values]
        category_to_index[column] = {category: i for i, category in enumerate(categories)}

    # Store the mapping in onehot_dict
    for column in onehot_params:
        onehot_dict[column] = category_to_index[column]

    return onehot_dict


def add_onehot_to_adata(adata, onehot_dict, onehot_params):
    # Initialize an array to store one-hot encoded vectors for all observations
    onehot_array = np.zeros((len(adata.obs), sum(len(onehot_dict[col]) for col in onehot_params)), dtype=np.float32)

    # Iterate over each observation in adata
    for i in range(len(adata.obs)):
        # Get the values for specified columns from adata
        values = [adata.obs[column].values[i] for column in onehot_params]

        # For each specified column, set the appropriate one-hot encoding
        for col_index, column in enumerate(onehot_params):
            value = values[col_index]
            category = f'{value}' if value is not None else ''
            if category in onehot_dict[column]:
                onehot_array[i, sum(len(onehot_dict[col]) for col in onehot_params[:col_index]) + onehot_dict[column][category]] = 1

    # Add the new 'onehots' layer to adata.obsm as a NumPy array
    adata.obsm['onehots'] = onehot_array

def prepare_onehot(adata, onehot_params, path_to_save):
        
    dep_path = os.path.join(os.path.dirname(os.path.dirname(path_to_save)), 'dependencies')
    onehot_dict_path = os.path.join(dep_path, 'onehot_dict.pkl')
    
    # If the onehot_dict.pkl file doesn't exist, create it and save the dictionary
    if not os.path.exists(onehot_dict_path):
        os.makedirs(dep_path, exist_ok=True)
        onehot_dict = get_onehot_dict(adata, onehot_params)
        with open(onehot_dict_path, 'wb') as f:
            pickle.dump(onehot_dict, f)
    else:
        # Load the onehot_dict if it exists
        with open(onehot_dict_path, 'rb') as f:
            onehot_dict = pickle.load(f)
        
    add_onehot_to_adata(adata, onehot_dict, onehot_params)
    
