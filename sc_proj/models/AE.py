import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import os
from utils.data_handler import split_train_valid, prepare_data_loader
from utils.common import calculate_cpm, load_model
from utils.prepare_onehot import prepare_onehot

class AE(nn.Module):
    def __init__(self, input_dim, hyperparams):
        super(AE, self).__init__()
        
        self.conditional_model = hyperparams['conditional_model']
        self.cond_dim = hyperparams['cond_dim'] if self.conditional_model else 0

        # Extract hyperparameters
        activation = hyperparams['activation']
        activation_out = hyperparams['activation_out']
        dropout = hyperparams['dropout']
        latent_dim = hyperparams['latent_dim']
        hidden_layers = hyperparams['hidden_layers']

        # Build the encoder
        encoder_layers = [nn.Linear(input_dim + self.cond_dim, hidden_layers[0]), get_activation(activation),
                          nn.Dropout(dropout)]
        for i in range(1, len(hidden_layers)):
            encoder_layers += [nn.Linear(hidden_layers[i - 1], hidden_layers[i]), get_activation(activation),
                               nn.Dropout(dropout)]
        encoder_layers.append(nn.Linear(hidden_layers[-1], latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build the decoder
        decoder_layers = [nn.Linear(latent_dim + self.cond_dim, hidden_layers[-1]), get_activation(activation_out),
                          nn.Dropout(dropout)]
        for i in range(len(hidden_layers) - 2, -1, -1):
            decoder_layers += [nn.Linear(hidden_layers[i + 1], hidden_layers[i]), get_activation(activation_out),
                               nn.Dropout(dropout)]
        decoder_layers.append(nn.Linear(hidden_layers[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x, c=None):
        if self.conditional_model:
            if c is None or c.shape[1] != self.cond_dim:
                raise ValueError(f"Condition variable has incorrect dimensions: {c.shape[1]}, expected {self.cond_dim}")
            x = torch.cat([x, c], 1)
    
        z = self.encoder(x)
    
        if self.conditional_model:
            z = torch.cat([z, c], 1)
    
        reconstruction = self.decoder(z)
        return reconstruction
    
def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()    
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(0.2)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'softplus':
        return nn.Softplus()
    elif activation == 'linear':
        return nn.Identity()  # Identity function for linear activation
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

        
def train_AE(train_adata, hyperparams, data_params, setting, ood_primary=None, path_to_save='.'):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # GET THE DATA
    train_adata = calculate_cpm(train_adata)
    
    if hyperparams['conditional_model']:
       conditional_params= [data_params['primary_variable'], data_params['modality_variable']]
       prepare_onehot(train_adata, conditional_params, path_to_save)       
       hyperparams['cond_dim'] = train_adata.obsm['onehots'].shape[1]   
    
    if hyperparams['validation']:
       train_adata, valid_adata = split_train_valid(train_adata, train_percentage=0.85)
    else:
        valid_adata = None
    
    train_loader = prepare_data_loader(train_adata, batch_size=hyperparams['batch_size'], shuffle=True)
    
    if valid_adata is not None:
       valid_loader = prepare_data_loader(valid_adata, batch_size=hyperparams['batch_size'], shuffle=True)
       
    model = AE(train_adata.n_vars, hyperparams).to(device)   
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    best_loss = float('inf')
    
    for epoch in range(1, hyperparams['train_epochs'] + 1):
        model.train()
        train_loss = 0
    
        for batch_idx, (data, condition) in enumerate(train_loader):
            data = data.to(device)
            if condition.numel() == 0:  
                condition = None
            else:
                condition = condition.to(device)
                
            optimizer.zero_grad()
                        
            reconstruction = model(data, condition)
            loss = F.mse_loss(reconstruction, data)         
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch: {epoch}, Average AE training loss: {avg_train_loss}')
        
        if valid_adata is not None:
            avg_valid_loss = evaluate_AE(model, valid_loader, device)
            if  avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                best_save_path = os.path.join(path_to_save, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_loss': best_loss,
                }, best_save_path)
    
    model_save_path = os.path.join(path_to_save, 'final_model.pth')
    torch.save({
        'epoch': hyperparams['train_epochs'],
        'model_state_dict': model.state_dict(),
        'loss': avg_train_loss,
    }, model_save_path)
    
    print("Training complete.")


def evaluate_AE(model, valid_loader, device):
    model.eval()
    total_recon_loss = 0

    with torch.no_grad():
        for data, condition in valid_loader:
            data = data.to(device)
            if condition.numel() == 0:  
                condition = None
            else:
                condition = condition.to(device)
            
            reconstruction = model(data, condition)
           
            # Reconstruction loss
            recon_loss = F.mse_loss(reconstruction, data) 
            total_recon_loss += recon_loss.item()

    avg_recon_loss = total_recon_loss / len(valid_loader)

    print(f'AE Evaluation loss in valid data {avg_recon_loss}')

    return avg_recon_loss


def predict_AE(train_adata, hyperparams, data_params, setting, ood_primary, ood_modality, path_to_save):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_adata = calculate_cpm(train_adata)
    
    # GET THE DATA 
    test_adata = train_adata[(train_adata.obs[data_params['primary_variable']] == ood_primary) & (train_adata.obs[data_params['modality_variable']] == data_params['control_key'])].copy() 
    print(test_adata.obs.shape)
    
    if hyperparams['conditional_model']:
       conditional_params= [data_params['primary_variable'], data_params['modality_variable']]
       test_adata.obs[data_params['modality_variable']] = ood_modality
       prepare_onehot(test_adata, conditional_params, path_to_save)       
       hyperparams['cond_dim'] = test_adata.obsm['onehots'].shape[1]   
    
    Modality = [str(condition) for condition in test_adata.obs[data_params['modality_variable']].unique()]
    Primary = [str(cell_type) for cell_type in test_adata.obs[data_params['primary_variable']].unique()]
    
    print(f"test_modality: {Modality}, test_primary: {Primary}")
    
    test_loader = prepare_data_loader(test_adata, batch_size=hyperparams['batch_size'], shuffle=False)
    model = AE(test_adata.n_vars, hyperparams).to(device)   
    
    model = load_model(model, path_to_save, hyperparams['validation'])     
      
    model.eval()
    reconstructed_data = []

    with torch.no_grad():
        for batch_idx, (data, condition) in enumerate(test_loader):
            data = data.to(device)
            if condition.numel() == 0: 
                condition = None
            else:
                condition = condition.to(device)
                        
            reconstruction = model(data, condition)

            reconstructed_data.append(reconstruction.cpu().numpy())

    predicted_data = np.concatenate(reconstructed_data, axis=0)
    predicted_data = np.clip(predicted_data, 0.0, predicted_data.max())
    test_adata.X = predicted_data
    pred_adata = test_adata.copy()
    pred_adata.obs[data_params['modality_variable']] = 'ae_pred' 
    
    print("Prediction complete.")
    
    pred_file_path = os.path.join(path_to_save, 'pred_adata.h5ad')
    # Save the predicted AnnData object
    pred_adata.write(pred_file_path)
    
