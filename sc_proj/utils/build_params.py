
def ae(config_model):
    
    # Hyperparameters
    hyperparams = {
        'hidden_layers': [int(layer) for layer in config_model.get('hyperparams', 'hidden_layers').split(',')],
        'dropout': config_model.getfloat('hyperparams', 'dropout'),
        'activation': config_model.get('hyperparams', 'activation'),
        'activation_out': config_model.get('hyperparams', 'activation_out'),
        'latent_dim': config_model.getint('hyperparams', 'latent_dim'),
        'conditional_model': config_model.getboolean('hyperparams', 'conditional_model'),
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
        'validation': config_model.get('hyperparams', 'validation'),
        'batch_size': config_model.getint('hyperparams', 'batch_size')
    }
    
    if config_model.has_option('hyperparams', 'conditional_params'):
        hyperparams['conditional_params'] = [item.strip() for item in config_model.get('hyperparams', 'conditional_params').split(',')]

    return hyperparams

def cae(config_model):

    # Hyperparameters
    hyperparams = {
        'hidden_layers': [int(layer) for layer in config_model.get('hyperparams', 'hidden_layers').split(',')],
        'dropout': config_model.getfloat('hyperparams', 'dropout'),
        'activation': config_model.get('hyperparams', 'activation'),
        'activation_out': config_model.get('hyperparams', 'activation_out'),
        'latent_dim': config_model.getint('hyperparams', 'latent_dim'),
        'conditional_model': config_model.getboolean('hyperparams', 'conditional_model'),
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
        'validation': config_model.get('hyperparams', 'validation'),
        'batch_size': config_model.getint('hyperparams', 'batch_size')
    }
    
    if config_model.has_option('hyperparams', 'conditional_params'):
        hyperparams['conditional_params'] = [item.strip() for item in config_model.get('hyperparams', 'conditional_params').split(',')]

    return hyperparams


def vae(config_model):

    # Hyperparameters
    hyperparams = {
        'hidden_layers': [int(layer) for layer in config_model.get('hyperparams', 'hidden_layers').split(',')],
        'dropout': config_model.getfloat('hyperparams', 'dropout'),
        'activation': config_model.get('hyperparams', 'activation'),
        'activation_out': config_model.get('hyperparams', 'activation_out'),
        'latent_dim': config_model.getint('hyperparams', 'latent_dim'),
        'conditional_model': config_model.getboolean('hyperparams', 'conditional_model'),
        'learning_rate': config_model.getfloat('hyperparams', 'learning_rate'),
        'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
        'train_func': config_model.get('hyperparams', 'train_func'),
        'predict_func': config_model.get('hyperparams', 'predict_func'),
        'validation': config_model.get('hyperparams', 'validation'),
        'batch_size': config_model.getint('hyperparams', 'batch_size'),
        'beta': config_model.getfloat('hyperparams', 'beta'),
        'gamma': config_model.getfloat('hyperparams', 'gamma'),
        'loss_plot_path': config_model.get('hyperparams', 'loss_plot_path')
    }
    
    if config_model.has_option('hyperparams', 'conditional_params'):
        hyperparams['conditional_params'] = [item.strip() for item in config_model.get('hyperparams', 'conditional_params').split(',')]

    return hyperparams


def noperturb(config_model):    
    
    hyperparams = {'predict_func': config_model.get('hyperparams', 'predict_func')}

    return hyperparams

def random_all(config_model):    
    
    hyperparams = {'predict_func': config_model.get('hyperparams', 'predict_func')}

    return hyperparams


def baseline_logfc(config_model):    
    
    hyperparams = {'predict_func': config_model.get('hyperparams', 'predict_func')}

    return hyperparams


def cpa(config_model):    
    
    hyperparams = {'predict_func': config_model.get('hyperparams', 'predict_func'),
                   'train_func': config_model.get('hyperparams', 'train_func')
                   }

    return hyperparams

def scPRAM(config_model):    
    
    hyperparams = {'predict_func': config_model.get('hyperparams', 'predict_func'),
                   'train_func': config_model.get('hyperparams', 'train_func'),
                   'train_epochs': config_model.getint('hyperparams', 'train_epochs'),
                   'validation': config_model.get('hyperparams', 'validation')
                   
                   }

    return hyperparams


def build_data_params(config_data):

    data_params = {
        'data': config_data.get('main', 'data'),
        'base_path': config_data.get('main', 'base_path'),
        'ood_primaries': [item.strip() for item in config_data.get('data', 'ood_primaries').split(',')],
        'ood_modality': [item.strip() for item in config_data.get('data', 'ood_modality').split(',')],
        'primary_variable': config_data.get('data', 'primary_variable'),
        'modality_variable': config_data.get('data', 'modality_variable'),
        'control_key': config_data.get('data', 'control_key')
        
    }

    if config_data.has_option('main', 'gene_subset'):
        data_params['gene_subset'] = config_data.get('main', 'gene_subset')
    
    return  data_params