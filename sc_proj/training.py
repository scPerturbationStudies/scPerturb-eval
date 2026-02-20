import os
import warnings
import argparse
from run_pipeline import train

# Filter out the specific UserWarnings
warnings.filterwarnings("ignore", category=UserWarning, module="anndata")

def process_file(models, datasets, config_path, save_folder, setting = 'ood', pid_percentage = 0.2):
    model_config_path = os.path.join(config_path, 'models')
    data_config_path = os.path.join(config_path, 'data')
    
    # Run the pipeline    
    train(models, datasets, model_config_path, data_config_path, save_folder, setting, pid_percentage)
    
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Process models and datasets for training.")

    # Add the arguments
    parser.add_argument('--models', nargs='+', required=True, help="List of models, e.g., model1 model2 model3")
    parser.add_argument('--datasets', nargs='+', required=True, help="List of datasets, e.g., dataset1 dataset2 dataset3")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the configuration directory")
    parser.add_argument('--save_folder', type=str, required=True, help="Path to the save folder")
    parser.add_argument('--setting', type=str, required=True, help="Define OOD or PID settings")
    parser.add_argument('--pid_percentage', type=float, required=False, help="The percentage of the ood cell to be seen by the model")

    # Parse the arguments
    args = parser.parse_args()

    # Call the process_file function with parsed arguments
    process_file(args.models, args.datasets, args.config_path, args.save_folder, args.setting, args.pid_percentage)
