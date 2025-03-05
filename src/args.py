import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description="CLIP Decoder Model Training and Inference Arguments")
    
    # Dataset & Output Paths
    parser.add_argument('--dataset_path', type=str, default="../MLDataset", help='Path to the dataset')
    parser.add_argument('--output_path', type=str,  default="outputs", help='Path to save the outputs')
    parser.add_argument('--exp_name', type=str, default="v1", help='Path to save the outputs')

    # Model Hyperparameters
    parser.add_argument('--num_groups', type=int, default=4, help='Number of learnable group queries')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--ff_dim', type=int, default=2048, help='Feed-forward hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training Parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for classification')
    parser.add_argument('--early_stop_patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--experimental_run', action='store_true', help='Apply Grid Search to find optimal hyperparameters')
    
    
    # Augmentation Option
    parser.add_argument('--augment', action='store_true', help='Apply image augmentation')
    
    # Inference Parameters
    parser.add_argument('--checkpoint_path', type=str, default="outputs/exp1/best_model.pth", help='Checkpoint Path to load the model')
    parser.add_argument('--classnames', type=str, default="../MLDataset/classnames.txt")
    parser.add_argument('--input_folder', type=str, default="../MLDataset/inference")


    return parser.parse_args()