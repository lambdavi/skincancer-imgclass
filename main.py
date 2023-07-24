from config import *
from train import run_train_loop, run_eval_loop
from test import make_inference
from args import get_parser

if __name__ == "__main__":
    # Args
    parser = get_parser()
    args = parser.parse_args()

    # Device configuration
    device = get_device()
    
    # Model configuration
    model = get_model(args)

    # Transformations configuration
    transform = get_transformations(args)

    # Dataset configuration
    dataset = get_dataset(transform)

    # Train loop
    run_train_loop(model, dataset, device)

    # Eval loop
    run_eval_loop(model, dataset, device)

    # Inference
    if args.pred_path:
        make_inference(args.pred_path, model, transform, device, dataset)
