import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from moving_mnist_dataset import MovingMNISTDataset
from moving_mnist_models import Seq2SeqFERNN
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
from train_eval_utils import train_epoch, eval_epoch, eval_len_generalization, eval_velocity_generalization


def main():
    parser = argparse.ArgumentParser(description="Train & evaluate RNN models on Moving MNIST")
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--input_frames', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--min_epochs', type=int, default=50, help='Minimum number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model', choices=['grnn', 'fernn'], default='fernn')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--decoder_conv_layers', type=int, default=4)
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples to use (default: use all)')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.0, help='Probability of using teacher forcing during training (0.0-1.0)')
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--v_range', type=int, default=2)
    parser.add_argument('--data_v_range', type=int, default=2)
    parser.add_argument('--gen_seq_len', type=int, default=40, help='Sequence length used **only** for length‑generalization evaluation (must be > seq_len)')
    parser.add_argument('--data_seed', type=int, default=42, help='Random seed for dataset splitting')
    parser.add_argument('--model_seed', type=int, default=None, help='Random seed for model initialization (default: random)')
    parser.add_argument('--run_name', type=str, default=None, help='Name of the run')
    parser.add_argument('--model_save_dir', type=str, default='./fernn/movmnist/', help='Directory to save model checkpoints')
    parser.add_argument('--load_model', type=str, default=None, help='Path to a saved model checkpoint to load for evaluation')
    parser.add_argument('--evaluate_only', action='store_true', help='Only evaluate the loaded model without training')
    parser.add_argument('--gen_vel_min', type=int, default=-2, help='Minimum integer pixel velocity evaluated along each axis')
    parser.add_argument('--gen_vel_max', type=int, default= 2, help='Maximum integer pixel velocity evaluated along each axis')
    parser.add_argument('--gen_vel_step', type=int, default= 1, help='Step size (integer) for the velocity grid')
    parser.add_argument('--gen_vel_n_seq', type=int, default=128, help='How many sequences to sample **per (vx,vy)** for velocitygeneralisation test')
    parser.add_argument('--run_velocity_generalization', action='store_true', help='Run velocity generalization test (default: False)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value (default: 1.0, None for no clipping)')
    args = parser.parse_args()

    # Set seeds for reproducibility
    # Data seed for consistent dataset splits
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)
    random.seed(args.data_seed)

    # Create model save directory if it doesn't exist
    os.makedirs(args.model_save_dir, exist_ok=True)

    # Initialize wandb
    wandb.init(entity="ENTITY", 
               project="FERNN", 
               dir='./tmp/',
               config=vars(args),
               name=f"moving_mnist_fp_{args.model}_{args.run_name}")

    assert args.input_frames < args.seq_len, "input_frames must be less than seq_len"
    pred_frames = args.seq_len - args.input_frames
    gen_pred_frames = args.gen_seq_len - args.input_frames

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and splits
    train_dataset = MovingMNISTDataset(
        root=args.root,
        train=True,
        seq_len=args.seq_len,
        image_size=args.image_size,
        velocity_range_x=(-args.data_v_range,args.data_v_range),
        velocity_range_y=(-args.data_v_range,args.data_v_range),
        num_digits=2
    )
    
    test_dataset = MovingMNISTDataset(
        root=args.root,
        train=False,
        seq_len=args.seq_len,
        image_size=args.image_size,
        velocity_range_x=(-args.data_v_range,args.data_v_range),
        velocity_range_y=(-args.data_v_range,args.data_v_range),
        num_digits=2
    )

    gen_test_dataset = MovingMNISTDataset(
            root=args.root,
            train=False,
            seq_len=args.gen_seq_len,
            image_size=args.image_size,
            velocity_range_x=(-args.data_v_range, args.data_v_range),
            velocity_range_y=(-args.data_v_range, args.data_v_range),
            num_digits=2,
            random=False
    )

    # Split training data into train and validation
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_ds, val_ds = random_split(train_dataset, [train_size, val_size], 
                                    generator=torch.Generator().manual_seed(args.data_seed))
    
    # Limit training samples if specified
    if args.max_train_samples is not None and args.max_train_samples < len(train_ds):
        indices = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(args.data_seed))[:args.max_train_samples]
        train_ds = Subset(train_ds, indices)
        print(f"Limited training to {args.max_train_samples} samples")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    gen_test_loader = DataLoader(gen_test_dataset,
                                batch_size=args.batch_size)


    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_dataset)}")
    print(f"Length‑gen test size: {len(gen_test_dataset)} | Eval sequence length: {args.gen_seq_len}")
    print(f"Input frames: {args.input_frames}, Pred frames: {pred_frames}")
    print(f"Model seed: {args.model_seed}")
    print(f"Data seed: {args.data_seed}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Min epochs: {args.min_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Model: {args.model}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Num layers: {args.num_layers}")
    print(f"Decoder conv layers: {args.decoder_conv_layers}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Teacher forcing ratio: {args.teacher_forcing_ratio}")
    print(f"Max train samples: {args.max_train_samples}")
    print(f"Image size: {args.image_size}")
    print(f"Velocity range: {args.v_range}")
    v_list = [(x, y) for x in range(-args.v_range, args.v_range + 1) for y in range(-args.v_range, args.v_range + 1)]
    num_v = len(v_list)
    print(f"Num v: {num_v}")

    # Set model initialization seed
    if args.model_seed is None:
        args.model_seed = int(time.time()) % 10000  # Use current time as seed if not provided
        wandb.config.update({"model_seed": args.model_seed}, allow_val_change=True)
    
    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)
    random.seed(args.model_seed)
    
    if args.model == "fernn":
        model = Seq2SeqFERNN(
                input_channels=1,
                hidden_channels=args.hidden_size,
                height=args.image_size,
                width=args.image_size,
                h_kernel_size=args.kernel_size,
                u_kernel_size=args.kernel_size,
                v_range=args.v_range,
                pool_type='max',
                decoder_conv_layers=args.decoder_conv_layers
            ).to(device)
    elif args.model == "grnn":
        assert args.v_range == 0, "v_range must be 0 for grnn"
        model = Seq2SeqFERNN(
                input_channels=1,
                hidden_channels=args.hidden_size,
                height=args.image_size,
                width=args.image_size,
                h_kernel_size=args.kernel_size,
                u_kernel_size=args.kernel_size,
                v_range=0,
                pool_type='max',
                decoder_conv_layers=args.decoder_conv_layers
            ).to(device)

    # Load model if specified
    if args.load_model is not None:
        print(f"Loading model from {args.load_model}")
        try:
            model.load_state_dict(torch.load(args.load_model))
            print(f"Successfully loaded model weights.")
        except Exception as e:
            print(f"Failed to load model weights {e}")

    # If evaluate_only is True, skip training and only run evaluation
    if args.evaluate_only:
        print("Running evaluation only...")
        criterion = nn.MSELoss()
        test_loss = eval_epoch(model, test_loader, criterion, device, args.input_frames, 0, split_name="test")
        print(f"Test Loss: {test_loss:.4f}")

        wandb.log({
            f"test_loss": test_loss,
            "epoch": 0
        })

        print("Running length generalization test...")
        gen_mean, gen_std = eval_len_generalization(model, gen_test_loader, device, args.input_frames, subsample_t=2)
        print(f"Length generalization mean MSE: {gen_mean.mean():.4f}")

        wandb.log({f"len_gen_mean_t{t+1}": gen_mean[t] for t in range(len(gen_mean))})
        wandb.log({f"len_gen_std_t{t+1}":  gen_std[t]  for t in range(len(gen_std))})
        wandb.log({f"len_gen_mean_mean_over_time": gen_mean.mean()})

        if args.run_velocity_generalization:
            print("Running velocity generalization test...")
            vx, vy, err = eval_velocity_generalization(model, device, args)
            print("Velocity generalization results:")
            for i, vx_i in enumerate(vx):
                for j, vy_j in enumerate(vy):
                    print(f"Velocity (vx={vx_i}, vy={vy_j}): MSE {err[j, i]:.4f}")
            
            wandb.log({f"vel_gen_err_vx{vx_i}_vy{vy_j}": err[j, i]
                    for i, vx_i in enumerate(vx)
                    for j, vy_j in enumerate(vy)})

            wandb.log({f"vel_gen_err_vx{vx_i}_vy{vy_j}": err[j, i]
                    for i, vx_i in enumerate(vx)
                    for j, vy_j in enumerate(vy)})

            fig, ax = plt.subplots()
            im = ax.imshow(err[::-1],  # flip y so +vy is up
                        extent=[vx[0]-0.5, vx[-1]+0.5, vy[0]-0.5, vy[-1]+0.5],
                        origin='lower')
            ax.set_xlabel('$v_x$  (pixels / frame)')
            ax.set_ylabel('$v_y$')
            ax.set_title('Velocity-generalisation MSE')
            fig.colorbar(im, ax=ax)
            wandb.log({"vel_gen_heatmap": wandb.Image(fig)})
            plt.close(fig)

        return

    # Optimizers & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'test_loss': []}
    best_val_losses = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.input_frames, args.teacher_forcing_ratio, args.grad_clip)
        val_loss = eval_epoch(model, val_loader, criterion, device, args.input_frames, epoch, split_name="val")
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch}/{args.epochs} | {args.model:^8} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Log metrics to wandb
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch
        })
        
        # Save model if it's the best so far
        if val_loss < best_val_losses:
            best_val_losses = val_loss
            model_filename = f"{args.model}_best_model_{wandb.run.id}.pth"
            model_path = os.path.join(args.model_save_dir, model_filename)
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model for {args.model} at {model_path}")
            
            # Evaluate on test set with best model
            test_loss = eval_epoch(model, test_loader, criterion, device, args.input_frames, epoch, split_name="test")

            history['test_loss'].append(test_loss)
            print(f"Test Loss: {test_loss:.4f}")

            gen_mean, gen_std = eval_len_generalization(model, gen_test_loader, device, args.input_frames)

            wandb.log({f"len_gen_mean_t{t+1}": gen_mean[t] for t in range(len(gen_mean))})
            wandb.log({f"len_gen_std_t{t+1}":  gen_std[t]  for t in range(len(gen_std))})
            wandb.log({f"len_gen_mean_mean_over_time": gen_mean.mean()})

            # ---- wandb line plot ----
            steps = np.arange(1, gen_pred_frames + 1)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(steps, gen_mean, label="MSE")
            ax.fill_between(steps,
                            (gen_mean - gen_std).clip(min=0),
                            gen_mean + gen_std,
                            alpha=0.3,
                            label="± std")
            ax.set_xlabel("Prediction horizon t")
            ax.set_ylabel("MSE")
            ax.set_title(f"Length generalization (seq_len = {args.gen_seq_len})")
            ax.legend()
            wandb.log({f"len_gen_curve": wandb.Image(fig)})
            plt.close(fig)
            
            if args.run_velocity_generalization:
                vx, vy, err = eval_velocity_generalization(model, device, args)

                wandb.log({f"vel_gen_err_vx{vx_i}_vy{vy_j}": err[j, i]
                        for i, vx_i in enumerate(vx)
                        for j, vy_j in enumerate(vy)})

                fig, ax = plt.subplots()
                im = ax.imshow(err[::-1],  # flip y so +vy is up
                            extent=[vx[0]-0.5, vx[-1]+0.5, vy[0]-0.5, vy[-1]+0.5],
                            origin='lower')
                ax.set_xlabel('$v_x$  (pixels / frame)')
                ax.set_ylabel('$v_y$')
                ax.set_title('Velocity-generalisation MSE')
                fig.colorbar(im, ax=ax)
                wandb.log({"vel_gen_heatmap": wandb.Image(fig)})
                plt.close(fig)

            # Log model path and metrics to wandb
            wandb.log({
                "best_model_path": model_path,
                "best_val_loss": val_loss,
                "test_loss": test_loss,
                "epoch": epoch
            })

    # Print final best model paths and test results
    print("\nBest model paths and test results:")
    model_filename = f"{args.model}_best_model_{wandb.run.id}.pth"
    model_path = os.path.join(args.model_save_dir, model_filename)
    test_loss = history['test_loss'][-1] if history['test_loss'] else None
    print(f"{args.model}: {model_path} | Test Loss: {test_loss:.4f}" if test_loss is not None else f"{args.model}: {model_path} | Test Loss: N/A")

if __name__ == '__main__':
    main()
