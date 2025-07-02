import time
import sys
import os
import math
import argparse
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import DataLoader, random_split, Subset
from rotating_mnist_dataset import RotatingMNISTDataset
from rotating_mnist_models import Seq2SeqRotationRNN
from train_eval_utils import train_epoch, eval_epoch, eval_len_generalization, eval_velocity_generalization


def main():
    parser = argparse.ArgumentParser(description="Train & evaluate RNN models on Moving MNIST")
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--gen_seq_len', type=int, default=80)
    parser.add_argument('--input_frames', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model', choices=['grnn', 'fernn'], default='fernn')
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--decoder_conv_layers', type=int, default=4)
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples to use (default: use all)')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.0, help='Probability of using teacher forcing during training (0.0-1.0)')
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--v_list', type=int, nargs='+', default=[-40, -30, -20, -10, 0, 10, 20, 30, 40], help='List of velocities for model training')
    parser.add_argument('--data_v_list', type=int, nargs='+', default=[-40, -30, -20, -10, 0, 10, 20, 30, 40], help='List of velocities for dataset generation')
    parser.add_argument('--data_seed', type=int, default=42, help='Random seed for dataset splitting')
    parser.add_argument('--model_seed', type=int, default=None, help='Random seed for model initialization (default: random)')
    parser.add_argument('--run_name', type=str, default=None, help='Name of the run')
    parser.add_argument('--run_len_generalization', action='store_true', help='Run length generalization test')
    parser.add_argument('--gen_vel_n_seq', type=int, default=512, help='How many sequences to sample **per (vx,vy)** for generalisation test')
    parser.add_argument('--gen_vel_list', type=int, nargs='+', default=[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], help='Angular velocities (deg/frame) to evaluate for velocity generalization')
    parser.add_argument('--run_velocity_generalization', action='store_true', help='Run velocity generalization test')
    parser.add_argument('--model_save_dir', type=str, default='./fernn/rotmnist/', help='Directory to save model checkpoints')
    parser.add_argument('--load_model', type=str, default=None, help='Path to a saved model checkpoint to load for evaluation')
    parser.add_argument('--evaluate_only', action='store_true', help='Only evaluate the loaded model without training')
    parser.add_argument('--num_digits', type=int, default=2, help='Number of digits to use in the dataset')
    parser.add_argument('--batches_per_eval', type=int, default=160, help='Number of batches between evaluations')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default="FERNN", help='Wandb project')
    parser.add_argument('--wandb_dir', type=str, default='./tmp/', help='Wandb directory')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb name')

    args = parser.parse_args()

    # Set seeds for reproducibility
    # Data seed for consistent dataset splits
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)
    random.seed(args.data_seed)
    
    # Create model save directory if it doesn't exist
    os.makedirs(args.model_save_dir, exist_ok=True)

    # Initialize wandb
    wandb.init(entity=args.wandb_entity, 
               project=args.wandb_project, 
               dir=args.wandb_dir,
               config=vars(args),
               name=args.wandb_name)

    assert args.input_frames < args.seq_len, "input_frames must be less than seq_len"
    pred_frames = args.seq_len - args.input_frames

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and splits
    full = RotatingMNISTDataset(
        root=args.root,
        train=True,
        seq_len=args.seq_len,
        image_size=args.image_size,
        angular_velocities=args.data_v_list,
        num_digits=args.num_digits
    )
    val_size = int(0.1 * len(full))
    train_size = len(full) - val_size
    train_ds, val_ds = random_split(full, [train_size, val_size], 
                                    generator=torch.Generator().manual_seed(args.data_seed))
    
    test_dataset = RotatingMNISTDataset(
        root=args.root,
        train=False,
        seq_len=args.seq_len,
        image_size=args.image_size,
        angular_velocities=args.data_v_list,
        num_digits=args.num_digits
    )

    gen_test_dataset = RotatingMNISTDataset(
            root=args.root,
            train=False,
            seq_len=args.gen_seq_len,
            image_size=args.image_size,
            angular_velocities=args.data_v_list,
            num_digits=args.num_digits
    )
    gen_test_loader = DataLoader(gen_test_dataset, batch_size=args.batch_size)
    gen_pred_frames = args.gen_seq_len - args.input_frames

    # Limit training samples if specified
    if args.max_train_samples is not None and args.max_train_samples < len(train_ds):
        indices = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(args.data_seed))[:args.max_train_samples]
        train_ds = Subset(train_ds, indices)
        print(f"Limited training to {args.max_train_samples} samples")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_dataset)}")
    print(f"Input frames: {args.input_frames}, Pred frames: {pred_frames}")
    print(f"Model seed: {args.model_seed}")
    print(f"Data seed: {args.data_seed}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Model: {args.model}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Num layers: {args.num_layers}")
    print(f"Decoder conv layers: {args.decoder_conv_layers}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Teacher forcing ratio: {args.teacher_forcing_ratio}")
    print(f"Max train samples: {args.max_train_samples}")
    print(f"Image size: {args.image_size}")
    print(f"Velocity list: {args.v_list}")
    print(f"Data velocity list: {args.data_v_list}")
    num_v = len(args.v_list)
    print(f"Num v: {num_v}")
    N_lifted = 36
    print(f"N_lifted: {N_lifted}")

    # Set model initialization seed
    if args.model_seed is None:
        args.model_seed = int(time.time()) % 10000  # Use current time as seed if not provided
        wandb.config.update({"model_seed": args.model_seed}, allow_val_change=True)
    
    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)
    random.seed(args.model_seed)
    
    if args.model == 'fernn':
        model = Seq2SeqRotationRNN(
            input_channels=1,
            hidden_channels=args.hidden_size,
            height=args.image_size,
            width=args.image_size,
            h_kernel_size=args.kernel_size,
            u_kernel_size=args.kernel_size,
            v_list=args.v_list,
            N=N_lifted,
            pool_type='max',
            decoder_conv_layers=args.decoder_conv_layers
        ).to(device)
    elif args.model == 'grnn':
        assert args.v_list == [0], "GRNN only supports v_list = [0]"
        model = Seq2SeqRotationRNN(
            input_channels=1,
            hidden_channels=args.hidden_size,
            height=args.image_size,
            width=args.image_size,
            h_kernel_size=args.kernel_size,
            u_kernel_size=args.kernel_size,
            v_list=[0],
            N=N_lifted,
            pool_type='max',
            decoder_conv_layers=args.decoder_conv_layers
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Load model if specified
    if args.load_model is not None:
        print(f"Loading model from {args.load_model}")
        try:
            model.load_state_dict(torch.load(args.load_model))
            print(f"Successfully loaded model weights")
        except Exception as e:
            print(f"Failed to load model weights: {e}")

    # If evaluate_only is True, skip training and only run evaluation
    if args.evaluate_only:
        print("Running evaluation only...")
        criterion = nn.MSELoss()
        print(f"\nEvaluating {args.model} model:")
        test_loss = eval_epoch(model, test_loader, criterion, device, args.input_frames, 0, split_name="test")
        print(f"Test Loss: {test_loss:.4f}")

        wandb.log({
            f"test_loss": test_loss,
            "epoch": 0
        })

        if args.run_len_generalization:
            print("Running length generalization test...")
            gen_mean, gen_std = eval_len_generalization(model, gen_test_loader, device, args.input_frames)
            print(f"Length generalization mean MSE: {gen_mean.mean():.4f}")

            wandb.log({f"len_gen_mean_t{t+1}": gen_mean[t] for t in range(len(gen_mean))})
            wandb.log({f"len_gen_std_t{t+1}":  gen_std[t]  for t in range(len(gen_std))})
            wandb.log({f"len_gen_mean_mean_over_time": gen_mean.mean()})


        if args.run_velocity_generalization:
            print("Running velocity generalization test...")
            v_vals, mse_vec = eval_velocity_generalization(model, device, args, use_subset=True)
            print(f"Velocity generalization results:")
            for v, mse in zip(v_vals, mse_vec):
                print(f"Velocity {v}: MSE {mse:.4f}")
            
            wandb.log({f"vel_gen_err_v{v_i}": mse_vec[v_i] for v_i in range(len(v_vals))})

            # Line-plot (MSE vs. angular velocity)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(v_vals, mse_vec, marker="o")
            ax.set_xlabel("Angular velocity (deg / frame)")
            ax.set_ylabel("MSE")
            ax.set_title(f"{args.model} – velocity generalization")
            ax.grid(True)

            # Heat-map (1-D for clarity, but still useful)
            fig_hm, ax_hm = plt.subplots(figsize=(6, 1.2))
            hm = ax_hm.imshow(mse_vec[None, :], aspect="auto", cmap="viridis")
            ax_hm.set_yticks([])  # 1-D heat-map – hide y-axis
            ax_hm.set_xticks(np.arange(len(v_vals)))
            ax_hm.set_xticklabels(v_vals, rotation=45, ha="right")
            ax_hm.set_title("Velocity-generalization heat-map")
            plt.colorbar(hm, ax=ax_hm, shrink=0.6, label="MSE")

            wandb.log({
                f"vel_gen_curve"  : wandb.Image(fig),
                f"vel_gen_heatmap": wandb.Image(fig_hm),
                f"vel_gen_raw"    : wandb.Table(
                    data=list(zip(v_vals, mse_vec)),
                    columns=["velocity_deg_per_frame", "mse"]
                )
            })

            plt.close(fig)
            plt.close(fig_hm)

        return

    # Optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'test_loss': []}
    best_val_loss = float('inf')
    
    
    global_batch_idx = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, global_batch_idx = train_epoch(model, train_loader, optimizer, criterion, device, args.input_frames, args.teacher_forcing_ratio, global_batch_idx, val_loader, args.batches_per_eval)
        val_loss = eval_epoch(model, val_loader, criterion, device, args.input_frames, epoch, split_name="val")
        
        # Check if loss is NaN or infinity
        if torch.isnan(torch.tensor(train_loss)) or torch.isinf(torch.tensor(train_loss)) or \
           torch.isnan(torch.tensor(val_loss)) or torch.isinf(torch.tensor(val_loss)):
            print(f"Exiting due to NaN or Inf loss: Train Loss: {train_loss}, Val Loss: {val_loss}")
            sys.exit(1)
            
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch}/{args.epochs} | {args.model:^8} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_filename = f"{args.model}_best_model_{wandb.run.id}.pth"
            model_path = os.path.join(args.model_save_dir, model_filename)
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model at {model_path}")

            # Evaluate on test set with best model
            test_loss = eval_epoch(model, test_loader, criterion, device, args.input_frames, epoch, split_name="test")
            history['test_loss'].append(test_loss)
            print(f"Test Loss: {test_loss:.4f}")

            if args.run_len_generalization:
                max_batches = None if epoch % 10 == 0 else 10
                gen_mean, gen_std = eval_len_generalization(model, gen_test_loader, device, args.input_frames, max_batches=max_batches)

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
                use_subset = False if epoch % 10 == 0 else True
                v_vals, mse_vec = eval_velocity_generalization(model, device, args, use_subset=use_subset)

                wandb.log({f"vel_gen_err_v{v_i}": mse_vec[v_i] for v_i in range(len(v_vals))})

                # Line-plot (MSE vs. angular velocity)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(v_vals, mse_vec, marker="o")
                ax.set_xlabel("Angular velocity (deg / frame)")
                ax.set_ylabel("MSE")
                ax.set_title(f"{args.model} – velocity generalization")
                ax.grid(True)

                # Heat-map (1-D for clarity, but still useful)
                fig_hm, ax_hm = plt.subplots(figsize=(6, 1.2))
                hm = ax_hm.imshow(mse_vec[None, :], aspect="auto", cmap="viridis")
                ax_hm.set_yticks([])  # 1-D heat-map – hide y-axis
                ax_hm.set_xticks(np.arange(len(v_vals)))
                ax_hm.set_xticklabels(v_vals, rotation=45, ha="right")
                ax_hm.set_title("Velocity-generalization heat-map")
                plt.colorbar(hm, ax=ax_hm, shrink=0.6, label="MSE")

                wandb.log({
                    f"vel_gen_curve"  : wandb.Image(fig),
                    f"vel_gen_heatmap": wandb.Image(fig_hm),
                    f"vel_gen_raw"    : wandb.Table(
                        data=list(zip(v_vals, mse_vec)),
                        columns=["velocity_deg_per_frame", "mse"]
                    )
                })

                plt.close(fig)
                plt.close(fig_hm)


            # Log model path to wandb
            wandb.log({
                f"best_model_path": model_path,
                f"best_val_loss": val_loss,
                f"test_loss": test_loss, # Log test_loss for this model
                "epoch": epoch
            })
        
        # Log metrics to wandb
        wandb.log({
            f"train_loss": train_loss,
            f"val_loss": val_loss,
            "epoch": epoch
        })
    

    # Print final best model path and test result
    print("\nBest model path and test result:")
    model_filename = f"{args.model}_best_model_{wandb.run.id}.pth"
    model_path = os.path.join(args.model_save_dir, model_filename)
    test_loss = history['test_loss'][-1] if history['test_loss'] else None # Get the last recorded test loss
    print(f"{args.model}: {model_path} | Test Loss: {test_loss:.4f}" if test_loss is not None else f"{args.model}: {model_path} | Test Loss: N/A")

if __name__ == '__main__':
    main()
