import torch
from tqdm import tqdm
import wandb
import numpy as np
import random
from torch.utils.data import DataLoader
from rotating_mnist_dataset import RotatingMNISTDataset
from visualization import log_sequence_predictions_new, log_sequence_predictions_vgen


def train_epoch(model, dataloader, optimizer, criterion, device, input_frames, teacher_forcing_ratio, global_batch_idx=0, val_loader=None, batches_per_eval=None):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for i, (seq, _) in enumerate(pbar):
        seq = seq.to(device)  # (B, seq_len, C, H, W)
        input_seq = seq[:, :input_frames]
        target_seq = seq[:, input_frames:]
        pred_len = target_seq.size(1)

        optimizer.zero_grad() 
        output_seq = model(
            input_seq,
            pred_len=pred_len,
            teacher_forcing_ratio=teacher_forcing_ratio,
            target_seq=target_seq
        )  # (B, pred_len, C, H, W)
        loss = criterion(output_seq, target_seq)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss * seq.size(0)
        pbar.set_postfix({"loss": f"{batch_loss:.4f}"})
        
        # Log batch loss to wandb with global batch index
        wandb.log({
            "avg_train_batch_loss": batch_loss / seq.size(0),
            "batch_idx": global_batch_idx + i
        })

        # Run evaluation if specified number of batches have been processed
        if batches_per_eval is not None and (i + 1) % batches_per_eval == 0:
            val_loss = eval_epoch(model, val_loader, criterion, device, input_frames, 0, split_name="val")
            print(f"Batch {i + 1} | Val Loss: {val_loss:.4f}")
            wandb.log({
                "val_loss_inter_epoch": val_loss,
                "val_batch_idx": global_batch_idx + i
            })
            model.train()  # Set back to training mode

        # Log images for visualization (first batch and every 5 epochs)
        if i == 0:
            log_sequence_predictions_new(input_seq, target_seq, output_seq, split_name="in_domain_train", num_samples=10, device=device, subsample_t=1)

    return running_loss / len(dataloader.dataset), global_batch_idx + len(dataloader)


def eval_epoch(model, dataloader, criterion, device, input_frames, epoch, split_name):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for i, (seq, _) in enumerate(pbar):
            seq = seq.to(device)
            input_seq = seq[:, :input_frames]
            target_seq = seq[:, input_frames:]
            pred_len = target_seq.size(1)

            output_seq = model(
                input_seq,
                pred_len=pred_len,
                teacher_forcing_ratio=0.0
            )
            loss = criterion(output_seq, target_seq)
            batch_loss = loss.item()
            running_loss += batch_loss * seq.size(0)
            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})
            
            # Log images for visualization (first batch and every 5 epochs)
            if i == 0:
                log_sequence_predictions_new(input_seq, target_seq, output_seq, split_name="in_domain_eval", num_samples=10, device=device, subsample_t=1)

    return running_loss / len(dataloader.dataset)


def eval_len_generalization(model, dataloader, device, input_frames, max_batches=None):
    """
    Returns:
        mean_err  – numpy array [T]  (MSE at each future step, averaged over test set)
        std_err   – numpy array [T]  (sample‑wise std at each step)
        
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the test data
        device: Device to run evaluation on
        input_frames: Number of input frames to use
        max_batches: Maximum number of batches to process (None = process all)
    """
    model.eval()
    first_pass = True
    with torch.no_grad():
        n_sequences = 0
        pbar = tqdm(dataloader, desc="Evaluating Length Generalization", leave=False)
        for i, (seq, _) in enumerate(pbar):
            # Stop if we've reached the maximum number of batches
            if max_batches is not None and i >= max_batches:
                break
                
            seq = seq.to(device)
            inp, tgt = seq[:, :input_frames], seq[:, input_frames:]
            T = tgt.size(1)
            pred = model(inp, pred_len=T, teacher_forcing_ratio=0.0)

            # MSE per example per timestep  →  [B, T]
            per_ex_t = ((pred - tgt)**2).mean(dim=(2, 3, 4))  # assume (B,T,C,H,W)
            if first_pass:
                sum_err  = per_ex_t.sum(dim=0)          # [T]
                sum_err2 = (per_ex_t**2).sum(dim=0)     # [T]
                first_pass, T_global = False, T

                log_sequence_predictions_new(inp, tgt, pred, split_name="len_gen", num_samples=10, device=device, subsample_t=2)

            else:
                sum_err  += per_ex_t.sum(dim=0)
                sum_err2 += (per_ex_t**2).sum(dim=0)

            n_sequences += per_ex_t.size(0)
            
            # Update progress bar with current batch size
            pbar.set_postfix({"loss": per_ex_t.mean()})

    mean = sum_err / n_sequences
    var  = sum_err2 / n_sequences - mean**2
    std  = torch.sqrt(torch.clamp(var, min=0.0))
    return mean.cpu().numpy(), std.cpu().numpy()



def eval_velocity_generalization(model, device, args, use_subset=True):
    """
    Returns
    -------
    v_vals  : np.ndarray [K]   (sorted angular velocities)
    err_vec : np.ndarray [K]   (mean MSE at each v)
    """
    v_vals = np.array(sorted(args.gen_vel_list))
    err_vec = np.zeros(len(v_vals))

    crit = torch.nn.MSELoss(reduction='none')
    model.eval()

    vel_pbar = tqdm(v_vals, desc="Evaluating Velocity Generalization", leave=False)
    for i, v in enumerate(vel_pbar):
        vel_pbar.set_postfix({"v": v})
        dataset = RotatingMNISTDataset(
            root=args.root,
            train=False,
            seq_len=args.seq_len,
            image_size=args.image_size,
            angular_velocities=[v],
            num_digits=args.num_digits,
            random=False
        )
        
        def seed_worker(worker_id):
            # derive a unique seed for each worker from the base seed
            worker_seed = args.data_seed + worker_id
            torch.manual_seed(worker_seed)
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # if use_subset:
        #     subset, _ = torch.utils.data.random_split(
        #         dataset,
        #         [min(args.gen_vel_n_seq, len(dataset)),
        #         max(0, len(dataset) - min(args.gen_vel_n_seq, len(dataset)))],
        #         generator=torch.Generator().manual_seed(42)
        #     )
        #     loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_worker)
        # else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_worker)

        mse_sum, n_seen = 0.0, 0
        with torch.no_grad():
            batch_pbar = tqdm(loader, desc=f"v={v}", leave=False)
            for seq, _ in batch_pbar:
                seq = seq.to(device)
                inp, tgt = seq[:, :args.input_frames], seq[:, args.input_frames:]
                pred = model(inp, pred_len=tgt.size(1), teacher_forcing_ratio=0.0)
                mse = crit(pred, tgt).mean(dim=(2, 3, 4))  # [B,T]
                batch_mse = mse.mean().item()
                mse_sum += batch_mse * mse.size(0)
                n_seen += mse.size(0)
                batch_pbar.set_postfix({"mse": f"{batch_mse:.4f}"})
                if batch_pbar.n == 0:  # first batch
                    log_sequence_predictions_vgen(inp, tgt, pred, split_name=f"vel_gen_v{v}", num_samples=10, device=device)
                if use_subset and n_seen > 100:
                    break

        err_vec[i] = mse_sum / n_seen
    vel_pbar.close()
    return v_vals, err_vec