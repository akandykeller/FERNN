import torch
from tqdm import tqdm
from moving_mnist_dataset import FixedVelocityMovingMNIST
from visualization import log_sequence_predictions, log_sequence_predictions_new
import numpy as np
from torch.utils.data import DataLoader


def train_epoch(model, dataloader, optimizer, criterion, device, input_frames, teacher_forcing_ratio, grad_clip=None):
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
        
        # Apply gradient clipping if specified
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss * seq.size(0)
        pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        # Log images for visualization (first batch and every 5 epochs)
        if i == 0:
            log_sequence_predictions(input_seq, target_seq, output_seq, split_name="train")

    return running_loss / len(dataloader.dataset)


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
                log_sequence_predictions(input_seq, target_seq, output_seq, split_name=split_name)

    return running_loss / len(dataloader.dataset)


def eval_len_generalization(model, dataloader, device, input_frames, subsample_t=1):
    """
    Returns:
        mean_err  – numpy array [T]  (MSE at each future step, averaged over test set)
        std_err   – numpy array [T]  (sample‑wise std at each step)
    """
    model.eval()
    first_pass = True
    with torch.no_grad():
        n_sequences = 0
        pbar = tqdm(dataloader, desc="Evaluating Length Generalization", leave=False)
        for seq, _ in pbar:
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

                log_sequence_predictions_new(inp, tgt, pred, split_name="len_gen", num_samples=10, device=device, subsample_t=subsample_t)

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


def eval_velocity_generalization(model, device, args):
    """
    Returns
    -------
    vx_vals : np.ndarray [K]   (sorted unique velocities on x-axis)
    vy_vals : np.ndarray [K]   (same on y-axis)
    err_mat : np.ndarray [K,K] (mean MSE at (vy, vx))
    """
    vx_vals = np.arange(args.gen_vel_min, args.gen_vel_max + 1, args.gen_vel_step)
    vy_vals = np.arange(args.gen_vel_min, args.gen_vel_max + 1, args.gen_vel_step)
    err_mat = np.zeros((len(vy_vals), len(vx_vals)))

    crit = torch.nn.MSELoss(reduction='none')
    model.eval()

    vel_pbar = tqdm(total=len(vy_vals)*len(vx_vals), desc="Evaluating Velocity Generalization", leave=False)
    for iy, vy in enumerate(vy_vals):
        for ix, vx in enumerate(vx_vals):
            vel_pbar.set_postfix({"vx": vx, "vy": vy})
            dataset = FixedVelocityMovingMNIST(
                vx=vx, vy=vy,
                root=args.root,
                train=False,
                seq_len=args.seq_len,
                image_size=args.image_size,
                num_digits=2,
                random=False)
            
            # Uncomment to use a subset of the dataset for faster evaluation
            # subset, _ = torch.utils.data.random_split(
            #     dataset, [args.gen_vel_n_seq, len(dataset) - args.gen_vel_n_seq],
            #     generator=torch.Generator().manual_seed(42))
            # loader = DataLoader(subset, batch_size=args.batch_size,
                                # shuffle=False)

            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            mse_sum, n_seen = 0.0, 0
            with torch.no_grad():
                batch_pbar = tqdm(loader, desc=f"vx={vx}, vy={vy}", leave=False)
                for seq, _ in batch_pbar:
                    seq = seq.to(device)
                    inp, tgt = seq[:, :args.input_frames], seq[:, args.input_frames:]
                    pred = model(inp, pred_len=tgt.size(1), teacher_forcing_ratio=0.0)
                    mse = crit(pred, tgt).mean(dim=(2, 3, 4))  # [B, T]
                    batch_mse = mse.mean().item()
                    mse_sum += batch_mse * mse.size(0)
                    n_seen  += mse.size(0)
                    batch_pbar.set_postfix({"mse": f"{batch_mse:.4f}"})
                    # Log sequence predictions for the first batch of this velocity pair
                    if batch_pbar.n == 0: # Check if it's the first batch
                        log_sequence_predictions_new(inp, tgt, pred, split_name=f"vel_gen_vx{vx}_vy{vy}", num_samples=10, device=device)
            err_mat[iy, ix] = mse_sum / n_seen
            vel_pbar.update(1)
    vel_pbar.close()

    return vx_vals, vy_vals, err_mat