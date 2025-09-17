import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import common_args
from dataset import Dataset
from spice.net_spice import TransformerSPICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _append_tag(path: str, tag: str) -> str:
    if not tag:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{tag}{ext}"


def _get_logits(outputs, action_dim: int):
    """Return logits tensor from model outputs (robust to dict return)."""
    if isinstance(outputs, dict):
        for k in ("policy_logits", "logits", "action_logits", "pi_logits"):
            if k in outputs:
                return outputs[k]
        # fallback: any tensor with trailing dim == action_dim
        for v in outputs.values():
            if torch.is_tensor(v) and v.dim() >= 2 and v.shape[-1] == action_dim:
                return v
        raise RuntimeError("Could not find logits in SPICE outputs dict.")
    return outputs


def _safe_beh_probs(batch):
    """
    Return behavior probabilities per action if present, else None.

    Accepted shapes:
      - [B, H, A]  -> average over H
      - [B, A]     -> use directly
    """
    p = batch.get("context_action_probs", None)
    if p is None:
        return None
    if torch.is_tensor(p):
        if p.dim() == 3:
            return p.float().mean(dim=1)
        if p.dim() == 2:
            return p.float()
    # numpy -> tensor
    p = torch.as_tensor(p)
    if p.dim() == 3:
        return p.float().mean(dim=1)
    return p.float()


def _weighted_ce(logits, labels_onehot, weights):
    """
    Self-normalized, per-sample weighted CE over positions.

    Args
    ----
    logits: [B, H, A]
    labels_onehot: [B, A]  (one-hot)
    weights: [B]           (>=0), arbitrary scale

    Returns
    -------
    scalar loss
    """
    B, H, A = logits.shape
    target_idx = labels_onehot.argmax(dim=-1)  # [B]
    ce_flat = F.cross_entropy(
        logits.reshape(B * H, A),
        target_idx.unsqueeze(1).repeat(1, H).reshape(-1),
        reduction="none",
    )  # [B*H]
    # self-normalize to keep average weight = 1
    w = weights / weights.mean().clamp_min(1e-8)   # [B]
    w_flat = w.repeat_interleave(H)                # [B*H]
    return (w_flat * ce_flat).mean()


def _policy_weights_from_q(Q, labels_onehot, beh_probs, action_dim, args):
    """
    Build per-sample weights using:
      - Importance (π_u / π_b)
      - Advantage (exp(adv / τ))
      - Epistemic (1 + λσ * std)

    Inputs
    ------
    Q: [B, H, K, A]
    labels_onehot: [B, A]
    beh_probs: [B, A] or None

    Returns
    -------
    w: [B] (>=0), unnormalized (we self-normalize in _weighted_ce)
    """
    B, H, K, A = Q.shape
    dev = Q.device
    idx = labels_onehot.argmax(dim=-1)  # [B]

    w = torch.ones(B, device=dev)

    # 1) Importance weighting (π_u / π_b)
    if args.get("policy_iw", False) and (beh_probs is not None):
        pb = beh_probs.to(dev).gather(1, idx.unsqueeze(1)).squeeze(1).clamp_min(1e-6)  # [B]
        pi_u = 1.0 / float(action_dim)
        w_is = torch.clamp(pi_u / pb, 0.0, float(args.get("iw_clip", 10.0)))
        w = w * w_is

    # 2) Advantage weighting
    if args.get("policy_adv", False):
        # mean over heads and time -> per-action value estimate
        q_mean = Q.mean(dim=2).mean(dim=1)                 # [B, A]
        v = q_mean.mean(dim=-1, keepdim=True)              # [B, 1] baseline
        a = q_mean.gather(1, idx.unsqueeze(1)) - v         # [B, 1]
        w_adv = torch.exp(a.squeeze(1) / float(args.get("adv_temp", 0.1)))
        w_adv = torch.clamp(w_adv, 1e-3, float(args.get("adv_clip", 10.0)))
        w = w * w_adv

    #  3) Epistemic weighting (disagreement across heads)
    if args.get("policy_epistemic", False):
        q_std = Q.std(dim=2).mean(dim=1)                   # [B, A]
        s = q_std.gather(1, idx.unsqueeze(1)).squeeze(1)   # [B]
        w_ep = 1.0 + float(args.get("epistemic_coef", 1.0)) * s
        w_ep = torch.clamp(w_ep, 1e-3, float(args.get("epistemic_clip", 10.0)))
        w = w * w_ep

    return w


def main():
    parser = argparse.ArgumentParser()
    # Standard project flags
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_train_args(parser)

    # Extras
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_tag", type=str, default="weak",
                        help="Suffix appended to dataset filenames (e.g., 'weak').")
    parser.add_argument("--models_dir", type=str, default="models_weak",
                        help="Directory to save SPICE checkpoints.")
    parser.add_argument("--spice_heads", type=int, default=7)
    parser.add_argument("--spice_prior", type=float, default=0.1)
    parser.add_argument("--train_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="")

    # Weighting knobs
    parser.add_argument("--policy_iw", action="store_true",
                        help="Use π_u/π_b importance weights when beh probs available.")
    parser.add_argument("--policy_adv", action="store_true",
                        help="Use exp(adv/τ) weighting.")
    parser.add_argument("--policy_epistemic", action="store_true",
                        help="Use (1+λσ·std) weighting.")
    parser.add_argument("--detach_policy_weights", action="store_true",
                        help="Stop gradients flowing through policy weights.")

    parser.add_argument("--iw_clip", type=float, default=10.0)
    parser.add_argument("--adv_temp", type=float, default=0.1)
    parser.add_argument("--adv_clip", type=float, default=10.0)
    parser.add_argument("--epistemic_coef", type=float, default=1.0)
    parser.add_argument("--epistemic_clip", type=float, default=10.0)

    # Q training & regularization
    parser.add_argument("--train_q", action="store_true",
                        help="Enable masked regression + shrinkage Q-loss.")
    parser.add_argument("--q_loss_coef", type=float, default=0.5)
    parser.add_argument("--anchor_coef", type=float, default=1e-6)

    args = vars(parser.parse_args())


    seed = int(args["seed"])
    torch.manual_seed(seed if seed >= 0 else 0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed if seed >= 0 else 0)
    np.random.seed(seed if seed >= 0 else 0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env = args["env"]
    n_envs = int(args["envs"])
    H = int(args["H"])
    dim = int(args["dim"])
    var = float(args["var"])
    cov = float(args["cov"])
    n_embd = int(args["embd"])
    n_head = int(args["head"])
    n_layer = int(args["layer"])
    dropout = float(args["dropout"])
    lr = float(args["lr"])
    num_epochs = int(args["num_epochs"])
    shuffle = bool(args["shuffle"])
    log_every = int(args.get("log_every", 200))

    if args["train_path"] and args["test_path"]:
        path_train = args["train_path"]
        path_test  = args["test_path"]
    else:
        ds_cfg = {"dim": dim, "var": var, "cov": cov, "type": "uniform", "horizon": H,
                  "n_hists": int(args["hists"]), "n_samples": int(args["samples"])}
        if env == "bandit":
            from utils import build_bandit_data_filename
            path_train = build_bandit_data_filename(env, n_envs, ds_cfg, mode=0)
            path_test  = build_bandit_data_filename(env, n_envs, ds_cfg, mode=1)
        elif env == "linear_bandit":
            from utils import build_linear_bandit_data_filename
            path_train = build_linear_bandit_data_filename(env, n_envs, ds_cfg, mode=0)
            path_test  = build_linear_bandit_data_filename(env, n_envs, ds_cfg, mode=1)
        elif env.startswith("darkroom"):
            from utils import build_darkroom_data_filename
            path_train = build_darkroom_data_filename(env, n_envs, ds_cfg, mode=0)
            path_test  = build_darkroom_data_filename(env, n_envs, ds_cfg, mode=1)
        else:
            raise NotImplementedError(f"env={env}")
        tag = args["dataset_tag"]
        path_train = _append_tag(path_train, tag)
        path_test  = _append_tag(path_test,  tag)

    if not os.path.exists(path_train):
        raise FileNotFoundError(f"[data] train file not found: {path_train}")
    if not os.path.exists(path_test):
        raise FileNotFoundError(f"[data] test file not found: {path_test}")

    print("========== SPICE training ==========", flush=True)
    print("Args:", args, flush=True)
    print("[cfg] env:", env, "| H:", H, "dim:", dim, "var:", var, "cov:", cov, flush=True)
    print("[cfg] model:", f"L{n_layer}/H{n_head}/E{n_embd}, dropout={dropout}", flush=True)
    print("[cfg] spice:", f"heads={args['spice_heads']} prior={args['spice_prior']}", flush=True)
    print("[data] train:", path_train, flush=True)
    print("[data] test :", path_test, flush=True)
    os.makedirs(args["models_dir"], exist_ok=True)
    os.makedirs("figs/loss", exist_ok=True)

    dataset_config = {
        "horizon": H,
        "state_dim": 1,
        "action_dim": dim,
        "shuffle": shuffle,
        "dropout": dropout,
        "test": False,
        "store_gpu": False,
    }
    train_dataset = Dataset(path_train, dataset_config)
    test_dataset  = Dataset(path_test,  dataset_config)

    loader_kwargs = {"batch_size": 64, "shuffle": shuffle, "num_workers": 0, "pin_memory": False}
    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_kwargs)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  **loader_kwargs)

    spice_cfg = dict(
        horizon=H, state_dim=1, action_dim=dim,
        n_layer=n_layer, n_embd=n_embd, n_head=n_head,
        dropout=dropout, shuffle=shuffle, test=False, store_gpu=False,
        spice_heads=int(args["spice_heads"]), spice_prior=float(args["spice_prior"]),
    )
    model = TransformerSPICE(spice_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce_sum = nn.CrossEntropyLoss(reduction="sum")  # for test logging

    tag = (f"bandit_spice_{args['dataset_tag']}_embd{n_embd}_layer{n_layer}_head{n_head}_"
           f"envs{n_envs}_H{H}_d{dim}_seed{seed}")
    log_path = os.path.join("figs", "loss", f"{tag}_logs.txt")
    open(log_path, "w").close()

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(str(msg) + "\n")

    log(f"[start] epochs={num_epochs} lr={lr} device={device}")

    # train
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        #  Eval (unweighted CE on test) 
        model.eval()
        test_sum = 0.0
        n_eval = 0
        with torch.no_grad():
            for i, batch in enumerate(test_loader, 1):
                if i % log_every == 0:
                    log(f"[eval] epoch {epoch:04d} | batch {i}/{len(test_loader)}")
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                logits = _get_logits(model(batch), dim)  # [B,H,A]
                true = batch["optimal_actions"].to(device)  # [B,A]
                B, H_, A = logits.shape
                target_idx = true.argmax(dim=-1)  # [B]
                loss = ce_sum(
                    logits.reshape(B * H_, A),
                    target_idx.unsqueeze(1).repeat(1, H_).reshape(-1),
                )
                test_sum += loss.item() / H_
                n_eval += B
        test_loss = test_sum / max(1, n_eval)

        #  Train 
        model.train()
        train_accum = 0.0
        n_train = 0
        for i, batch in enumerate(train_loader, 1):
            if i % log_every == 0:
                log(f"[train] epoch {epoch:04d} | batch {i}/{len(train_loader)}")
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            # Forward with Q
            logits, Q = model(batch, return_q=True)  # logits:[B,H,A], Q:[B,H,K,A]
            true = batch["optimal_actions"].to(device)  # [B,A]
            B, H_, A = logits.shape

            # Build policy weights
            beh_probs = _safe_beh_probs(batch)            # [B,A] or None
            w = _policy_weights_from_q(Q, true, beh_probs, A, args)  # [B]
            if args.get("detach_policy_weights", False):
                w = w.detach()

            # Weighted policy CE
            policy_loss = _weighted_ce(logits, true, w)

            # Q supervised loss (masked regression + shrinkage)
            q_loss = torch.zeros((), device=device)
            if args.get("train_q", False):
                # Q mean across heads (time-aware for masked term)
                q_mean_t = Q.mean(dim=2)                 # [B,H,A]
                q_mean   = q_mean_t.mean(dim=1)          # [B,A]

                # Context one-hots and rewards
                ca  = batch["context_actions"]           # [B,H,A] one-hot
                idx = ca.argmax(dim=-1).long()           # [B,H]
                r   = batch["context_rewards"]
                if r.dim() == 3: r = r.squeeze(-1)       # [B,H]

                # (1) Masked regression on taken actions: Q(h_t, a_t) ≈ r_t
                q_taken = q_mean_t.gather(2, idx.unsqueeze(-1)).squeeze(-1)  # [B,H]
                loss_masked = F.mse_loss(q_taken, r)

                # (2) Simple Bayesian shrinkage per arm
                dev = q_mean_t.device
                counts = torch.zeros(B, A, device=dev)
                sums   = torch.zeros(B, A, device=dev)
                counts.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float, device=dev))  # [B,A]
                sums.scatter_add_(1, idx, r)                                                     # [B,A]
                emp = sums / counts.clamp_min(1.0)                                               # [B,A]

                prior_mean = 0.5
                prior_var  = 1.0 / 12.0
                obs_var    = float(var) ** 2
                post_w     = obs_var / (obs_var + counts * prior_var)       # [B,A]
                post_mean  = post_w * prior_mean + (1.0 - post_w) * emp     # [B,A]

                mask = (counts > 0).float()                                 # [B,A]
                # Proper masked mean:
                diff = (q_mean - post_mean) * mask
                den  = mask.sum().clamp_min(1.0)
                loss_shrink = (diff.pow(2).sum()) / den

                q_loss = 0.5 * loss_masked + 0.5 * loss_shrink

            # Tiny anchor (keep heads near init to preserve diversity)
            anchor_loss = torch.zeros((), device=device)
            if args.get("anchor_coef", 0.0) > 0:
                anchor_loss = model.ensemble_q.anchor_loss()

            total = policy_loss + args.get("q_loss_coef", 0.5) * q_loss + args.get("anchor_coef", 0.0) * anchor_loss

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            train_accum += policy_loss.detach().item()
            n_train += 1

        train_loss = train_accum / max(1, n_train)
        dt = time.time() - t0
        log(f"Epoch {epoch:04d} | trainCE {train_loss:.6f} | testCE {test_loss:.6f} | "
            f"q {q_loss.detach().item():.6f} | {dt:.1f}s")

        # Save every 50 epochs and at the end
        if (epoch % 50) == 0 or (epoch == num_epochs):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt = os.path.join(args["models_dir"], f"bandit_spice_epoch{epoch}_{ts}.pt")
            torch.save(model.state_dict(), ckpt)
            log(f"[save] {ckpt}")

    # Final safeguard
    if num_epochs % 50 != 0:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt = os.path.join(args["models_dir"], f"bandit_spice_final_{ts}.pt")
        torch.save(model.state_dict(), ckpt)
        log(f"[save-final] {ckpt}")


if __name__ == "__main__":
    main()
