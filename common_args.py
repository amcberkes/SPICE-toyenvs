"""
Shared CLI argument builders used by train.py and spice/train_spice.py.

"""

from argparse import ArgumentTypeError


def str2bool(v):
    """Accept many boolean spellings and also support flag-only usage."""
    if isinstance(v, bool):
        return v
    if v is None:
        return True  # when used as `--flag` with nargs='?' and no value
    v = str(v).strip().lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise ArgumentTypeError("Boolean value expected (True/False).")


def add_dataset_args(p):
    p.add_argument("--envs", type=int, default=100000,
                   help="Number of training environments (trajectories).")
    p.add_argument("--envs_eval", type=int, default=100,
                   help="Number of eval environments.")
    p.add_argument("--hists", type=int, default=1,
                   help="Number of histories per env in the dataset.")
    p.add_argument("--samples", type=int, default=1,
                   help="Samples per history (keep 1 for standard bandit).")

    # Environment configuration
    p.add_argument("--H", type=int, default=500, help="Horizon (episodes).")
    p.add_argument("--dim", type=int, default=5, help="Num arms / action dim for bandits.")
    p.add_argument("--lin_d", type=int, default=2, help="Latent linear bandit arm dimension.")
    p.add_argument("--var", type=float, default=0.3, help="Reward noise std for bandits.")
    p.add_argument("--cov", type=float, default=0.0, help="Correlation across arms (if applicable).")

    p.add_argument("--env", type=str, required=True,
                   help="One of {'bandit','bandit_thompson','linear_bandit','darkroom*','miniworld'}.")

    # (Used by MiniWorld generator; harmless for others)
    p.add_argument("--env_id_start", type=int, default=-1)
    p.add_argument("--env_id_end", type=int, default=-1)


def add_model_args(p):
    p.add_argument("--embd", type=int, default=64, help="Transformer embedding size.")
    p.add_argument("--head", type=int, default=1, help="Number of attention heads.")
    p.add_argument("--layer", type=int, default=6, help="Number of transformer layers.")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate (AdamW).")
    p.add_argument("--dropout", type=float, default=0.0, help="Dropout probability.")

    p.add_argument(
        "--shuffle",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Shuffle trajectories when forming batches (True/False)."
    )


def add_train_args(p):
    # Train loop
    p.add_argument("--num_epochs", type=int, default=300, help="Training epochs.")


    p.add_argument(
        "--wandb",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable W&B logging (True/False)."
    )
    p.add_argument("--wandb_entity", type=str, default=None,
                   help="W&B entity (e.g., 'david-rolnick').")
    p.add_argument("--wandb_project", type=str, default=None,
                   help="W&B project name (e.g., 'SPICE-').")
    p.add_argument("--wandb_run_name", type=str, default="",
                   help="Optional W&B run name override.")
    p.add_argument("--wandb_tags", type=str, default="",
                   help="Comma-separated tags (e.g., 'bandit,spice').")
    p.add_argument("--log_every", type=int, default=100,
                   help="Per-iteration W&B log frequency in steps.")
