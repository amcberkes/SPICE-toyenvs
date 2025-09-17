# spice/fig2_bandit_weak.py
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import re
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

try:
    import wandb
except Exception:
    wandb = None

from envs.bandit_env import BanditEnv, BanditEnvVec
from evals.eval_bandit import deploy_online_vec
from ctrls.ctrl_bandit import (
    BanditTransformerController, EmpMeanPolicy, ThompsonSamplingPolicy,
    UCBPolicy, PessMeanPolicy, OptPolicy
)
from net import Transformer
from utils import build_bandit_data_filename

# ---- SPICE (optional) ----
SPICE_AVAILABLE = True
try:
    from spice.ctrls_spice import SPICEBanditController
    from spice.net_spice import TransformerSPICE
except Exception:
    SPICE_AVAILABLE = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------- helpers ----------
def _append_tag(path, tag):
    if not tag:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{tag}{ext}"

def load_eval_trajs_from_cfg(env, n_eval, H, dim, var, cov=0.0, dataset_tag=""):
    cfg = {'horizon': H, 'dim': dim, 'var': var, 'cov': cov, 'type': 'uniform'}
    p = build_bandit_data_filename(env, n_eval, cfg, mode=2)
    p = _append_tag(p, dataset_tag)
    if not os.path.exists(p):
        raise FileNotFoundError(f"[eval] could not find eval set: {p}")
    with open(p, 'rb') as f:
        trajs = pickle.load(f)
    return trajs[:n_eval], p

def load_eval_trajs_from_path(eval_path, n_eval):
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"[eval] direct path not found: {eval_path}")
    with open(eval_path, 'rb') as f:
        trajs = pickle.load(f)
    return trajs[:n_eval], eval_path

def make_envs(eval_trajs, H, var):
    envs = [BanditEnv(tr['means'], H, var=var) for tr in eval_trajs]
    return envs, BanditEnvVec(envs)

def maybe_parse_arch_from_ckpt_path(path, default_embd, default_layer, default_head):
    embd, layer, head = default_embd, default_layer, default_head
    m = re.search(r'embd(\d+)_layer(\d+)_head(\d+)', os.path.basename(path))
    if m:
        embd = int(m.group(1)); layer = int(m.group(2)); head = int(m.group(3))
        print(f"[auto] inferred arch from '{os.path.basename(path)}': embd={embd} layer={layer} head={head}")
    else:
        print(f"[auto] using arch defaults for '{os.path.basename(path)}': embd={embd} layer={layer} head={head}")
    return embd, layer, head

def load_ckpt(ckpt_path):
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and 'state_dict' in sd:
        return sd['state_dict'], sd.get('config', None)
    return sd, None

def infer_n_layer_from_sd(sd, default=4):
    idxs = []
    for k in sd.keys():
        m = re.search(r'transformer\.h\.(\d+)\.', k)
        if m: idxs.append(int(m.group(1)))
    return max(idxs) + 1 if idxs else default

def infer_n_embd_from_sd(sd, default=64):
    if 'transformer.wte.weight' in sd and sd['transformer.wte.weight'].dim() == 2:
        return int(sd['transformer.wte.weight'].shape[1])
    k = 'transformer.h.0.attn.c_attn.weight'
    if k in sd and sd[k].dim() == 2:
        return int(sd[k].shape[1])
    return default

def infer_spice_heads_from_state_dict(sd, default=7):
    head_idxs = set()
    for k in sd.keys():
        m = re.search(r'ensemble_q\.heads\.(\d+)\.', k)
        if m: head_idxs.add(int(m.group(1)))
    return max(head_idxs) + 1 if head_idxs else default

def model_arch_str(cfg):
    return f"L{cfg.get('n_layer','?')}/H{cfg.get('n_head','?')}/E{cfg.get('n_embd','?')}"

def resolve_ckpt_path(path_or_dir, prefer_epoch=None):
    if os.path.isfile(path_or_dir):
        return path_or_dir
    if not os.path.isdir(path_or_dir):
        raise FileNotFoundError(f"Checkpoint '{path_or_dir}' is neither a file nor a directory.")
    cands = [os.path.join(path_or_dir, f) for f in os.listdir(path_or_dir) if f.endswith('.pt')]
    if not cands:
        raise FileNotFoundError(f"No .pt files found in directory '{path_or_dir}'.")
    def epoch_of(fn):
        m = re.search(r'epoch(\d+)', os.path.basename(fn))
        return int(m.group(1)) if m else -1
    if prefer_epoch is not None:
        for f in cands:
            if f"epoch{prefer_epoch}_" in os.path.basename(f) or os.path.basename(f).endswith(f"epoch{prefer_epoch}.pt"):
                return f
    return max(cands, key=epoch_of)

def is_weak_path(p: str) -> bool:
    p = (p or "").lower()
    return ("models_weak" in p) or ("weak" in os.path.basename(p))

def _to_1d(x):
    x = np.asarray(x); return x.reshape(-1)


# ---------- exploration wrapper (online only) ----------
class ExploreWrapper:
    def __init__(self, ctrl, horizon, warmup_steps=0, eps=0.0, anneal=False):
        self.ctrl = ctrl
        self.horizon = int(horizon)
        self.warm = int(warmup_steps)
        self.eps = float(eps)
        self.anneal = bool(anneal)
        self._t = 0

    def reset(self):
        self._t = 0
        if hasattr(self.ctrl, "reset"):
            self.ctrl.reset()

    def set_batch_numpy_vec(self, batch):
        if hasattr(self.ctrl, "set_batch_numpy_vec"):
            self.ctrl.set_batch_numpy_vec(batch)
        try:
            t = batch.get("context_actions", None)
            if t is not None:
                self._t = int(t.shape[1])
        except Exception:
            pass

    def _rand_onehot(self, B, A):
        idx = np.random.randint(0, A, size=(B,))
        onehot = np.zeros((B, A), dtype=np.float32)
        onehot[np.arange(B), idx] = 1.0
        return onehot

    def act_numpy_vec(self, states):
        base = self.ctrl.act_numpy_vec(states)  # [B,A]
        B, A = base.shape
        if self._t < self.warm:
            out = self._rand_onehot(B, A)
        else:
            out = base.copy()
            if self.eps > 0.0:
                eps_t = self.eps * max(0.0, 1.0 - self._t / float(self.horizon)) if self.anneal and self.horizon > 0 else self.eps
                flip = (np.random.rand(B) < eps_t)
                if flip.any():
                    out[flip] = self._rand_onehot(flip.sum(), A)
        self._t += 1
        return out


# ---------- plots ----------
def fig_offline(eval_trajs, controllers_dict, H, var, savepath,
                title_suffix, subtitle, caption, skip_set):
    skip_set = set(s.lower() for s in (skip_set or []))

    cs  = np.stack([t['context_states'][:H]      for t in eval_trajs], axis=0)
    ca  = np.stack([t['context_actions'][:H]     for t in eval_trajs], axis=0)
    cn  = np.stack([t['context_next_states'][:H] for t in eval_trajs], axis=0)
    cr  = np.stack([t['context_rewards'][:H]     for t in eval_trajs], axis=0)
    if cr.ndim == 2: cr = cr[..., None]
    batch_full = {'context_states':cs, 'context_actions':ca,
                  'context_next_states':cn, 'context_rewards':cr}

    envs, vec = make_envs(eval_trajs, H, var)
    N = len(envs)

    opt = OptPolicy(envs, batch_size=N)

    # Build offline controllers (deterministic)
    name2ctrl = {
        'Emp': EmpMeanPolicy(envs[0], online=False, batch_size=N),
        'TS': ThompsonSamplingPolicy(envs[0], std=var, sample=False,
                                     prior_mean=0.5, prior_var=1/12., warm_start=False,
                                     batch_size=N),
        'LCB': PessMeanPolicy(envs[0], const=0.8, batch_size=N),
    }
    # add user-provided ones (already BanditTransformerController / SPICEBanditController)
    for k, ctor in controllers_dict.items():
        if k.lower() in skip_set: continue
        # offline = no sampling for DPT/SPICE "policy_logits"
        if isinstance(ctor, BanditTransformerController):
            name2ctrl[k] = BanditTransformerController(ctor.model, sample=False, batch_size=N)
        elif SPICE_AVAILABLE and isinstance(ctor, SPICEBanditController):
            kw = ctor.kwargs if hasattr(ctor, "kwargs") else {}
            name2ctrl[k] = SPICEBanditController(ctor.model, batch_size=N, policy_sample=False, **kw)
        else:
            name2ctrl[k] = ctor

    horizons = np.linspace(1, H, 50, dtype=int)
    reg_hist = {k: [] for k in name2ctrl.keys()}

    for h in horizons:
        b_h = {k: v[:, :h] for k, v in batch_full.items()}
        for c in [opt] + list(name2ctrl.values()):
            c.set_batch_numpy_vec(b_h)
        rs_opt = _to_1d( vec.deploy_eval(opt)[-1] )
        for name, ctrl in name2ctrl.items():
            rs = _to_1d( vec.deploy_eval(ctrl)[-1] )
            reg_hist[name].append(rs_opt - rs)

    for k in reg_hist:
        reg_hist[k] = np.stack(reg_hist[k], axis=0)

    eps = 1e-4
    plt.figure(figsize=(6.8, 4.2))
    for label, vals in reg_hist.items():
        vals = np.maximum(vals, eps)
        m  = vals.mean(axis=1)
        se = vals.std(axis=1) / np.sqrt(vals.shape[1])
        plt.plot(horizons, m, label=label)
        plt.fill_between(horizons, m - se, m + se, alpha=0.2)

    plt.yscale('log'); plt.xlabel('Dataset size'); plt.ylabel('Suboptimality')
    plt.title(f'Offline Bandit — {title_suffix}')
    if subtitle: plt.suptitle(subtitle, y=1.02, fontsize=9)
    if caption:  plt.figtext(0.01, 0.01, caption, ha='left', va='bottom', fontsize=8)
    plt.legend(fontsize=8)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, bbox_inches='tight'); plt.close()


def fig_online(eval_trajs, controllers_dict, H, var, savepath,
               title_suffix, subtitle, caption,
               warmup_steps=0, eps_greedy=0.0, eps_anneal=False, skip_set=None):
    skip_set = set(s.lower() for s in (skip_set or []))

    envs, vec = make_envs(eval_trajs, H, var)
    N = len(envs)

    controllers = {
        'opt':   OptPolicy(envs, batch_size=N),
        'Emp':   EmpMeanPolicy(envs[0], online=True, batch_size=N),
        'UCB1.0':UCBPolicy(envs[0], const=1.0, batch_size=N),
        'TS':    ThompsonSamplingPolicy(envs[0], std=var, sample=True,
                                        prior_mean=0.5, prior_var=1/12., warm_start=False,
                                        batch_size=N),
    }
    # add user-provided ones in sampling mode
    for k, ctor in controllers_dict.items():
        if k.lower() in skip_set: continue
        if isinstance(ctor, BanditTransformerController):
            controllers[k] = BanditTransformerController(ctor.model, sample=True, batch_size=N)
        elif SPICE_AVAILABLE and isinstance(ctor, SPICEBanditController):
            kw = ctor.kwargs if hasattr(ctor, "kwargs") else {}
            controllers[k] = SPICEBanditController(ctor.model, batch_size=N, policy_sample=True, **kw)
        else:
            controllers[k] = ctor

    # Optional exploration wrapper
    if warmup_steps > 0 or eps_greedy > 0.0:
        for name in list(controllers.keys()):
            if name == 'opt': continue
            controllers[name] = ExploreWrapper(controllers[name], H, warmup_steps, eps_greedy, eps_anneal)

    # run
    all_means = {}
    for name, ctrl in controllers.items():
        print("Deploying online vectorized...")
        cm = deploy_online_vec(vec, ctrl, H)  # [H, N]
        print("Deployed online vectorized")
        all_means[name] = cm.T                # [N, H]

    opt = all_means['opt']
    regrets = {k: np.cumsum(opt - v, axis=1) for k, v in all_means.items() if k != 'opt'}

    plt.figure(figsize=(7.4,4.2))
    for k, v in regrets.items():
        m = v.mean(0); s = v.std(0) / np.sqrt(v.shape[0])
        plt.plot(m, label=k); plt.fill_between(np.arange(H), m - s, m + s, alpha=0.2)
    plt.xlabel('Episodes'); plt.ylabel('Cumulative Regret')
    plt.title(f'Online Bandit — {title_suffix}')
    if subtitle: plt.suptitle(subtitle, y=1.02, fontsize=9)
    if caption:  plt.figtext(0.01, 0.01, caption, ha='left', va='bottom', fontsize=8)
    plt.legend(fontsize=8)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, bbox_inches='tight'); plt.close()


def fig_robustness(eval_trajs, controllers_dict, H, var_list, savepath,
                   title_suffix, subtitle, caption, skip_set=None):
    skip_set = set(s.lower() for s in (skip_set or []))
    keys = []
    # stable order: baselines then provided
    base = ['Emp', 'UCB1.0', 'TS']
    keys.extend([k for k in base if k.lower() not in skip_set])
    keys.extend([k for k in controllers_dict.keys() if k.lower() not in skip_set])

    final = {k: [] for k in keys}

    for v in var_list:
        envs, vec = make_envs(eval_trajs, H, v)
        N = len(envs)
        ctrls = {}
        if 'Emp' in final:    ctrls['Emp']    = EmpMeanPolicy(envs[0], online=True, batch_size=N)
        if 'UCB1.0' in final: ctrls['UCB1.0'] = UCBPolicy(envs[0], const=1.0, batch_size=N)
        if 'TS' in final:     ctrls['TS']     = ThompsonSamplingPolicy(envs[0], std=v, sample=True,
                                                                       prior_mean=0.5, prior_var=1/12., warm_start=False,
                                                                       batch_size=N)
        for k in controllers_dict.keys():
            if k not in final: continue
            ctor = controllers_dict[k]
            if isinstance(ctor, BanditTransformerController):
                ctrls[k] = BanditTransformerController(ctor.model, sample=True, batch_size=N)
            elif SPICE_AVAILABLE and isinstance(ctor, SPICEBanditController):
                kw = ctor.kwargs if hasattr(ctor, "kwargs") else {}
                ctrls[k] = SPICEBanditController(ctor.model, batch_size=N, policy_sample=True, **kw)

        opt = deploy_online_vec(vec, OptPolicy(envs, batch_size=N), H).T
        for name, ctrl in ctrls.items():
            cm = deploy_online_vec(vec, ctrl, H).T
            reg = np.cumsum(opt - cm, axis=1)
            final[name].append(reg[:, -1].mean())

    x = np.arange(len(var_list)); W = 0.8 / max(1, len(keys))
    plt.figure(figsize=(7.4,4.2))
    for i, name in enumerate(keys):
        vals = final[name]
        plt.bar(x + i*W, vals, width=W, label=name)
    plt.xticks(x + (len(keys)-1)*W/2, [str(v) for v in var_list])
    plt.xlabel('Noise standard deviation'); plt.ylabel('Final cumulative regret')
    plt.title(f'Online Robustness — {title_suffix}')
    if subtitle: plt.suptitle(subtitle, y=1.02, fontsize=9)
    if caption:  plt.figtext(0.01, 0.01, caption, ha='left', va='bottom', fontsize=8)
    plt.legend(fontsize=8)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, bbox_inches='tight'); plt.close()


# ---------- main ----------
if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='bandit')
    ap.add_argument('--H', type=int, default=500)
    ap.add_argument('--dim', type=int, default=5)
    ap.add_argument('--var', type=float, default=0.3)       # noise std at evaluation
    ap.add_argument('--cov', type=float, default=0.0)       # affects which eval dataset is loaded when not using --eval_path
    ap.add_argument('--n_eval', type=int, default=200)

    # Checkpoints (file or directory). Order does not matter; the one under models_weak/ is auto-labeled as DPT-weak.
    ap.add_argument('--dpt_ckpt', type=str, default='', help="Path/dir to a DPT model")
    ap.add_argument('--dpt_weak_ckpt', type=str, default='', help="Path/dir to a second DPT model (often under models_weak/)")
    ap.add_argument('--spice_ckpt', type=str, default='', help="Path/dir to a SPICE checkpoint (optional)")
    ap.add_argument('--prefer_epoch', type=int, default=None, help="If directories are passed, prefer this epoch number when selecting a ckpt")

    # Architecture hints (auto-parsed from filenames when possible)
    ap.add_argument('--embd', type=int, default=64)
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--head', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--infer_arch_from_ckpt', action='store_true', default=True)

    # SPICE legacy override (rare)
    ap.add_argument('--spice_prior_override', type=float, default=None)

    # Dataset selection
    ap.add_argument('--dataset_tag', type=str, default='',
                    help="Suffix for tagged eval dataset (e.g., 'weak'). Ignored if --eval_path is given.")
    ap.add_argument('--eval_path', type=str, default='',
                    help="Direct path to eval .pkl; if provided, overrides var/cov/dataset_tag path building.")

    # Exploration aids (online only)
    ap.add_argument('--warmup_steps', type=int, default=0)
    ap.add_argument('--eps_greedy', type=float, default=0.0)
    ap.add_argument('--eps_anneal', action='store_true')

    # Filtering / speed
    ap.add_argument('--skip_baselines', type=str, default='',
                    help="Comma-list among Emp,UCB1.0,TS,LCB,DPT,DPT-weak,SPICE(...) to hide (opt is kept).")
    ap.add_argument('--spice_include', type=str, default='all',
                    help="Comma-list of SPICE variants to include (e.g., 'posterior_ctx,ucb_ctx'). Use 'all' for everything.")
    ap.add_argument('--skip_offline', action='store_true')
    ap.add_argument('--skip_online', action='store_true')
    ap.add_argument('--skip_robustness', action='store_true')
    ap.add_argument('--robust_vars', type=str, default='0.0,0.3,0.5',
                    help="Comma-separated list of noise stds for robustness bar plot")

    # NEW: allow overriding where figures are saved (default keeps prior behaviour)
    ap.add_argument('--outdir_prefix', type=str, default=os.path.join('figs','fig2_weak'),
                    help="Directory prefix where figures will be saved (default: figs/fig2_weak)")

    ap.add_argument('--epoch_label', required=True)
    ap.add_argument('--wandb_eval', action='store_true')
    args = ap.parse_args()

    skip_set = set(s.strip().lower() for s in args.skip_baselines.split(",") if s.strip())
    spice_include_set = set(s.strip().lower() for s in args.spice_include.split(",") if s.strip())

    # Resolve eval dataset
    if args.eval_path:
        eval_trajs, eval_used = load_eval_trajs_from_path(args.eval_path, args.n_eval)
    else:
        eval_trajs, eval_used = load_eval_trajs_from_cfg(args.env, args.n_eval, args.H, args.dim, args.var,
                                                         cov=args.cov, dataset_tag=args.dataset_tag)
    print(f"[eval] using dataset: {eval_used}")

    # Resolve ckpt paths (accept directories). Some may be omitted.
    paths = []
    if args.dpt_ckpt: paths.append(resolve_ckpt_path(args.dpt_ckpt, args.prefer_epoch))
    if args.dpt_weak_ckpt: paths.append(resolve_ckpt_path(args.dpt_weak_ckpt, args.prefer_epoch))
    if not paths:
        raise ValueError("Provide at least one of --dpt_ckpt or --dpt_weak_ckpt.")
    path_a = paths[0]
    path_b = paths[1] if len(paths) > 1 else None

    # Auto-assign DPT vs DPT-weak using folder name (models_weak/)
    if path_b is not None:
        a_is_weak, b_is_weak = is_weak_path(path_a), is_weak_path(path_b)
        if a_is_weak == b_is_weak:
            # both weak or both not weak: use second as weak for naming consistency
            weak_path, dpt_path = path_b, path_a
        else:
            weak_path = path_a if a_is_weak else path_b
            dpt_path  = path_b if a_is_weak else path_a
        print(f"[labeling] DPT path   : {dpt_path}")
        print(f"[labeling] DPT-weak   : {weak_path}")
    else:
        # only one path; name it DPT (or DPT-weak if it looks weak)
        if is_weak_path(path_a):
            dpt_path, weak_path = None, path_a
            print(f"[labeling] DPT-weak only: {weak_path}")
        else:
            dpt_path, weak_path = path_a, None
            print(f"[labeling] DPT only: {dpt_path}")

    # ----- Load DPT (standard) -----
    dpt_model, dpt_cfg, dpt_name = None, None, None
    if dpt_path is not None:
        embd, layer, head = args.embd, args.layer, args.head
        if args.infer_arch_from_ckpt:
            embd, layer, head = maybe_parse_arch_from_ckpt_path(dpt_path, embd, layer, head)
        dpt_sd, dpt_cfg_from_ckpt = load_ckpt(dpt_path)
        if dpt_cfg_from_ckpt is None:
            layer = infer_n_layer_from_sd(dpt_sd, default=layer)
            embd  = infer_n_embd_from_sd(dpt_sd, default=embd)
            dpt_cfg = dict(horizon=args.H, state_dim=1, action_dim=args.dim,
                           n_layer=layer, n_embd=embd, n_head=head,
                           dropout=args.dropout, test=True)
        else:
            dpt_cfg = dpt_cfg_from_ckpt.copy(); dpt_cfg.update(dict(test=True))
        dpt_model = Transformer(dpt_cfg).to(device)
        dpt_model.load_state_dict(dpt_sd); dpt_model.eval()
        dpt_name = os.path.basename(dpt_path)

    # ----- Load DPT (weak) -----
    dptw_model, dptw_cfg, dptw_name = None, None, None
    if weak_path is not None:
        embd, layer, head = args.embd, args.layer, args.head
        if args.infer_arch_from_ckpt:
            embd, layer, head = maybe_parse_arch_from_ckpt_path(weak_path, embd, layer, head)
        dptw_sd, dptw_cfg_from_ckpt = load_ckpt(weak_path)
        if dptw_cfg_from_ckpt is None:
            layer = infer_n_layer_from_sd(dptw_sd, default=layer)
            embd  = infer_n_embd_from_sd(dptw_sd, default=embd)
            dptw_cfg = dict(horizon=args.H, state_dim=1, action_dim=args.dim,
                            n_layer=layer, n_embd=embd, n_head=head,
                            dropout=args.dropout, test=True)
        else:
            dptw_cfg = dptw_cfg_from_ckpt.copy(); dptw_cfg.update(dict(test=True))
        dptw_model = Transformer(dptw_cfg).to(device)
        dptw_model.load_state_dict(dptw_sd); dptw_model.eval()
        dptw_name = os.path.basename(weak_path)

    # ----- Load SPICE (optional) -----
    spice_model, spice_cfg, spice_name = None, None, None
    if args.spice_ckpt:
        if not SPICE_AVAILABLE:
            raise RuntimeError("SPICE code not available, but --spice_ckpt was provided.")
        spice_path = resolve_ckpt_path(args.spice_ckpt, args.prefer_epoch)
        spice_sd, spice_cfg_from_ckpt = load_ckpt(spice_path)
        if spice_cfg_from_ckpt is None:
            inferred_K = infer_spice_heads_from_state_dict(spice_sd)
            inferred_L = infer_n_layer_from_sd(spice_sd, default=args.layer)
            inferred_E = infer_n_embd_from_sd(spice_sd, default=args.embd)
            spice_prior = args.spice_prior_override if args.spice_prior_override is not None else 0.1
            spice_cfg = dict(horizon=args.H, state_dim=1, action_dim=args.dim,
                             n_layer=inferred_L, n_embd=inferred_E, n_head=args.head,
                             dropout=args.dropout, test=True,
                             spice_heads=inferred_K, spice_prior=spice_prior,
                             alphaA=None, lambda_sigma=None, c_sigma=None, lambda_q=None)
            print(f"[legacy] inferred SPICE arch: L={inferred_L} E={inferred_E} H={args.head} | heads={inferred_K} prior={spice_prior}")
        else:
            spice_cfg = spice_cfg_from_ckpt.copy()
            spice_cfg.update(dict(test=True))
            print(f"[ckpt] using SPICE config from checkpoint.")
        spice_model = TransformerSPICE(spice_cfg).to(device)
        spice_model.load_state_dict(spice_sd); spice_model.eval()
        spice_name = os.path.basename(spice_path)

    # ---- Build controller map we will reuse across figures ----
    controllers_dict = {}
    if dpt_model is not None:
        controllers_dict['DPT'] = BanditTransformerController(dpt_model, sample=False, batch_size=len(eval_trajs))
    if dptw_model is not None:
        controllers_dict['DPT-weak'] = BanditTransformerController(dptw_model, sample=False, batch_size=len(eval_trajs))

    # Optional SPICE variants (filterable)
    if spice_model is not None:
        want_all = ('all' in spice_include_set)
        def want(token): return want_all or any(token in s for s in spice_include_set)

        if want('policy_logits'):
            controllers_dict['SPICE(policy_logits)'] = SPICEBanditController(spice_model, batch_size=len(eval_trajs),
                                                                             variant='policy_logits', policy_sample=False)
        if want('ts_noctx'):
            controllers_dict['SPICE(ts_noctx)'] = SPICEBanditController(spice_model, batch_size=len(eval_trajs),
                                                                        variant='ts_noctx', sample_heads=True, resample_every_step=False)
        if want('ts_ctx'):
            controllers_dict['SPICE(ts_ctx)'] = SPICEBanditController(spice_model, batch_size=len(eval_trajs),
                                                                      variant='ts_ctx', sample_heads=True, resample_every_step=False)
        if want('ucb_ctx'):
            controllers_dict['SPICE(ucb_ctx,β=0.5)'] = SPICEBanditController(spice_model, batch_size=len(eval_trajs),
                                                                             variant='ucb_ctx', beta_ucb=0.5)
        if want('posterior_ctx'):
            controllers_dict['SPICE(posterior_ctx)'] = SPICEBanditController(spice_model, batch_size=len(eval_trajs),
                                                                             variant='posterior_ctx', posterior_noise_var=args.var**2)
        if want('adapter_ctx'):
            controllers_dict['SPICE(adapter_ctx,η=0.5)'] = SPICEBanditController(spice_model, batch_size=len(eval_trajs),
                                                                                  variant='adapter_ctx', posterior_noise_var=args.var**2, adapter_eta=0.5)
        if want('qmean_ctx'):
            controllers_dict['SPICE(qmean_ctx)'] = SPICEBanditController(spice_model, batch_size=len(eval_trajs),
                                                                         variant='qmean_ctx')

    # captions / titles
    parts = []
    if dpt_name: parts.append(f"DPT: {dpt_name}")
    if dptw_name: parts.append(f"DPT-weak: {dptw_name}")
    if spice_name: parts.append(f"SPICE: {spice_name}")
    caption = "   |   ".join(parts) if parts else ""
    subtitle = []
    if dpt_cfg:  subtitle.append(f"DPT: {model_arch_str(dpt_cfg)}")
    if dptw_cfg: subtitle.append(f"Weak: {model_arch_str(dptw_cfg)}")
    if spice_cfg:
        K = spice_cfg.get('spice_heads','?'); prior = spice_cfg.get('spice_prior','?')
        subtitle.append(f"SPICE: {model_arch_str(spice_cfg)}, heads={K}, prior={prior}")
    subtitle = "  |  ".join(subtitle)

    # output paths
    tag_bits = []
    if args.epoch_label: tag_bits.append(args.epoch_label)
    if dptw_name: tag_bits.append(os.path.splitext(dptw_name)[0])
    if spice_name: tag_bits.append(os.path.splitext(spice_name)[0])
    tag = "_".join(tag_bits) if tag_bits else args.epoch_label
    outdir = os.path.join(args.outdir_prefix, args.epoch_label, (os.path.splitext(dptw_name)[0] if dptw_name else 'no_dptweak'))
    os.makedirs(outdir, exist_ok=True)

    # ---- PLOTS ----
    if not args.skip_offline:
        fig_offline(eval_trajs, controllers_dict, args.H, args.var,
                    os.path.join(outdir, f'a_offline_{tag}.png'),
                    title_suffix=tag, subtitle=subtitle, caption=caption, skip_set=skip_set)

    if not args.skip_online:
        fig_online(eval_trajs, controllers_dict, args.H, args.var,
                   os.path.join(outdir, f'b_online_{tag}.png'),
                   title_suffix=tag, subtitle=subtitle, caption=caption,
                   warmup_steps=args.warmup_steps, eps_greedy=args.eps_greedy, eps_anneal=args.eps_anneal,
                   skip_set=skip_set)

    if not args.skip_robustness:
        var_list = [float(x) for x in args.robust_vars.split(",") if x.strip()]
        fig_robustness(eval_trajs, controllers_dict, args.H, var_list,
                       savepath=os.path.join(outdir, f'c_robust_{tag}.png'),
                       title_suffix=tag, subtitle=subtitle, caption=caption, skip_set=skip_set)

    print(f"Saved figs under {outdir}/")

    if args.wandb_eval and (wandb is not None):
        run_name = f"eval_weak_{tag}"
        wandb.init(entity=os.environ.get('WANDB_ENTITY','david-rolnick'),
                   project=os.environ.get('WANDB_PROJECT','SPICE-'),
                   name=run_name,
                   config={'epoch_label': args.epoch_label,
                           'dpt_ckpt': dpt_path if dpt_path else '',
                           'dpt_weak_ckpt': weak_path if weak_path else '',
                           'spice_ckpt': args.spice_ckpt,
                           'eval_path': args.eval_path,
                           'dataset_tag': args.dataset_tag,
                           'var': args.var, 'cov': args.cov, 'H': args.H, 'dim': args.dim,
                           'skip_baselines': args.skip_baselines,
                           'spice_include': args.spice_include,
                           'robust_vars': args.robust_vars,
                           'outdir_prefix': args.outdir_prefix})
        for fn in [f'a_offline_{tag}.png', f'b_online_{tag}.png', f'c_robust_{tag}.png']:
            p = os.path.join(outdir, fn)
            if os.path.exists(p): wandb.log({fn: wandb.Image(p)})
        wandb.finish()
