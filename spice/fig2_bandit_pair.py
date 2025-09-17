# spice/fig2_bandit_pair.py
import os, re, pickle, argparse, numpy as np, matplotlib.pyplot as plt, torch
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
from spice.ctrls_spice import SPICEBanditController
from spice.net_spice import TransformerSPICE
from net import Transformer
from utils import build_bandit_data_filename

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _append_tag(path, tag):
    if not tag:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{tag}{ext}"


# ---------- helpers ----------
def load_eval_trajs(env, n_eval, H, dim, var, cov=0.0, dataset_tag=""):
    cfg = {'horizon': H, 'dim': dim, 'var': var, 'cov': cov, 'type': 'uniform'}
    p = build_bandit_data_filename(env, n_eval, cfg, mode=2)
    p = _append_tag(p, dataset_tag)
    if not os.path.exists(p):
        raise FileNotFoundError(f"[eval] could not find eval set: {p}")
    with open(p, 'rb') as f:
        trajs = pickle.load(f)
    return trajs[:n_eval]

def make_envs(eval_trajs, H, var):
    envs = [BanditEnv(tr['means'], H, var=var) for tr in eval_trajs]
    return envs, BanditEnvVec(envs)

def maybe_parse_dpt_arch_from_ckpt_path(path, default_embd, default_layer, default_head):
    embd, layer, head = default_embd, default_layer, default_head
    m = re.search(r'embd(\d+)_layer(\d+)_head(\d+)', os.path.basename(path))
    if m:
        embd = int(m.group(1)); layer = int(m.group(2)); head = int(m.group(3))
        print(f"[auto] inferred DPT arch from ckpt name: embd={embd} layer={layer} head={head}")
    else:
        print(f"[auto] using DPT arch defaults: embd={embd} layer={layer} head={head}")
    return embd, layer, head

def load_ckpt(ckpt_path):
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and 'state_dict' in sd:
        return sd['state_dict'], sd.get('config', None)
    return sd, None

def infer_spice_heads_from_state_dict(sd):
    head_idxs = set()
    for k in sd.keys():
        m = re.search(r'ensemble_q\.heads\.(\d+)\.', k)
        if m: head_idxs.add(int(m.group(1)))
    return max(head_idxs) + 1 if head_idxs else 7

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

def spice_desc_from_cfg(cfg):
    parts = []
    for k in ['spice_heads','spice_prior','alphaA','lambda_sigma','c_sigma','lambda_q']:
        if k in cfg and cfg[k] is not None:
            parts.append(f"{k}={cfg[k]}")
    arch = f"L{cfg.get('n_layer','?')}/H{cfg.get('n_head','?')}/E{cfg.get('n_embd','?')}"
    parts.append(f"arch={arch}")
    return "SPICE: " + ", ".join(parts)

def _to_1d(x):
    x = np.asarray(x)
    return x.reshape(-1)


# ---------- lightweight exploration wrapper (for online only) ----------
class ExploreWrapper:
    def __init__(self, ctrl, horizon, warmup_steps=0, eps=0.0, anneal=False):
        self.ctrl = ctrl
        self.horizon = int(horizon)
        self.warm = int(warmup_steps)
        self.eps = float(eps)
        self.anneal = bool(anneal)
        self._t = 0
        self._last_A = None

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
        self._last_A = A
        if self._t < self.warm:
            out = self._rand_onehot(B, A)
        else:
            out = base.copy()
            if self.eps > 0.0:
                if self.anneal and self.horizon > 0:
                    eps_t = self.eps * max(0.0, 1.0 - self._t / float(self.horizon))
                else:
                    eps_t = self.eps
                flip = (np.random.rand(B) < eps_t)
                if flip.any():
                    out[flip] = self._rand_onehot(flip.sum(), A)
        self._t += 1
        return out


def _apply_filters_to_controllers(controllers, spice_include_set, skip_baselines_set):
    out = {}
    for name, ctrl in controllers.items():
        lname = name.lower()
        is_spice = lname.startswith("spice(")
        is_baseline = not is_spice and lname in {"emp", "ucb1.0", "ts", "dpt", "opt"}
        if is_baseline and lname in skip_baselines_set:
            continue
        if is_spice and spice_include_set and "all" not in spice_include_set:
            keep = any(token in lname for token in spice_include_set)
            if not keep:
                continue
        out[name] = ctrl
    return out


# ---------- figures ----------
def fig_offline(eval_trajs, dpt_model, spice_model, H, var, savepath,
                title_suffix, subtitle, caption,
                spice_include_set=None, skip_baselines_set=None):
    spice_include_set = set() if spice_include_set is None else spice_include_set
    skip_baselines_set = set() if skip_baselines_set is None else skip_baselines_set

    cs  = np.stack([t['context_states'][:H]      for t in eval_trajs], axis=0)
    ca  = np.stack([t['context_actions'][:H]     for t in eval_trajs], axis=0)
    cn  = np.stack([t['context_next_states'][:H] for t in eval_trajs], axis=0)
    cr  = np.stack([t['context_rewards'][:H]     for t in eval_trajs], axis=0)
    if cr.ndim == 2: cr = cr[..., None]
    batch_full = {'context_states':cs, 'context_actions':ca,
                  'context_next_states':cn, 'context_rewards':cr}

    envs, vec = make_envs(eval_trajs, H, var)
    N = len(envs)

    # controllers (deterministic offline)
    opt    = OptPolicy(envs, batch_size=N)
    emp    = EmpMeanPolicy(envs[0], online=False, batch_size=N)
    thmp   = ThompsonSamplingPolicy(envs[0], std=var, sample=False,
                                    prior_mean=0.5, prior_var=1/12., warm_start=False,
                                    batch_size=N)
    lcb    = PessMeanPolicy(envs[0], const=0.8, batch_size=N)
    dpt    = BanditTransformerController(dpt_model, sample=False, batch_size=N)
    spice0 = SPICEBanditController(spice_model, batch_size=N, variant='policy_logits', policy_sample=False)
    spice1 = SPICEBanditController(spice_model, batch_size=N, variant='ts_noctx', sample_heads=True, resample_every_step=False)
    spice2 = SPICEBanditController(spice_model, batch_size=N, variant='ucb_ctx', beta_ucb=0.5)
    spice3 = SPICEBanditController(spice_model, batch_size=N, variant='posterior_ctx', posterior_noise_var=var**2)
    spice4 = SPICEBanditController(spice_model, batch_size=N, variant='adapter_ctx', posterior_noise_var=var**2, adapter_eta=0.5)
    spice5 = SPICEBanditController(spice_model, batch_size=N, variant='ts_ctx', sample_heads=True, resample_every_step=False)
    spice6 = SPICEBanditController(spice_model, batch_size=N, variant='qmean_ctx')

    name2ctrl = {
        'Emp': emp, 'TS': thmp, 'LCB': lcb, 'DPT': dpt,
        'SPICE(policy_logits)': spice0,
        'SPICE(ts_noctx det)':  spice1,
        'SPICE(ucb_ctx,β=0.5)': spice2,
        'SPICE(posterior_ctx)': spice3,
        'SPICE(adapter_ctx,η=0.5)': spice4,
        'SPICE(old no-TS ctx)': spice5,
        'SPICE(qmean_ctx)': spice6,
    }
    filtered = _apply_filters_to_controllers(name2ctrl, spice_include_set, skip_baselines_set)

    horizons = np.linspace(1, H, 50, dtype=int)
    reg_hist = {k: [] for k in filtered.keys()}

    for h in horizons:
        b_h = {k: v[:, :h] for k, v in batch_full.items()}
        for c in [opt] + list(filtered.values()):
            c.set_batch_numpy_vec(b_h)
        rs_opt = _to_1d( vec.deploy_eval(opt)[-1] )
        for name, ctrl in filtered.items():
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


def fig_online(eval_trajs, dpt_model, spice_model, H, var, savepath,
               title_suffix, subtitle, caption,
               warmup_steps=0, eps_greedy=0.0, eps_anneal=False,
               spice_include_set=None, skip_baselines_set=None):
    spice_include_set = set() if spice_include_set is None else spice_include_set
    skip_baselines_set = set() if skip_baselines_set is None else skip_baselines_set

    envs, vec = make_envs(eval_trajs, H, var)
    N = len(envs)

    controllers = {
        'opt':   OptPolicy(envs, batch_size=N),
        'Emp':   EmpMeanPolicy(envs[0], online=True, batch_size=N),
        'UCB1.0':UCBPolicy(envs[0], const=1.0, batch_size=N),
        'TS':    ThompsonSamplingPolicy(envs[0], std=var, sample=True,
                                        prior_mean=0.5, prior_var=1/12., warm_start=False,
                                        batch_size=N),
        'DPT':   BanditTransformerController(dpt_model, sample=True, batch_size=N),
        'SPICE(policy_logits)': SPICEBanditController(spice_model, batch_size=N, variant='policy_logits', policy_sample=True),
        'SPICE(ts_noctx)':     SPICEBanditController(spice_model, batch_size=N, variant='ts_noctx', sample_heads=True, resample_every_step=False),
        'SPICE(ts_ctx)':       SPICEBanditController(spice_model, batch_size=N, variant='ts_ctx',    sample_heads=True, resample_every_step=False),
        'SPICE(ucb_ctx,β=0.5)': SPICEBanditController(spice_model, batch_size=N, variant='ucb_ctx', beta_ucb=0.5),
        'SPICE(posterior_ctx)': SPICEBanditController(spice_model, batch_size=N, variant='posterior_ctx', posterior_noise_var=var**2),
        'SPICE(adapter_ctx,η=0.5)': SPICEBanditController(spice_model, batch_size=N, variant='adapter_ctx', posterior_noise_var=var**2, adapter_eta=0.5),
        'SPICE(qmean_ctx)':    SPICEBanditController(spice_model, batch_size=N, variant='qmean_ctx'),
    }
    controllers = _apply_filters_to_controllers(controllers, spice_include_set, skip_baselines_set)

    if warmup_steps > 0 or eps_greedy > 0.0:
        controllers = {k: ExploreWrapper(v, H, warmup_steps, eps_greedy, eps_anneal) for k, v in controllers.items()}

    all_means = {}
    for name, ctrl in controllers.items():
        cm = deploy_online_vec(vec, ctrl, H)  # [H, N]
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


def fig_robustness(eval_trajs, dpt_model, spice_model, H, var_list, savepath,
                   title_suffix, subtitle, caption,
                   warmup_steps=0, eps_greedy=0.0, eps_anneal=False,
                   spice_include_set=None, skip_baselines_set=None):
    spice_include_set = set() if spice_include_set is None else spice_include_set
    skip_baselines_set = set() if skip_baselines_set is None else skip_baselines_set

    want_ts_noctx = ("all" in spice_include_set) or any("ts_noctx" in s for s in spice_include_set)
    want_posterior = ("all" in spice_include_set) or any("posterior_ctx" in s for s in spice_include_set)
    want_adapter = ("all" in spice_include_set) or any("adapter_ctx" in s for s in spice_include_set)

    final = {'Emp':[], 'UCB1.0':[], 'TS':[], 'DPT':[]}
    if want_ts_noctx: final['SPICE(ts_noctx)'] = []
    if want_posterior: final['SPICE(posterior_ctx)'] = []
    if want_adapter: final['SPICE(adapter_ctx,η=0.5)'] = []

    for v in var_list:
        envs, vec = make_envs(eval_trajs, H, v)
        N = len(envs)
        ctrls = {
            'Emp':   EmpMeanPolicy(envs[0], online=True, batch_size=N),
            'UCB1.0':UCBPolicy(envs[0], const=1.0, batch_size=N),
            'TS':    ThompsonSamplingPolicy(envs[0], std=v, sample=True,
                                            prior_mean=0.5, prior_var=1/12., warm_start=False,
                                            batch_size=N),
            'DPT':   BanditTransformerController(dpt_model, sample=True, batch_size=N),
        }
        if want_ts_noctx:
            ctrls['SPICE(ts_noctx)'] = SPICEBanditController(spice_model, batch_size=N,
                                                             variant='ts_noctx', sample_heads=True, resample_every_step=False)
        if want_posterior:
            ctrls['SPICE(posterior_ctx)'] = SPICEBanditController(spice_model, batch_size=N,
                                                                  variant='posterior_ctx', posterior_noise_var=v**2)
        if want_adapter:
            ctrls['SPICE(adapter_ctx,η=0.5)'] = SPICEBanditController(spice_model, batch_size=N,
                                                                      variant='adapter_ctx', posterior_noise_var=v**2, adapter_eta=0.5)
        ctrls = _apply_filters_to_controllers(ctrls, spice_include_set, skip_baselines_set)

        opt = deploy_online_vec(vec, OptPolicy(envs, batch_size=N), H).T
        for name, ctrl in ctrls.items():
            cm = deploy_online_vec(vec, ctrl, H).T
            reg = np.cumsum(opt - cm, axis=1)
            final.setdefault(name, []).append(reg[:, -1].mean())

    keys = list(final.keys())
    x = np.arange(len(var_list)); W = 0.8 / max(1, len(keys))
    plt.figure(figsize=(7.4,4.2))
    for i,(name) in enumerate(keys):
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
    ap.add_argument('--var', type=float, default=0.3)
    ap.add_argument('--n_eval', type=int, default=200)

    ap.add_argument('--dpt_ckpt', required=True)
    ap.add_argument('--spice_ckpt', required=True)

    # DPT arch (may be auto-parsed from ckpt filename)
    ap.add_argument('--embd', type=int, default=64)
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--head', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--infer_dpt_arch_from_ckpt', action='store_true', default=True)

    # Legacy SPICE ckpts
    ap.add_argument('--spice_prior_override', type=float, default=None)

    # Exploration aids (online only)
    ap.add_argument('--warmup_steps', type=int, default=0)
    ap.add_argument('--eps_greedy', type=float, default=0.0)
    ap.add_argument('--eps_anneal', action='store_true')

    # Filtering
    ap.add_argument('--spice_include', type=str, default='all',
                    help="Comma-list of SPICE variants to include (e.g., 'posterior_ctx'). Use 'all' for everything.")
    ap.add_argument('--skip_baselines', type=str, default='',
                    help="Comma-list among Emp,UCB1.0,TS,DPT to hide (opt is kept).")

    # NEW: dataset tag for eval set
    ap.add_argument('--dataset_tag', type=str, default='',
                    help="Suffix for tagged eval dataset (e.g., 'weak').")

    ap.add_argument('--epoch_label', required=True)
    ap.add_argument('--wandb_eval', action='store_true')
    args = ap.parse_args()

    spice_include_set = set(s.strip().lower() for s in args.spice_include.split(",") if s.strip())
    skip_baselines_set = set(s.strip().lower() for s in args.skip_baselines.split(",") if s.strip())

    # DPT model
    if args.infer_dpt_arch_from_ckpt:
        args.embd, args.layer, args.head = maybe_parse_dpt_arch_from_ckpt_path(
            args.dpt_ckpt, args.embd, args.layer, args.head)
    dpt_cfg = dict(horizon=args.H, state_dim=1, action_dim=args.dim,
                   n_layer=args.layer, n_embd=args.embd, n_head=args.head,
                   dropout=args.dropout, test=True)
    dpt_model = Transformer(dpt_cfg).to(device)
    dpt_sd, _ = load_ckpt(args.dpt_ckpt)
    dpt_model.load_state_dict(dpt_sd); dpt_model.eval()

    # SPICE model
    spice_sd, spice_cfg_from_ckpt = load_ckpt(args.spice_ckpt)
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
        subtitle = spice_desc_from_cfg(spice_cfg)
        print(f"[legacy] inferred SPICE arch: L={inferred_L} E={inferred_E} H={args.head} | heads={inferred_K} prior={spice_prior}")
    else:
        spice_cfg = spice_cfg_from_ckpt.copy()
        spice_cfg.update(dict(test=True))
        subtitle = spice_desc_from_cfg(spice_cfg)
        print(f"[ckpt] using SPICE config from checkpoint: {subtitle}")

    spice_model = TransformerSPICE(spice_cfg).to(device)
    spice_model.load_state_dict(spice_sd); spice_model.eval()

    # eval set
    eval_trajs = load_eval_trajs(args.env, args.n_eval, args.H, args.dim, args.var, dataset_tag=args.dataset_tag)

    # output paths
    dpt_name   = os.path.basename(args.dpt_ckpt)
    spice_name = os.path.basename(args.spice_ckpt)
    spice_tag  = os.path.splitext(spice_name)[0]
    outdir     = os.path.join('figs', 'fig2_pair', args.epoch_label, spice_tag)
    os.makedirs(outdir, exist_ok=True)
    tag = f"{args.epoch_label}_{spice_tag}"

    caption = f"DPT: {dpt_name}   |   SPICE: {spice_name}"

    # plots
    fig_offline(eval_trajs, dpt_model, spice_model, args.H, args.var,
                os.path.join(outdir, f'a_offline_{tag}.png'),
                title_suffix=tag, subtitle=subtitle, caption=caption,
                spice_include_set=spice_include_set, skip_baselines_set=skip_baselines_set)
    fig_online(eval_trajs, dpt_model, spice_model, args.H, args.var,
               os.path.join(outdir, f'b_online_{tag}.png'),
               title_suffix=tag, subtitle=subtitle, caption=caption,
               warmup_steps=args.warmup_steps, eps_greedy=args.eps_greedy, eps_anneal=args.eps_anneal,
               spice_include_set=spice_include_set, skip_baselines_set=skip_baselines_set)
    fig_robustness(eval_trajs, dpt_model, spice_model, args.H, var_list=[0.0, 0.3, 0.5],
                   savepath=os.path.join(outdir, f'c_robust_{tag}.png'),
                   title_suffix=tag, subtitle=subtitle, caption=caption,
                   warmup_steps=args.warmup_steps, eps_greedy=args.eps_greedy, eps_anneal=args.eps_anneal,
                   spice_include_set=spice_include_set, skip_baselines_set=skip_baselines_set)
    print(f"Saved figs under {outdir}/")

    if args.wandb_eval and (wandb is not None):
        run_name = f"eval_bandit_{tag}"
        wandb.init(entity=os.environ.get('WANDB_ENTITY','david-rolnick'),
                   project=os.environ.get('WANDB_PROJECT','SPICE-'),
                   name=run_name,
                   config={'epoch_label': args.epoch_label,
                           'dpt_ckpt': args.dpt_ckpt,
                           'spice_ckpt': args.spice_ckpt,
                           'warmup_steps': args.warmup_steps,
                           'eps_greedy': args.eps_greedy,
                           'eps_anneal': args.eps_anneal,
                           'spice_include': args.spice_include,
                           'skip_baselines': args.skip_baselines,
                           'dataset_tag': args.dataset_tag})
        for fn in [f'a_offline_{tag}.png', f'b_online_{tag}.png', f'c_robust_{tag}.png']:
            wandb.log({fn: wandb.Image(os.path.join(outdir, fn))})
        wandb.finish()
