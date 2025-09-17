import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def _to_torch(x, device, dtype=torch.float32):
    """Safely move numpy/torch to target device/dtype (no grad)."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


class SPICEBanditController:
    """
    Context-aware SPICE controller for bandits with multiple selection variants.

    Variants (set via 'variant'):
      - 'ts_noctx'      : Thompson sampling via head sampling, IGNORE context.
      - 'policy_logits' : Use policy head logits (softmax sample or argmax).
      - 'ts_ctx'        : Thompson via head sampling, ALWAYS use context.
      - 'ucb_ctx'       : Ensemble-UCB with context (mean_Q + beta_ucb * std_Q).
      - 'posterior_ctx' : Bayesian fusion (ensemble prior + empirical means from context).
      - 'adapter_ctx'   : posterior_ctx + light online adapter toward empirical mean.
      - 'qmean_ctx'     : greedy on mean Q across heads (with context).

    Works with envs.bandit_env.BanditEnvVec:
        reset()
        act_numpy_vec(states) -> np.ndarray[float] of shape [B, A] (one-hot)
    """

    def __init__(
        self,
        model,
        batch_size=1,
        variant: str = "ts_ctx",
        sample_heads: bool = True,
        resample_every_step: bool = False,
        device=None,
        # policy head
        policy_sample: bool = True,
        # UCB
        beta_ucb: float = 0.0,
        # posterior fusion
        posterior_noise_var: float = 0.1,
        posterior_prior_floor: float = 1e-4,
        # adapter
        adapter_eta: float = 0.0,
    ):
        self.model = model.eval()
        self.batch_size = int(batch_size)
        self.variant = str(variant).lower()
        assert self.variant in {
            "ts_noctx", "policy_logits", "ts_ctx", "ucb_ctx",
            "posterior_ctx", "adapter_ctx", "qmean_ctx"
        }, f"Unknown variant: {self.variant}"

        self.sample_heads = bool(sample_heads)
        self.resample_every_step = bool(resample_every_step)  # typically False for bandits
        self.device = device or next(self.model.parameters()).device

        # selection params
        self.policy_sample = bool(policy_sample)
        self.beta_ucb = float(beta_ucb)
        self.posterior_noise_var = float(posterior_noise_var)
        self.posterior_prior_floor = float(posterior_prior_floor)
        self.adapter_eta = float(adapter_eta)

        # per-episode state
        self._k = None                        # sampled head index per env
        self._cached_batch = None             # latest context prefix (numpy)
        self._context_t = 0                   # step index in current episode

    #  context management 
    def set_batch_numpy_vec(self, batch):
        """
        Cache the prefix batch for the current step (online) or the full batch (offline).
        Saves the context / history so far. 
        Expected numpy keys:
          - context_states      [B,t,S]
          - context_actions     [B,t,A] one-hot
          - context_next_states [B,t,S]
          - context_rewards     [B,t,1] or [B,t]
        """
        self._cached_batch = batch

    #  episode control 
    def reset(self):
        """Sample one head per env for 'ts_*' variants; reset step counter."""
        self._context_t = 0
        if self.variant.startswith("ts") and self.sample_heads:
            K = int(self.model.config.get("spice_heads", 1))
            self._k = np.random.randint(0, K, size=(self.batch_size,))
        else:
            self._k = np.zeros((self.batch_size,), dtype=np.int64)

    #  vectorized act , public API to act
    def act_numpy_vec(self, states=None, *_, **__):
        """Return one-hot actions for each env."""
        return self._act_with_states(states)

    def _coerce_states_to_query(self, states):
        """Coerce states -> torch.FloatTensor [B,S] on device."""
        B = self.batch_size
        S = int(self.model.config.get("state_dim", 1))

        if states is None:
            return torch.zeros((B, S), dtype=torch.float32, device=self.device)

        if isinstance(states, torch.Tensor):
            st = states.to(self.device).float()
        else:
            if isinstance(states, (list, tuple)):
                states = np.asarray(states)
            st = torch.as_tensor(states, device=self.device, dtype=torch.float32)

        if st.dim() == 1:            # [B]
            st = st.view(B, 1)
        elif st.dim() >= 2:
            st = st.view(B, -1)

        # pad/truncate to S
        if st.size(-1) != S:
            if st.size(-1) > S:
                st = st[:, :S]
            else:
                st = F.pad(st, (0, S - st.size(-1)), value=0.0)
        return st

    def _uses_context(self) -> bool:
        """Whether this variant should consume the context."""
        return (
            self.variant.endswith("_ctx")
            or self.variant in ("adapter_ctx", "policy_logits", "qmean_ctx")
        )

    def _maybe_add_context_from_cache(self, batch_out: dict):
        """
        Inject context_* tensors if cached AND if this variant uses context.
        (Batch already contains empty context placeholders.)
        """
        if not self._uses_context() or self._cached_batch is None:
            return
        # Convert cached arrays to torch tensors on device; ensure rewards shape is [B, t, 1] when needed. Why: the model expects these keys to compute context-aware logits/Q.
        keys = ["context_states", "context_actions", "context_next_states", "context_rewards"]
        for key in keys:
            if key not in self._cached_batch:
                continue
            tx = _to_torch(self._cached_batch[key], device=self.device)
            if tx is None:
                continue
            if key == "context_rewards" and tx.dim() == 2:
                tx = tx.unsqueeze(-1)
            batch_out[key] = tx

    def _empirical_counts_and_means_from_context(self):
        """
        Compute per-env per-arm (counts, empirical mean reward) from cached context.
        For each arm, compute:
        counts: how many times it was pulled.
        sums: total reward obtained when it was pulled.
        means = sums / counts (safe divide by ≥1).
        Why: needed by posterior_ctx and adapter_ctx.

        Returns
        -------
        counts : np.ndarray [B,A]  (zeros if no samples)
        means  : np.ndarray [B,A]  (0 where count==0)
        """
        if self._cached_batch is None:
            return None, None
        acts = self._cached_batch.get("context_actions", None)  # [B,t,A]
        rews = self._cached_batch.get("context_rewards", None)  # [B,t,1] or [B,t]
        if acts is None or rews is None:
            return None, None

        if rews.ndim == 3:
            rews = rews[..., 0]
        B, t, A = acts.shape
        a_idx = acts.argmax(axis=-1)  # [B,t]

        counts = np.zeros((B, A), dtype=np.float32)
        sums   = np.zeros((B, A), dtype=np.float32)
        for a in range(A):
            mask = (a_idx == a)              # [B,t]
            counts[:, a] = mask.sum(axis=1)
            sums[:, a]   = (rews * mask).sum(axis=1)
        means = sums / np.maximum(1.0, counts)
        return counts, means

    def _compose_posterior_from_ensemble_and_context(self, Q_mean, Q_std):
        """
        Gaussian-Gaussian fusion per arm: prior (Q_mean, Q_std) + empirical (n, r̄) -> posterior mean.

        Q_mean, Q_std : [B,A] tensors
        Returns       : post_mean [B,A]
        """
        counts_np, means_np = self._empirical_counts_and_means_from_context()
        if counts_np is None or means_np is None:
            return Q_mean

        counts = torch.as_tensor(counts_np, device=Q_mean.device, dtype=Q_mean.dtype)
        rbar   = torch.as_tensor(means_np,  device=Q_mean.device, dtype=Q_mean.dtype)

        prior_var = torch.clamp(Q_std**2, min=self.posterior_prior_floor)  # [B,A]
        noise_var = torch.tensor(self.posterior_noise_var, device=Q_mean.device, dtype=Q_mean.dtype)

        num = Q_mean / prior_var + counts * rbar / noise_var
        den = 1.0 / prior_var + counts / noise_var
        post_mean = num / torch.clamp(den, min=1e-8)
        return post_mean

    @torch.no_grad()
    def _forward_eval_logits_Q(self, states):
        """
        Build eval batch and query model for (logits, Q).

        Always provides EMPTY context placeholders (shape [B,0,*]) so the model
        can safely index keys even for no-context variants. For *_ctx/policy_logits/qmean_ctx,
        cached context (if any) will overwrite these placeholders.
        """
        B = self.batch_size
        S = int(self.model.config.get("state_dim", 1))
        A = int(self.model.config.get("action_dim", 1))

        qs = self._coerce_states_to_query(states)  # [B,S]
        zeros = torch.zeros((B, S*S + A + 1), dtype=torch.float32, device=self.device)

        # Empty placeholders for context (length 0). These prevent KeyError in the model.
        batch = {
            "query_states": qs,
            "zeros": zeros,
            "context_states":      torch.zeros((B, 0, S), dtype=torch.float32, device=self.device),
            "context_actions":     torch.zeros((B, 0, A), dtype=torch.float32, device=self.device),
            "context_next_states": torch.zeros((B, 0, S), dtype=torch.float32, device=self.device),
            "context_rewards":     torch.zeros((B, 0, 1), dtype=torch.float32, device=self.device),
        }

        # Overwrite with cached prefix if this variant uses context
        self._maybe_add_context_from_cache(batch)

        logits, Q = self.model(batch, return_q=True)  # logits [B,A], Q [B,K,A] or [B,A]
        if Q.dim() == 2:  # [B,A] -> add head dim
            Q = Q.unsqueeze(1)
        return logits, Q

    @torch.no_grad()
    def _act_with_states(self, states):
        # Manage head sampling for TS variants
        if self._k is None:
            self.reset()
        if self.resample_every_step and self.variant.startswith("ts") and self.sample_heads:
            K = int(self.model.config.get("spice_heads", 1))
            self._k = np.random.randint(0, K, size=(self.batch_size,))

        logits, Q = self._forward_eval_logits_Q(states)  # logits [B,A], Q [B,K,A]
        A = logits.size(-1)
        B, K, _ = Q.shape

        if self.variant == "policy_logits":
            # Use policy head; sample or argmax (slightly sharper)
            if self.policy_sample:
                probs = torch.softmax(logits / 0.7, dim=-1)
                a_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                a_idx = torch.argmax(logits, dim=-1)

        else:
            # Use Q ensemble
            Q_mean = Q.mean(dim=1)  # [B,A]
            Q_std  = Q.std(dim=1)   # [B,A]

            if self.variant in ("ts_noctx", "ts_ctx"):
                idx = torch.arange(B, device=Q.device)
                k = torch.as_tensor(self._k, device=Q.device, dtype=torch.long).clamp_(0, K-1)
                Qa = Q[idx, k, :]         # [B,A]
                scores = Qa
                a_idx = torch.argmax(scores, dim=-1)

            elif self.variant == "ucb_ctx":
                scores = Q_mean + self.beta_ucb * Q_std
                a_idx = torch.argmax(scores, dim=-1)

            elif self.variant == "posterior_ctx":
                scores = self._compose_posterior_from_ensemble_and_context(Q_mean, Q_std)
                a_idx = torch.argmax(scores, dim=-1)

            elif self.variant == "adapter_ctx":
                base = self._compose_posterior_from_ensemble_and_context(Q_mean, Q_std)  # [B,A]
                counts_np, means_np = self._empirical_counts_and_means_from_context()
                if counts_np is not None and means_np is not None and self.adapter_eta > 0.0:
                    counts = torch.as_tensor(counts_np, device=base.device, dtype=base.dtype)
                    rbar   = torch.as_tensor(means_np,  device=base.device, dtype=base.dtype)
                    # move fraction eta toward rbar where we have data
                    w = (counts > 0).to(base.dtype)
                    scores = base * (1.0 - self.adapter_eta * w) + rbar * (self.adapter_eta * w)
                else:
                    scores = base
                a_idx = torch.argmax(scores, dim=-1)

            elif self.variant == "qmean_ctx":
                scores = Q_mean
                a_idx = torch.argmax(scores, dim=-1)

            else:
                raise RuntimeError(f"Unknown variant {self.variant}")

        one_hot = torch.zeros((B, A), device=logits.device, dtype=torch.float32)
        one_hot.scatter_(1, a_idx.view(-1, 1), 1.0)

        self._context_t += 1
        return one_hot.cpu().numpy()
