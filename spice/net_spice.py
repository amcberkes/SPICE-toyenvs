import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

from spice.value_heads import make_ensemble_q

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TransformerSPICE(nn.Module):
    """
    TransformerSPICE: a DPT-style sequence model with a SPICE value-ensemble head.

    Overview
    --------
    This module encodes a short sequence of transitions
    (s_t, a_t, s_{t+1}, r_t) with a GPT-style transformer and exposes:
      1) A policy head that produces action logits.
      2) A SPICE ensemble head that produces K Q-value heads for every action.

    Shape contract
    --------------
    Let:
      B: batch size (parallel envs)
      H: context length (number of transitions in context)
      S: state_dim
      A: action_dim
      K: number of SPICE heads (spice_heads)
      D: embedding dim (n_embd)

    - Train mode (config['test'] == False):
        logits: [B, H, A]         # one per context step (positions 1..H)
        if return_q=True: Q: [B, H, K, A]
    - Eval mode (config['test'] == True):
        logits: [B, A]            # at the query position (last)
        if return_q=True: Q: [B, K, A]

    Inputs (forward)
    ----------------
    x: dict containing:
      'query_states'          [B, S]
      'context_states'        [B, H, S]
      'context_actions'       [B, H, A]
      'context_next_states'   [B, H, S]
      'context_rewards'       [B, H, 1] or [B, H]
      'zeros'                 [B, S^2 + A + 1]
        A SPICE/DPT convention: slices of this vector are used to provide
        dummy placeholders at the query position (for action/next_state/reward).

    Config (required keys)
    ----------------------
    config['horizon']     -> int, context length H
    config['n_embd']      -> int, transformer hidden size D
    config['n_layer']     -> int, number of transformer blocks
    config['n_head']      -> int, number of attention heads (kept at 1 to match DPT)
    config['state_dim']   -> int, S
    config['action_dim']  -> int, A
    Optional:
      config['test']          -> bool, switches train/eval output shapes
      config['dropout']       -> float, dropout prob
      config['spice_heads']   -> int, K (default 7)
      config['spice_prior']   -> float, SPICE prior alpha (default 0.1)

    Returns
    -------
    If return_q is False:
      - Train: logits [B, H, A]
      - Eval : logits [B, A]

    If return_q is True:
      - Train: (logits [B, H, A], Q [B, H, K, A])
      - Eval : (logits [B, A],     Q [B, K, A])

    Notes
    -----
    - The first transformer position corresponds to the query step; we insert
      the current query state and dummy features for a, s', r.
    - Positions 1..H correspond to the provided context transitions.
    - The policy head is trained in parallel with the value ensemble; downstream
      controllers may ignore logits and act via Q (e.g., Thompson sampling).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # DPT-compatible config
        self.test = bool(config.get('test', False))
        self.horizon = int(config['horizon'])
        self.n_embd = int(config['n_embd'])
        self.n_layer = int(config['n_layer'])
        self.n_head = int(config['n_head'])
        self.state_dim = int(config['state_dim'])
        self.action_dim = int(config['action_dim'])
        self.dropout = float(config.get('dropout', 0.0))

        # SPICE ensemble params
        self.K = int(config.get('spice_heads', 7))
        self.prior_alpha = float(config.get('spice_prior', 0.1))

        # GPT-2 trunk (matches original DPT net.py choices)
        gpt_cfg = GPT2Config(
            n_positions=4 * (1 + self.horizon),
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,  # DPT repo keeps this at 1 even if 'head' flag exists
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(gpt_cfg)

        # (s, a, s', r) → embed per step
        self.embed_transition = nn.Linear(
            2 * self.state_dim + self.action_dim + 1, self.n_embd
        )
        # policy (classification over actions)
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)

        # SPICE value ensemble over [h_s; one_hot(a)]
        self.ensemble_q = make_ensemble_q(
            K=self.K,
            d_model=self.n_embd,
            action_dim=self.action_dim,
            hidden=256,
            alpha_prior=self.prior_alpha,
        )


    def _q_from_h(self, h: torch.Tensor) -> torch.Tensor:
        """
        Construct Q-values for all actions from hidden states, with an explicit head dim.

        Parameters
        ----------
        h : torch.Tensor
            Hidden representation(s) from the transformer.
            - If shape is [B, D], returns Q with shape [B, K, A].
            - If shape is [B, T, D], returns Q with shape [B, T, K, A].

        Returns
        -------
        torch.Tensor
            Q-values with an explicit head dimension K and action dimension A.

        Mechanism
        ---------
        For each hidden vector h and action a, we concatenate [h; one_hot(a)]
        and pass the result through `self.ensemble_q` to obtain K Q-values.
        Doing this for all actions yields [K, A] (or [T, K, A] for sequences).
        """
        A = self.action_dim
        D = self.n_embd
        dev = h.device

        if h.dim() == 2:
            # [B, D] → [B, K, A]
            B = h.size(0)
            eye = torch.eye(A, device=dev)                # [A, A] build one hots for actions
            h_rep = h.unsqueeze(1).expand(B, A, D)       # [B, A, D] broadcast hidden to pair with each action
            eye_rep = eye.unsqueeze(0).expand(B, A, A)   # [B, A, A]
            x = torch.cat([h_rep, eye_rep], dim=-1)      # [B, A, D+A] concatenate per action
            q = self.ensemble_q(x.reshape(B * A, D + A)) # [B*A, K] feed to ensmble
            q = q.view(B, A, self.K).permute(0, 2, 1)    # [B, K, A] reshape back 
            return q

        if h.dim() == 3:
            # [B, T, D] → [B, T, K, A]
            B, T, _ = h.shape
            eye = torch.eye(A, device=dev)                       # [A, A]
            h_rep = h.unsqueeze(2).expand(B, T, A, D)            # [B, T, A, D]
            eye_rep = eye.view(1, 1, A, A).expand(B, T, A, A)    # [B, T, A, A]
            x = torch.cat([h_rep, eye_rep], dim=-1)              # [B, T, A, D+A]
            q = self.ensemble_q(x.reshape(B * T * A, D + A))     # [B*T*A, K]
            q = q.view(B, T, A, self.K).permute(0, 1, 3, 2)      # [B, T, K, A]
            return q

        raise RuntimeError(f"Unexpected h shape {tuple(h.shape)}")


    def forward(self, x: dict, return_q: bool = False):
        """
        Forward pass.

        Parameters
        ----------
        x : dict
            Must contain:
              - 'query_states'          : [B, S]
              - 'context_states'        : [B, H, S]
              - 'context_actions'       : [B, H, A]
              - 'context_next_states'   : [B, H, S]
              - 'context_rewards'       : [B, H, 1] or [B, H]
              - 'zeros'                 : [B, S^2 + A + 1]  (DPT convention; used in slices)
        return_q : bool, optional
            If True, also compute and return SPICE ensemble Q-values alongside logits.

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            If return_q is False:
              - Train (test == False): logits [B, H, A]
              - Eval  (test == True) : logits [B, A]
            If return_q is True:
              - Train: (logits [B, H, A], Q [B, H, K, A])
              - Eval : (logits [B, A],     Q [B, K, A])

        Notes
        -----
        - Position 0 (prepended) is the query step:
            * states: query_states
            * actions/next_states/reward: dummy zeros slices
        - Positions 1..H are the provided context transitions.
        - In train mode, outputs correspond to positions 1..H.
          In eval mode, outputs correspond to position -1 (the query).
        """
        query_states = x['query_states'][:, None, :]  # [B,1,S]
        zeros = x['zeros'][:, None, :]               # [B,1,S^2+A+1]

        state_seq = torch.cat([query_states, x['context_states']], dim=1)  # [B,1+H,S]
        action_seq = torch.cat(
            [zeros[:, :, :self.action_dim], x['context_actions']], dim=1
        )  # [B,1+H,A]
        next_state_seq = torch.cat(
            [zeros[:, :, :self.state_dim], x['context_next_states']], dim=1
        )  # [B,1+H,S]
        # ensure rewards has shape [B,*,1]
        rewards = x['context_rewards']
        if rewards.dim() == 2:
            rewards = rewards.unsqueeze(-1)
        reward_seq = torch.cat([zeros[:, :, :1], rewards], dim=1)          # [B,1+H,1]

        seq = torch.cat([state_seq, action_seq, next_state_seq, reward_seq], dim=2)
        emb = self.embed_transition(seq)                                   # [B,1+H,D]
        hidden = self.transformer(inputs_embeds=emb).last_hidden_state     # [B,1+H,D]
        preds = self.pred_actions(hidden)                                   # [B,1+H,A]

        if self.test: # we only care about the query token (the “last” position in the sequence).Slice out the logits at that position → shape [B, A]. These are the action scores the policy will use to act.
            # EVAL path: query at last position
            logits = preds[:, -1, :]  # [B, A]
            if not return_q:
                return logits
            hq = hidden[:, -1, :]     # [B, D]
            Q = self._q_from_h(hq)    # [B, K, A] per-action Q estimates for each head.
            assert Q.dim() == 3 and Q.shape[1] == self.K
            return logits, Q

        else: # In training, we don’t predict at the query (position 0).Instead, we train to predict the action taken at each context transition (positions 1..H). Slice out those logits → shape [B, H, A].
            # TRAIN path: use positions 1..H
            logits = preds[:, 1:, :]   # [B, H, A]
            if not return_q:
                return logits
            hs = hidden[:, 1:, :]      # [B, H, D]
            Q = self._q_from_h(hs)     # [B, H, K, A]
            assert Q.dim() == 4 and Q.shape[2] == self.K
            return logits, Q
