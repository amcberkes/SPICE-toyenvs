# evals/eval_bandit.py
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from ctrls.ctrl_bandit import (
    BanditTransformerController,
    GreedyOptPolicy,
    EmpMeanPolicy,
    OptPolicy,
    PessMeanPolicy,
    ThompsonSamplingPolicy,
    UCBPolicy,
)
from envs.bandit_env import BanditEnv, BanditEnvVec
from utils import convert_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def deploy_online(env, controller, horizon):
    context_states = torch.zeros((1, horizon, env.dx)).float().to(device)
    context_actions = torch.zeros((1, horizon, env.du)).float().to(device)
    context_next_states = torch.zeros((1, horizon, env.dx)).float().to(device)
    context_rewards = torch.zeros((1, horizon, 1)).float().to(device)

    cum_means = []
    for h in range(horizon):
        batch = {
            'context_states': context_states[:, :h, :],
            'context_actions': context_actions[:, :h, :],
            'context_next_states': context_next_states[:, :h, :],
            'context_rewards': context_rewards[:, :h, :],
        }

        controller.set_batch(batch)
        states_lnr, actions_lnr, next_states_lnr, rewards_lnr = env.deploy(controller)

        context_states[0, h, :] = convert_to_tensor(states_lnr[0])
        context_actions[0, h, :] = convert_to_tensor(actions_lnr[0])
        context_next_states[0, h, :] = convert_to_tensor(next_states_lnr[0])
        context_rewards[0, h, :] = convert_to_tensor(rewards_lnr[0])

        actions = actions_lnr.flatten()
        mean = env.get_arm_value(actions)
        cum_means.append(mean)

    return np.array(cum_means)


def deploy_online_vec(vec_env, controller, horizon, include_meta=False):
    num_envs = vec_env.num_envs

    context_states = np.zeros((num_envs, horizon, vec_env.dx))
    context_actions = np.zeros((num_envs, horizon, vec_env.du))
    context_next_states = np.zeros((num_envs, horizon, vec_env.dx))
    context_rewards = np.zeros((num_envs, horizon, 1))

    cum_means = []
    print("Deploying online vectorized...")
    for h in range(horizon):
        batch = {
            'context_states': context_states[:, :h, :],
            'context_actions': context_actions[:, :h, :],
            'context_next_states': context_next_states[:, :h, :],
            'context_rewards': context_rewards[:, :h, :],
        }
        controller.set_batch_numpy_vec(batch)

        states_lnr, actions_lnr, next_states_lnr, rewards_lnr = vec_env.deploy(controller)

        context_states[:, h, :] = states_lnr
        context_actions[:, h, :] = actions_lnr
        context_next_states[:, h, :] = next_states_lnr
        context_rewards[:, h, :] = rewards_lnr[:, None]

        mean = vec_env.get_arm_value(actions_lnr)
        cum_means.append(mean)

    print("Deployed online vectorized")
    cum_means = np.array(cum_means)
    if not include_meta:
        return cum_means
    else:
        meta = {
            'context_states': context_states,
            'context_actions': context_actions,
            'context_next_states': context_next_states,
            'context_rewards': context_rewards,
        }
        return cum_means, meta


def online(eval_trajs, model, n_eval, horizon, var, bandit_type):
    all_means = {}

    envs = []
    for i_eval in range(n_eval):
        traj = eval_trajs[i_eval]
        means = traj['means']
        env = BanditEnv(means, horizon, var=var)
        envs.append(env)

    vec_env = BanditEnvVec(envs)

    controller = OptPolicy(envs, batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    all_means['opt'] = cum_means

    controller = BanditTransformerController(model, sample=True, batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    all_means['Lnr'] = cum_means

    controller = EmpMeanPolicy(envs[0], online=True, batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    all_means['Emp'] = cum_means

    controller = UCBPolicy(envs[0], const=1.0, batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    all_means['UCB1.0'] = cum_means

    controller = ThompsonSamplingPolicy(
        envs[0], std=var, sample=True, prior_mean=0.5, prior_var=1/12.0,
        warm_start=False, batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    all_means['Thomp'] = cum_means

    all_means = {k: np.array(v) for k, v in all_means.items()}
    all_means_diff = {k: all_means['opt'] - v for k, v in all_means.items()}

    means = {k: np.mean(v, axis=0) for k, v in all_means_diff.items()}
    sems = {k: scipy.stats.sem(v, axis=0) for k, v in all_means_diff.items()}

    cumulative_regret = {k: np.cumsum(v, axis=1) for k, v in all_means_diff.items()}
    regret_means = {k: np.mean(v, axis=0) for k, v in cumulative_regret.items()}
    regret_sems = {k: scipy.stats.sem(v, axis=0) for k, v in cumulative_regret.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for key in means.keys():
        if key == 'opt':
            ax1.plot(means[key], label=key, linestyle='--', color='black', linewidth=2)
            ax1.fill_between(np.arange(horizon),
                             means[key] - sems[key], means[key] + sems[key],
                             alpha=0.2, color='black')
        else:
            ax1.plot(means[key], label=key)
            ax1.fill_between(np.arange(horizon),
                             means[key] - sems[key], means[key] + sems[key], alpha=0.2)

    ax1.set_yscale('log')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Suboptimality')
    ax1.set_title('Online Evaluation')
    ax1.legend()

    for key in regret_means.keys():
        if key != 'opt':
            ax2.plot(regret_means[key], label=key)
            ax2.fill_between(np.arange(horizon),
                             regret_means[key] - regret_sems[key],
                             regret_means[key] + regret_sems[key], alpha=0.2)

    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Regret Over Time')
    ax2.legend()


def offline(eval_trajs, model, n_eval, horizon, var, bandit_type):
    num_envs = len(eval_trajs)

    tmp_env = BanditEnv(eval_trajs[0]['means'], horizon, var=var)
    context_states = np.zeros((num_envs, horizon, tmp_env.dx))
    context_actions = np.zeros((num_envs, horizon, tmp_env.du))
    context_next_states = np.zeros((num_envs, horizon, tmp_env.dx))
    context_rewards = np.zeros((num_envs, horizon, 1))

    envs = []
    for i_eval in range(n_eval):
        traj = eval_trajs[i_eval]
        means = traj['means']
        env = BanditEnv(means, horizon, var=var)
        envs.append(env)

        context_states[i_eval, :, :] = traj['context_states'][:horizon]
        context_actions[i_eval, :, :] = traj['context_actions'][:horizon]
        context_next_states[i_eval, :, :] = traj['context_next_states'][:horizon]
        context_rewards[i_eval, :, :] = traj['context_rewards'][:horizon, None]

    vec_env = BanditEnvVec(envs)
    batch = {
        'context_states': context_states,
        'context_actions': context_actions,
        'context_next_states': context_next_states,
        'context_rewards': context_rewards,
    }

    opt_policy = OptPolicy(envs, batch_size=num_envs)
    emp_policy = EmpMeanPolicy(envs[0], online=False, batch_size=num_envs)
    lnr_policy = BanditTransformerController(model, sample=False, batch_size=num_envs)
    thomp_policy = ThompsonSamplingPolicy(
        envs[0], std=var, sample=False, prior_mean=0.5, prior_var=1/12.0,
        warm_start=False, batch_size=num_envs)
    # --- OFFLINE LCB baseline (not UCB) ---
    lcb_policy = PessMeanPolicy(envs[0], const=.8, batch_size=len(envs))

    for c in [opt_policy, emp_policy, thomp_policy, lcb_policy, lnr_policy]:
        c.set_batch_numpy_vec(batch)

    _, _, _, rs_opt  = vec_env.deploy_eval(opt_policy)
    _, _, _, rs_emp  = vec_env.deploy_eval(emp_policy)
    _, _, _, rs_lnr  = vec_env.deploy_eval(lnr_policy)
    _, _, _, rs_lcb  = vec_env.deploy_eval(lcb_policy)
    _, _, _, rs_thmp = vec_env.deploy_eval(thomp_policy)

    baselines = {
        'opt':  np.array(rs_opt),
        'lnr':  np.array(rs_lnr),
        'emp':  np.array(rs_emp),
        'thmp': np.array(rs_thmp),
        'lcb':  np.array(rs_lcb),
    }

    # quick bar plot utility
    baselines_means = {k: np.mean(v) for k, v in baselines.items()}
    colors = plt.cm.viridis(np.linspace(0, 1, len(baselines_means)))
    plt.bar(baselines_means.keys(), baselines_means.values(), color=colors)
    plt.title(f'Mean Reward on {n_eval} Trajectories')
    return baselines


def offline_graph(eval_trajs, model, n_eval, horizon, var, bandit_type):
    """
    Plot offline suboptimality vs dataset size with correct averaging and uncertainty bands.
    """
    horizons = np.linspace(1, horizon, 50, dtype=int)

    # For each horizon, compute regret arrays across tasks, then mean & SEM.
    series_mean = {k: [] for k in ['lnr', 'emp', 'thmp', 'lcb']}
    series_sem  = {k: [] for k in ['lnr', 'emp', 'thmp', 'lcb']}

    for h in horizons:
        baselines = offline(eval_trajs, model, n_eval=n_eval, horizon=h, var=var, bandit_type=bandit_type)
        plt.clf()

        rs_opt = baselines['opt']  # shape [n_eval]
        for key in ['lnr', 'emp', 'thmp', 'lcb']:
            rs = baselines[key]
            regrets = rs_opt - rs  # suboptimality per task
            series_mean[key].append(regrets.mean())
            series_sem[key].append(scipy.stats.sem(regrets))

    # Plot each curve with horizon-wise SEM
    for key in ['lnr', 'emp', 'thmp', 'lcb']:
        m = np.array(series_mean[key])
        s = np.array(series_sem[key])
        label = {'lnr':'Lnr', 'emp':'Emp', 'thmp':'TS', 'lcb':'LCB(0.8)'}[key]
        plt.plot(horizons, m, label=label)
        plt.fill_between(horizons, m - s, m + s, alpha=0.2)

    plt.legend()
    plt.yscale('log')
    plt.xlabel('Dataset size')
    plt.ylabel('Suboptimality')
