import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)

import argparse
import os
import time
from datetime import datetime

try:
    import wandb
except Exception:
    wandb = None

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms

import numpy as np
import common_args
import random
from dataset import Dataset, ImageDataset
from net import Transformer, ImageTransformer
from utils import (
    build_bandit_data_filename,
    build_bandit_model_filename,
    build_linear_bandit_data_filename,
    build_linear_bandit_model_filename,
    build_darkroom_data_filename,
    build_darkroom_model_filename,
    build_miniworld_data_filename,
    build_miniworld_model_filename,
    worker_init_fn,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _append_tag(path, tag):
    if not tag:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{tag}{ext}"


if __name__ == '__main__':
    if not os.path.exists('figs/loss'):
        os.makedirs('figs/loss', exist_ok=True)

    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_train_args(parser)

    parser.add_argument('--dataset_tag', type=str, default='',
                        help="Suffix to select tagged datasets (e.g., 'weak').")
    parser.add_argument('--models_dir', type=str, default='models',
                        help="Directory to save checkpoints (e.g., models_weak).")

    parser.add_argument('--seed', type=int, default=0)

    args = vars(parser.parse_args())
    print("Args: ", args)

    args.setdefault('wandb', False)
    args.setdefault('shuffle', False)
    args.setdefault('wandb_entity', os.environ.get('WANDB_ENTITY', 'david-rolnick'))
    args.setdefault('wandb_project', os.environ.get('WANDB_PROJECT', 'SPICE-'))
    args.setdefault('wandb_tags', '')
    args.setdefault('log_every', 100)

    env = args['env']
    n_envs = args['envs']
    n_hists = args['hists']
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']
    state_dim = dim
    action_dim = dim
    n_embd = args['embd']
    n_head = args['head']
    n_layer = args['layer']
    lr = args['lr']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    cov = args['cov']
    num_epochs = args['num_epochs']
    seed = args['seed']
    lin_d = args['lin_d']
    dataset_tag = args.get('dataset_tag', '')
    models_dir = args.get('models_dir', 'models')

    tmp_seed = 0 if seed == -1 else seed
    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(tmp_seed)
        torch.cuda.manual_seed_all(tmp_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(tmp_seed)
    random.seed(tmp_seed)

    if shuffle and env == 'linear_bandit':
        raise Exception("Shuffling linear bandit data is not recommended (adaptive collection bias).")

    dataset_config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
    }
    model_config = {
        'shuffle': shuffle,
        'lr': lr,
        'dropout': dropout,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_envs': n_envs,
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
        'seed': seed,
    }

    # Resolve dataset paths
    if env == 'bandit':
        state_dim = 1
        dataset_config.update({'var': var, 'cov': cov, 'type': 'uniform'})
        path_train = build_bandit_data_filename(env, n_envs, dataset_config, mode=0)
        path_test  = build_bandit_data_filename(env, n_envs, dataset_config, mode=1)
        # apply tag
        path_train = _append_tag(path_train, dataset_tag)
        path_test  = _append_tag(path_test,  dataset_tag)
        model_config.update({'var': var, 'cov': cov})
        filename = build_bandit_model_filename(env, model_config)

    elif env == 'bandit_thompson':
        state_dim = 1
        dataset_config.update({'var': var, 'cov': cov, 'type': 'bernoulli'})
        path_train = build_bandit_data_filename(env, n_envs, dataset_config, mode=0)
        path_test  = build_bandit_data_filename(env, n_envs, dataset_config, mode=1)
        path_train = _append_tag(path_train, dataset_tag)
        path_test  = _append_tag(path_test,  dataset_tag)
        model_config.update({'var': var, 'cov': cov})
        filename = build_bandit_model_filename(env, model_config)

    elif env == 'linear_bandit':
        state_dim = 1
        dataset_config.update({'lin_d': lin_d, 'var': var, 'cov': cov})
        path_train = build_linear_bandit_data_filename(env, n_envs, dataset_config, mode=0)
        path_test  = build_linear_bandit_data_filename(env, n_envs, dataset_config, mode=1)
        path_train = _append_tag(path_train, dataset_tag)
        path_test  = _append_tag(path_test,  dataset_tag)
        model_config.update({'lin_d': lin_d, 'var': var, 'cov': cov})
        filename = build_linear_bandit_model_filename(env, model_config)

    elif env.startswith('darkroom'):
        state_dim = 2
        action_dim = 5
        dataset_config.update({'rollin_type': 'uniform'})
        path_train = build_darkroom_data_filename(env, n_envs, dataset_config, mode=0)
        path_test  = build_darkroom_data_filename(env, n_envs, dataset_config, mode=1)
        path_train = _append_tag(path_train, dataset_tag)
        path_test  = _append_tag(path_test,  dataset_tag)
        filename = build_darkroom_model_filename(env, model_config)

    elif env == 'miniworld':
        state_dim = 2
        action_dim = 4
        dataset_config.update({'rollin_type': 'uniform'})

        increment = 5000
        starts = np.arange(0, n_envs, increment)
        starts = np.array(starts)
        ends = starts + increment - 1

        paths_train = []
        paths_test = []
        for start_env_id, end_env_id in zip(starts, ends):
            p_train = build_miniworld_data_filename(env, start_env_id, end_env_id, dataset_config, mode=0)
            p_test  = build_miniworld_data_filename(env, start_env_id, end_env_id, dataset_config, mode=1)
            # note: miniworld tagging omitted here (image shards)
            paths_train.append(p_train)
            paths_test.append(p_test)

        filename = build_miniworld_model_filename(env, model_config)
        print(f"Generate filename: {filename}")
    else:
        raise NotImplementedError

    config = {
        'horizon': horizon,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'shuffle': shuffle,
        'dropout': dropout,
        'test': False,
        'store_gpu': False,
    }

    if env == 'miniworld':
        config.update({'image_size': 25, 'store_gpu': False})
        model = ImageTransformer(config).to(device)
    else:
        model = Transformer(config).to(device)

    use_wandb = bool(args.get('wandb', False)) and (wandb is not None)
    if use_wandb:
        run_name = args.get('wandb_run_name') or (
            f"DPT_{env}_embd{n_embd}_layer{n_layer}_head{n_head}"
            f"_lr{lr}_drop{dropout}_seed{seed}_tag{dataset_tag or 'none'}"
        )
        wandb.init(entity=args.get('wandb_entity','david-rolnick'),
                   project=args.get('wandb_project','SPICE-'),
                   name=run_name,
                   config=args)
        wandb.watch(model, log='all', log_freq=200)

    params = {
        'batch_size': 64,
        'shuffle': bool(args['shuffle']),
    }

    os.makedirs('figs/loss', exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    log_filename = f'figs/loss/{filename}_logs.txt'
    with open(log_filename, 'w') as f:
        pass

    def printw(string):
        print(string)
        with open(log_filename, 'a') as f:
            print(string, file=f)

    if env == 'miniworld':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        params.update({
            'num_workers': 16,
            'prefetch_factor': 2,
            'persistent_workers': True,
            'pin_memory': True,
            'batch_size': 64,
            'worker_init_fn': worker_init_fn,
        })
        printw("Loading miniworld data...")
        train_dataset = ImageDataset(paths_train, config, transform)
        test_dataset  = ImageDataset(paths_test,  config, transform)
        printw("Done loading miniworld data")
    else:
        train_dataset = Dataset(path_train, config)
        test_dataset  = Dataset(path_test,  config)

    train_loader = torch.utils.data.DataLoader(train_dataset, **params)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  **params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    test_loss = []
    train_loss = []

    printw("Num train batches: " + str(len(train_loader)))
    printw("Num test batches: " + str(len(test_loader)))

    for epoch in range(num_epochs):
        # EVAL
        printw(f"Epoch: {epoch + 1}")
        start_time = time.time()
        with torch.no_grad():
            epoch_test_loss = 0.0
            for i, batch in enumerate(test_loader):
                print(f"Batch {i} of {len(test_loader)}", end='\r')
                batch = {k: v.to(device) for k, v in batch.items()}
                true_actions = batch['optimal_actions']
                pred_actions = model(batch)
                true_actions = true_actions.unsqueeze(1).repeat(1, pred_actions.shape[1], 1)
                true_actions = true_actions.reshape(-1, action_dim)
                pred_actions = pred_actions.reshape(-1, action_dim)
                loss = loss_fn(pred_actions, true_actions)
                epoch_test_loss += loss.item() / horizon
        test_loss.append(epoch_test_loss / len(test_dataset))
        end_time = time.time()
        printw(f"\tTest loss: {test_loss[-1]}")
        printw(f"\tEval time: {end_time - start_time}")

        # TRAIN
        epoch_train_loss = 0.0
        start_time = time.time()
        for i, batch in enumerate(train_loader):
            print(f"Batch {i} of {len(train_loader)}", end='\r')
            batch = {k: v.to(device) for k, v in batch.items()}
            true_actions = batch['optimal_actions']
            pred_actions = model(batch)
            true_actions = true_actions.unsqueeze(1).repeat(1, pred_actions.shape[1], 1)
            true_actions = true_actions.reshape(-1, action_dim)
            pred_actions = pred_actions.reshape(-1, action_dim)

            optimizer.zero_grad()
            loss = loss_fn(pred_actions, true_actions)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() / horizon
        train_loss.append(epoch_train_loss / len(train_dataset))
        end_time = time.time()
        printw(f"\tTrain loss: {train_loss[-1]}")
        printw(f"\tTrain time: {end_time - start_time}")

        # SAVE
        if (epoch + 1) % 50 == 0 or (env == 'linear_bandit' and (epoch + 1) % 10 == 0):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = os.path.join(models_dir, f'{filename}_epoch{epoch+1}_{timestamp}.pt')
            torch.save(model.state_dict(), ckpt_path)
            if use_wandb:
                wandb.save(ckpt_path)

        # PLOT
        if (epoch + 1) % 10 == 0:
            printw(f"Epoch: {epoch + 1}")
            printw(f"Test Loss:  {test_loss[-1]}")
            printw(f"Train Loss: {train_loss[-1]}\n")
            plt.yscale('log')
            plt.plot(train_loss[1:], label="Train Loss")
            plt.plot(test_loss[1:],  label="Test Loss")
            plt.legend()
            plt.savefig(f"figs/loss/{filename}_train_loss.png")
            plt.clf()

    # Final checkpoint
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_ckpt_path = os.path.join(models_dir, f'{filename}_final_{timestamp}.pt')
    torch.save(model.state_dict(), final_ckpt_path)
    if use_wandb:
        wandb.save(final_ckpt_path)
        wandb.finish()
    print("Done.")
