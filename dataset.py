# dataset.py
import pickle
import numpy as np
import torch
from utils import convert_to_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dataset(torch.utils.data.Dataset):
    """Dataset class."""

    def __init__(self, path, config):
        self.shuffle = config['shuffle']
        self.horizon = config['horizon']
        self.store_gpu = config['store_gpu']
        self.config = config

        if not isinstance(path, list):
            path = [path]

        self.trajs = []
        for p in path:
            with open(p, 'rb') as f:
                self.trajs += pickle.load(f)

        context_states = []
        context_actions = []
        context_next_states = []
        context_rewards = []
        query_states = []
        optimal_actions = []
        means_list = []      # optional (bandits)
        cap_list = []        # optional context_action_probs [H,A] how likely each arm was to be chosen when the data was collected.

        for traj in self.trajs:
            context_states.append(traj['context_states'])
            context_actions.append(traj['context_actions'])
            context_next_states.append(traj['context_next_states'])
            context_rewards.append(traj['context_rewards'])

            query_states.append(traj['query_state'])
            optimal_actions.append(traj['optimal_action'])

            if 'means' in traj:
                means_list.append(traj['means'])
            if 'context_action_probs' in traj:
                cap_list.append(traj['context_action_probs'])

        context_states = np.array(context_states)
        context_actions = np.array(context_actions)
        context_next_states = np.array(context_next_states)
        context_rewards = np.array(context_rewards)
        if len(context_rewards.shape) < 3:
            context_rewards = context_rewards[:, :, None]
        query_states = np.array(query_states)
        optimal_actions = np.array(optimal_actions)

        self.dataset = {
            'query_states': convert_to_tensor(query_states, store_gpu=self.store_gpu),
            'optimal_actions': convert_to_tensor(optimal_actions, store_gpu=self.store_gpu),
            'context_states': convert_to_tensor(context_states, store_gpu=self.store_gpu),
            'context_actions': convert_to_tensor(context_actions, store_gpu=self.store_gpu),
            'context_next_states': convert_to_tensor(context_next_states, store_gpu=self.store_gpu),
            'context_rewards': convert_to_tensor(context_rewards, store_gpu=self.store_gpu),
        }

        if len(means_list) == len(self.trajs):
            means_arr = np.array(means_list, dtype=np.float32)
            self.dataset['means'] = convert_to_tensor(means_arr, store_gpu=self.store_gpu)

        if len(cap_list) == len(self.trajs):
            caps = np.array(cap_list, dtype=np.float32)
            self.dataset['context_action_probs'] = convert_to_tensor(caps, store_gpu=self.store_gpu)

        self.zeros = np.zeros(
            config['state_dim'] ** 2 + config['action_dim'] + 1
        )
        self.zeros = convert_to_tensor(self.zeros, store_gpu=self.store_gpu)

    def __len__(self):
        return len(self.dataset['query_states'])

    def __getitem__(self, index):
        res = {
            'context_states': self.dataset['context_states'][index],
            'context_actions': self.dataset['context_actions'][index],
            'context_next_states': self.dataset['context_next_states'][index],
            'context_rewards': self.dataset['context_rewards'][index],
            'query_states': self.dataset['query_states'][index],
            'optimal_actions': self.dataset['optimal_actions'][index],
            'zeros': self.zeros,
        }

        if 'means' in self.dataset:
            res['means'] = self.dataset['means'][index]
        if 'context_action_probs' in self.dataset:
            res['context_action_probs'] = self.dataset['context_action_probs'][index]

        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_next_states'] = res['context_next_states'][perm]
            res['context_rewards'] = res['context_rewards'][perm]
            if 'context_action_probs' in res:
                res['context_action_probs'] = res['context_action_probs'][perm]

        return res


class ImageDataset(Dataset):
    """"Dataset class for image-based data."""

    def __init__(self, paths, config, transform):
        config['store_gpu'] = False
        super().__init__(paths, config)
        self.transform = transform
        self.config = config

        context_filepaths = []
        query_images = []

        for traj in self.trajs:
            context_filepaths.append(traj['context_images'])
            query_image = self.transform(traj['query_image']).float()
            query_images.append(query_image)

        self.dataset.update({
            'context_filepaths': context_filepaths,
            'query_images': torch.stack(query_images),
        })

    def __getitem__(self, index):
        filepath = self.dataset['context_filepaths'][index]
        context_images = np.load(filepath)
        context_images = [self.transform(images) for images in context_images]
        context_images = torch.stack(context_images).float()

        query_images = self.dataset['query_images'][index]

        res = {
            'context_images': context_images,
            'context_states': self.dataset['context_states'][index],
            'context_actions': self.dataset['context_actions'][index],
            'context_next_states': self.dataset['context_next_states'][index],
            'context_rewards': self.dataset['context_rewards'][index],
            'query_images': query_images,
            'query_states': self.dataset['query_states'][index],
            'optimal_actions': self.dataset['optimal_actions'][index],
            'zeros': self.zeros,
        }

        if 'means' in self.dataset:
            res['means'] = self.dataset['means'][index]

        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['context_images'] = res['context_images'][perm]
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_next_states'] = res['context_next_states'][perm]
            res['context_rewards'] = res['context_rewards'][perm]

        return res
