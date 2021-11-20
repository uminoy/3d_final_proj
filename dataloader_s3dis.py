import pickle, time, warnings
import numpy as np
from helper_ply import read_ply
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler, BatchSampler
from os.path import join
import time, pickle, argparse, glob, os

from utils.helper_tool import ConfigS3DIS as cfg
from utils.helper_tool import DataProcessing as DP
from utils.helper_tool import Plot

class PointCloudsDataset(Dataset):
    def __init__(self, test_area_idx=5, split="training"):
        self.name = 'S3DIS'
        self.path = './data'
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = glob.glob(join(self.path, 'original_ply', '*.ply'))
        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.load_sub_sampled_clouds(0.040)
        self.split = split

    def __len__(self):
        return len(self.paths)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        print("tree path : ", tree_path)
        for i, file_path in enumerate(self.all_files):
            print(file_path)
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))
    
    def __getitem__(self, idx):
        split = self.split
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size

        self.possibility[split] = []
        self.min_possibility[split] = []
        # Random initialize
        for i, tree in enumerate(self.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

            # Generator loop
        for i in range(num_per_epoch):
            # Choose the cloud with the lowest probability
            cloud_idx = int(np.argmin(self.min_possibility[split]))
            # choose the point with the minimum of possibility in the cloud as query point
            point_ind = np.argmin(self.possibility[split][cloud_idx])
            # Get all points within the cloud from tree structure
            points = np.array(self.input_trees[split][cloud_idx].data, copy=False)
            # Center point of input region
            center_point = points[point_ind, :].reshape(1, -1)
            # Add noise to the center point
            noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
            pick_point = center_point + noise.astype(center_point.dtype)
            # Check if the number of points in the selected cloud is less than the predefined num_points
            if len(points) < cfg.num_points:
                # Query all points within the cloud
                queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
            else:
                # Query the predefined number of points
                queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]
            # Shuffle index
            queried_idx = DP.shuffle_idx(queried_idx)
            # Get corresponding points and colors based on the index
            queried_pc_xyz = points[queried_idx]
            queried_pc_xyz = queried_pc_xyz - pick_point
            queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx]
            queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx]
            # Update the possibility of the selected points
            dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
            delta = np.square(1 - dists / np.max(dists))
            self.possibility[split][cloud_idx][queried_idx] += delta
            self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))
            # up_sampled with replacement
            if len(points) < cfg.num_points:
                queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                    DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)
            # if True:
            #     yield (queried_pc_xyz.astype(np.float32),
            #            queried_pc_colors.astype(np.float32),
            #            queried_pc_labels,
            #            queried_idx.astype(np.int32),
            #            np.array([cloud_idx], dtype=np.int32))
                

        # gen_types = (torch.float32, torch.float32, torch.int32, torch.int32, torch.int32)
        # gen_shapes = ([None, 3], [None, 3], [None], [None], [None])

        return torch.from_numpy(queried_pc_xyz.astype(np.float32)),\
            torch.from_numpy(queried_pc_colors.astype(np.float32)),\
                torch.from_numpy(queried_pc_labels.astype(np.int32)),\
                    torch.from_numpy(queried_idx.astype(np.int32)),\
                        torch.from_numpy(np.array([cloud_idx], dtype=np.int32))

def data_loaders(**kwargs):
    train_dataset = PointCloudsDataset(split='training')
    val_dataset = PointCloudsDataset(split='validation')
    return DataLoader(train_dataset, shuffle=True, **kwargs), DataLoader(val_dataset, **kwargs)

if __name__ == '__main__':
    dataset = PointCloudsDataset(split='training')
    for data in dataset:
        xyz, colors, labels, idx, cloud_idx = data
        print('Number of points:', len(xyz))
        print('Point position:', xyz[1])
        print('Color:', colors[1])
        print('Label:', labels[1])
        print('Index of point:', idx[1])
        print('Cloud index:', cloud_idx)
        break
