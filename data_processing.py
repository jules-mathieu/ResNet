from torch.utils.data import random_split
from torchvision import datasets
import torch
import tqdm as tqdm
import numpy as np
from torch.utils.data import WeightedRandomSampler
import kagglehub

def get_balanced_sampler(dataset_subset):
    # 1. Get all labels in the subset
    # Since it's a subset, we look at the underlying dataset's targets at specific indices
    indices = dataset_subset.indices
    targets = [dataset_subset.dataset.targets[i] for i in indices]
    
    # 2. Calculate class weights
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
    weight = 1. / class_sample_count
    
    # 3. Map weights to every sample in the subset
    samples_weight = np.array([weight[t] for t in targets])
    samples_weight = torch.from_numpy(samples_weight).double()
    
    # 4. Create the sampler
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

class ApplyTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index] # x is a PIL image
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)
    

path = kagglehub.dataset_download("joebeachcapital/defungi")

print("Path to dataset files:", path)

full_dataset = datasets.ImageFolder(root=path)

# Proportions du split
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
n_total = len(full_dataset)
n_train = int(n_total * train_ratio)
n_val   = int(n_total * val_ratio)
n_test  = n_total - n_train - n_val

train_set, val_set, test_set = random_split(
    full_dataset,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

print("Train :", len(train_set))
print("Val   :", len(val_set))
print("Test  :", len(test_set))