from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import os
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW

from torch.utils.data import DataLoader, Dataset
import cv2
from externals.utils import cfg_init
from externals.models import Unet3D_full3d_shallow
from externals.metrics import AverageMeter
from externals.training_procedures import get_scheduler, scheduler_step
import wandb
import h5py
import time
mse = nn.MSELoss()

def criterion(y_preds, y_true):
    return (
        # dl(y_preds, y_true) 
        # + \
        # bce(y_preds, y_true)
        # cl(y_preds, y_true)
        mse(y_preds, y_true)
        )
    
class CFG:
    is_multiclass = True
    
    # edit these so they match your local data path
    comp_name = 'vesuvius_3d'
    comp_dir_path = './input'
    comp_folder_name = 'vesuvius-challenge-ink-detection'
    comp_dataset_path = f'{comp_dir_path}{comp_folder_name}/'
    # ========================
    
    exp_name = 'pretrain'
    # ============== pred target =============
    target_size = 1
    # ============== model cfg =============
    model_name = '3d_unet'
    # ============== training cfg =============
    size = 128
    tile_size = 128
    in_chans = 1

    train_batch_size = 64
    valid_batch_size = train_batch_size
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    epochs = 200
    valid_id = "856"
    # adamW warmup
    warmup_factor = 1
    lr = 1e-4 / warmup_factor
    # ============== fixed =============
    min_lr = 1e-6
    weight_decay = 1e-5
    max_grad_norm = 10
    num_workers = 4
    seed = int(time.time())
    # ============== set dataset path =============
    print('set dataset path')

    outputs_path = f'working/outputs/{comp_name}/{exp_name}/'

    submission_dir = outputs_path + 'submissions/'
    submission_path = submission_dir + f'submission_{exp_name}.csv'

    model_dir = outputs_path + \
        f'{comp_name}-models/'

    figures_dir = outputs_path + 'figures/'

    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'


cfg_init(CFG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Unet3D_full3d_shallow(CFG)

class CustomDataset(Dataset):
    def __init__(self, volume_path, cfg, labels=None, transform=None, mode="test", size=1000, coords=None, cache_size=10000):
        self.volumes = volume_path
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.mode = mode
        self.size = size
        self.coords = coords
        self.file = h5py.File("/mnt/aged-star/volume.hdf5", 'r')
        self.shape = self.file["20230205180739"].shape
        self.cache_size = cache_size
        self.cache = {}
        self.cache_indices = []

    def __len__(self):
        if self.coords is not None:
            return len(self.coords)
        else:
            return self.size

    def preload_cache(self):
        # Clear cache
        self.cache = {}
        self.cache_indices = []

        # Preload new cache elements
        indices = random.sample(range(len(self.coords)), min(self.cache_size, len(self.coords)))
        for idx in indices:
            self.cache[idx] = self._load_data(idx)
            self.cache_indices.append(idx)

    def _load_data(self, idx):
        invalid_volume = True
        while invalid_volume:
            coords = [np.random.randint(0, self.shape[0] - self.cfg.size),
                      np.random.randint(0, self.shape[1] - self.cfg.size),
                      np.random.randint(0, self.shape[2] - self.cfg.size)]
            volume = self.file["20230205180739"][coords[0]: (coords[0] + (self.cfg.size)),
                                              coords[1]: (coords[1] + ((self.cfg.size))),
                                              coords[2]: (coords[2] + ((self.cfg.size)))] / 255.
            if volume.max() > 0.7 and volume.min() < 0.3:
                invalid_volume = False
        volume = volume.astype(np.float16)
        image = volume.copy()
        for _ in range(4):
            size_x = random.randint(self.cfg.size // 2, self.cfg.size // 1.3)
            size_y = random.randint(self.cfg.size // 2, self.cfg.size // 1.3)
            size_z = random.randint(self.cfg.size // 2, self.cfg.size // 1.3)
            x = random.randint(0, volume.shape[0] - size_x)
            y = random.randint(0, volume.shape[1] - size_y)
            z = random.randint(0, volume.shape[2] - size_z)
            image[x:x+size_x, y:y+size_y, z:z+size_z] = -1
            
        return image[None], volume[None]

    def __getitem__(self, idx):
        if (random.random() > 0.5) and (len(self.cache_indices) > 1000):
            while True:
                try:
                    random_idx = random.choice(self.cache_indices)
                    random_sample = self.cache[random_idx]
                    break
                except:
                    pass
            return random_sample
        else:
            data = self._load_data(idx)
            if len(self.cache) < self.cache_size:
                self.cache[idx] = data
                self.cache_indices.append(idx)
            else:
                replace_idx = random.choice(self.cache_indices)
                try:
                    del self.cache[replace_idx]
                except:
                    pass
                self.cache[idx] = data
                self.cache_indices[self.cache_indices.index(replace_idx)] = idx
            return data
        
    
def train_fn(train_loader, model, criterion, optimizer, device):
    model.train()
    model.to(device)
    scaler = GradScaler(enabled=CFG.use_amp)
    losses = AverageMeter()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (images, labels) in pbar:
        images = images.to(torch.float16).to(device)
        labels = labels.to(torch.float16).to(device)
        batch_size = labels.size(0)
        with torch.autocast(device_type="cuda"):
            y_preds = model(images)
            loss = criterion(y_preds[images == -1], labels[images == -1]).mean()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm)
        pbar.set_description_str(str(losses.avg))
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    return losses.avg

def valid_fn(valid_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for step, (images, labels) in pbar:
        os.makedirs(f"./volume_predictions/{step}", exist_ok=True)
        os.makedirs(f"./volume_labels/{step}", exist_ok=True)
        os.makedirs(f"./volume/{step}", exist_ok=True)
        batch_size = labels.size(0)
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                images = images.to(torch.float16).to(device)
                labels = labels.to(torch.float16).to(device)
                y_preds = model(images)
                loss = criterion(y_preds[images == -1], labels[images == -1]).mean()
        pbar.set_description_str(str(losses.avg))
        losses.update(loss.mean().item(), batch_size)
        for layer_num, layer in enumerate(images[0][0]):
            layer[layer < 0] = 0
            cv2.imwrite(f"./volume/{step}/{layer_num}.jpg", ((layer)*255.).detach().cpu().numpy())
        for layer_num, layer in enumerate(labels[0][0]):
            layer[layer < 0] = 0
            cv2.imwrite(f"./volume_labels/{step}/{layer_num}.jpg", ((layer)*255.).detach().cpu().numpy())
        for layer_num, layer in enumerate(y_preds[0][0]):
            cv2.imwrite(f"./volume_predictions/{step}/{layer_num}.jpg", (layer*255.).detach().cpu().numpy())
    return losses.avg

import random
training_dataset = CustomDataset(volume_path="/data/volume.hdf5", labels="depth_narrow_train.hdf5", cfg=CFG, transform=None, mode="train", size = 1000000, coords=None)
sampler = torch.utils.data.RandomSampler(training_dataset, replacement=True, num_samples=100000)
train_loader = DataLoader(training_dataset, batch_size=CFG.train_batch_size, shuffle=False, num_workers=16, pin_memory=False, drop_last=True, sampler=sampler)

cfg_pairs = {value:CFG.__dict__[value] for value in dir(CFG) if value[1] != "_"}
model_name = f"{CFG.exp_name}_{CFG.model_name}"

# if os.path.exists("/home/ryanc/kaggle/working/outputs/vesuvius_3d/pretrain/vesuvius_3d-models/pretrain_3d_unet.pth"):
#     print("/home/ryanc/kaggle/working/outputs/vesuvius_3d/pretrain/vesuvius_3d-models/pretrain_3d_unet.pth")
#     model.load_state_dict(torch.load("/home/ryanc/kaggle/working/outputs/vesuvius_3d/pretrain/vesuvius_3d-models/pretrain_3d_unet.pth"))

model = torch.nn.DataParallel(model)
model.to(device)

optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
scheduler = get_scheduler(CFG, optimizer)

wandb.init(
    project="LSM"
)

for epoch in range(CFG.epochs):
    # train
    avg_loss = train_fn(train_loader, model, criterion, optimizer, device)
    torch.save(model.module.state_dict(),
            CFG.model_dir + f"{model_name}.pth")
    wandb.log({"avg_train_loss":avg_loss})
    print({"avg_train_loss":avg_loss})
    scheduler_step(scheduler, None, epoch)
