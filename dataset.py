#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CMALA-Net 数据集加载器（修复路径解析 + 禁止随机数兜底）
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SUBTYPE_NAMES = ['Luminal A', 'Luminal B(HER2-)', 'Luminal B(HER2+)', 'HER2-enriched', 'TNBC']
BIOMARKER_NAMES = ['ER', 'PR', 'HER2', 'Ki67']
NUM_CLASSES = 5

CENTER_TO_ID = {'A': 0, 'B': 1, 'C': 2}


class PathologyDataset(Dataset):
    def __init__(self, csv_path, img_size=224, is_train=True,
                 us_dir=None, path_dir=None):
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size
        self.is_train = is_train
        self.us_dir = us_dir
        self.path_dir = path_dir

        # ==== 关键：保存 CSV 所在目录，用于路径解析 ====
        self.csv_path = csv_path
        self.csv_dir = os.path.dirname(os.path.abspath(csv_path))

        # 只打印一次 warning
        self._warned_us = False
        self._warned_path = False

        if is_train:
            self.us_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            self.path_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.us_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            self.path_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, p, base_dir):
        """解析相对路径：依次尝试 base_dir / CSV 目录 / 原始路径"""
        if not isinstance(p, str) or not p:
            return p
        if os.path.isabs(p) and os.path.exists(p):
            return p

        candidates = []
        # 1) 原样
        candidates.append(p)
        # 2) base_dir + basename
        if base_dir:
            candidates.append(os.path.join(base_dir, os.path.basename(p)))
            candidates.append(os.path.join(base_dir, p))
        # 3) CSV 目录 + 原路径
        candidates.append(os.path.join(self.csv_dir, p))
        # 4) CSV 目录 + basename
        candidates.append(os.path.join(self.csv_dir, os.path.basename(p)))

        for c in candidates:
            if c and os.path.exists(c):
                return c
        return p   # 都不存在，返回原值让上层报错

    def _load_nrrd(self, path):
        """加载 US 图像；失败时返回零 tensor（不再返回随机数）"""
        # 尝试 nrrd
        try:
            import nrrd
            data, _ = nrrd.read(path)
            if data.ndim == 3:
                mid = data.shape[2] // 2
                img = data[:, :, mid]
            else:
                img = data
            img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return img
        except Exception:
            pass

        # 尝试 cv2
        try:
            if os.path.exists(path):
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8),
                                   cv2.IMREAD_COLOR)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

        # 失败：打印 warning（只一次），返回零图
        if not self._warned_us:
            print(f"[US LOAD FAIL] {repr(path)}  ->  using zeros")
            self._warned_us = True
        return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

    def _load_path(self, path_str):
        """加载病理图像；所有子图失败时返回零 tensor"""
        paths = [p.strip() for p in str(path_str).split(';') if p.strip()]
        tensors = []
        for p in paths:
            resolved = self._resolve_path(p, self.path_dir)
            if not os.path.exists(resolved):
                if not self._warned_path:
                    print(f"[PATH LOAD FAIL] not exists: {repr(p)} "
                          f"(resolved={repr(resolved)})")
                    self._warned_path = True
                continue
            img = cv2.imdecode(np.fromfile(resolved, dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                if not self._warned_path:
                    print(f"[PATH LOAD FAIL] cv2 read failed: {repr(resolved)}")
                    self._warned_path = True
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = self.path_transform(img)
            tensors.append(t)

        if not tensors:
            return torch.zeros(3, self.img_size, self.img_size)
        return torch.stack(tensors).mean(dim=0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        us_path = self._resolve_path(row['us_path'], self.us_dir)
        us_img = self._load_nrrd(us_path)
        us_tensor = self.us_transform(us_img)

        path_path = self._resolve_path(row['path_path'], self.path_dir)
        path_tensor = self._load_path(path_path)

        label = torch.tensor(int(row['subtype']), dtype=torch.long)

        biomarkers = torch.tensor([
            float(row['ER']) if 'ER' in row.index else 1.0,
            float(row['PR']) if 'PR' in row.index else 1.0,
            float(row['HER2']) if 'HER2' in row.index else 0.0,
            (float(row['Ki67']) / 100.0) if 'Ki67' in row.index else 0.2,
        ], dtype=torch.float32)

        center = row['center'] if 'center' in row.index else 'A'
        domain = torch.tensor(CENTER_TO_ID.get(str(center), 0), dtype=torch.long)

        return {
            'us_img': us_tensor,
            'path_img': path_tensor,
            'subtype': label,
            'biomarkers': biomarkers,
            'domain': domain,
        }


def create_dataloaders(train_csv, val_csv, test_csvs=None,
                       batch_size=8, num_workers=0, img_size=224,
                       pin_memory=True, **kwargs):
    if test_csvs is None and 'test_csv' in kwargs:
        test_csvs = kwargs['test_csv']
    us_dir = kwargs.get('us_dir', None)
    path_dir = kwargs.get('path_dir', None)

    train_ds = PathologyDataset(train_csv, img_size=img_size, is_train=True,
                                us_dir=us_dir, path_dir=path_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              drop_last=True)

    val_loader = None
    if val_csv is not None:
        val_ds = PathologyDataset(val_csv, img_size=img_size, is_train=False,
                                  us_dir=us_dir, path_dir=path_dir)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)

    test_loaders = {}
    if test_csvs is not None:
        def _add(name, path):
            ds = PathologyDataset(path, img_size=img_size, is_train=False,
                                  us_dir=us_dir, path_dir=path_dir)
            test_loaders[name] = DataLoader(ds, batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=pin_memory)

        if isinstance(test_csvs, dict):
            for n, p in test_csvs.items():
                _add(n, p)
        elif isinstance(test_csvs, (list, tuple)):
            for i, p in enumerate(test_csvs):
                _add(f'test_{i}', p)
        else:
            _add('test', test_csvs)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loaders if test_loaders else None
    }