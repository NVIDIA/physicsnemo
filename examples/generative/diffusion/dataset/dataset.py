# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import json
import os
import zipfile

import numpy as np
import PIL.Image
import torch
from physicsnemo.utils.generative import EasyDict

try:
    import pyspng
except ImportError:
    pyspng = None


class Dataset(torch.utils.data.Dataset):
    """
    Abstract base class for datasets
    """

    def __init__(
        self,
        name,  # Name of the dataset.
        raw_shape,  # Shape of the raw image data (NCHW).
        max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
        xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed=0,  # Random seed to use when applying max_size.
        cache=False,  # Cache images in CPU memory?
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict()  # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = int(self._xflip[idx]) != 0
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


class ImageFolderDataset(Dataset):
    """
    Dataset subclass that loads images recursively from the specified directory
    or ZIP file.
    """

    def __init__(
        self,
        path,  # Path to directory or zip.
        resolution=None,  # Ensure specific resolution, None = highest available.
        use_pyspng=True,  # Use pyspng if available?
        **super_kwargs,  # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = "dir"
            self._all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self._path)
                for root, _dirs, files in os.walk(self._path)
                for fname in files
            }
        elif self._file_ext(self._path) == ".zip":
            self._type = "zip"
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError("Path must point to a directory or zip")

        PIL.Image.init()
        self._image_fnames = sorted(
            fname
            for fname in self._all_fnames
            if self._file_ext(fname) in PIL.Image.EXTENSION
        )
        if len(self._image_fnames) == 0:
            raise IOError("No image files found in the specified path")

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (
            raw_shape[2] != resolution or raw_shape[3] != resolution
        ):
            raise IOError("Image files do not match the specified resolution")
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == "zip"
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == "dir":
            return open(os.path.join(self._path, fname), "rb")
        if self._type == "zip":
            return self._get_zipfile().open(fname, "r")
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if (
                self._use_pyspng
                and pyspng is not None
                and self._file_ext(fname) == ".png"
            ):
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = "dataset.json"
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)["labels"]
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace("\\", "/")] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels


class KolmogorovFlowDataset(Dataset):
    """Kolmogorov flow dataset"""

    def __init__(
        self,
        path,
        train_ratio=0.9,
        test=False,
        stat_path=None,
        max_cache_len=4000,
        **super_kwargs,  # Additional arguments for the Dataset base class.
    ):
        np.random.seed(1)
        self.all_data = np.load(path)
        # self.all_data = self.all_data[:,:,::4,::4]
        print("Dataset shape: ", self.all_data.shape)
        idxs = np.arange(self.all_data.shape[0])
        num_of_training_seeds = int(train_ratio * len(idxs))
        self.train_idx_lst = idxs[:num_of_training_seeds]
        self.test_idx_lst = idxs[num_of_training_seeds:]
        self.time_step_lst = np.arange(self.all_data.shape[1] - 2)
        if not test:
            self.idx_lst = self.train_idx_lst[:]
        else:
            self.idx_lst = self.test_idx_lst[:]
        self.cache = {}
        self.max_cache_len = max_cache_len

        if stat_path is not None:
            self.stat_path = stat_path
            self.stat = np.load(stat_path)
        else:
            self.stat = {}
            self.prepare_data()

        self._name = "KolmogorovFlow"
        B, T, H, W = self.all_data.shape
        self._raw_shape = [B * T, 3, H, W]
        self._raw_labels = None
        self._label_shape = None
        self._use_labels = False

    def __len__(self):
        return len(self.idx_lst) * len(self.time_step_lst)

    def prepare_data(self):
        # load all training data and calculate their statistics
        self.stat["mean"] = np.mean(self.all_data[self.train_idx_lst[:]].reshape(-1, 1))
        self.stat["scale"] = np.std(self.all_data[self.train_idx_lst[:]].reshape(-1, 1))
        data_mean = self.stat["mean"]
        data_scale = self.stat["scale"]
        print(f"Data statistics, mean: {data_mean}, scale: {data_scale}")

    def preprocess_data(self, data):
        # normalize data

        s = data.shape[-1]

        data = (data - self.stat["mean"]) / (self.stat["scale"])
        return data.astype(np.float32)

    def save_data_stats(self, out_dir):
        # save data statistics to out_dir
        np.savez(out_dir, mean=self.stat["mean"], scale=self.stat["scale"])

    def __getitem__(self, idx):
        seed = self.idx_lst[idx // len(self.time_step_lst)]
        frame_idx = idx % len(self.time_step_lst)
        id = idx

        if id in self.cache.keys():
            # return self.cache[id]
            return self.cache[id], torch.empty((0))
        else:
            frame0 = self.preprocess_data(self.all_data[seed, frame_idx])
            frame1 = self.preprocess_data(self.all_data[seed, frame_idx + 1])
            frame2 = self.preprocess_data(self.all_data[seed, frame_idx + 2])

            frame = np.concatenate(
                (frame0[None, ...], frame1[None, ...], frame2[None, ...]), axis=0
            )
            self.cache[id] = frame

            if len(self.cache) > self.max_cache_len:
                self.cache.pop(
                    list(self.cache.keys())[np.random.choice(len(self.cache.keys()))]
                )
            return frame, torch.empty((0))
