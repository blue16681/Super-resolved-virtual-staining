from PIL import Image
import blobfile as bf
try:
    from mpi4py import MPI
except:
    MPI = None
import numpy as np
import os, glob
import scipy.io as sio
from scipy.ndimage import zoom
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_wavelets import DWTForward, DWTInverse
from scipy.interpolate import interp1d
import random

import numpy as np
import cv2

def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
    
def fill_nan(A):
    """
    Interpolates data to fill nan values for multi-dimensional arrays.
    """

    if A.ndim == 1:
        inds = np.arange(A.shape[0])
        good = np.where(np.isfinite(A))
        f = interp1d(inds[good], A[good], bounds_error=False, fill_value="extrapolate")
        return np.where(np.isfinite(A), A, f(inds))

    elif A.ndim > 1:
        for dim in range(A.ndim):
            indices = np.indices(A.shape)
            for index in np.ndindex(*A.shape[:dim] + A.shape[dim + 1:]):
                slice_idx = tuple(index[:dim] + (slice(None),) + index[dim:])
                A_slice = A[slice_idx]

                if np.all(np.isnan(A_slice)):
                    continue

                if np.all(np.isfinite(A_slice)):
                    continue

                inds = indices[dim][slice_idx]
                good = np.isfinite(A_slice)
                f = interp1d(inds[good], A_slice[good], bounds_error=False, fill_value="extrapolate")
                A[slice_idx] = np.where(good, A_slice, f(inds))

        return A

    else:
        return A

def interpolate(input_arr, target_arr):
    y_zoom = target_arr.shape[0]/input_arr.shape[0]
    x_zoom = target_arr.shape[1]/input_arr.shape[1]
    zoom_factors = (y_zoom, x_zoom, 1)
    # zoom_factors = np.array(target_arr.shape) / np.array(input_arr.shape)
    interpolated_arr = zoom(input_arr, zoom_factors, order = 3)
    return interpolated_arr

def extract_prefix(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def load_paired_mat_data_test(
    *, input_dir, target_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not input_dir:
        raise ValueError("unspecified data directory")
    all_inputs = glob.glob(os.path.join(input_dir, '*.mat'))
    if not target_dir:
        raise ValueError("unspecified data directory")
    all_targets = glob.glob(os.path.join(target_dir, '*.mat'))

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_targets]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = PairedMATDataset_test(
        image_size,
        all_inputs,
        all_targets,
        classes=classes,
        shard=0,
        num_shards=1,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
        )
    while True:
        yield from loader

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def load_paired_npy_data(
    *, input_dir, target_dir, batch_size, image_size,scale_factor, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not input_dir:
        raise ValueError("unspecified data directory")
    all_inputs = glob.glob(os.path.join(input_dir, '*.npy'))
    if not target_dir:
        raise ValueError("unspecified data directory")
    all_targets = glob.glob(os.path.join(target_dir, '*.npy'))

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_targets]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    # dataset = PairedNPYDataset(
    #     image_size,
    #     all_inputs,
    #     all_targets,
    #     classes=classes,
    #     shard=0,
    #     num_shards=1,
    # )

    dataset = PairedImageDataset(
        image_size,
        input_dir = input_dir,
        target_dir = target_dir,
        scale_factor=scale_factor,
        classes=classes,
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
        )
    while True:
        yield from loader


def load_paired_eval_data(
    *,
    input_dir,
    target_dir,
    batch_size,
    image_size,
    class_cond=False,
    crop_mode="center",
    num_workers=2,
):
    """
    Create a deterministic paired image loader for validation/testing.
    This loader does not use random crop or random augmentation.
    """
    if not input_dir:
        raise ValueError("unspecified input_dir")
    if not target_dir:
        raise ValueError("unspecified target_dir")

    classes = None
    if class_cond:
        target_files = sorted(
            [p for p in glob.glob(os.path.join(target_dir, "*")) if os.path.isfile(p)]
        )
        class_names = [bf.basename(path).split("_")[0] for path in target_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    dataset = PairedImageEvalDataset(
        image_size,
        input_dir=input_dir,
        target_dir=target_dir,
        classes=classes,
        crop_mode=crop_mode,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

class PairedMATDataset_test(Dataset):
    def __init__(self, resolution, input_images, target_images, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.input_images = input_images[shard:][::num_shards]
        self.target_images = target_images[shard:][::num_shards]
        self.input_fnames = [os.path.basename(fp) for fp in self.input_images]
        self.target_fnames = [os.path.basename(fp) for fp in self.target_images]
        self.common_fnames = [f for f in self.input_fnames if f in self.target_fnames]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.common_fnames)

    def __getitem__(self, idx):
        path = self.input_images[self.input_fnames.index(self.common_fnames[idx])]
        with bf.BlobFile(path, "rb") as f:
            # inp = np.load(f).astype('float32')
            # inp = inp1[8:inp1.shape[1]-8,8:inp1.shape[1]-8,:]
            try:
                inp = sio.loadmat(f)['input'].astype('float32')
            except:
                inp = sio.loadmat(f)['input_tile'].astype('float32')
        path = self.target_images[self.target_fnames.index(self.common_fnames[idx])]
        with bf.BlobFile(path, "rb") as f:

            tag = sio.loadmat(f)['target'].astype('float32')
            tag = tag * 255
        if inp.ndim < 3:
            inp = np.expand_dims(inp, axis=0)
        if tag.ndim < 3:
            tag = np.expand_dims(tag, axis=0)
        
        crop_y = np.random.randint(0, inp.shape[0]-self.resolution)
        crop_x = np.random.randint(0, inp.shape[1]-self.resolution)
        
        inp_copy = inp[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution, ...].copy()
        tag_copy = tag[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution, ...].copy()
            
        inp_copy = pixel_binning_5x(inp_copy)
        inp_copy = (inp_copy - inp_copy.mean()) / (inp_copy.std() + 1e-6)

        tag_copy = tag_copy.astype(np.float32)/ 127.5 - 1
        
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            
        return tag_copy.transpose([2,0,1]), inp_copy.transpose([2,0,1]), out_dict, os.path.basename(path), os.path.basename(os.path.dirname(path))


class PairedNPYDataset(Dataset):
    def __init__(self, resolution, input_images, target_images, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.input_images = input_images[shard:][::num_shards]
        self.target_images = target_images[shard:][::num_shards]
        self.input_fnames = [os.path.basename(fp) for fp in self.input_images]
        self.target_fnames = [os.path.basename(fp) for fp in self.target_images]
        self.common_fnames = [f for f in self.input_fnames if f in self.target_fnames]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.common_fnames)

    def __getitem__(self, idx):
        path = self.input_images[self.input_fnames.index(self.common_fnames[idx])]
        with bf.BlobFile(path, "rb") as f:
            inp = np.load(f).astype('float32')
        path = self.target_images[self.target_fnames.index(self.common_fnames[idx])]
        with bf.BlobFile(path, "rb") as f:
            tag = np.load(f).astype('float32')
            tag = tag / 255.0
        if inp.ndim < 3:
            inp = np.expand_dims(inp, axis=0)
        if tag.ndim < 3:
            tag = np.expand_dims(tag, axis=0)
        
        tag_copy = np.ones_like(tag)
        n = 0
        while np.mean(tag_copy) * 255 >= 230:
            crop_y = np.random.randint(0, inp.shape[0]-self.resolution)
            crop_x = np.random.randint(0, inp.shape[1]-self.resolution)

            tag_copy = tag[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution, ...].copy()
            n = n+1
            if n >5:
                break
            
        inp = inp[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution, ...]

        mode = random.randint(0, 7)
        inp, tag_copy = augment_img(inp, mode=mode), augment_img(tag_copy, mode=mode)

        # inp = pixel_binning(inp, 3)
        
        inp = (inp - inp.mean()) / (inp.std() + 1e-6)
        tag_copy = tag_copy.astype(np.float32) * 255 / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return tag_copy.transpose([2,0,1]), inp.transpose([2,0,1]), out_dict




import os
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image


class PairedImageDataset(Dataset):
    def __init__(self, resolution, input_dir, target_dir, scale_factor=1,classes=None):
        super().__init__()
        self.resolution = resolution
        self.scale_factor = scale_factor
        # 获取文件名
        self.input_files = sorted(os.listdir(input_dir))
        self.target_files = sorted(os.listdir(target_dir))

        # 取交集（保证配对）
        self.common_fnames = list(set(self.input_files) & set(self.target_files))
        self.common_fnames.sort()

        self.input_paths = [os.path.join(input_dir, f) for f in self.common_fnames]
        self.target_paths = [os.path.join(target_dir, f) for f in self.common_fnames]

        self.local_classes = classes

    def __len__(self):
        return len(self.common_fnames)

    def __getitem__(self, idx):
        # ===== 1. 读取图片 =====
        inp = np.array(Image.open(self.input_paths[idx])).astype(np.float32)
        tag = np.array(Image.open(self.target_paths[idx])).astype(np.float32) / 255.0

        # ===== 2. 保证是 HWC =====
        if inp.ndim == 2:
            inp = np.expand_dims(inp, axis=-1)
        if tag.ndim == 2:
            tag = np.expand_dims(tag, axis=-1)

        # ===== 3. 随机裁剪（保留原逻辑）=====
        tag_copy = np.ones_like(tag)
        n = 0

        while np.mean(tag_copy) * 255 >= 230:
            h, w = inp.shape[:2]

            crop_y = np.random.randint(0, h - self.resolution + 1)
            crop_x = np.random.randint(0, w - self.resolution + 1)

            tag_copy = tag[crop_y:crop_y+self.resolution,
                           crop_x:crop_x+self.resolution].copy()
            n += 1
            if n > 5:
                break

        inp = inp[crop_y:crop_y+self.resolution,
                  crop_x:crop_x+self.resolution]

        # ===== 4. 数据增强 =====
        mode = random.randint(0, 7)
        inp = augment_img(inp, mode)
        tag_copy = augment_img(tag_copy, mode)

        # ===== 5. 模拟低分辨率（SR关键）=====
        if self.scale_factor > 1:
            inp = pixel_binning(inp, 3)

        # ===== 6. normalize（非常关键）=====
        inp = (inp - inp.mean()) / (inp.std() + 1e-6)
        tag_copy = tag_copy.astype(np.float32) * 255 / 127.5 - 1

        # ===== 7. 转成 CHW =====
        inp = inp.transpose(2, 0, 1)
        tag_copy = tag_copy.transpose(2, 0, 1)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return tag_copy, inp, out_dict


class PairedImageEvalDataset(Dataset):
    def __init__(self, resolution, input_dir, target_dir, classes=None, crop_mode="center"):
        super().__init__()
        self.resolution = resolution
        self.crop_mode = crop_mode

        input_files = sorted(
            [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        )
        target_files = sorted(
            [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
        )

        self.common_fnames = sorted(list(set(input_files) & set(target_files)))
        self.input_paths = [os.path.join(input_dir, f) for f in self.common_fnames]
        self.target_paths = [os.path.join(target_dir, f) for f in self.common_fnames]
        self.local_classes = classes

        if len(self.common_fnames) == 0:
            raise ValueError(
                f"No paired files found between {input_dir} and {target_dir}"
            )

    def __len__(self):
        return len(self.common_fnames)

    def _deterministic_crop_pair(self, inp, tag):
        h, w = inp.shape[:2]
        if h < self.resolution or w < self.resolution:
            raise ValueError(
                f"Image too small for crop: {(h, w)} < {self.resolution}"
            )

        if h == self.resolution and w == self.resolution:
            return inp, tag

        if self.crop_mode == "center":
            y0 = (h - self.resolution) // 2
            x0 = (w - self.resolution) // 2
        elif self.crop_mode == "top_left":
            y0, x0 = 0, 0
        else:
            raise ValueError(f"Unsupported crop_mode: {self.crop_mode}")

        y1, x1 = y0 + self.resolution, x0 + self.resolution
        return inp[y0:y1, x0:x1], tag[y0:y1, x0:x1]

    def __getitem__(self, idx):
        inp = np.array(Image.open(self.input_paths[idx]).convert("RGB"), dtype=np.float32)
        tag = np.array(Image.open(self.target_paths[idx]).convert("RGB"), dtype=np.float32) / 255.0

        if inp.shape[:2] != tag.shape[:2]:
            raise ValueError(
                f"Shape mismatch for {self.common_fnames[idx]}: "
                f"HE {inp.shape[:2]} vs IHC {tag.shape[:2]}"
            )

        inp, tag = self._deterministic_crop_pair(inp, tag)

        inp = (inp - inp.mean()) / (inp.std() + 1e-6)
        tag = tag.astype(np.float32) * 255 / 127.5 - 1

        inp = inp.transpose(2, 0, 1)
        tag = tag.transpose(2, 0, 1)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return tag, inp, out_dict

def pixel_binning(input_array, binning_factor=4):
    """
    Perform pixel binning on an input array of shape (256, 256, 4).
    
    Args:
        input_array (numpy.ndarray): Input array of shape (256, 256, 4).
        
    Returns:
        numpy.ndarray: Binned array of shape (256/4, 256/4, 4).
    """
    # Ensure the input dimensions are compatible
    # if input_array.shape[0] != 256 or input_array.shape[1] != 256:
    #     raise ValueError("Input dimensions must be 256x256.")
    
    # # Crop the array to make it divisible by 4
    # crop_size = 4 * (input_array.shape[0] // 4)  # Find the largest dimension divisible by 4
    # cropped_array = input_array[:crop_size, :crop_size, :]
    cropped_array = input_array
    
    # Reshape and take the mean across the 4x4 blocks
    binned_array = cropped_array.reshape(cropped_array.shape[0]//binning_factor, binning_factor, cropped_array.shape[1]//binning_factor, binning_factor, 3).mean(axis=(1,3))
    
    return binned_array
