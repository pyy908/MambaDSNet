import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

try:
    import SimpleITK as sitk

    sitk_available = True
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
except Exception:
    sitk = None
    sitk_available = False

# Fallback reader for .nii.gz if SimpleITK is unavailable
try:
    import nibabel as nib

    nib_available = True
except Exception:
    nib = None
    nib_available = False


def read_nii(path):
    """Read NIfTI image, prefer SimpleITK, fallback to nibabel; raise if neither exists.
    
    注意: SimpleITK 返回 (D, H, W)，nibabel 返回 (H, W, D)
    为保持一致，统一转换为 (D, H, W) 格式
    """
    if sitk_available:
        return sitk.GetArrayFromImage(sitk.ReadImage(path))
    if nib_available:
        img = nib.load(path)
        data = np.asanyarray(img.get_fdata())
        # nibabel 返回 (H, W, D)，转换为 (D, H, W) 以匹配 SimpleITK
        if data.ndim == 3:
            data = np.transpose(data, (2, 0, 1))
        return data
    raise ImportError("SimpleITK or nibabel is required to read NIfTI; install one of them.")


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def _acdc_intensity_normalize(volume: np.ndarray, clip_percentiles=(0.5, 99.5), eps=1e-8) -> np.ndarray:
    """对 ACDC MRI 体素做稳健归一化：
    1) 按百分位裁剪极端值；2) 用非零区域做 z-score；若全零则直接返回。"""
    vol = volume.astype(np.float32)
    if vol.max() == 0 and vol.min() == 0:
        return vol
    lo, hi = np.percentile(vol, clip_percentiles)
    vol = np.clip(vol, lo, hi)
    nonzero = vol != 0
    if nonzero.any():
        mean = vol[nonzero].mean()
        std = vol[nonzero].std()
    else:
        mean = vol.mean()
        std = vol.std()
    std = std if std > eps else eps
    vol = (vol - mean) / std
    return vol


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32)).long()

        sample = {'image': image, 'label': label}
        return sample


class ValGenerator(object):
    """验证/测试时的数据处理器，只做resize，不做数据增强"""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32)).long()

        return {'image': image, 'label': label}


class Synapse_dataset(Dataset):
    """Synapse 数据集"""

    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)

            image = np.asarray(data['image'], dtype=np.float32)
            label = np.asarray(data['label'], dtype=np.float32)

            sample = {'image': image, 'label': label}
            if self.transform:
                sample = self.transform(sample)
            sample['case_name'] = self.sample_list[idx].strip('\n')
            return sample
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = os.path.join(self.data_dir, 'test_vol_h5', "{}.npy.h5".format(vol_name))
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.float32))

            sample = {'image': image, 'label': label}
            if self.transform:
                sample = self.transform(sample)
            sample['case_name'] = self.sample_list[idx].strip('\n')
            return sample


class ACDC_dataset(Dataset):
    """
    ACDC 数据集

    参数:
        use_all_slices: 是否使用所有切片（推荐True）
        slice_range: 使用切片的范围，例如 (2, -2) 表示去掉头尾各2张
    """

    def __init__(self, base_dir, list_dir, split, transform=None,
                 num_classes=4, use_all_slices=False, slice_range=None,
                 filter_pure_background=False, min_fg_pixels=1):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(base_dir, 'list.txt')).readlines()
        self.data_dir = base_dir
        self.num_classes = num_classes
        self.use_all_slices = use_all_slices and (self.split != 'test')
        self.slice_range = slice_range  # (start, end) 例如 (2, -2)
        self.filter_pure_background = filter_pure_background
        self.min_fg_pixels = min_fg_pixels

        # 仅在需要全切片训练时构建索引
        if self.use_all_slices:
            self.all_slices = self._build_slice_index()

    def _build_slice_index(self):
        """构建所有切片的索引 (case_idx, slice_idx)"""
        all_slices = []
        dropped_background = 0
        for case_idx, case_name in enumerate(self.sample_list):
            case_name = case_name.strip('\n')
            pth = os.path.join(self.data_dir, case_name)
            img_pth = os.path.join(pth, case_name + '.nii.gz')
            lab_pth = os.path.join(pth, case_name + '_gt.nii.gz')

            # 读取图像获取切片数量
            img = read_nii(img_pth)
            num_slices = img.shape[0]

            lab = None
            if self.filter_pure_background:
                lab = read_nii(lab_pth)

            # 确定切片范围
            if self.slice_range:
                start_idx = max(0, self.slice_range[0])
                end_idx = min(num_slices,
                              num_slices + self.slice_range[1] if self.slice_range[1] < 0 else self.slice_range[1])
            else:
                start_idx = 0
                end_idx = num_slices

            # 添加所有有效切片
            for slice_idx in range(start_idx, end_idx):
                if self.filter_pure_background and lab is not None:
                    if np.count_nonzero(lab[slice_idx]) < self.min_fg_pixels:
                        dropped_background += 1
                        continue
                all_slices.append((case_idx, slice_idx))

        if self.filter_pure_background:
            print(f"ACDC {self.split}: {len(self.sample_list)} cases, {len(all_slices)} kept slices, {dropped_background} pure-background slices filtered")
        else:
            print(f"ACDC {self.split}: {len(self.sample_list)} cases, {len(all_slices)} total slices")
        return all_slices

    def __len__(self):
        if self.use_all_slices:
            return len(self.all_slices)
        else:
            return len(self.sample_list)

    def __getitem__(self, idx):
        # 测试模式：返回完整的 3D volume
        if self.split == 'test':
            case_name = self.sample_list[idx].strip('\n')
            pth = os.path.join(self.data_dir, case_name)
            img_pth = os.path.join(pth, case_name + '.nii.gz')
            lab_pth = os.path.join(pth, case_name + '_gt.nii.gz')

            # [D, H, W] 读取完整 3D volume
            image = read_nii(img_pth).astype(np.float32)
            image = _acdc_intensity_normalize(image)
            label = read_nii(lab_pth).astype(np.float32)

            # 返回完整 volume 用于测试
            image = torch.from_numpy(image)
            label = torch.from_numpy(label)

            sample = {'image': image, 'label': label}
            sample['case_name'] = case_name
            return sample

        # 训练/验证模式：返回单个 2D 切片
        if self.use_all_slices:
            case_idx, slice_idx = self.all_slices[idx]
            case_name = self.sample_list[case_idx].strip('\n')
        else:
            case_name = self.sample_list[idx].strip('\n')
            slice_idx = None  # 将使用中间切片

        pth = os.path.join(self.data_dir, case_name)
        img_pth = os.path.join(pth, case_name + '.nii.gz')
        lab_pth = os.path.join(pth, case_name + '_gt.nii.gz')

        # [D,H,W] 读取NIfTI文件
        img = read_nii(img_pth)
        lab = read_nii(lab_pth)

        # 体素强度归一化（对 AC/DC MRI 稳健 z-score）
        img = _acdc_intensity_normalize(img)

        # 选择切片
        if slice_idx is None:
            slice_idx = img.shape[0] // 2  # 中间切片

        image = img[slice_idx].astype(np.float32)
        label = lab[slice_idx].astype(np.float32)

        # 构建样本
        sample = {'image': image, 'label': label}

        # 应用transform
        if self.transform:
            sample = self.transform(sample)

        # case_name 添加切片索引
        sample['case_name'] = f"{case_name}_slice{slice_idx:03d}"
        return sample