"""Datasets
"""

import glob
import os
import random
from typing import List

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset  # noqa: F401

ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.ornament import OrnamentAugmentation, OrnamentGetter  # noqa: E402
from src.utils import is_image_file  # noqa: E402


def build_transform(name: str, **params):
    if hasattr(T, name):
        return getattr(T, name)(**params)
    elif name == 'OrnamentAugmentation':
        return OrnamentAugmentation(**params)
    raise UserWarning(f'torchvision.transforms.{name} does not exist.')


def build_transform_pipe(configs: List[dict]):
    """make transform from list of TransformConfigs
    Usage:
        transforms = make_transform_from_config(
            [
                {'name': 'Resize', 'size': (224, 224)},
                {'name': 'ToTensor'},
                {'name': 'Normalize', 'mean': 0.5, 'std': 0.5}
            ]
        )
    Arguments:
        configs: list[dict]
            List of dicts containing a least 'name' key.
    """
    transform = []
    for config in configs:
        transform.append(build_transform(**config))
    return T.Compose(transform)


def _collect_and_filter(root):
    temp_paths = glob.glob(os.path.join(root, '**', '*'), recursive=True)
    image_paths = [path for path in temp_paths if is_image_file(path)]
    return image_paths


class ImageImage(Dataset):
    def __init__(self, data_root1, data_root2, transform1, transform2) -> None:
        super().__init__()
        image_paths1 = _collect_and_filter(data_root1)
        image_paths2 = _collect_and_filter(data_root2)
        min_length = min(map(len, [image_paths1, image_paths2]))
        random.seed(3407)
        random.shuffle(image_paths1)
        random.shuffle(image_paths2)
        self.image_paths1 = image_paths1[:min_length]
        self.image_paths2 = image_paths2[:min_length]

        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.image_paths1)

    def __getitem__(self, index):
        image_path1 = self.image_paths1[index]
        image_path2 = self.image_paths2[index]

        image1 = Image.open(image_path1).convert('RGB')
        image2 = Image.open(image_path2).convert('RGB')

        image1 = self.transform1(image1)
        image2 = self.transform2(image2)

        return image1, image2


class ImageImageMask(Dataset):
    def __init__(self, data_root, ornament_root, image_size) -> None:
        super().__init__()
        self.image_paths = _collect_and_filter(data_root)
        self.image_size = image_size
        self.getter = OrnamentGetter(ornament_root)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        image = TF.resize(image, (self.image_size, self.image_size))
        ornament, mask = self.getter(image)

        if random.random() < 0.5:
            image = TF.hflip(image)

        image = TF.to_tensor(image)
        ornament = TF.to_tensor(ornament)
        mask = TF.to_tensor(mask)

        image = TF.normalize(image, 0.5, 0.5)
        ornament = TF.normalize(image, 0.5, 0.5)
        return image, ornament, mask
