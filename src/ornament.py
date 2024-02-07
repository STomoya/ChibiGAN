"""Ornament augmentation
"""

import glob
import os
import random

import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from skimage.color import rgb2hsv

from src.utils import is_image_file


class OrnamentGetter:
    def __init__(self, data_root, color_range=(0, 255), alpha_blur_kernel_size=5) -> None:
        ornaments = glob.glob(os.path.join(data_root, '**', '*'), recursive=True)
        self.ornaments = [item for item in ornaments if is_image_file(item)]
        assert len(self.ornaments) > 0, f'{self.__class__.__name__}: There should be more than 1 image.'

        self.color_range = color_range
        self.blur_kernel_size = alpha_blur_kernel_size

        mask = Image.open('./static/ornament_mask.png').convert('LA').split()[-1]
        # shrink -> pad to orignal size
        # this is done because of some changes in the dataset preparation,
        # but the mask was made before those changes.
        self.mask = TF.center_crop(TF.resize(mask, tuple((int(x * 0.6) for x in mask.size))), mask.size)

        self.affine_transform = T.RandomAffine(20, None, (0.5, 1.0), (0.9, 8.0, 0.9, 8.0))

    def __call__(self, image: Image.Image):
        shape = image.size

        # load ornament
        ornament_path = random.choice(self.ornaments)
        ornament = Image.open(ornament_path).convert('LA')
        ornament = ornament.resize(shape, Image.BILINEAR)

        # augmentation
        ornament = self.affine_transform(ornament)

        if random.random() < 0.5:
            ornament = TF.hflip(ornament)

        # we first confirm the position.
        trans_matrix = [1, 0, 0, 0, 1, 0]
        position_mask = TF.resize(self.mask, shape)
        position_mask = np.array(position_mask)
        while True:
            x, y = np.random.choice(min(shape), 2)
            if position_mask[y][x] > 0:
                break
        trans_matrix[2] = x - shape[0] // 2
        trans_matrix[5] = -(y - shape[1] // 2)
        ornament = ornament.transform(shape, Image.AFFINE, tuple(trans_matrix))

        # random color
        gray, mask = ornament.split()
        black = (0, 0, 0)
        white = np.random.uniform(*self.color_range, (3,)).astype(int).tolist()
        rgb = ImageOps.colorize(gray, black, white)
        if random.random() < 0.8:
            # adjust saturation and brightness depending on the target image
            img_array = np.array(image)
            img_hsv = rgb2hsv(img_array)
            itm_array = np.array(rgb)
            itm_hsv = rgb2hsv(itm_array)[itm_array.mean(axis=2) != 0]  # exclude transparent pixels
            saturation_scale = img_hsv[:, :, 1].mean() / (itm_hsv[:, 1].mean() + 1e-8)
            brightness_scale = img_hsv[:, :, 2].mean() / (itm_hsv[:, 2].mean() + 1e-8)
            # add random brightness/saturation
            if np.random.random() < 0.8:
                saturation_scale += np.random.uniform(-0.3, 0.3)
            if np.random.random() < 0.8:
                brightness_scale += np.random.uniform(-0.3, 0.3)
            rgb = TF.adjust_saturation(rgb, saturation_scale)
            rgb = TF.adjust_brightness(rgb, brightness_scale)

        if random.random() < 0:
            mask = TF.gaussian_blur(mask, kernel_size=self.blur_kernel_size)

        return rgb, mask


class OrnamentAugmentation:
    def __init__(
        self,
        data_root,
        p,
        color_range=(0, 255),
        alpha_blur_kernel_size=5,
        return_mask=False,
    ) -> None:
        self.p = p
        self.getter = OrnamentGetter(data_root, color_range, alpha_blur_kernel_size)
        self.return_mask = return_mask

    def __call__(self, image):
        if random.random() < self.p:
            ornament, mask = self.getter(image)
            alpha = TF.invert(mask)
            image = Image.composite(image, ornament, alpha)

            if self.return_mask:
                return image, alpha

        return image
