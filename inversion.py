"""from https://github.com/rosinality/stylegan2-pytorch/blob/f938922c5ba7f9c325e5e13795da8ee93e849250/projector.py
modified by Tomoya Sawada
"""

import argparse
import glob
import math
import os

import lpips
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import save_image

from src.utils import construct_class_by_name


def load_image(path):
    """load single image"""
    image = Image.open(path).convert('RGB')
    image = TF.resize(image, 128)
    image = TF.center_crop(image, 128)
    image = TF.to_tensor(image)
    image = TF.normalize(image, 0.5, 0.5)
    image = image.unsqueeze(0)
    return image


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    """learning rate"""
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def get_schedule(t, rampdown=0.5):
    """lambda schedule"""
    ramp = min(1, (1 - t) / rampdown)
    ramp = 0.5 - 0.5 * math.cos(ramp * math.pi)
    return ramp


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


_lpips_fn = None


def get_lpips():
    global _lpips_fn
    if _lpips_fn is None:
        _lpips_fn = lpips.LPIPS(net='vgg')
    return _lpips_fn


def run(
    G: nn.Module,  # Generator. Output should be: (deform, detail)
    id: str,  # output filename will be {original filename}_{id}.png
    output: str,  # output folder name
    input: str = './unseen',  # folder name where input images are
    latent_dim: int = 512,  # latent dim input size
    iterations: int = 1000,  # iterations
    init_lr=0.1,  # initial learning rate
    lambda_0=2.0,  # lambda for deform
    noise_lambda=0.00005,
    device=torch.device('cuda'),  # device  # noqa: B008
    save_last_only=True,  # only save the result at end
    only_result=True,  # save only results to image
):
    assert isinstance(input, (list, tuple)) or os.path.exists(input)

    lpips_fn = get_lpips().to(device)
    downsample = nn.AvgPool2d(2)

    if isinstance(input, (list, tuple)):
        images = input
    elif os.path.isdir(input):
        images = sorted(glob.glob(os.path.join(input, '*')))
    elif os.path.isfile(input):
        images = [input]

    for image in images:
        filename = os.path.splitext(os.path.basename(image))[0]

        target = load_image(image).to(device)

        with torch.no_grad():
            latent = torch.randn(10000, latent_dim, device=device)
            style = G.mapping(latent)
            latent_mean = style.mean(0)
            latent_std = ((latent - latent_mean).pow(2).sum() / 10000) ** 0.5
        noises = G.make_noise(1, device)

        latent_in = latent_mean.detach().clone().unsqueeze(0)
        latent_in = latent_in.unsqueeze(1).repeat(1, G.num_layers, 1)

        latent_in.requires_grad = True
        for noise in noises:
            noise.requires_grad = True

        optimizer = optim.Adam([latent_in, *noises], lr=init_lr)

        for i in range(iterations):
            t = i / iterations
            lr = get_lr(t, init_lr)
            optimizer.param_groups[0]['lr'] = lr

            noise_strength = latent_std * 0.05 * max(0, 1 - t / 0.75) ** 2
            latent_n = latent_in + torch.randn_like(latent_in) * noise_strength.item()

            image_deform, image_detail = G(latent_n, noise=noises)

            d_lambda_effect = lambda_0 * get_schedule(t)
            lpips_detail = lpips_fn(image_detail, target)
            lpips_deform = lpips_fn(downsample(image_deform), downsample(target)) * d_lambda_effect
            loss = lpips_detail + lpips_deform
            if noise_lambda > 0:
                loss = loss + noise_regularize(noises) * noise_lambda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize(noises)

            if not save_last_only:
                image_deform, image_detail = G(latent_in, noise=noises)
                if only_result:
                    image_tensor = image_deform
                else:
                    image_tensor = torch.cat([target, image_deform, image_detail], dim=0)
                save_image(
                    image_tensor,
                    os.path.join(output, f'{filename}_{id}.png'),
                    normalize=True,
                    value_range=(-1, 1),
                    padding=0,
                )

        if save_last_only:
            image_deform, image_detail = G(latent_in, noise=noises)
            if only_result:
                image_tensor = image_deform
            else:
                image_tensor = torch.cat([target, image_deform, image_detail], dim=0)
            save_image(
                image_tensor,
                os.path.join(output, f'{filename}.{id}.png'),
                normalize=True,
                value_range=(-1, 1),
                padding=0,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Image to translate.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('weights', help='Trained weights')
    parser.add_argument('--output', default='./results', help='Folder to save translation results.')
    parser.add_argument('--device', default='cuda', help='Device. Either "cpu" or "cuda".')
    parser.add_argument('--num-trials', default=1, type=int, help='Number of trials.')
    parser.add_argument('--total-steps', default=2000, type=int, help='Number of total optimization steps.')
    parser.add_argument('--lambda0', default=2.0, type=float, help='Initial value of lambda')
    parser.add_argument('--only-results', default=False, action='store_true', help='Save only the translation result.')
    args = parser.parse_args()

    device = torch.device(args.device)

    # build model
    gconfig = OmegaConf.load(args.config).config.model.generator
    G = construct_class_by_name(**gconfig)
    G.eval()
    for parm in G.parameters():
        parm.requires_grad = False
    state_dict = torch.load(args.weights, map_location='cpu')
    G.load_state_dict(state_dict)
    G.to(device)
    if args.device == 'cuda':
        # build CUDA kernels before running
        G(torch.randn(3, gconfig.latent_dim))

    os.makedirs(args.output, exist_ok=True)

    for trial in range(args.num_trials):
        run(
            G,
            f'trail{trial+1:05}',
            args.output,
            args.input,
            gconfig.latent_dim,
            args.total_steps,
            0.1,
            args.lambda0,
            0.00005,
            device,
            save_last_only=True,
            only_result=args.only_results,
        )


if __name__ == '__main__':
    main()
