"""training loop for Harmonizer
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image

from src import data, loss, utils
from src.status import Status
from src.torchops import make_image_grid, optimizer_step


def setup_folder(config, child_folders: dict):
    folder = utils.Folder(utils.Path(config.ckpt_folder) / config.name / utils.get_now_string())
    folder.add_children(**child_folders)
    folder.mkdir()
    return folder


def train(config):
    folder = setup_folder(config.config.experiment, dict(image='images', model='models'))
    utils.save_hydra_config(config, folder.root / 'config.yaml')
    cfg = config.config
    tcfg = cfg.train

    # env
    device = torch.device(cfg.env.device)
    amp = cfg.env.amp

    # dataset
    dcfg = cfg.data
    dataset = data.ImageImageMask(dcfg.data_root, dcfg.ornament_root, dcfg.image_size)
    dataset = data.DataLoader(dataset, **dcfg.loader)

    # model
    mcfg = cfg.model
    H = utils.construct_class_by_name(**mcfg.generator)
    D = utils.construct_class_by_name(**mcfg.discriminator)
    H.to(device)
    D.to(device)

    # optimizer
    optcfg = tcfg.optimizer
    optim_H = utils.construct_class_by_name(H.parameters(), **optcfg.generator)
    optim_D = utils.construct_class_by_name(D.parameters(), **optcfg.discriminator)

    # criterion
    adv_fn = loss.LeastSquare()
    l1_fn = nn.L1Loss()
    vgg_fn = loss.VGGLoss(device)

    # status
    status = Status(
        len(dataset) * tcfg.epochs,
        folder.root / cfg.experiment.log_file,
        cfg.experiment.log_interval,
        'Harmonizer',
        log_nvidia_smi_at=100,
    )
    status.log_stuff(config, dataset, H, optim_H, D, optim_D)

    # others
    scaler = GradScaler() if amp else None

    while not status.is_end():
        for image, ornament, mask in dataset:
            image = image.to(device)
            ornament = ornament.to(device)
            mask = mask.to(device)

            with autocast(amp):
                image_w_orna = ornament * mask + image * (1 - mask)
                # G forward
                fake = H(image_w_orna)
                identity = H(image)

                # D forward
                real_logits = D(image)
                fake_logits = D(fake.detach())

                # loss
                D_loss = adv_fn.d_loss(real_logits, fake_logits)

            optimizer_step(
                D_loss,
                optim_D,
                scaler,
                zero_grad=True,
                set_to_none=True,
                update_scaler=False,
            )

            with autocast(amp):
                # D forward with gradient
                fake_logits = D(fake)

                # loss
                adv_loss = adv_fn.g_loss(fake_logits) * tcfg.gan_lambda
                id_loss = l1_fn(identity, image) * tcfg.id_lambda
                style_loss = vgg_fn.style_loss(image, fake)
                G_loss = adv_loss + id_loss + style_loss

            optimizer_step(
                G_loss,
                optim_H,
                scaler,
                zero_grad=True,
                set_to_none=True,
                update_scaler=True,
            )

            if tcfg.running > 0 and status.batches_done % tcfg.running == 0:
                save_image(
                    make_image_grid(image, image_w_orna, fake),
                    folder.root / 'running.jpg',
                    nrow=6,
                    normalize=True,
                    value_range=(-1, 1),
                )

            if status.batches_done % tcfg.save == 0:
                kbatch = status.get_kbatches()
                save_image(
                    make_image_grid(image, image_w_orna, fake, identity),
                    folder.image / f'{kbatch}.jpg',
                    nrow=2 * 4,
                    normalize=True,
                    value_range=(-1, 1),
                )
                torch.save(H.state_dict(), folder.model / f'{kbatch}.torch')

            status.update(**{'Loss/H': G_loss.item(), 'Loss/D': D_loss.item()})

            if status.is_end():
                break
