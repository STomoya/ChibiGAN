"""Training loop for generator
"""

import copy
import os
from functools import partial
from itertools import chain

import torch
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image

from src import data, loss, utils
from src.DiffAugment import DiffAugment
from src.status import Status
from src.torchops import freeze, make_image_grid, optimizer_step, update_ema


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
    transform1 = data.build_transform_pipe(dcfg.transforms)
    transform2 = data.build_transform_pipe(dcfg.transforms)
    dataset = data.ImageImage(dcfg.data_root[0], dcfg.data_root[1], transform1, transform2)
    dataset = data.DataLoader(dataset, **dcfg.loader)

    # model
    mcfg = cfg.model
    latent_dim = mcfg.generator.latent_dim
    G = utils.construct_class_by_name(**mcfg.generator)
    G.initialize_torgb()
    G_ema = copy.deepcopy(G)
    Ddet = utils.construct_class_by_name(**mcfg.discriminator)
    Ddef = utils.construct_class_by_name(**mcfg.discriminator)
    G.to(device)
    G_ema.to(device)
    Ddet.to(device)
    Ddef.to(device)

    # load pre-trained Hramonizer weights
    assert mcfg.h_config_file is not None and mcfg.h_weight_file is not None, 'specify a pre-trained Blender'
    assert os.path.exists(mcfg.h_config_file) and os.path.exists(
        mcfg.h_weight_file
    ), 'You seem to not have a pre-trained model, or the path is incorrect'
    hcfg = OmegaConf.load(mcfg.h_config_file).config.model
    H = utils.construct_class_by_name(**hcfg.generator)
    h_state_dict = torch.load(mcfg.h_weight_file, map_location='cpu')
    H.load_state_dict(h_state_dict)
    freeze(H)
    H.to(device)

    # optimizer
    optcfg = tcfg.optimizer
    optim_G = utils.construct_class_by_name(G.parameters4optim(optcfg.generator.lr), **optcfg.generator)
    optim_D = utils.construct_class_by_name(chain(Ddet.parameters(), Ddef.parameters()), **optcfg.discriminator)

    # criterion
    adv_fn = loss.NonSaturating()
    r1_fn = loss.r1_regularize

    # diffaugment
    augment = partial(DiffAugment, policy=tcfg.diffaugment_policy)

    # status
    status = Status(
        len(dataset) * tcfg.epochs,
        folder.root / cfg.experiment.log_file,
        cfg.experiment.log_interval,
        'ChibiGAN',
        log_nvidia_smi_at=100,
    )
    status.log_stuff(config, dataset, G, optim_G, Ddet, optim_D, H)

    # others
    scaler = GradScaler() if amp else None
    const_z = torch.randn((tcfg.test_sample, latent_dim), device=device)

    while not status.is_end():
        for detailed, deformed in dataset:
            detailed = detailed.to(device)
            deformed = deformed.to(device)

            tempz = torch.randn((detailed.size(0), 2, latent_dim), device=device)
            z_det, z_def = tempz.unbind(dim=1)

            with autocast(amp):
                # B forward (we don't need grad)
                with torch.no_grad():
                    detailed = H(detailed)
                    deformed = H(deformed)

                # G forward
                _, x_det = G(z_det)
                x_def, _ = G(z_def)

                # diffaugment
                detailed_aug = augment(detailed)
                deformed_aug = augment(deformed)
                x_det_aug = augment(x_det)
                x_def_aug = augment(x_def)

                # D forward
                real_det_logits = Ddet(detailed_aug)
                real_def_logits = Ddef(deformed_aug)
                fake_det_logits = Ddet(x_det_aug.detach())
                fake_def_logits = Ddef(x_def_aug.detach())

                # loss
                adv_det_loss = adv_fn.d_loss(real_det_logits, fake_det_logits)
                adv_def_loss = adv_fn.d_loss(real_def_logits, fake_def_logits)
                adv_loss = adv_det_loss + adv_def_loss
                gp_loss = 0
                if tcfg.gp_lambda > 0 and status.batches_done % tcfg.gp_every == 0:
                    gp_det_loss = r1_fn(detailed, Ddet, scaler) * tcfg.gp_lambda
                    gp_def_loss = r1_fn(deformed, Ddef, scaler) * tcfg.gp_lambda
                    gp_loss = gp_det_loss + gp_def_loss
                D_loss = adv_loss + gp_loss

            optimizer_step(D_loss, optim_D, scaler, zero_grad=True, set_to_none=True, update_scaler=False)

            with autocast(amp):
                # D forward
                fake_det_logits = Ddet(x_det_aug.detach())
                fake_def_logits = Ddef(x_def_aug.detach())

                # loss
                adv_det_loss = adv_fn.g_loss(fake_det_logits)
                adv_def_loss = adv_fn.g_loss(fake_def_logits)
                G_loss = adv_det_loss + adv_def_loss

            optimizer_step(G_loss, optim_G, scaler, zero_grad=True, set_to_none=True, update_scaler=True)
            update_ema(G, G_ema, tcfg.ema_decay, copy_buffers=True)

            # save running samples
            if tcfg.running > 0 and status.batches_done % tcfg.running == 0:
                save_image(
                    make_image_grid(x_det, x_def), folder.root / 'running.jpg', normalize=True, value_range=(-1, 1)
                )

            # save weights and images at that iteration.
            if status.batches_done % tcfg.save == 0:
                kbatch = status.get_kbatches()
                with torch.no_grad():
                    x_def, x_det = G_ema(const_z)
                save_image(
                    make_image_grid(x_det, x_def),
                    folder.image / f'{kbatch}.jpg',
                    nrow=2 * 4,
                    normalize=True,
                    value_range=(-1, 1),
                )
                torch.save(G_ema.state_dict(), folder.model / f'{kbatch}.torch')

            status.update(G=G_loss.item(), D=D_loss.item())

            if status.is_end():
                break
