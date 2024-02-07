from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from thirdparty.stylegan2_ops.ops import bias_act, conv2d_resample, upfirdn2d

#############################################   UTILS   #########################################


def _default(value, default_value):
    return value if value is not None else default_value


def make_blur_kernel(filter_size=4):
    def _binomial_filter():
        """Pascal's triangle."""

        def c(n, k):
            if k <= 0 or n <= k:
                return 1
            else:
                return c(n - 1, k - 1) + c(n - 1, k)

        return [c(filter_size - 1, j) for j in range(filter_size)]

    filter = torch.tensor(_binomial_filter(), dtype=torch.float32)
    kernel = torch.outer(filter, filter)
    kernel /= kernel.sum()
    return kernel


#############################################   LAYERS   #########################################


class LinearAct(nn.Module):
    """linear -> bias -> activation"""

    def __init__(self, in_features, out_features, bias=True, act_name='lrelu') -> None:
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._bias = bias
        self._act_name = act_name

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = 1 / (self.weight[0].numel() ** 0.5)

    def forward(self, x):
        # linear
        weight = self.weight * self.scale
        x = F.linear(x, weight)
        # bias + act
        b = self.bias.to(x) if self.bias is not None else None
        dim = 2 if x.ndim == 3 else 1
        x = bias_act.bias_act(x, b, dim, self._act_name)
        return x

    def extra_repr(self):
        return (
            f'in_features={self._in_features}, out_features={self._out_features}, '
            + f'bias={self._bias}, act_name={self._act_name}'
        )


class Conv2dAct(nn.Module):
    """conv -> bias -> activation"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        bias=True,
        up=1,
        down=1,
        filter_size=4,
        act_name='lrelu',
        act_gain=None,
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._bias = bias
        self._up = up
        self._down = down
        self._filter_size = filter_size
        self._act_name = act_name
        self._act_gain = bias_act.activation_funcs[act_name].def_gain if act_gain is None else act_gain

        self._padding = kernel_size // 2
        if kernel_size % 2 == 0:
            self._padding -= 1  # for PatchGAN
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.scale = 1 / (self.weight[0].numel() ** 0.5)

        if up > 1 or down > 1:
            self.register_buffer('kernel', make_blur_kernel(filter_size))
        else:
            self.kernel = None

    def forward(self, x):
        # convolution
        weight = self.weight * self.scale
        x = conv2d_resample.conv2d_resample(x, weight.to(x), self.kernel, self._up, self._down, self._padding)
        # bias + act
        b = self.bias.to(x) if self.bias is not None else None
        x = bias_act.bias_act(x, b, 1, self._act_name, gain=self._act_gain)
        return x

    def extra_repr(self):
        return (
            f'in_channels={self._in_channels}, out_channels={self._out_channels}, bias={self._bias}, up={self._up}, '
            + f'down={self._down}, filter_size={self._filter_size}, act_name={self._act_name}, act_gain={self._act_gain}'  # noqa: E501
        )


class ModulatedConv2d(nn.Module):
    """modulate -> conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, up=1, filter_size=4, demod=True) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._up = up
        self._filter_size = filter_size
        self._demod = demod

        self._padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.scale = 1 / (self.weight[0].numel() ** 0.5)

        if up > 1:
            self.register_buffer('kernel', make_blur_kernel(filter_size))
        else:
            self.kernel = None
        self._flip_weight = up == 1

    def forward(self, x, w):
        B, _, H, W = x.size()

        # modulate
        weight = self.weight[None, ...] * w[:, None, :, None, None] * self.scale
        # demodulate
        if self._demod:
            d = weight.pow(2).sum([2, 3, 4], keepdim=True).add(1e-8).rsqrt()
            weight = weight * d

        # reshape for group conv implementation
        x = x.reshape(1, -1, H, W)
        _, _, *weight_size = weight.size()
        weight = weight.reshape(B * self._out_channels, *weight_size)

        # conv (+ upsample)
        x = conv2d_resample.conv2d_resample(
            x, weight.to(x), self.kernel, self._up, padding=self._padding, groups=B, flip_weight=self._flip_weight
        )

        return x.reshape(B, self._out_channels, H * self._up, W * self._up)

    def extra_repr(self):
        return (
            f'in_channels={self._in_channels}, out_channel={self._out_channels}, kernel_size={self._kernel_size}, '
            + f'up={self._up}, filter_size={self._filter_size}, demod={self._demod}'
        )


class InjectNoise(nn.Module):
    """noise"""

    def __init__(self, size) -> None:
        super().__init__()
        self._size = size

        self.register_buffer('const_noise', torch.randn(1, 1, size, size))
        self.scale = nn.Parameter(torch.zeros([]))

    def forward(self, x, noise='random'):
        if isinstance(noise, torch.Tensor):
            pass
        elif noise == 'random':
            B, _, H, W = x.size()
            noise = torch.randn(B, 1, H, W, device=x.device)
        elif noise == 'const':
            noise = self.const_noise.expand(x.size(0), -1, -1, -1)

        return x + noise * self.scale

    def make_noise(self, batch_size, device):
        return torch.randn(batch_size, 1, self._size, self._size, device=device)

    def extra_repr(self):
        return f'size={self._size}'


class PixelNorm(nn.Module):
    """pixel normalization"""

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input**2, dim=1, keepdim=True).add(1e-8))


class Upsample(nn.Module):
    """upsample"""

    def __init__(self, filter_size=4) -> None:
        super().__init__()
        self._filter_size = filter_size
        self.register_buffer('kernel', make_blur_kernel(filter_size))

    def forward(self, x):
        x = upfirdn2d.upsample2d(x, self.kernel)
        return x

    def extra_repr(self):
        return f'filter_size={self._filter_size}'


#############################################   BLOCKS   #########################################


class StyleLayer(nn.Module):
    """modconv -> noise -> bias -> activation
    affine --^
    """

    def __init__(self, in_channels, style_dim, out_channels, resolution, up=1, filter_size=4, act_name='lrelu') -> None:
        super().__init__()
        self._in_channels = in_channels
        self._style_dim = style_dim
        self._out_channels = out_channels
        self._resolution = resolution * up
        self._up = up
        self._filter_size = filter_size
        self._act_name = act_name

        self.affine = LinearAct(style_dim, in_channels, act_name='linear')
        self.affine.bias.data.fill_(1.0)

        self.conv = ModulatedConv2d(in_channels, out_channels, 3, up, filter_size)
        self.add_noise = InjectNoise(self._resolution)

        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.make_noise = self.add_noise.make_noise

    def forward(self, x, w, noise='random'):
        w = self.affine(w)
        # mod conv
        x = self.conv(x, w)
        # noise
        x = self.add_noise(x, noise)
        # bias + act
        x = bias_act.bias_act(x, self.bias.to(x), 1, self._act_name)
        return x


class ToRGB(nn.Module):
    """modconv -> bias -> act -> modconv
    affine --^  affine ----------------^
    w -^-----------^
    """

    def __init__(self, in_channels, style_dim, out_channels, act_name='lrelu') -> None:
        super().__init__()
        self._in_channels = in_channels
        self._style_dim = style_dim
        self._out_channels = out_channels
        self._act_name = act_name

        self.affine1 = LinearAct(style_dim, in_channels, act_name='linear')
        self.affine1.bias.data.fill_(1.0)
        self.conv1 = ModulatedConv2d(in_channels, in_channels, 3)
        self.bias1 = nn.Parameter(torch.zeros(in_channels))
        self.affine2 = LinearAct(style_dim, in_channels, act_name='linear')
        self.affine2.bias.data.fill_(1.0)
        self.conv2 = ModulatedConv2d(in_channels, out_channels, 1, demod=False)

    def forward(self, x, w):
        x = self.conv1(x, self.affine1(w))
        x = bias_act.bias_act(x, self.bias1.to(x), 1, self._act_name)
        x = self.conv2(x, self.affine2(w))
        return x


class ConstInput(nn.Module):
    def __init__(self, channels, size) -> None:
        super().__init__()
        self._channels = channels
        self._size = size
        self.input = nn.Parameter(torch.randn(1, channels, size, size))

    def forward(self, x):
        B = x.size(0)
        return self.input.repeat(B, 1, 1, 1)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, act_name='lrelu') -> None:
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._filter_size = filter_size
        self._act_name = act_name

        self.conv1 = Conv2dAct(in_channels, out_channels, act_name=act_name)
        self.conv2 = Conv2dAct(
            out_channels, out_channels, down=2, filter_size=filter_size, act_name=act_name, act_gain=1.0
        )
        self.skip = Conv2dAct(
            in_channels, out_channels, 1, bias=False, down=2, filter_size=filter_size, act_name='linear', act_gain=1.0
        )

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        x = self.skip(x)
        return h + x


class MinibatchStdDev(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = self.group_size if N % self.group_size == 0 else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)
        y = y - y.mean(dim=0)
        y = y.square().mean(dim=0)
        y = (y + 1e-8).sqrt()
        y = y.mean(dim=[2, 3, 4])
        y = y.reshape(-1, F, 1, 1)
        y = y.repeat(G, 1, H, W)
        x = torch.cat([x, y], dim=1)
        return x


class DiscEpilogue(nn.Sequential):
    def __init__(self, channels, bottom, mbsd_group_size=4, mbsd_channels=1, act_name='lrelu') -> None:
        self._channels = channels
        self._bottom = bottom
        self._mbsd_group_size = mbsd_group_size
        self._mbsd_channels = mbsd_channels
        self._act_name = act_name

        super().__init__(
            MinibatchStdDev(mbsd_group_size, mbsd_channels) if mbsd_channels > 0 else nn.Identity(),
            Conv2dAct(channels + mbsd_channels, channels, act_name=act_name),
            nn.Flatten(),
            LinearAct(channels * bottom**2, channels, act_name=act_name),
            LinearAct(channels, 1, act_name='linear'),
        )


#############################################   NETWORKS   #########################################


class Mapping(nn.Module):
    def __init__(self, latent_dim, style_dim, num_layers, pixel_norm=True, act_name='lrelu', ema_decay=0.998) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self._style_dim = style_dim
        self._num_layers = num_layers
        self._pixel_norm = pixel_norm
        self._act_name = act_name
        self._ema_decay = ema_decay

        self.norm = PixelNorm() if pixel_norm else nn.Identity()
        layers = [LinearAct(latent_dim, style_dim, act_name=act_name)]
        for _ in range(num_layers - 1):
            layers.append(LinearAct(style_dim, style_dim, act_name=act_name))
        self.map = nn.Sequential(*layers)
        self.register_buffer('w_avg', torch.zeros(style_dim))

    def forward(self, z, truncation_psi=1.0):
        z = self.norm(z)
        w = self.map(z)

        if self.training:
            stats = w.detach().to(torch.float32).mean(dim=0)
            self.w_avg.copy_(stats.lerp(self.w_avg, self._ema_decay))

        if truncation_psi != 1.0:
            w = self.w_avg.lerp(w, truncation_psi)

        return w


class Synthesis(nn.Module):
    def __init__(
        self,
        image_size,
        style_dim,
        in_channels=None,
        out_channels=None,
        channels=64,
        max_channels=512,
        bottom=4,
        filter_size=4,
        act_name='lrelu',
    ) -> None:
        super().__init__()
        in_channels = _default(in_channels, style_dim)
        out_channels = _default(out_channels, 3)

        self._image_size = image_size
        self._style_dim = style_dim
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._channels = channels
        self._max_channels = max_channels
        self._bottom = bottom
        self._filter_size = filter_size
        self._act_name = act_name

        num_ups = int(math.log2(image_size) - math.log2(bottom))

        self.const_input = ConstInput(in_channels, bottom)
        channels = channels * 2**num_ups
        ochannels = min(max_channels, channels)
        self.input = StyleLayer(in_channels, style_dim, ochannels, bottom, act_name=act_name)
        self.input_to_det = ToRGB(ochannels, style_dim, out_channels, act_name)
        self.input_to_def = ToRGB(ochannels, style_dim, out_channels, act_name)
        self._num_layers = 1

        self.style_layers1 = nn.ModuleList()
        self.style_layers2 = nn.ModuleList()
        self.to_dets = nn.ModuleList()
        self.to_defs = nn.ModuleList()

        resl = bottom
        for _ in range(num_ups):
            channels //= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            self.style_layers1.append(StyleLayer(ichannels, style_dim, ochannels, resl, 1, filter_size, act_name))
            self.style_layers2.append(StyleLayer(ochannels, style_dim, ochannels, resl, 2, filter_size, act_name))
            self.to_dets.append(ToRGB(ochannels, style_dim, out_channels, act_name))
            self.to_defs.append(ToRGB(ochannels, style_dim, out_channels, act_name))
            resl *= 2
            self._num_layers += 2

        self.upsample = Upsample(filter_size)
        self.register_buffer('empty_image', torch.zeros(1, out_channels, bottom, bottom))
        self._num_noise = self._num_layers

        self.init_torgbs()

    def init_torgbs(self):
        """initialize the two ToRGBs to have the same initial weights"""
        for to_det, to_def in zip(self.to_dets, self.to_defs):
            for det_param, def_param in zip(to_det.parameters(), to_def.parameters()):
                det_param.data.copy_(def_param.data)

    def forward(self, w, noise, return_feat: bool = False):
        assert len(w) == self._num_layers
        assert len(noise) == self._num_noise

        x = self.const_input(w[0])
        x = self.input(x, w[0], noise[0])
        det_img = self.input_to_det(x, w[0])
        def_img = self.input_to_def(x, w[0])

        for i, (style_layer1, style_layer2, to_det, to_def) in enumerate(
            zip(self.style_layers1, self.style_layers2, self.to_dets, self.to_defs)
        ):
            index = (i * 2 + 1, i * 2 + 2)
            x = style_layer1(x, w[index[0]], noise[index[0]])
            x = style_layer2(x, w[index[1]], noise[index[1]])
            det_img = to_det(x, w[index[1]]) + self.upsample(det_img)
            def_img = to_def(x, w[index[1]]) + self.upsample(def_img)

        if return_feat:
            return def_img, det_img, x
        return def_img, det_img

    def make_noise(self, batch_size, device):
        noise = [self.input.make_noise(batch_size, device)]
        for style_layer1, style_layer2 in zip(self.style_layers1, self.style_layers2):
            noise.extend([style_layer1.make_noise(batch_size, device), style_layer2.make_noise(batch_size, device)])
        return noise


#############################################   GENERATOR   #########################################


class Generator(nn.Module):
    def __init__(
        self,
        image_size,
        latent_dim,
        style_dim=None,
        syn_in_channels=None,
        channels=64,
        max_channels=512,
        bottom=4,
        filter_size=4,
        map_num_layers=8,
        pixel_norm=True,
        act_name='lrelu',
        map_lr_scale=0.01,
    ) -> None:
        super().__init__()
        style_dim = _default(style_dim, latent_dim)
        syn_in_channels = _default(syn_in_channels, latent_dim)

        self._image_size = image_size
        self._latent_dim = latent_dim
        self._style_dim = style_dim
        self._syn_in_channels = syn_in_channels
        self._channels = channels
        self._max_channels = max_channels
        self._bottom = bottom
        self._filter_size = filter_size
        self._map_num_layers = map_num_layers
        self._pixel_norm = pixel_norm
        self._act_name = act_name

        self.map_lr_scale = map_lr_scale

        self.mapping = Mapping(latent_dim, style_dim, map_num_layers, pixel_norm, act_name)
        self.synthesis = Synthesis(
            image_size, style_dim, syn_in_channels, None, channels, max_channels, bottom, filter_size, act_name
        )
        self.num_layers = self.synthesis._num_layers
        self.num_noise = self.synthesis._num_noise
        self.make_noise = self.synthesis.make_noise
        self.initialize_torgb = self.synthesis.init_torgbs

    def forward(
        self,
        z: torch.Tensor,
        noise='random',
        truncation_psi: float = 1.0,
        return_w: bool = False,
        switch_indices: list | None = None,
    ):
        w = self.mapping(z, truncation_psi)

        if w.ndim == 2:
            w = w[:, None, :].repeat(1, self.num_layers, 1)
        elif w.ndim == 3 and w.size(1) != self.num_layers:
            assert switch_indices is not None
            assert (w.size(1) - 1) == len(switch_indices)
            end_indices = [*switch_indices, self.num_layers]
            w_parts = w.chunk(w.size(1), dim=1)
            start = 0
            w = []
            for index, end in enumerate(end_indices):
                w.extend([w_parts[index].clone() for _ in range(start, end)])
                start = end
            w = torch.cat(w, dim=1)
        if isinstance(noise, str):
            noise = [noise for _ in range(self.num_noise)]

        ws = w.unbind(dim=1)
        def_image, det_image = self.synthesis(ws, noise)

        if return_w:
            return def_image, det_image, w
        return def_image, det_image

    def parameters4optim(self, lr):
        return [
            {'params': self.synthesis.parameters()},
            {'params': self.mapping.parameters(), 'lr': lr * self.map_lr_scale},
        ]


#############################################   DISC   #########################################


class Discriminator(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels=None,
        channels=64,
        max_channels=512,
        mbsd_group_size=4,
        mbsd_channels=1,
        bottom=4,
        filter_size=4,
        act_name='lrelu',
    ) -> None:
        super().__init__()
        in_channels = _default(in_channels, 3)

        self._image_size = image_size
        self._in_channels = in_channels
        self._channels = channels
        self._max_channels = max_channels
        self._mbsd_group_size = mbsd_group_size
        self._mbsd_channels = mbsd_channels
        self._bottom = bottom
        self._filter_size = filter_size
        self._act_name = act_name

        num_downs = int(math.log2(image_size) - math.log2(bottom))

        ochannels = channels
        self.from_rgb = Conv2dAct(in_channels, ochannels, act_name=act_name)

        resblocks = []
        for _ in range(num_downs):
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            resblocks.append(ResBlock(ichannels, ochannels, filter_size, act_name))
        self.resblocks = nn.Sequential(*resblocks)
        self.epilogue = DiscEpilogue(ochannels, bottom, mbsd_group_size, mbsd_channels, act_name)

    def forward(self, x):
        x = self.from_rgb(x)
        x = self.resblocks(x)
        x = self.epilogue(x)
        return x


#############################################   BLENDER   #########################################


class FlatResBlock(nn.Module):
    def __init__(self, channels, act_name) -> None:
        super().__init__()
        self.conv1 = Conv2dAct(channels, channels, act_name=act_name)
        self.conv2 = Conv2dAct(channels, channels, act_name=act_name, act_gain=1.0)

    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + skip


class Harmonizer(nn.Module):
    def __init__(
        self,
        image_size,
        bottom=None,
        channels=64,
        max_channels=512,
        resblocks=2,
        act_name='lrelu',
        filter_size=4,
        io_channels=3,
    ) -> None:
        super().__init__()
        self._image_size = image_size
        self._bottom = _default(bottom, image_size // 4)  # default to 2 downsampling blocks
        self._channels = channels
        self._max_channels = max_channels
        self._act_name = act_name
        self._filter_size = filter_size
        self._io_channels = io_channels

        num_sampling = int(math.log2(self._image_size) - math.log2(self._bottom))

        ochannels = channels
        self.input = Conv2dAct(io_channels, ochannels, act_name=act_name)

        self.downs = nn.ModuleList()
        dfeat_dims = []
        for _ in range(num_sampling - 1):
            channels *= 2
            ichannels, ochannels = ochannels, min(max_channels, channels)
            dfeat_dims.append(ochannels)
            self.downs.append(
                nn.Sequential(
                    Conv2dAct(ichannels, ochannels, act_name=act_name),
                    Conv2dAct(ochannels, ochannels, down=2, filter_size=filter_size, act_name=act_name, act_gain=1.0),
                )
            )

        self.middle = nn.Sequential(
            Conv2dAct(ochannels, ochannels * 2, down=2, filter_size=filter_size, act_name=act_name),
            *[FlatResBlock(ochannels * 2, act_name) for _ in range(resblocks)],
            Conv2dAct(ochannels * 2, ochannels, up=2, filter_size=filter_size, act_name=act_name, act_gain=1.0),
        )

        self.ups = nn.ModuleList()
        for dfeat_dim in reversed(dfeat_dims):
            channels //= 2
            ichannels, ochannels = ochannels + dfeat_dim, min(max_channels, channels)
            self.ups.append(
                nn.Sequential(
                    Conv2dAct(ichannels, ochannels, up=2, filter_size=filter_size, act_name=act_name),
                    Conv2dAct(ochannels, ochannels, act_name=act_name, act_gain=1.0),
                )
            )

        self.output = Conv2dAct(ochannels, io_channels, act_name='linear')
        self.output.weight.data.copy_(self.output.weight.data * 1e-2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.input(x)
        dfeats = []
        for down in self.downs:
            x = down(x)
            dfeats.append(x)
        x = self.middle(x)
        for up, dfeat in zip(self.ups, reversed(dfeats)):
            x = torch.cat([x, dfeat], dim=1)
            x = up(x)
        x = self.output(x)
        x = self.tanh(x)
        return x


#############################################   PATCHD   #########################################


class PatchDiscriminator(nn.Module):
    def __init__(self, num_layers=3, channels=64, act_name='lrelu', filter_size=4, in_channels=3) -> None:
        super().__init__()

        layers = [Conv2dAct(in_channels, channels, 4, down=2, filter_size=filter_size, act_name=act_name)]
        for _ in range(num_layers - 1):
            layers.append(Conv2dAct(channels, channels * 2, 4, down=2, filter_size=filter_size, act_name=act_name))
            channels *= 2
        layers.append(Conv2dAct(channels, channels * 2, 4, act_name=act_name))
        layers.append(Conv2dAct(channels * 2, 1, 4, act_name='linear'))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
