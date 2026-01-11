import math
import os
import random
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.amp as amp
import torch.nn.functional as F
import torchvision.transforms as T
from imageio import imread, mimsave
from PIL import Image
from torch import nn
from torch_optimizer import AdamP, DiffGrad
from tqdm import tqdm, trange

from .clip import load, tokenize

# =============================================================================
# Activation Functions for Implicit Neural Representations
# =============================================================================


class SineActivation(nn.Module):
    """Standard SIREN activation: sin(w0 * x)"""

    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class GaborActivation(nn.Module):
    """
    WIRE activation: sin(w0 * x) * exp(-s0 * x^2)
    Gabor wavelet with Gaussian envelope for local support.
    """

    def __init__(self, w0=10.0, s0=10.0):
        super().__init__()
        self.w0 = w0
        self.s0 = s0

    def forward(self, x):
        return torch.sin(self.w0 * x) * torch.exp(-self.s0 * (x**2))


class INRLayer(nn.Module):
    """Single layer for implicit neural representation with configurable activation."""

    def __init__(self, in_features, out_features, activation, is_first=False, w0=30.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

        # SIREN-style initialization
        with torch.no_grad():
            bound = 1 / in_features if is_first else math.sqrt(6 / in_features) / w0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return self.activation(self.linear(x))


class INRNet(nn.Module):
    """
    Exact functional clone of siren-pytorch.SirenNet with added Gabor support.
    Supports different activations: 'siren' (sine) or 'gabor' (WIRE).
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        activation="siren",
        w0=30.0,
        w0_initial=30.0,
        s0=10.0,
        use_bias=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.w0 = w0

        # Build activation factory
        def make_activation(is_first=False):
            freq = w0_initial if is_first else w0
            if activation == "gabor":
                return GaborActivation(w0=freq, s0=s0)
            return SineActivation(w0=freq)

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            # For the first layer, w0 is w0_initial, otherwise it's the hidden w0
            current_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(
                INRLayer(
                    in_features=layer_dim_in,
                    out_features=dim_hidden,
                    activation=make_activation(is_first),
                    is_first=is_first,
                    w0=current_w0,
                )
            )

        # Final linear layer (exact match to siren-pytorch)
        self.last_layer = nn.Linear(dim_hidden, dim_out, bias=use_bias)
        with torch.no_grad():
            bound = math.sqrt(6 / dim_hidden) / w0
            self.last_layer.weight.uniform_(-bound, bound)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.last_layer(x)


class INRWrapper(nn.Module):
    """Wraps INRNet to generate images from a coordinate grid."""

    def __init__(self, net, image_width, image_height):
        super().__init__()
        self.net = net
        self.image_width = image_width
        self.image_height = image_height

        # Pre-compute normalized coordinate grid [-1, 1]
        y = torch.linspace(-1, 1, steps=image_height)
        x = torch.linspace(-1, 1, steps=image_width)
        grid = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)
        self.register_buffer("grid", grid.reshape(-1, 2))

    def forward(self, img=None):
        out = self.net(self.grid)
        out = out.reshape(1, self.image_height, self.image_width, -1).permute(0, 3, 1, 2)
        out = norm_siren_output(out)
        if img is not None:
            return F.mse_loss(out, img)
        return out


# =============================================================================
# Helpers
# =============================================================================


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def interpolate(image, size):
    return F.interpolate(image, (size, size), mode="bilinear", align_corners=False)


def rand_cutout(image, size, center_bias=False, center_focus=2, offset_x=None, offset_y=None):
    width = image.shape[-1]
    min_offset = 0
    max_offset = width - size

    if offset_x is None or offset_y is None:
        if center_bias:
            # sample around image center
            center = max_offset / 2
            std = center / center_focus
            offset_x = int(random.gauss(mu=center, sigma=std))
            offset_y = int(random.gauss(mu=center, sigma=std))
            # resample uniformly if over boundaries
            offset_x = (
                random.randint(min_offset, max_offset) if (offset_x > max_offset or offset_x < min_offset) else offset_x
            )
            offset_y = (
                random.randint(min_offset, max_offset) if (offset_y > max_offset or offset_y < min_offset) else offset_y
            )
        else:
            offset_x = random.randint(min_offset, max_offset)
            offset_y = random.randint(min_offset, max_offset)

    cutout = image[:, :, offset_x : offset_x + size, offset_y : offset_y + size]
    return cutout, (offset_x, offset_y)


def augment_piece(piece, input_resolution):
    """Applies random augmentations to a cutout piece."""
    # We use .item() to convert to python scalars because torchvision.transforms.functional.affine
    # requires them. We also accept that this might cause a graph break in torch.compile,
    # which is preferable to a crash or a recompile of the entire graph.
    orig_dtype = piece.dtype
    piece = piece.float()

    device = piece.device
    angle = (torch.rand([], device=device) * 20 - 10).item()
    trans_x = (torch.rand([], device=device) * 0.2 - 0.1).item() * input_resolution
    trans_y = (torch.rand([], device=device) * 0.2 - 0.1).item() * input_resolution

    # Resize to CLIP input resolution
    piece = interpolate(piece, input_resolution)

    # Apply affine transform
    piece = T.functional.affine(
        piece,
        angle=angle,
        translate=[trans_x, trans_y],
        scale=1.0,
        shear=[0.0, 0.0],
        interpolation=T.InterpolationMode.BILINEAR,
    )

    return piece.to(orig_dtype)


def create_clip_img_transform(image_width):
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    transform = T.Compose(
        [
            # T.ToPILImage(),
            T.Resize(image_width),
            T.CenterCrop((image_width, image_width)),
            T.ToTensor(),
            T.Normalize(mean=clip_mean, std=clip_std),
        ]
    )
    return transform


def open_folder(path):
    if os.path.isfile(path):
        path = os.path.dirname(path)

    if not os.path.isdir(path):
        return

    cmd_list = None
    if sys.platform == "darwin":
        cmd_list = ["open", "--", path]
    elif sys.platform == "linux2" or sys.platform == "linux":
        cmd_list = ["xdg-open", path]
    elif sys.platform in ["win32", "win64"]:
        cmd_list = ["explorer", path.replace("/", "\\")]
    if cmd_list is None:
        return

    try:
        subprocess.check_call(cmd_list)
    except subprocess.CalledProcessError:
        pass
    except OSError:
        pass


def norm_siren_output(img):
    return ((img + 1) * 0.5).clamp(0.0, 1.0)


def total_variation_loss(img):
    """Total Variation loss for smoothing images."""
    bs, c, h, w = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (bs * c * h * w)


def create_text_path(context_length, text=None, img=None, encoding=None, separator=None):
    if text is not None:
        if separator is not None and separator in text:
            # Reduces filename to first epoch text
            text = text[
                : text.index(
                    separator,
                )
            ]
        input_name = text.replace(" ", "_")[:context_length]
    elif img is not None:
        if isinstance(img, str):
            input_name = "".join(img.replace(" ", "_").split(".")[:-1])
        else:
            input_name = "PIL_img"
    else:
        input_name = "your_encoding"
    return input_name


class DeepDaze(nn.Module):
    def __init__(
        self,
        clip_perceptor,
        clip_norm,
        input_res,
        total_batches,
        batch_size,
        num_layers=8,
        image_width=512,
        loss_coef=100,
        theta_initial=None,
        theta_hidden=None,
        lower_bound_cutout=0.1,  # should be smaller than 0.8
        upper_bound_cutout=1.0,
        saturate_bound=False,
        gauss_sampling=False,
        gauss_mean=0.6,
        gauss_std=0.2,
        do_cutout=True,
        center_bias=False,
        center_focus=2,
        hidden_size=256,
        averaging_weight=0.3,
        use_gabor=False,
        gabor_scale=10.0,
        aug_both=False,
        tv_coef=100.0,
        do_aug=False,
    ):
        super().__init__()
        # load clip
        self.perceptor = clip_perceptor
        self.input_resolution = input_res
        self.normalize_image = clip_norm

        self.loss_coef = loss_coef
        self.tv_coef = tv_coef
        self.image_width = image_width

        self.batch_size = batch_size
        self.total_batches = total_batches
        self.register_buffer("num_batches_processed", torch.tensor(0, dtype=torch.long))

        self.do_aug = do_aug

        w0 = default(theta_hidden, 30.0)
        w0_initial = default(theta_initial, 30.0)

        activation = "gabor" if use_gabor else "siren"
        siren = INRNet(
            dim_in=2,
            dim_hidden=hidden_size,
            dim_out=3,
            num_layers=num_layers,
            activation=activation,
            w0=w0,
            w0_initial=w0_initial,
            s0=gabor_scale,
        )
        self.model = INRWrapper(siren, image_width=image_width, image_height=image_width)

        self.saturate_bound = saturate_bound
        self.saturate_limit = 0.75  # cutouts above this value lead to destabilization
        self.lower_bound_cutout = lower_bound_cutout
        self.upper_bound_cutout = upper_bound_cutout
        self.gauss_sampling = gauss_sampling
        self.gauss_mean = gauss_mean
        self.gauss_std = gauss_std
        self.do_cutout = do_cutout
        self.center_bias = center_bias
        self.center_focus = center_focus
        self.averaging_weight = averaging_weight
        self.aug_both = aug_both

    def reset_weights(self):
        """Reset all weights in the SIREN/INR model."""

        def _reset(m):
            if hasattr(m, "_init_weights"):
                m._init_weights()
            elif isinstance(m, nn.Linear):
                # Fallback for standard layers
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.model.net.apply(_reset)

    def generate_random_params(self, batch_size, device, lower_bound=0.1):
        """Generates random sampling parameters entirely on the GPU."""
        # Sample scales
        if self.gauss_sampling:
            scales = torch.zeros(batch_size, device=device).normal_(mean=self.gauss_mean, std=self.gauss_std)
            scales = scales.clamp(lower_bound, self.upper_bound_cutout)
        else:
            scales = torch.empty(batch_size, device=device).uniform_(lower_bound, self.upper_bound_cutout)

        # Sample translations (must stay within bounds so we don't sample outside the image)
        # Max translation is (1 - scale) in normalized [-1, 1] coordinates
        max_trans = (1.0 - scales).clamp(min=0.0)

        if self.center_bias:
            # Sample from a normal distribution centered at 0
            # Standard deviation is scaled by center_focus
            std = 1.0 / self.center_focus
            tx = torch.zeros(batch_size, device=device).normal_(mean=0, std=std)
            ty = torch.zeros(batch_size, device=device).normal_(mean=0, std=std)
            # Clamp and scale to max_trans
            tx = tx.clamp(-1, 1) * max_trans
            ty = ty.clamp(-1, 1) * max_trans
        else:
            tx = torch.empty(batch_size, device=device).uniform_(-1, 1) * max_trans
            ty = torch.empty(batch_size, device=device).uniform_(-1, 1) * max_trans

        # Sample rotations (if augmentation is enabled)
        if self.do_aug:
            angles = torch.empty(batch_size, device=device).uniform_(-math.pi / 18, math.pi / 18)  # ~ +-10 degrees
        else:
            angles = torch.zeros(batch_size, device=device)

        return scales, tx, ty, angles

    def build_affine_matrices(self, scales, tx, ty, angles):
        """Builds a batch of affine matrices on the GPU."""
        batch_size = scales.shape[0]

        # F.affine_grid expects 2x3 matrices:
        # [ s*cos(a)  -s*sin(a)  tx ]
        # [ s*sin(a)   s*cos(a)  ty ]
        # Note: We use 'scales' as a multiplier for the grid coordinates.
        # To zoom IN (make cutout), the grid coordinates should be scaled DOWN.
        # So we use the scale directly in the matrix.

        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        # Construct rows
        # Row 1: [scale * cos_a, -scale * sin_a, tx]
        # Row 2: [scale * sin_a,  scale * cos_a, ty]
        m00 = scales * cos_a
        m01 = -scales * sin_a
        m02 = tx
        m10 = scales * sin_a
        m11 = scales * cos_a
        m12 = ty

        matrices = torch.stack([m00, m01, m02, m10, m11, m12], dim=1).reshape(batch_size, 2, 3)
        return matrices

    def forward(self, text_embed, target_image=None, return_loss=True, dry_run=False, input_moments=None):
        # Profiling dictionary
        self.last_timings = {}

        # Check if we should profile (skip if compiling to avoid recompiles)
        is_compiling = False
        try:
            import torch.compiler

            is_compiling = torch.compiler.is_compiling()
        except (ImportError, AttributeError):
            pass

        def sync():
            if is_compiling:
                return
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif torch.backends.mps.is_available():
                torch.mps.synchronize()

        if not is_compiling:
            start_siren = time.time()
        out = self.model()
        if not is_compiling:
            sync()
            self.last_timings["siren"] = time.time() - start_siren

        if not return_loss:
            return out

        # determine upper and lower sampling bound
        if not is_compiling:
            start_cutouts = time.time()

        lower_bound = self.lower_bound_cutout
        if self.saturate_bound:
            # self.num_batches_processed is now a tensor, so we need to use it as such
            # We use .float() to ensure floating point division
            progress_fraction = self.num_batches_processed.float() / self.total_batches
            lower_bound = lower_bound + (self.saturate_limit - self.lower_bound_cutout) * progress_fraction

        device = out.device
        batch_size = self.batch_size

        # 1. Generate random parameters on GPU
        scales, tx, ty, angles = self.generate_random_params(batch_size, device, lower_bound=lower_bound)

        # 2. Build affine matrices
        matrices = self.build_affine_matrices(scales, tx, ty, angles)

        # 3. Create sampling grid
        # We want to output [batch, 3, input_resolution, input_resolution] for CLIP
        grid = F.affine_grid(
            matrices, [batch_size, 3, self.input_resolution, self.input_resolution], align_corners=False
        )

        # 4. Sample image pieces using grid_sample (Vectorized!)
        # We expand the single 'out' image to batch_size
        image_pieces = F.grid_sample(out.expand(batch_size, -1, -1, -1), grid, align_corners=False)

        # Apply normalization
        image_pieces = self.normalize_image(image_pieces)

        # Inject Global View (Full Image) to prevent ghosting
        # We downsample the full 'out' image to CLIP resolution
        full_view = interpolate(out, self.input_resolution)
        full_view = self.normalize_image(full_view)
        # Concatenate global view to the batch (batch_size + 1)
        image_pieces = torch.cat([image_pieces, full_view], dim=0)

        if self.aug_both and target_image is not None:
            # Apply same sampling to target image
            target_pieces = F.grid_sample(target_image.expand(batch_size, -1, -1, -1), grid, align_corners=False)
            target_pieces = self.normalize_image(target_pieces)

            # Add target global view
            target_full = interpolate(target_image, self.input_resolution)
            target_full = self.normalize_image(target_full)
            target_pieces = torch.cat([target_pieces, target_full], dim=0)

        if not is_compiling:
            sync()
            self.last_timings["cutouts"] = time.time() - start_cutouts

        # calc image embedding
        if not is_compiling:
            start_clip = time.time()
        device_type = "cuda" if "cuda" in str(device) else "mps" if "mps" in str(device) else "cpu"

        # Use bfloat16 for MPS and CUDA (if supported) for better stability/speed
        autocast_kwargs = {"device_type": device_type, "enabled": True}
        if device_type == "mps":
            autocast_kwargs["dtype"] = torch.bfloat16
        elif device_type == "cuda" and torch.cuda.is_bf16_supported():
            autocast_kwargs["dtype"] = torch.bfloat16

        with amp.autocast(**autocast_kwargs):
            image_embed = self.perceptor.encode_image(image_pieces)
            if self.aug_both and target_image is not None:
                target_embed = self.perceptor.encode_image(target_pieces)
            else:
                target_embed = text_embed

        if not is_compiling:
            sync()
            self.last_timings["clip"] = time.time() - start_clip

        # calc loss
        # loss over averaged features of cutouts
        avg_image_embed = image_embed.mean(dim=0).unsqueeze(0)

        # If we have target_embed for each cutout, we average it too for the averaged_loss
        if self.aug_both and target_image is not None:
            avg_target_embed = target_embed.mean(dim=0).unsqueeze(0)
            averaged_loss = -self.loss_coef * torch.cosine_similarity(avg_target_embed, avg_image_embed, dim=-1).mean()
            general_loss = -self.loss_coef * torch.cosine_similarity(target_embed, image_embed, dim=-1).mean()
        else:
            averaged_loss = -self.loss_coef * torch.cosine_similarity(text_embed, avg_image_embed, dim=-1).mean()
            general_loss = -self.loss_coef * torch.cosine_similarity(text_embed, image_embed, dim=-1).mean()

        # merge losses
        loss = averaged_loss * (self.averaging_weight) + general_loss * (1 - self.averaging_weight)

        # add TV loss
        if self.tv_coef > 0:
            loss = loss + self.tv_coef * total_variation_loss(out)

        # Color Moment Loss (Fixes Sepia/Grayness)
        if input_moments is not None:
            # input_moments is (mean, std) of the webcam input
            in_mean, in_std = input_moments
            gen_mean = out.mean(dim=(2, 3))
            gen_std = out.std(dim=(2, 3))

            # Loss = L1 distance between moments
            moment_loss = F.l1_loss(gen_mean, in_mean) + F.l1_loss(gen_std, in_std)
            loss = loss + 10.0 * moment_loss
        else:
            # Fallback: simple saturation boost
            rgb_std = out.std(dim=1).mean()
            loss = loss - 10.0 * rgb_std

        # count batches
        if not dry_run:
            self.num_batches_processed += self.batch_size

        return out, loss


class Imagine(nn.Module):
    def __init__(
        self,
        *,
        text=None,
        img=None,
        clip_encoding=None,
        lr=1e-5,
        batch_size=4,
        gradient_accumulate_every=4,
        save_every=100,
        image_width=512,
        num_layers=16,
        epochs=20,
        iterations=1050,
        save_progress=True,
        seed=None,
        open_folder=True,
        save_date_time=False,
        start_image_path=None,
        start_image_train_iters=10,
        start_image_lr=3e-4,
        theta_initial=None,
        theta_hidden=None,
        model_name="ViT-B/32",
        lower_bound_cutout=0.1,  # should be smaller than 0.8
        upper_bound_cutout=1.0,
        saturate_bound=False,
        averaging_weight=0.3,
        create_story=False,
        story_start_words=5,
        story_words_per_epoch=5,
        story_separator=None,
        gauss_sampling=False,
        gauss_mean=0.6,
        gauss_std=0.2,
        do_cutout=True,
        center_bias=False,
        center_focus=2,
        optimizer="AdamP",
        jit=True,
        hidden_size=256,
        save_gif=False,
        save_video=False,
        use_gabor=False,
        gabor_scale=10.0,
        aug_both=False,
        tv_coef=100.0,
        do_aug=False,
    ):

        super().__init__()

        if exists(seed):
            tqdm.write(f"setting seed: {seed}")
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True

        # fields for story creation:
        self.create_story = create_story
        self.words = None
        self.separator = str(story_separator) if story_separator is not None else None
        if self.separator is not None and text is not None:
            # exit if text is just the separator
            if str(text).replace(" ", "").replace(self.separator, "") == "":
                print(
                    "Exiting because the text only consists of the separator! Needs words or phrases that are separated by the separator."
                )
                exit()
            # adds a space to each separator and removes double spaces that might be generated
            text = text.replace(self.separator, self.separator + " ").replace("  ", " ").strip()
        self.all_words = text.split(" ") if text is not None else None
        self.num_start_words = story_start_words
        self.words_per_epoch = story_words_per_epoch
        if create_story:
            assert text is not None, "We need text input to create a story..."
            # overwrite epochs to match story length
            num_words = len(self.all_words)
            self.epochs = 1 + (num_words - self.num_start_words) / self.words_per_epoch
            # add one epoch if not divisible
            self.epochs = int(self.epochs) if int(self.epochs) == self.epochs else int(self.epochs) + 1
            if self.separator is not None:
                if self.separator not in text:
                    print("Separator '" + self.separator + "' will be ignored since not in text!")
                    self.separator = None
                else:
                    self.epochs = len(list(filter(None, text.split(self.separator))))
            print(
                "Running for",
                self.epochs,
                "epochs"
                + (" (split with '" + self.separator + "' as the separator)" if self.separator is not None else ""),
            )
        else:
            self.epochs = epochs

        # jit models only compatible with version 1.7.1
        jit = False

        # Load CLIP
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            # Enable TensorFloat32 for better performance on NVIDIA GPUs (Ampere+)
            torch.set_float32_matmul_precision("high")
        else:
            self.device = torch.device("cpu")

        clip_perceptor, norm = load(model_name, jit=False, device=self.device)
        self.perceptor = clip_perceptor.eval()
        for param in self.perceptor.parameters():
            param.requires_grad = False
        if not jit:
            input_res = clip_perceptor.visual.input_resolution
        else:
            input_res = clip_perceptor.input_resolution.item()
        self.clip_transform = create_clip_img_transform(input_res)
        self.target_transform = T.Compose(
            [
                T.Resize(image_width),
                T.CenterCrop((image_width, image_width)),
                T.ToTensor(),
            ]
        )

        self.iterations = iterations
        self.image_width = image_width
        self.aug_both = aug_both
        total_batches = self.epochs * self.iterations * batch_size * gradient_accumulate_every
        model = DeepDaze(
            self.perceptor,
            norm,
            input_res,
            total_batches,
            batch_size=batch_size,
            image_width=image_width,
            num_layers=num_layers,
            theta_initial=theta_initial,
            theta_hidden=theta_hidden,
            lower_bound_cutout=lower_bound_cutout,
            upper_bound_cutout=upper_bound_cutout,
            saturate_bound=saturate_bound,
            gauss_sampling=gauss_sampling,
            gauss_mean=gauss_mean,
            gauss_std=gauss_std,
            do_cutout=do_cutout,
            center_bias=center_bias,
            center_focus=center_focus,
            hidden_size=hidden_size,
            averaging_weight=averaging_weight,
            use_gabor=use_gabor,
            gabor_scale=gabor_scale,
            aug_both=aug_both,
            tv_coef=tv_coef,
            do_aug=do_aug,
        ).to(self.device)
        self.model = model

        # Use GradScaler only for CUDA float16. BFloat16 and MPS do not need it.
        self.use_scaler = "cuda" in str(self.device) and not torch.cuda.is_bf16_supported()
        self.scaler = amp.GradScaler(enabled=self.use_scaler)
        siren_params = model.model.parameters()
        if optimizer == "AdamP":
            self.optimizer = AdamP(siren_params, lr)
        elif optimizer == "Adam":
            self.optimizer = torch.optim.Adam(siren_params, lr)
        elif optimizer == "DiffGrad":
            self.optimizer = DiffGrad(siren_params, lr)
        self.gradient_accumulate_every = gradient_accumulate_every

        self.save_every = save_every
        self.save_date_time = save_date_time
        self.open_folder = open_folder
        self.save_progress = save_progress
        self.text = text
        self.image = img
        self.textpath = create_text_path(
            self.perceptor.context_length, text=text, img=img, encoding=clip_encoding, separator=story_separator
        )
        self.filename = self.image_output_path()

        # create coding to optimize for
        self.clip_encoding = self.create_clip_encoding(text=text, img=img, encoding=clip_encoding)

        self.start_image = None
        self.start_image_train_iters = start_image_train_iters
        self.start_image_lr = start_image_lr
        if exists(start_image_path):
            file = Path(start_image_path)
            assert file.exists(), f"file does not exist at given starting image path {start_image_path}"
            image = Image.open(str(file))
            start_img_transform = T.Compose(
                [T.Resize(image_width), T.CenterCrop((image_width, image_width)), T.ToTensor()]
            )
            image_tensor = start_img_transform(image).unsqueeze(0).to(self.device)
            self.start_image = image_tensor

        self.save_gif = save_gif
        self.save_video = save_video
        self.measured_memory = False

    def create_clip_encoding(self, text=None, img=None, encoding=None):
        self.text = text
        self.img = img
        if encoding is not None:
            encoding = encoding.to(self.device)
        elif self.create_story:
            encoding = self.update_story_encoding(epoch=0, iteration=1)
        elif text is not None and img is not None:
            encoding = (self.create_text_encoding(text) + self.create_img_encoding(img)) / 2
        elif text is not None:
            encoding = self.create_text_encoding(text)
        elif img is not None:
            encoding = self.create_img_encoding(img)
        return encoding

    def create_text_encoding(self, text):
        tokenized_text = tokenize(text).to(self.device)
        with torch.no_grad():
            text_encoding = self.perceptor.encode_text(tokenized_text).detach()
        return text_encoding

    def create_img_encoding(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        normed_img = self.clip_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_encoding = self.perceptor.encode_image(normed_img).detach()
        return img_encoding

    def set_clip_encoding(self, text=None, img=None, encoding=None):
        encoding = self.create_clip_encoding(text=text, img=img, encoding=encoding)
        self.clip_encoding = encoding.to(self.device)

        if self.aug_both and img is not None:
            if isinstance(img, str):
                img_pil = Image.open(img)
            else:
                img_pil = img
            self.target_image_tensor = self.target_transform(img_pil).unsqueeze(0).to(self.device)
        else:
            self.target_image_tensor = None

    def index_of_first_separator(self) -> int:
        for c, word in enumerate(self.all_words):
            if self.separator in str(word):
                return c + 1

    def update_story_encoding(self, epoch, iteration):
        if self.separator is not None:
            self.words = " ".join(self.all_words[: self.index_of_first_separator()])
            # removes separator from epoch-text
            self.words = self.words.replace(self.separator, "")
            self.all_words = self.all_words[self.index_of_first_separator() :]
        else:
            if self.words is None:
                self.words = " ".join(self.all_words[: self.num_start_words])
                self.all_words = self.all_words[self.num_start_words :]
            else:
                # add words_per_epoch new words
                count = 0
                while count < self.words_per_epoch and len(self.all_words) > 0:
                    new_word = self.all_words[0]
                    self.words = " ".join(self.words.split(" ") + [new_word])
                    self.all_words = self.all_words[1:]
                    count += 1
                # remove words until it fits in context length
                while len(self.words) > self.perceptor.context_length:
                    # remove first word
                    self.words = " ".join(self.words.split(" ")[1:])
        # get new encoding
        print("Now thinking of: ", '"', self.words, '"')
        sequence_number = self.get_img_sequence_number(epoch, iteration)
        # save new words to disc
        with open("story_transitions.txt", "a") as f:
            f.write(f"{epoch}, {sequence_number}, {self.words}\n")

        encoding = self.create_text_encoding(self.words)
        return encoding

    def image_output_path(self, sequence_number=None):
        """
        Returns underscore separated Path.
        A current timestamp is prepended if `self.save_date_time` is set.
        Sequence number left padded with 6 zeroes is appended if `save_every` is set.
        :rtype: Path
        """
        output_path = self.textpath
        if sequence_number:
            sequence_number_left_padded = str(sequence_number).zfill(6)
            output_path = f"{output_path}.{sequence_number_left_padded}"
        if self.save_date_time:
            current_time = datetime.now().strftime("%y%m%d-%H%M%S_%f")
            output_path = f"{current_time}_{output_path}"
        return Path(f"{output_path}.jpg")

    def train_step(self, epoch, iteration, input_moments=None):
        total_loss = 0
        all_timings = []

        def sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif torch.backends.mps.is_available():
                torch.mps.synchronize()

        device_type = "cuda" if "cuda" in str(self.device) else "mps" if "mps" in str(self.device) else "cpu"
        autocast_kwargs = {"device_type": device_type, "enabled": True}
        if device_type == "mps":
            autocast_kwargs["dtype"] = torch.bfloat16
        elif device_type == "cuda" and torch.cuda.is_bf16_supported():
            autocast_kwargs["dtype"] = torch.bfloat16

        for _ in range(self.gradient_accumulate_every):
            with amp.autocast(**autocast_kwargs):
                out, loss = self.model(
                    self.clip_encoding, target_image=self.target_image_tensor, input_moments=input_moments
                )

            t_fwd = self.model.last_timings.copy()

            loss = loss / self.gradient_accumulate_every
            total_loss += loss

            start_back = time.time()
            if self.use_scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            sync()
            t_fwd["backward"] = time.time() - start_back
            all_timings.append(t_fwd)

        out = out.cpu().float().clamp(0.0, 1.0)

        if self.use_scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

        # Safety Check: If we hit NaNs, the LR is too high or weights exploded
        if torch.isnan(loss):
            print("\n!!! WARNING: Loss is NaN. Optimization exploded. !!!")
            print("Action: Resetting model weights and suggesting lower learning rate.")
            self.reset_weights()

        if not self.measured_memory:
            self.measured_memory = True
            sync()
            ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On macOS, ru_maxrss is in bytes, on Linux it is in KiB
            ram_mib = ram / (1024 * 1024) if sys.platform == "darwin" else ram / 1024

            vram_mib = 0
            if torch.cuda.is_available():
                vram_mib = torch.cuda.memory_allocated() / (1024 * 1024)
            elif torch.backends.mps.is_available():
                try:
                    vram_mib = torch.mps.current_allocated_memory() / (1024 * 1024)
                except Exception:
                    vram_mib = -1

            vram_str = f"{vram_mib:.2f} MiB" if vram_mib >= 0 else "Unsupported"
            print(f"\n[Memory Report After First Step] RAM: {ram_mib:.2f} MiB | VRAM: {vram_str}")

        if (iteration % self.save_every == 0) and self.save_progress:
            self.save_image(epoch, iteration, img=out)

        # Average timings
        avg_timings = {}
        if all_timings:
            for key in all_timings[0].keys():
                avg_timings[key] = sum(t[key] for t in all_timings) / len(all_timings)

        return out, total_loss, avg_timings

    def get_img_sequence_number(self, epoch, iteration):
        current_total_iterations = epoch * self.iterations + iteration
        sequence_number = current_total_iterations // self.save_every
        return sequence_number

    @torch.no_grad()
    def save_image(self, epoch, iteration, img=None):
        sequence_number = self.get_img_sequence_number(epoch, iteration)

        if img is None:
            img = self.model(self.clip_encoding, return_loss=False).cpu().float().clamp(0.0, 1.0)
        self.filename = self.image_output_path(sequence_number=sequence_number)

        pil_img = T.ToPILImage()(img.squeeze())
        pil_img.save(self.filename, quality=95, subsampling=0)
        pil_img.save(f"{self.textpath}.jpg", quality=95, subsampling=0)

        tqdm.write(f'image updated at "./{str(self.filename)}"')

    def generate_gif(self):
        images = []
        for file_name in sorted(os.listdir("./")):
            if file_name.startswith(self.textpath) and file_name != f"{self.textpath}.jpg":
                images.append(imread(os.path.join("./", file_name)))

        if self.save_video:
            mimsave(f"{self.textpath}.mp4", images)
            print(f"Generated image generation animation at ./{self.textpath}.mp4")
        if self.save_gif:
            mimsave(f"{self.textpath}.gif", images)
            print(f"Generated image generation animation at ./{self.textpath}.gif")

    def forward(self):
        if exists(self.start_image):
            tqdm.write("Preparing with initial image...")
            optim = DiffGrad(self.model.model.parameters(), lr=self.start_image_lr)
            pbar = trange(self.start_image_train_iters, desc="iteration")
            try:
                for _ in pbar:
                    loss = self.model.model(self.start_image)
                    loss.backward()
                    pbar.set_description(f"loss: {loss.item():.2f}")

                    optim.step()
                    optim.zero_grad()
            except KeyboardInterrupt:
                print("interrupted by keyboard, gracefully exiting")
                return exit()

            del self.start_image
            del optim

        tqdm.write(f'Imagining "{self.textpath}" from the depths of my weights...')

        # do one warmup step due to potential issue with CLIP and CUDA
        # We don't use no_grad here to match the requires_grad state of training
        self.model(self.clip_encoding, dry_run=True)
        self.model.zero_grad()

        if self.open_folder:
            open_folder("./")
            self.open_folder = False

        try:
            for epoch in trange(self.epochs, desc="epochs"):
                pbar = trange(self.iterations, desc="iteration")
                for i in pbar:
                    _, loss, _ = self.train_step(epoch, i)
                    pbar.set_description(f"loss: {loss.item():.2f}")

                # Update clip_encoding per epoch if we are creating a story
                if self.create_story:
                    self.clip_encoding = self.update_story_encoding(epoch, i)
        except KeyboardInterrupt:
            print("interrupted by keyboard, gracefully exiting")
            return

        self.save_image(epoch, i)  # one final save at end

        if (self.save_gif or self.save_video) and self.save_progress:
            self.generate_gif()
