import yaml
import json
import cv2
import os
import torch
from Stage2.Network import Network
import numpy as np
from torchvision.utils import make_grid
import math
from PIL import Image


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.cpu().clamp_(*min_max)  # move to CPU before numpy conversion
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = ((img_np+1) * 127.5).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type).squeeze()


# ---------------------------------------------------------------------------
# Refactored Stage2 class — follows Stage-Class-Refactor-Contract (CLAUDE.md)
# ---------------------------------------------------------------------------
class Stage2:
    """Contour-to-contour diffusion stage (60-step DDPM).

    All heavy model loading happens in ``__init__``; ``run`` performs pure
    inference only.
    """

    def __init__(self, device: torch.device, config_path: str, *, mode: str = "C2C2T_v2", state: str = None):
        """Load config, network weights, and generator once at init time.

        Parameters
        ----------
        device : torch.device
            Target device (``cuda`` / ``cpu``).
        config_path : str
            Path to the YAML config file for this mode.  Ignored when *mode*
            is provided — in that case the canonical config is resolved
            automatically.  Kept for contract compatibility.
        mode : str
            One of the supported Stage2 modes.
        state : str
            Path to the checkpoint ``.pth`` file.
        """
        self.device = device
        self.mode = mode

        # --- resolve config & generator class from mode -----------------------
        if mode in ['M2M2T']:
            from Stage2.Generator import Mask2MaskGenerator as Generator
            resolved_config_path = config_path or "./Stage2/config/config_Mask2Mask.yaml"
        elif mode in ['C2C2T_v1', 'C2C2T_v2', 'C2C2T_v2_facecolor_teethcolor',
                       'C2C2T_v2_facecolor_lightcolor', 'C2C2T_v2_fourier']:
            from Stage2.Generator import Contour2ContourGenerator as Generator
            resolved_config_path = config_path or "./Stage2/config/config_Contour2Contour.yaml"
        else:
            raise ValueError(f"Unsupported Stage2 mode: {mode}")

        with open(resolved_config_path, 'r') as f:
            generator_config = yaml.load(f, Loader=yaml.SafeLoader)['GeneratorConfig']

        # --- load network weights (heavy, one-time) ---------------------------
        self.netG = Network(generator_config['unet'], generator_config['beta_schedule'])
        if state is not None:
            self.netG.load_state_dict(
                torch.load(state, map_location=self.device), strict=False
            )
        self.netG.to(self.device)
        self.netG.eval()

        # --- build generator --------------------------------------------------
        self.generator = Generator(self.netG)

    @torch.no_grad()
    def run(self, input_data, *, sample_num: int = 5, if_visual: bool = False, **params) -> dict:
        """Run pure inference — no weight loading allowed here.

        Parameters
        ----------
        input_data
            Pre-processed data dict expected by the generator.
        sample_num : int
            Number of intermediate diffusion samples to return.
        if_visual : bool
            If ``True``, write a debug image to ``./result_vis/``.

        Returns
        -------
        dict
            ``generated_contour``  — numpy BGR uint8 image.
            ``stage2_visuals``     — list of numpy BGR uint8 intermediate images.
        """
        prediction, visuals = self.generator.predict_with_visuals(
            input_data, sample_num=sample_num
        )
        generated_contour = tensor2img(prediction)  # numpy_BGR_uint8 (0-255)

        # chuyển các ảnh trung gian từ tensor sang numpy BGR uint8
        visual_imgs = [tensor2img(v) for v in visuals]

        if if_visual:
            os.makedirs('./result_vis', exist_ok=True)
            cv2.imwrite(os.path.join('./result_vis', 'teeth_mask_align.png'), generated_contour)

        return {
            "generated_contour": generated_contour,   # contract key (CLAUDE.md)
            "crop_teeth_align": generated_contour,     # backward-compat alias
            "stage2_visuals": visual_imgs,             # list of numpy_BGR_uint8
        }


# ---------------------------------------------------------------------------
# Backward-compatible wrapper — keeps old call-sites working until they are
# migrated to the class-based API.
# ---------------------------------------------------------------------------
def Stage2_Mask2Mask(data, mode, state, if_visual=False):
    """Legacy function wrapper.  Instantiates :class:`Stage2` per call.

    .. deprecated::
        Use :class:`Stage2` directly for production / Triton workloads so that
        model weights are loaded once in ``__init__``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage = Stage2(device=device, config_path=None, mode=mode, state=state)
    return stage.run(data, if_visual=if_visual)
