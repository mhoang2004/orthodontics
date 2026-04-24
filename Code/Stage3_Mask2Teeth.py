import yaml
import json
import cv2
import os
import torch
from Stage3.Network import Network
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
    tensor = tensor.clamp_(*min_max)  # clamp
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
# Mode → (Generator class import path, config YAML path) mapping
# ---------------------------------------------------------------------------
_MODE_REGISTRY = {
    'M2M2T': (
        'Stage3.Generator', 'Mask2TeethGenerator',
        './Stage3/config/config_Mask2Teeth.yaml',
    ),
    'C2C2T_v1': (
        'Stage3.Generator', 'Contour2TeethGenerator',
        './Stage3/config/config_Contour2Teeth.yaml',
    ),
    'C2C2T_v2': (
        'Stage3.Generator', 'Contour2TeethGenerator',
        './Stage3/config/config_Contour2Teeth.yaml',
    ),
    'C2C2T_v2_facecolor_teethcolor': (
        'Stage3.Generator', 'Contour2ToothGenerator_FaceColor_TeethColor',
        './Stage3/config/config_Contour2Tooth_facecolor_teethcolor.yaml',
    ),
    'C2C2T_v2_facecolor_lightcolor': (
        'Stage3.Generator', 'Contour2ToothGenerator_FaceColor_LightColor',
        './Stage3/config/config_Contour2Tooth_facecolor_lightcolor.yaml',
    ),
    'C2C2T_v2_fourier': (
        'Stage3.Generator', 'Contour2ToothGenerator_Fourier',
        './Stage3/config/config_Contour2Tooth_Fourier.yaml',
    ),
}


def _resolve_generator_class(module_path: str, class_name: str):
    """Dynamically import and return a Generator class."""
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class Stage3:
    """Stage 3 — Mask / Contour → Teeth image diffusion.

    Follows the Stage-Class-Refactor-Contract:
      • __init__  : all heavy model loading (weights, generator, config)
      • run()     : pure inference, returns dict with named outputs
      • self.device is respected everywhere
    """

    def __init__(self, device: torch.device, config_path: str, **kwargs):
        """
        Args:
            device:      torch.device for model placement.
            config_path: Path to the checkpoint (.pth) file.
            **kwargs:
                mode (str): One of the keys in _MODE_REGISTRY.
                            Defaults to ``'C2C2T_v2'``.
                custom_timesteps (int | None): Override n_timestep in
                            beta_schedule config. Applied once at init.
        """
        self.device = device
        self.config_path = config_path

        mode = kwargs.get('mode', 'C2C2T_v2')
        custom_timesteps = kwargs.get('custom_timesteps', None)

        if mode not in _MODE_REGISTRY:
            raise ValueError(
                f"Unknown mode '{mode}'. "
                f"Supported modes: {list(_MODE_REGISTRY.keys())}"
            )

        module_path, class_name, yaml_path = _MODE_REGISTRY[mode]

        # ---- Load config ------------------------------------------------
        with open(yaml_path, 'r') as f:
            self._generator_config = yaml.load(f, Loader=yaml.SafeLoader)['GeneratorConfig']

        # Override timesteps if requested (interactive mode)
        if custom_timesteps is not None:
            self._generator_config['beta_schedule']['n_timestep'] = int(custom_timesteps)

        # ---- Load network weights (heavy) --------------------------------
        self._netG = Network(
            self._generator_config['unet'],
            self._generator_config['beta_schedule'],
        )
        self._netG.load_state_dict(
            torch.load(config_path, map_location=self.device),
            strict=False,
        )
        self._netG.to(self.device)
        self._netG.eval()

        # ---- Instantiate generator --------------------------------------
        GeneratorClass = _resolve_generator_class(module_path, class_name)
        self._generator = GeneratorClass(self._netG)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run(self, input_data, **params) -> dict:
        """Run Stage 3 inference.

        Args:
            input_data: Conditioning data expected by the Generator's
                        ``predict`` / ``predict_interactive`` method
                        (typically a dict of tensors produced by Stage 2).
            **params:
                whiteness (float | None): 0.0–1.0 tooth whiteness control.
                alignment_strength (float | None): Alignment adjustment.
                if_visual (bool): Save debug visualisation to disk.

        Returns:
            dict with key ``"teeth_image"`` (np.ndarray, BGR uint8).
                  Also includes legacy keys for backward compatibility:
                  ``"crop_mouth_align"`` and ``"cond_teeth_color"``.
        """
        whiteness = params.get('whiteness', None)
        alignment_strength = params.get('alignment_strength', None)
        if_visual = params.get('if_visual', False)

        # Build interactive kwargs
        interactive_kwargs = {}
        if whiteness is not None:
            interactive_kwargs['whiteness'] = whiteness
        if alignment_strength is not None:
            interactive_kwargs['alignment_strength'] = alignment_strength

        # Run prediction
        if interactive_kwargs and hasattr(self._generator, 'predict_interactive'):
            prediction, cond_teeth_color = self._generator.predict_interactive(
                input_data, **interactive_kwargs,
            )
        else:
            prediction, cond_teeth_color = self._generator.predict(input_data)

        # Tensor → numpy (must move to CPU before numpy conversion)
        mouth_align = tensor2img(prediction.cpu())

        if if_visual:
            cv2.imwrite(os.path.join('./result_vis', 'mouth_align.png'), mouth_align)

        return {
            # Contract key (used by model.py / Triton)
            "teeth_image": mouth_align,
            # Legacy keys (backward compatibility)
            "crop_mouth_align": mouth_align,
            "cond_teeth_color": cond_teeth_color,
        }


# ======================================================================
# Backward-compatible free-function API
# ======================================================================

def Stage3_Mask2Teeth(data, mode, state, if_visual=False,
                      whiteness=None, alignment_strength=None, custom_timesteps=None):
    """Legacy wrapper — creates a one-shot Stage3 instance and runs it.

    Kept for backward compatibility with ``Code/main.py`` and any other
    callers that rely on the original function signature.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stage = Stage3(
        device=device,
        config_path=state,
        mode=mode,
        custom_timesteps=custom_timesteps,
    )
    return stage.run(
        data,
        whiteness=whiteness,
        alignment_strength=alignment_strength,
        if_visual=if_visual,
    )
