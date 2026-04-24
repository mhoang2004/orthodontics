import torch
import numpy as np

from Stage1.DetectFace.DetectFace import DetectFace, load_face_models
from Stage1.DetectMouth.DetectMouth import DetectMouth, CropMouth
from Stage1.DetectMouth.test import load_bisenet
from Stage1.SegmentTeeth.DetectContour import MaskingMouth
from Stage1.SegmentToothContour.SegmentToothContour import SegmentToothContour, load_unet


# ---------------------------------------------------------------------------
# Refactored Stage1 class — follows Stage-Class-Refactor-Contract (CLAUDE.md)
# ---------------------------------------------------------------------------
class Stage1Class:
    """Face detection → Mouth parsing → Tooth contour segmentation.

    All heavy model loading happens in ``__init__``; ``run`` performs pure
    inference only.

    Model placement rules (per SPECIAL RULES):
    - dlib detector/predictor → **CPU only** (dlib has no CUDA support)
    - BiSeNet (mouth parsing) → ``self.device``
    - UNet (tooth contour)    → ``self.device``
    """

    def __init__(self, device: torch.device, config_path: str = None, *,
                 mode: str = 'C2C2T_v2',
                 state: str = None,
                 bisenet_cp: str = 'Stage1/DetectMouth/cp/79999_iter.pth',
                 dlib_weight_path: str = './Stage1/DetectFace/ckpts/shape_predictor_68_face_landmarks.dat'):
        """Load ALL model weights once.

        Parameters
        ----------
        device : torch.device
            Target device for GPU-capable models (BiSeNet, UNet).
        config_path : str, optional
            Unused — kept for contract compatibility with ``StageN.__init__``
            signature.
        mode : str
            Pipeline mode.
        state : str, optional
            Path to UNet checkpoint ``.pth`` file.
        bisenet_cp : str
            Path to BiSeNet checkpoint.
        dlib_weight_path : str
            Path to dlib shape predictor ``.dat`` file.
        """
        self.device = device
        self.mode = mode

        # --- dlib face detector (CPU-only — dlib has no CUDA support) ---------
        self.face_detector, self.shape_predictor = load_face_models(dlib_weight_path)

        # --- BiSeNet mouth parser (on self.device) ----------------------------
        self.bisenet = load_bisenet(bisenet_cp, device=self.device)

        # --- UNet tooth contour segmenter (on self.device) --------------------
        self.unet = None
        if state is not None:
            self.unet = load_unet(state, device=self.device)

    @torch.no_grad()
    def run(self, input_data, *, if_visual: bool = False, **params) -> dict:
        """Run Stage1 inference — no weight loading here.

        Parameters
        ----------
        input_data : str
            Path to the input image file (``img_path``).
        if_visual : bool
            If ``True``, write debug images to ``./result_vis/``.

        Returns
        -------
        dict
            Keys: ``contour_mask``, ``crop_info``, ``debug_image`` (contract),
            plus backward-compat keys: ``ori_face``, ``detect_face``,
            ``info``, ``crop_face``, ``crop_mouth``, ``crop_teeth``,
            ``crop_mask``.
        """
        img_path = input_data

        # ---------------------------------------------------------------
        # Step 1: Face detection & crop mouth
        # ---------------------------------------------------------------
        ori_img, face, info_detectface = DetectFace(
            img_path, newsize=(512, 512),
            face_detector=self.face_detector,
            shape_predictor=self.shape_predictor,
        )
        face, mouth_mask, mouth_color = DetectMouth(
            face, bisenet=self.bisenet, device=self.device,
        )
        crop_face, crop_mask, info_cropmouth = CropMouth(
            face, mouth_mask, crop_size=(256, 128), if_visual=if_visual,
        )

        # ---------------------------------------------------------------
        # Step 2: Tooth contour segmentation (mode-gated)
        # ---------------------------------------------------------------
        mouth_masking = None
        crop_teeth = None

        if self.mode in ['C2C2T_v2', 'C2C2T_v2_facecolor_teethcolor',
                         'C2C2T_v2_facecolor_lightcolor', 'C2C2T_v2_fourier']:
            mouth_masking = MaskingMouth(crop_face, crop_mask, if_visual=if_visual)
            mouth_masking = np.uint8(mouth_masking)

            teeth_contour = SegmentToothContour(
                mouth_masking, state=None, if_visual=if_visual,
                model=self.unet, device=self.device,
            )
            teeth_contour = np.uint8(crop_mask / 255 * teeth_contour)
            crop_teeth = teeth_contour

        # ---------------------------------------------------------------
        # Build result dict
        # ---------------------------------------------------------------
        return {
            # --- Contract keys (CLAUDE.md) --------------------------------
            "contour_mask": crop_teeth,
            "crop_info": {0: info_detectface, 1: info_cropmouth},
            "debug_image": face,

            # --- Backward-compatible keys ---------------------------------
            "ori_face": ori_img,            # numpy_BGR_uint8
            "detect_face": face,            # numpy_BGR_uint8
            "info": {0: info_detectface, 1: info_cropmouth},
            "crop_face": crop_face,         # numpy_BGR_uint8
            "crop_mouth": mouth_masking,    # numpy_BGR_uint8
            "crop_teeth": crop_teeth,       # numpy_BGR_uint8
            "crop_mask": crop_mask,         # numpy_BGR_uint8
        }


# ---------------------------------------------------------------------------
# Backward-compatible wrapper — keeps old call-sites working until they are
# migrated to the class-based API.
# ---------------------------------------------------------------------------
def Stage1(img_path, mode, state, if_visual=False):
    """Legacy function wrapper.  Instantiates :class:`Stage1Class` per call.

    .. deprecated::
        Use :class:`Stage1Class` directly for production / Triton workloads so
        that model weights are loaded once in ``__init__``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage = Stage1Class(device=device, mode=mode, state=state)
    return stage.run(img_path, if_visual=if_visual)


if __name__ == "__main__":
    img_path = r'C:\IDEA_Lab\Project_tooth_photo\Img2Img\Data\118_199fcc33faec4b39bb0fe2efc9e09cf3.jpg'
    mode = 1
    state = "Stage1/ToothContourDetect/ckpt/ckpt_4800.pth"
