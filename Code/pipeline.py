import os
import yaml
import types
import uuid
import numpy as np

# Change current working directory to where the source files are
# so that relative paths (e.g. 'Stage2/ckpt/...') resolve correctly.
if os.path.exists("/app/src"):
    os.chdir("/app/src")
elif os.path.exists("/app/Code"):
    os.chdir("/app/Code")

from Stage1_ToothSegm import Stage1
from Stage2_Mask2Mask import Stage2_Mask2Mask
from Stage3_Mask2Teeth import Stage3_Mask2Teeth
from Restore.Restore import Restore

class DentalPipeline:
    def __init__(
        self, 
        config_path=None, 
        config_key="C2C2T_v2_facecolor_lightcolor", 
        tmp_dir=None
    ):
        if config_path is None:
            config_path = os.environ.get("DENTAL_CONFIG_PATH", "./Config.yaml")
        if tmp_dir is None:
            tmp_dir = os.environ.get("TRITON_TMP_DIR", "/tmp/triton_io")
            
        with open(config_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)[config_key]
        self.args = types.SimpleNamespace(**cfg)
        
        self.tmp_dir = tmp_dir
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir, exist_ok=True)

    def _run_pipeline(self, img_path: str, whiteness=1.0, alignment=1.0, timesteps=60) -> np.ndarray:
        stage1_data = Stage1(img_path, mode=self.args.mode, state=self.args.stage1, if_visual=False)
        stage2_data = Stage2_Mask2Mask(stage1_data, mode=self.args.mode, state=self.args.stage2, if_visual=False)
        stage2_data.update(stage1_data)
        stage3_data = Stage3_Mask2Teeth(
            stage2_data, mode=self.args.mode, state=self.args.stage3, if_visual=False,
            whiteness=whiteness,
            alignment_strength=alignment,
            custom_timesteps=timesteps
        )
        stage3_data.update(stage2_data)
        pred = Restore(stage3_data['crop_mouth_align'], stage3_data)
        return pred['pred_ori_face']

    def run_from_path(self, img_path: str, whiteness=1.0, alignment=1.0, timesteps=60) -> np.ndarray:
        return self._run_pipeline(img_path, whiteness, alignment, timesteps)

    def run(self, image_bytes: bytes, whiteness=1.0, alignment=1.0, timesteps=60) -> np.ndarray:
        tmp_path = os.path.join(self.tmp_dir, f"triton_in_{uuid.uuid4().hex}.jpg")
        try:
            with open(tmp_path, "wb") as f:
                f.write(image_bytes)
            
            import cv2
            if cv2.imread(tmp_path) is None:
                raise RuntimeError(f"Failed to read image! size={len(image_bytes)}, first 10 bytes={image_bytes[:10]}")
                
            return self._run_pipeline(tmp_path, whiteness, alignment, timesteps)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def run_with_intermediates(self, image_bytes: bytes, whiteness=1.0, alignment=1.0, timesteps=60) -> dict:
        tmp_path = os.path.join(self.tmp_dir, f"triton_in_{uuid.uuid4().hex}.jpg")
        try:
            with open(tmp_path, "wb") as f:
                f.write(image_bytes)
            
            import cv2
            if cv2.imread(tmp_path) is None:
                raise RuntimeError(f"Failed to read image! size={len(image_bytes)}, first 10 bytes={image_bytes[:10]}")
            
            stage1_data = Stage1(tmp_path, mode=self.args.mode, state=self.args.stage1, if_visual=False)
            stage2_data = Stage2_Mask2Mask(stage1_data, mode=self.args.mode, state=self.args.stage2, if_visual=False)
            stage2_data.update(stage1_data)
            stage3_data = Stage3_Mask2Teeth(
                stage2_data, mode=self.args.mode, state=self.args.stage3, if_visual=False,
                whiteness=whiteness,
                alignment_strength=alignment,
                custom_timesteps=timesteps
            )
            stage3_data.update(stage2_data)
            pred = Restore(stage3_data['crop_mouth_align'], stage3_data)
            
            return {
                "pred_ori_face": pred['pred_ori_face'],
                "intermediates": stage3_data
            }
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
