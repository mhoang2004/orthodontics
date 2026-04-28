"""
main_interactive.py
-------------------
Interactive inference pipeline. Accepts tunable parameters via CLI:
  --whiteness       : Intensity of light-color conditioning (0.0 - 2.0, default 1.0)
  --alignment       : Strength of contour alignment signal (0.0 - 2.0, default 1.0)
  --timesteps       : Number of diffusion timesteps for Stage 3 (10 - 200, default 60)

These parameters modify how Stage 3 (Contour→Teeth) generates teeth,
turning the model into an interactive editing tool.
"""

import yaml
import argparse
import os
import cv2
import json
import sys

# Thêm thư mục Code vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Stage1_ToothSegm import Stage1
from Stage2_Mask2Mask import Stage2_Mask2Mask
from Stage3_Mask2Teeth import Stage3_Mask2Teeth
from Restore.Restore import Restore


def run_interactive(img_path, out_path, whiteness=1.0, alignment=1.0, timesteps=60):
    """
    Chạy pipeline với các tham số tùy chỉnh.
    
    Parameters
    ----------
    img_path    : str   - Đường dẫn ảnh đầu vào
    out_path    : str   - Thư mục output  
    whiteness   : float - Cường độ face_light_color (0.0-2.0, mặc định 1.0)
    alignment   : float - Cường độ contour alignment (0.0-2.0, mặc định 1.0)
    timesteps   : int   - Số bước diffusion Stage 3 (10-200, mặc định 60)
    
    Returns
    -------
    dict với các key: 'pred_ori_face', 'pred_detect_face', 'pred_crop_face'
    """
    with open("./Config.yaml", 'r') as f:
        GeneratorConfig = yaml.load(f, Loader=yaml.SafeLoader)['C2C2T_v2_facecolor_lightcolor']

    mode = GeneratorConfig['mode']
    stage1_ckpt = GeneratorConfig['stage1']
    stage2_ckpt = GeneratorConfig['stage2']
    stage3_ckpt = GeneratorConfig['stage3']

    img_name = os.path.basename(img_path).split('.')[0]

    # ── Stage 1: Tooth Segmentation ──────────────────────────────────────────
    stage1_data = Stage1(img_path, mode=mode, state=stage1_ckpt, if_visual=False)

    # ── Stage 2: Contour Refinement (Diffusion) ─────────────────────────────
    stage2_data = Stage2_Mask2Mask(stage1_data, mode=mode, state=stage2_ckpt, if_visual=False)
    stage2_data.update(stage1_data)

    # ── Stage 3: Teeth Generation (Diffusion) — với interactive params ──────
    stage3_data = Stage3_Mask2Teeth(
        stage2_data, mode=mode, state=stage3_ckpt, if_visual=False,
        whiteness=whiteness,
        alignment_strength=alignment,
        custom_timesteps=timesteps
    )
    stage3_data.update(stage2_data)

    # ── Restore ─────────────────────────────────────────────────────────────
    pred = Restore(stage3_data['crop_mouth_align'], stage3_data)
    pred_face = pred['pred_ori_face']

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs(os.path.join(out_path, 'prediction'), exist_ok=True)
    result_path = os.path.join(out_path, 'prediction', img_name + '.png')
    cv2.imwrite(result_path, pred_face)
    print(f"[interactive] Saved → {result_path}")

    return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive Orthodontics Inference')
    parser.add_argument('-i', '--img_path', type=str, required=True,
                        help='Path to input facial photograph')
    parser.add_argument('--whiteness', type=float, default=1.0,
                        help='Teeth whiteness intensity (0.0-2.0, default=1.0)')
    parser.add_argument('--alignment', type=float, default=1.0,
                        help='Contour alignment strength (0.0-2.0, default=1.0)')
    parser.add_argument('--timesteps', type=int, default=60,
                        help='Diffusion timesteps for Stage 3 (10-200, default=60)')
    parser.add_argument('--out_path', type=str, default='../Output',
                        help='Output directory')

    args = parser.parse_args()

    # Clamp values
    args.whiteness = max(0.0, min(2.0, args.whiteness))
    args.alignment = max(0.0, min(2.0, args.alignment))
    args.timesteps = max(10, min(200, args.timesteps))

    run_interactive(
        img_path=args.img_path,
        out_path=args.out_path,
        whiteness=args.whiteness,
        alignment=args.alignment,
        timesteps=args.timesteps
    )
