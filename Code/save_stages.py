"""
save_stages.py
--------------
Xuất ảnh minh họa từng giai đoạn (giải đoạn) trong quá trình infer của pipeline.

Cấu trúc thư mục đầu ra:
    <out_path>/giaidoan/<img_name>/
        00_original.png          - Ảnh gốc đầu vào
        01_detect_face.png       - Face detected (512x512, sau bbox crop)
        02_crop_face.png         - Vùng miệng được crop (256x128)
        03_crop_mouth_masked.png - Vùng miệng sau khi masking (chỉ giữ răng)
        04_crop_teeth_contour.png- Contour của răng (Stage 1 output)
        05_stage2_output.png     - Contour được tinh chỉnh bởi Stage 2 (diffusion)
        06_stage3_output.png     - Ảnh răng được sinh ra bởi Stage 3 (diffusion)
        07_restore_crop.png      - Ảnh crop face sau khi ghép răng mới
        08_restore_detect.png    - Ảnh face 512x512 sau khi ghép răng mới
        09_final_result.png      - Ảnh gốc kích thước đầy đủ sau khi ghép răng
        pipeline_overview.png    - Tổng quan toàn bộ pipeline (tất cả trong 1 ảnh)
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_dir(path: str) -> None:
    """Tạo thư mục nếu chưa tồn tại."""
    os.makedirs(path, exist_ok=True)


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Chuyển đổi BGR (OpenCV) sang RGB (matplotlib)."""
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 3:
        return img[:, :, ::-1]
    return img  # grayscale / single-channel – giữ nguyên


def _save(img: np.ndarray, path: str) -> None:
    """Lưu ảnh numpy (BGR) ra file bằng OpenCV."""
    if img is None:
        return
    cv2.imwrite(path, img)


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    """
    Đảm bảo ảnh có 3 kênh BGR.
    - Nếu grayscale (H,W) → stack thành (H,W,3)
    - Nếu đã là (H,W,3) → giữ nguyên
    """
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    return img


def _safe_img(img: np.ndarray, label: str = "") -> np.ndarray:
    """Trả về ảnh placeholder nếu img là None."""
    if img is None:
        placeholder = np.full((128, 256, 3), 40, dtype=np.uint8)
        cv2.putText(placeholder, label if label else "N/A",
                    (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return placeholder
    return _ensure_bgr(img)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def save_stage_images(stage1_data: dict,
                      stage2_data: dict,
                      stage3_data: dict,
                      pred: dict,
                      out_path: str,
                      img_name: str) -> None:
    """
    Xuất ảnh từng bước infer ra <out_path>/giaidoan/<img_name>/.

    Parameters
    ----------
    stage1_data : dict   – output của Stage1()
    stage2_data : dict   – output của Stage2_Mask2Mask() đã .update(stage1_data)
    stage3_data : dict   – output của Stage3_Mask2Teeth() đã .update(stage2_data)
    pred        : dict   – output của Restore()
    out_path    : str    – thư mục gốc (giá trị args.out_path)
    img_name    : str    – tên ảnh đầu vào (không có extension)
    """
    save_dir = os.path.join(out_path, 'giaidoan', img_name)
    _make_dir(save_dir)

    # ── Thu thập các ảnh từ mỗi giai đoạn ────────────────────────────────────

    imgs = {
        "00_original":           stage1_data.get("ori_face"),
        "01_detect_face":        stage1_data.get("detect_face"),
        "02_crop_face":          stage1_data.get("crop_face"),
        "03_crop_mouth_masked":  stage1_data.get("crop_mouth"),
        "04_crop_teeth_contour": stage1_data.get("crop_teeth"),
        "05_stage2_output":      stage2_data.get("crop_teeth_align"),
        "06_stage3_output":      stage3_data.get("crop_mouth_align"),
        "07_restore_crop":       pred.get("pred_crop_face"),
        "08_restore_detect":     pred.get("pred_detect_face"),
        "09_final_result":       pred.get("pred_ori_face"),
    }

    labels = {
        "00_original":           "Gốc (original)",
        "01_detect_face":        "Face detected (512×512)",
        "02_crop_face":          "Crop miệng (256×128)",
        "03_crop_mouth_masked":  "Mouth masking",
        "04_crop_teeth_contour": "Contour răng (Stage 1)",
        "05_stage2_output":      "Contour tinh chỉnh (Stage 2 – Diffusion)",
        "06_stage3_output":      "Răng mới (Stage 3 – Diffusion)",
        "07_restore_crop":       "Crop ghép răng mới",
        "08_restore_detect":     "Face 512×512 ghép răng mới",
        "09_final_result":       "Kết quả cuối (full size)",
    }

    # ── Lưu từng ảnh riêng lẻ ────────────────────────────────────────────────
    for key, img in imgs.items():
        if img is not None:
            _save(_ensure_bgr(img), os.path.join(save_dir, f"{key}.png"))

    # ── Lưu 5 ảnh bitmap trung gian diffusion Stage 2 ──────────────────────
    stage2_visuals = stage2_data.get("stage2_visuals", [])
    diffusion_dir = os.path.join(save_dir, 'stage2_diffusion')
    _make_dir(diffusion_dir)

    n_visuals = len(stage2_visuals)
    if n_visuals > 0:
        # Chọn 5 bước đều nhau từ danh sách visuals (bỏ bước 0 – noise thuần)
        # visuals[0] = noise ban đầu, visuals[-1] = kết quả cuối
        if n_visuals <= 5:
            selected_indices = list(range(n_visuals))
        else:
            # Lấy 5 bước đều nhau (bao gồm bước đầu và cuối)
            selected_indices = [int(round(i * (n_visuals - 1) / 4)) for i in range(5)]

        selected_visuals = [(idx, stage2_visuals[idx]) for idx in selected_indices]

        for step_i, (orig_idx, vis_img) in enumerate(selected_visuals):
            vis_bgr = _ensure_bgr(vis_img)
            bmp_path = os.path.join(diffusion_dir,
                                    f"step{step_i}_t{orig_idx}.bmp")
            _save(vis_bgr, bmp_path)

        # Tạo ảnh overview cho quá trình diffusion Stage 2
        _save_diffusion_overview(selected_visuals, diffusion_dir,
                                 img_name, n_visuals)
        print(f"[save_stages] Đã lưu {len(selected_visuals)} ảnh diffusion Stage 2 → {diffusion_dir}")

    # ── Tạo ảnh tổng quan pipeline ─────────────────────────────────────────────
    _save_pipeline_overview(imgs, labels, save_dir, img_name)


def _save_diffusion_overview(selected_visuals: list, save_dir: str,
                              img_name: str, total_steps: int) -> None:
    """
    Vẽ ảnh minh họa quá trình denoise của Stage 2 Diffusion.
    Hiển thị 5 bước: từ noise → contour cuối cùng.
    """
    n = len(selected_visuals)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), facecolor='#1a1a2e')
    fig.suptitle(f"Stage 2 Diffusion Denoising — {img_name}\n"
                 f"({total_steps} bước tổng cộng, hiển thị {n} bước đại diện)",
                 fontsize=13, fontweight='bold', color='white', y=1.02)

    if n == 1:
        axes = [axes]

    step_labels = []
    for step_i, (orig_idx, _) in enumerate(selected_visuals):
        if step_i == 0:
            step_labels.append(f"Bước {orig_idx}\n(Noise)")
        elif step_i == n - 1:
            step_labels.append(f"Bước {orig_idx}\n(Kết quả)")
        else:
            step_labels.append(f"Bước {orig_idx}\n(Đang denoise)")

    for ax_i, (ax, (orig_idx, vis_img)) in enumerate(zip(axes, selected_visuals)):
        vis_bgr = _safe_img(vis_img, f"step {orig_idx}")
        rgb = _bgr_to_rgb(vis_bgr)
        ax.imshow(rgb, aspect='auto')
        ax.set_title(step_labels[ax_i], fontsize=10, color='#e0e0e0', pad=6)
        ax.axis('off')

        # Viền cam cho Stage 2
        for spine in ax.spines.values():
            spine.set_edgecolor('#FF9800')
            spine.set_linewidth(2.5)
            spine.set_visible(True)

        # Mũi tên giữa các subplot
        if ax_i < n - 1:
            ax.annotate('→', xy=(1.08, 0.5), xycoords='axes fraction',
                        fontsize=22, color='#FF9800', fontweight='bold',
                        ha='center', va='center')

    overview_path = os.path.join(save_dir, 'diffusion_overview.png')
    fig.savefig(overview_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_pipeline_overview(imgs: dict, labels: dict,
                             save_dir: str, img_name: str) -> None:
    """
    Vẽ bảng tổng quan: 2 cột × 5 hàng với mũi tên chỉ luồng xử lý.
    """
    keys = list(imgs.keys())     # 10 ảnh
    n_rows, n_cols = 5, 2

    fig = plt.figure(figsize=(16, 22), facecolor='#1a1a2e')
    fig.suptitle(f"Pipeline Inference Overview — {img_name}",
                 fontsize=16, fontweight='bold', color='white', y=0.99)

    gs = gridspec.GridSpec(n_rows, n_cols,
                           figure=fig,
                           hspace=0.45, wspace=0.15,
                           left=0.05, right=0.95,
                           top=0.96, bottom=0.02)

    for idx, key in enumerate(keys):
        row, col = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[row, col])

        img = _safe_img(imgs[key], labels.get(key, key))
        rgb = _bgr_to_rgb(img)
        ax.imshow(rgb, aspect='auto')
        ax.set_title(f"[{key[:2]}] {labels.get(key, key)}",
                     fontsize=9, color='#e0e0e0', pad=4)
        ax.axis('off')

        # Viền màu theo giai đoạn
        stage_colors = {
            '00': '#4CAF50', '01': '#4CAF50',           # xanh lá – input
            '02': '#2196F3', '03': '#2196F3', '04': '#2196F3',  # xanh dương – Stage 1
            '05': '#FF9800',                             # cam – Stage 2
            '06': '#E91E63',                             # hồng – Stage 3
            '07': '#9C27B0', '08': '#9C27B0', '09': '#9C27B0',  # tím – Restore
        }
        color = stage_colors.get(key[:2], '#888888')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
            spine.set_visible(True)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_info = [
        ("#4CAF50", "Input / Detect Face"),
        ("#2196F3", "Stage 1 – Tooth Segmentation"),
        ("#FF9800", "Stage 2 – Contour Refinement (Diffusion)"),
        ("#E91E63", "Stage 3 – Teeth Generation (Diffusion)"),
        ("#9C27B0", "Restore – Ghép vào ảnh gốc"),
    ]
    legend_ax = fig.add_axes([0.05, 0.005, 0.90, 0.018], facecolor='none')
    legend_ax.axis('off')
    x_pos = 0.0
    for color, text in legend_info:
        legend_ax.add_patch(plt.Rectangle((x_pos, 0.0), 0.015, 1.0,
                                          color=color, transform=legend_ax.transAxes))
        legend_ax.text(x_pos + 0.02, 0.5, text,
                       transform=legend_ax.transAxes,
                       fontsize=7.5, color='white', va='center')
        x_pos += 0.20

    overview_path = os.path.join(save_dir, 'pipeline_overview.png')
    fig.savefig(overview_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[save_stages] Đã lưu pipeline_overview → {overview_path}")
