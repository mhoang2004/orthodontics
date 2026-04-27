import io
import json
import logging
import os
import sys
import cv2
from typing import Any, Dict, Iterable

CODE_DIR = os.path.join(os.path.dirname(__file__), "libs")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from PIL import Image


LOGGER = logging.getLogger("orthodontics.triton")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)


class TritonPythonModel:
    """Triton Python backend wrapper for the full orthodontics pipeline."""

    def initialize(self, args: Dict[str, Any]) -> None:
        """Load all stage models once at Triton startup."""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_config = self._parse_model_config(args)
            stage_cfg = self._read_stage_config(self.model_config)

            from Restore.Restore import Restore as restore
            from Stage1_ToothSegm import Stage1Class as Stage1
            from Stage2_Mask2Mask import Stage2
            from Stage3_Mask2Teeth import Stage3

            self.stage1 = Stage1(device=self.device, **stage_cfg["stage1"])
            self.stage2 = Stage2(device=self.device, **stage_cfg["stage2"])
            self.stage3 = Stage3(device=self.device, **stage_cfg["stage3"])
            self.restore_fn = restore

            LOGGER.info("Orthodontics pipeline initialized on device=%s", self.device)
        except Exception as exc:
            LOGGER.exception("Failed to initialize orthodontics pipeline")
            raise pb_utils.TritonModelException(
                f"Initialization failed for orthodontics_pipeline: {exc}"
            )

    def execute(self, requests: list) -> list:
        """Run inference for each request and return one response per request."""
        responses = []
        for request in requests:
            try:
                responses.append(self._execute_single(request))
            except Exception as exc:
                LOGGER.exception("Inference request failed")
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"Inference failed: {exc}")
                    )
                )
        return responses

    def finalize(self) -> None:
        """Release references and best-effort GPU cache cleanup."""
        try:
            LOGGER.info("Finalizing orthodontics pipeline model")
            for name in ("stage1", "stage2", "stage3", "restore_fn"):
                if hasattr(self, name):
                    setattr(self, name, None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            LOGGER.exception("Finalize encountered an error")

    def _execute_single(self, request: Any) -> pb_utils.InferenceResponse:
        image_bytes = self._get_bytes_input(request, "image_bytes")
        whiteness = self._get_scalar_input(request, "whiteness", float)
        alignment = self._get_scalar_input(request, "alignment", float)
        sample_num = self._get_scalar_input(request, "sample_num", int)
        seed = self._get_scalar_input(request, "seed", int)

        if sample_num <= 0:
            raise ValueError("sample_num must be > 0")

        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)

        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to parse image_bytes into image data")

        with torch.no_grad():
            stage1_out = self.stage1.run(image)
            self._validate_stage_output(
                stage1_out,
                required_keys=("contour_mask", "crop_info", "debug_image"),
                stage_name="Stage1",
            )

            stage2_out = self.stage2.run(
                stage1_out["contour_mask"],
                alignment=alignment,
                sample_num=sample_num,
            )
            self._validate_stage_output(
                stage2_out,
                required_keys=("generated_contour",),
                stage_name="Stage2",
            )

            stage3_out = self.stage3.run(
                stage2_out["generated_contour"],
                whiteness=whiteness,
                sample_num=sample_num,
            )
            self._validate_stage_output(
                stage3_out,
                required_keys=("teeth_image",),
                stage_name="Stage3",
            )

            restore_out = self.restore_fn(
                mouth_align=stage3_out["teeth_image"],
                data=stage1_out
            )
            result_image = restore_out["pred_ori_face"]

        result_bytes = self._encode_png_bytes(result_image)
        debug_bytes = self._encode_png_bytes(stage1_out["debug_image"])

        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(
                    "result_image", np.array([result_bytes], dtype=object)
                ),
                pb_utils.Tensor(
                    "stage1_debug", np.array([debug_bytes], dtype=object)
                ),
            ]
        )

    def _parse_model_config(self, args: Dict[str, Any]) -> Dict[str, Any]:
        raw_model_config = args.get("model_config", "{}")
        if isinstance(raw_model_config, dict):
            return raw_model_config
        if isinstance(raw_model_config, str):
            return json.loads(raw_model_config)
        raise ValueError("Unsupported model_config format from Triton args")

    def _read_stage_config(self, model_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        parameters = model_config.get("parameters", {}) or {}

        stage1_config_path = self._resolve_code_path(
            self._param_str(parameters, "stage1_config_path", "Config.yaml")
        )
        stage2_config_path = self._resolve_code_path(
            self._param_str(
                parameters,
                "stage2_config_path",
                "Stage2/config/config_Contour2Contour.yaml",
            )
        )
        stage3_config_path = self._resolve_code_path(
            self._param_str(
                parameters,
                "stage3_config_path",
                "Stage3/config/config_Contour2Tooth_facecolor_lightcolor.yaml",
            )
        )

        stage1_kwargs = self._param_json_dict(parameters, "stage1_kwargs_json")
        stage2_kwargs = self._param_json_dict(parameters, "stage2_kwargs_json")
        stage3_kwargs = self._param_json_dict(parameters, "stage3_kwargs_json")

        return {
            "stage1": {"config_path": stage1_config_path, **stage1_kwargs},
            "stage2": {"config_path": stage2_config_path, **stage2_kwargs},
            "stage3": {"config_path": stage3_config_path, **stage3_kwargs},
        }

    def _param_str(
        self,
        parameters: Dict[str, Any],
        name: str,
        default_value: str,
    ) -> str:
        entry = parameters.get(name)
        if not isinstance(entry, dict):
            return default_value
        value = entry.get("string_value", default_value)
        if value is None:
            return default_value
        return str(value)

    def _param_json_dict(self, parameters: Dict[str, Any], name: str) -> Dict[str, Any]:
        payload = self._param_str(parameters, name, "")
        if not payload:
            return {}
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError(f"Parameter '{name}' must be a JSON object")
        return parsed

    def _resolve_code_path(self, maybe_relative_path: str) -> str:
        if os.path.isabs(maybe_relative_path):
            return maybe_relative_path
        return os.path.normpath(os.path.join(CODE_DIR, maybe_relative_path))

    def _get_input_array(self, request: Any, name: str) -> np.ndarray:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            raise KeyError(f"Missing required input tensor '{name}'")
        value = tensor.as_numpy()
        if value is None:
            raise ValueError(f"Input tensor '{name}' is empty")
        return value

    def _get_scalar_input(self, request: Any, name: str, cast_type: Any) -> Any:
        raw = self._get_input_array(request, name)
        flat = np.asarray(raw).reshape(-1)
        if flat.size != 1:
            raise ValueError(
                f"Input tensor '{name}' must contain exactly one element; got {flat.size}"
            )
        return cast_type(flat[0])

    def _get_bytes_input(self, request: Any, name: str) -> bytes:
        raw = self._get_input_array(request, name)
        flat = np.asarray(raw, dtype=object).reshape(-1)
        if flat.size != 1:
            raise ValueError(
                f"Input tensor '{name}' must contain exactly one bytes item; got {flat.size}"
            )

        value = flat[0]
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        if isinstance(value, memoryview):
            return value.tobytes()
        if isinstance(value, str):
            return value.encode("utf-8")
        raise TypeError(f"Input tensor '{name}' must be bytes-like; got {type(value)!r}")

    def _validate_stage_output(
        self,
        output: Any,
        required_keys: Iterable[str],
        stage_name: str,
    ) -> None:
        if not isinstance(output, dict):
            raise TypeError(f"{stage_name}.run(...) must return dict, got {type(output)!r}")

        missing = [key for key in required_keys if key not in output]
        if missing:
            raise KeyError(f"{stage_name} output missing required keys: {missing}")

    def _encode_png_bytes(self, image_obj: Any) -> bytes:
        if isinstance(image_obj, torch.Tensor):
            image_obj = image_obj.detach().cpu().numpy()
            
        if isinstance(image_obj, np.ndarray):
            array = image_obj
            if array.dtype != np.uint8:
                if np.issubdtype(array.dtype, np.floating):
                    if np.max(array) <= 1.0:
                        array = array * 255.0
                    array = np.clip(array, 0.0, 255.0).astype(np.uint8)
                else:
                    array = np.clip(array, 0, 255).astype(np.uint8)
            
            success, buffer = cv2.imencode('.png', array)
            if success:
                return buffer.tobytes()
        
        raise TypeError(f"Unsupported image type for PNG encoding: {type(image_obj)!r}")

    def _coerce_to_pil_image(self, image_obj: Any) -> Image.Image:
        if isinstance(image_obj, Image.Image):
            return image_obj

        if isinstance(image_obj, torch.Tensor):
            image_obj = image_obj.detach().cpu().numpy()

        if isinstance(image_obj, np.ndarray):
            array = image_obj
            if array.dtype != np.uint8:
                if np.issubdtype(array.dtype, np.floating):
                    if np.max(array) <= 1.0:
                        array = array * 255.0
                    array = np.clip(array, 0.0, 255.0).astype(np.uint8)
                else:
                    array = np.clip(array, 0, 255).astype(np.uint8)

            if array.ndim == 2:
                return Image.fromarray(array, mode="L")
            if array.ndim == 3 and array.shape[2] in (3, 4):
                return Image.fromarray(array)
            raise ValueError(f"Unsupported numpy image shape: {array.shape}")

        raise TypeError(f"Unsupported image type for PNG encoding: {type(image_obj)!r}")
