import os
import json
import numpy as np
import cv2
import torch
import triton_python_backend_utils as pb_utils

import sys
if "/app/Code" not in sys.path:
    sys.path.insert(0, "/app/Code")
    
from pipeline import DentalPipeline

class TritonPythonModel:
    def initialize(self, args: dict) -> None:
        config_path = os.environ.get("DENTAL_CONFIG_PATH", "/app/Code/Config.yaml")
        tmp_dir = os.environ.get("TRITON_TMP_DIR", "/tmp/triton_io")
        self.pipeline = DentalPipeline(config_path=config_path, tmp_dir=tmp_dir)

    def execute(self, requests: list) -> list:
        responses = []
        for request in requests:
            try:
                responses.append(self._process_single(request))
            except Exception as e:
                responses.append(pb_utils.InferenceResponse([], error=pb_utils.TritonError(str(e))))
        return responses

    def _process_single(self, request):
        t_img = pb_utils.get_input_tensor_by_name(request, "IMAGE")
        if t_img is None:
            raise ValueError("IMAGE tensor is required")
        img_bytes = t_img.as_numpy().flat[0]
        
        # New interactive parameters
        t_whiteness = pb_utils.get_input_tensor_by_name(request, "whiteness")
        whiteness = t_whiteness.as_numpy().flat[0] if t_whiteness is not None else 1.0
        
        t_alignment = pb_utils.get_input_tensor_by_name(request, "alignment")
        alignment = t_alignment.as_numpy().flat[0] if t_alignment is not None else 1.0
        
        t_timesteps = pb_utils.get_input_tensor_by_name(request, "sample_num")
        timesteps = t_timesteps.as_numpy().flat[0] if t_timesteps is not None else 60
        
        # Ensure timesteps < num_timesteps (which is 60 in current config)
        # We can get the actual num_timesteps from the pipeline's network
        max_allowed = self.pipeline.args.stage3_timesteps if hasattr(self.pipeline.args, "stage3_timesteps") else 60
        if hasattr(self.pipeline, "pipeline") and hasattr(self.pipeline.pipeline, "num_timesteps"):
             max_allowed = self.pipeline.pipeline.num_timesteps
        
        # Simple clamp to avoid 'num_timesteps must greater than sample_num'
        if timesteps >= 60: 
            timesteps = 59
        if timesteps < 1:
            timesteps = 1
        
        t_seed = pb_utils.get_input_tensor_by_name(request, "seed")
        seed = int(t_seed.as_numpy().flat[0]) if t_seed is not None else -1
        
        # If seed is -1 or negative, pick a random one
        import random
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        t_inter = pb_utils.get_input_tensor_by_name(request, "RETURN_INTERMEDIATES")
        return_intermediates = t_inter.as_numpy().flat[0] if t_inter is not None else False
        
        if return_intermediates:
            res = self.pipeline.run_with_intermediates(
                img_bytes, 
                whiteness=whiteness, 
                alignment=alignment, 
                timesteps=timesteps
            )
            pred_face = res["pred_ori_face"]
            
            # Extract basic metadata
            meta = {"shape": list(pred_face.shape)}
            metadata = json.dumps(meta)
        else:
            pred_face = self.pipeline.run(
                img_bytes,
                whiteness=whiteness, 
                alignment=alignment, 
                timesteps=timesteps
            )
            meta = {"shape": list(pred_face.shape)}
            metadata = json.dumps(meta)
            
        _, buf = cv2.imencode(".png", pred_face)
        
        out_tensors = [
            pb_utils.Tensor("PRED_IMAGE", np.array([buf.tobytes()], dtype=object)),
            pb_utils.Tensor("METADATA", np.array([metadata], dtype=object))
        ]
        return pb_utils.InferenceResponse(out_tensors)

    def finalize(self) -> None:
        if hasattr(self, 'pipeline'):
            del self.pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
