import torch
import os
import cv2
import numpy as np
from torchvision import transforms
from Stage1.SegmentToothContour.Model import UNet

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
])


def load_unet(state, device=None):
    """Load UNet weights once and return the model on *device*.

    Parameters
    ----------
    state : str
        Checkpoint path.
    device : torch.device, optional
        Target device.  Falls back to ``cuda`` if available, else ``cpu``.

    Returns
    -------
    model : UNet
        Loaded and eval-ready model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_classes=2)
    model.load_state_dict(torch.load(state, map_location=device))
    model.to(device)
    model.eval()
    return model


def SegmentToothContour(mouth, state, if_visual=True, model=None, device=None):
    """Segment tooth contour from mouth image.

    Parameters
    ----------
    mouth : numpy.ndarray
        Input mouth image (BGR, uint8).
    state : str
        Checkpoint path.  Ignored when *model* is supplied.
    if_visual : bool
        If ``True``, write debug image to ``./result_vis/``.
    model : UNet, optional
        Pre-loaded UNet model (avoids re-loading per call).
    device : torch.device, optional
        Target device for inference.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### build model (only if not injected)
    if model is None:
        model = load_unet(state, device)

    ### initialize data
    mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB)    # numpy_RGB_uint8
    mouth = transform(mouth)
    mouth = mouth.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(mouth)
        pred = pred[0].cpu().numpy().argmax(0)
        pred = np.uint8(pred*255)
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    if if_visual == True:
        cv2.imwrite(os.path.join('./result_vis', 'result_ToothContour.png'), pred)
    return pred          #numpy_BGR_uint8
