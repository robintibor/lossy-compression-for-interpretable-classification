from torch import nn
import numpy as np
from PIL import Image
from lossy.util import np_to_th

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO


def compress_im(im, quality, optimize, format='jpeg', ):
    #https://stackoverflow.com/a/41818645/1469195
    out = BytesIO()
    im.save(out, format=format,quality=quality, optimize=optimize)
    out.seek(0)
    return out


class JPGCompress(nn.Module):
    def __init__(self, quality, optimize):
        super().__init__()
        self.quality = quality
        self.optimize = optimize

    def forward(self, X):
        old_device = X.device
        X = X.cpu()
        xs = []
        for x in X:
            im = Image.fromarray(
                np.uint8(np.round(np.array(x) * 255).transpose(
                    1, 2, 0)))
            im_bytes = compress_im(
                im, quality=self.quality, optimize=self.optimize)
            im = Image.open(im_bytes)
            xs.append(np.array(im).astype(np.float32).transpose(
                2, 0, 1) / 255.0)
        new_X = np_to_th(np.stack(xs), device=old_device)
        return new_X
