import itertools

import PIL.ImageDraw as ImageDraw
import numpy as np
from PIL import Image


def create_bw_image(image_cells):
    rows = image_cells.shape[0]
    cols = image_cells.shape[1]
    blank_image = Image.new(
        "L", (image_cells.shape[3] * cols, image_cells.shape[2] * rows)
    )
    for i_row, i_col in itertools.product(range(rows), range(cols)):
        x = image_cells[i_row, i_col]
        x = np.clip(255 - np.round(x * 255), 0, 255).astype(np.uint8)
        blank_image.paste(
            Image.fromarray(x),
            (i_col * image_cells.shape[3], i_row * image_cells.shape[2]),
        )
    return blank_image


def create_rgb_image(image_cells):
    rows = image_cells.shape[0]
    cols = image_cells.shape[1]
    blank_image = Image.new(
        "RGB", (image_cells.shape[4] * cols, image_cells.shape[3] * rows)
    )
    for i_row, i_col in itertools.product(range(rows), range(cols)):
        x = image_cells[i_row, i_col]
        x = np.clip(np.round(x * 255), 0, 255).astype(np.uint8).transpose(1, 2, 0)
        blank_image.paste(
            Image.fromarray(x),
            (i_col * image_cells.shape[4], i_row * image_cells.shape[3]),
        )
    return blank_image


def create_image_with_label(X, y):
    im = create_bw_image(X)
    im = im.convert("RGB")
    draw = ImageDraw.Draw(im)
    for i_row in range(len(y)):
        for i_col in range(len(y[i_row])):
            draw.text(
                (i_col * 28, i_row * 28),
                str(y[i_row, i_col]),
                (255, 50, 255),
            )
    return im


def rescale(im, scale_factor, resample=Image.BICUBIC):
    return im.resize(
        (int(round(im.width * scale_factor)), int(round(im.height * scale_factor))),
        resample=resample,
    )


def stack_images_in_rows(*batch_images, n_cols):
    n_rows_per_batch = batch_images[0].shape[0] // n_cols
    reshaped_batches = [
        b.reshape(n_rows_per_batch, n_cols, *b.shape[1:]) for b in batch_images
    ]
    n_rows = n_rows_per_batch * len(batch_images)

    return np.stack(reshaped_batches, axis=1).reshape(
        n_rows, n_cols, *batch_images[0].shape[1:]
    )
