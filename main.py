u"""sumary_line

conda create -n STIFMaps -y python=3.10
conda activate STIFMaps
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
"""

import os

import sys

import click

# import cv2 and torch before STIFMap
import cv2
import numpy as np
import torch

from PIL import Image
from skimage import io

from loguru import logger

import STIFMap_generation
from misc import get_step

__DIR__ = os.path.dirname(os.path.abspath(__file__))

# Specify the DAPI and collagen images to be overlaid
dapi = '/NAS/yzy/project/MIBI/histocat/Ca2_05_001/Ir191_DNA1_1.tiff'
collagen = '/NAS/yzy/project/MIBI/histocat/Ca2_05_001/Tm169_collagen1_1.tiff'


# Networks were trained at a microscopy resolution of 4.160 pixels/micron (0.2404 microns/pixel)
# Provide a scale factor to resize the input images to this resolution
# Ex: Images at 2.308 pixels/micron require a scale_factor of 1.802
@click.command(context_settings=dict(help_option_names=['-h', '--help']), no_args_is_help=False)
@click.option("-d", "--dapi", type=click.Path(exists=True), default=dapi, help="the path to dapi")
@click.option("-c", "--collagen", type=click.Path(exists=True), default=collagen, help = "the path to collagen")
@click.option("-o", "--output", type=click.Path(), help="the output image path")
@click.option("-m", "--model-dir", type=click.Path(exists=True),
              default=os.path.join(__DIR__, "trained_models"), show_default=True,
              help="the path to directory contains models")
@click.option("--scale-factor", type=float, default=-1, show_default=True,
              help="a scale factor to resize the input images to this resolution. "
                   "Ex: Images at 2.308 pixels/micron require a scale_factor of 4.160 / 2.308 = 1.802")
@click.option("--step", type=int, default=40, show_default=True, help="the default step to iter over image")
@click.option("--batch-size", type=int, default=100, show_default=True, help="batch size")
def main(
        dapi: str, collagen: str, model_dir: str, output: str,
        scale_factor: float, step: int, batch_size: int
):
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    name = "test"

    if scale_factor < 0:
        im = io.imread(dapi)
        scale_factor = 1694 / im.shape[1] * (4.16 / 2.308)
        logger.info(f"estimate scale factor = {scale_factor}")

    # Specify the models to use for stiffness predictions:
    models = [os.path.join(model_dir, model) for model in os.listdir(model_dir)]

    # Given the scale_factor, what is the actual step size (in pixels) from one square to the next?
    step = get_step(step, scale_factor)

    logger.info(f"step size if {step} pixels")

    # Get the actual side length of one square
    # The models expect input squares that are 224 x 224 pixels.
    # Given the scale_factor, how many pixels is that in these images?
    square_side = get_step(224, scale_factor)

    logger.info('Side length for a square is ' + str(square_side) + ' pixels')

    # Generate the stiffness predictions
    z_out = STIFMap_generation.generate_STIFMap(dapi, collagen, name, step, models=models,
                                                mask=False, batch_size=batch_size, save_dir=False)
    col_colored = STIFMap_generation.collagen_paint(dapi, collagen, z_out, name, step,
                                                    mask=False, scale_percent=100, save_dir=False)

    im = Image.fromarray((col_colored * 255).astype(np.uint8))
    logger.info(f"Saved image size: {col_colored.shape}")
    im.save(output)

    # # # Specify the staining file to use
    # # stain = '/NAS/yzy/software/STIFMaps-main/test_cases/with_stain/test1_stain.TIF'
    # # # The pixel threshold to use when comparing STIFMaps/DAPI/collagen vs stain intensity
    # # quantile = .99
    # # # Correlate the stain intensity with the intensity of collagen, DAPI, and predicted stiffness
    # # z_stain_corr, collagen_stain_corr, dapi_stain_corr = STIFMap_generation.correlate_signals_with_stain(
    # #     dapi, collagen, z_out, stain, step,
    # #     mask=False, square_side=square_side,
    # #     scale_percent=100, quantile=quantile)
    # #
    # # io.imsave("./z_stain_corr.tif", z_stain_corr)


if __name__ == '__main__':
    main()
