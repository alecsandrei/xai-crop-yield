from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f'PROJ_ROOT path is: {PROJ_ROOT}')

DATA_DIR = PROJ_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'

MODELS_DIR = PROJ_ROOT / 'models'

REPORTS_DIR = PROJ_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 0

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True)
except ModuleNotFoundError:
    pass
