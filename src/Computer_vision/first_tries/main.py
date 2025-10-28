import cv2 as cv
import numpy as np
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR: Path = Path(__file__).parent
IMAGES_DIR: Path = SCRIPT_DIR / 'images'

# Define image paths
GAMEPLAY_PATH: str = str(IMAGES_DIR / 'play.png')
PLAYER_PATH: str = str(IMAGES_DIR / 'player.png')

# Read images using proper path handling
gameplay_img: np.ndarray = cv.imread(GAMEPLAY_PATH, cv.IMREAD_COLOR)
player_img: np.ndarray = cv.imread(PLAYER_PATH, cv.IMREAD_COLOR)

result = cv.matchTemplate(gameplay_img, player_img, cv.TM_CCOEFF_NORMED)

THRESHOLD: float = 0.4

locations = np.where(result >= THRESHOLD)
locations = list(zip(*locations[::-1]))

print(locations)
