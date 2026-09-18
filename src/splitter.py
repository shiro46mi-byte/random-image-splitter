"""
Voronoi分割を担当するモジュール。
"""

from PIL import Image
import numpy as np

from src.piece import Piece


def create_voronoi_map(
    image: Image.Image,
    piece_count: int,
) -> np.ndarray:
    """
    画像サイズに対するVoronoiラベルマップを生成する。

    Args:
        image:
            元画像

        piece_count:
            分割数

    Returns:
        shape=(height, width) の整数配列
    """
    if piece_count <= 0:
        raise ValueError("piece_countは1以上で指定してください。")

    width, height = image.size

    rng = np.random.default_rng()

    seeds = np.column_stack(
        (
            rng.integers(0, width, piece_count),
            rng.integers(0, height, piece_count),
        )
    )

    yy, xx = np.ogrid[:height, :width]

    distances = (
    (xx[:, :, np.newaxis] - seeds[:, 0]) ** 2
    + (yy[:, :, np.newaxis] - seeds[:, 1]) ** 2
    )
    

    labels = np.argmin(distances, axis=-1)

    return labels

def extract_pieces(
    image: Image.Image,
    labels: np.ndarray,
) -> list[Piece]:
    """
    Voronoiラベルマップをもとに画像片を生成する。

    Args:
        image:
            RGBA形式の元画像。

        labels:
            Voronoi領域を表すラベルマップ。
            それぞれのピクセルが何番の画像辺に所属するかを記録する。

    Returns:
        生成したPieceのリスト。
    """
    pieces: list[Piece] = []

    image_array = np.array(image)

    for piece_id in np.unique(labels):
        mask = labels == piece_id

        y_coords, x_coords = np.nonzero(mask)

        left = int(x_coords.min())
        top = int(y_coords.min())
        right = int(x_coords.max()) + 1
        bottom = int(y_coords.max()) + 1

        cropped_image = image_array[top:bottom, left:right].copy()
        cropped_mask = mask[top:bottom, left:right]

        cropped_image[~cropped_mask, 3] = 0

        piece_image = Image.fromarray(
            cropped_image,
            mode="RGBA",
        )

        pieces.append(
            Piece(
                id=int(piece_id),
                image=piece_image,
                bbox=(left, top, right, bottom),
            )
        )

    return pieces

