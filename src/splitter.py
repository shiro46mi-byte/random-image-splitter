"""
Voronoi分割を担当するモジュール。
"""

from PIL import Image
import numpy as np

from src.piece import Piece


def create_seeds(
    width: int,
    height: int,
    piece_count: int,
    shape_variation: int,
) -> np.ndarray:
    """
    Voronoi分割用のシード座標を生成する。

    Args:
        width:
            画像幅。

        height:
            画像高さ。

        piece_count:
            分割数。

        shape_variation:
            形状ばらつき (0～100)。

    Returns:
        shape=(piece_count, 2) の座標配列。
    """
    rng = np.random.default_rng()

    cols = int(np.ceil(np.sqrt(piece_count)))
    rows = int(np.ceil(piece_count / cols))

    cell_width = width / cols
    cell_height = height / rows

    variation_ratio = shape_variation / 100.0

    seeds: list[tuple[int, int]] = []

    for row in range(rows):
        for col in range(cols):

            if len(seeds) >= piece_count:
                break

            center_x = (col + 0.5) * cell_width
            center_y = (row + 0.5) * cell_height

            max_dx = (cell_width / 2) * variation_ratio
            max_dy = (cell_height / 2) * variation_ratio

            x = center_x + rng.uniform(
                -max_dx,
                max_dx,
            )

            y = center_y + rng.uniform(
                -max_dy,
                max_dy,
            )

            x = np.clip(
                x,
                0,
                width - 1,
            )

            y = np.clip(
                y,
                0,
                height - 1,
            )

            seeds.append(
                (
                    int(x),
                    int(y),
                )
            )

    return np.asarray(
        seeds,
        dtype=np.int32,
    )


def create_voronoi_map(
    image: Image.Image,
    piece_count: int,
    shape_variation: int,
) -> np.ndarray:
    """
    画像サイズに対するVoronoiラベルマップを生成する。

    Args:
        image:
            元画像

        piece_count:
            分割数

        shape_variation:
            形状ばらつき(0～100)

    Returns:
        shape=(height, width) の整数配列
    """
    if piece_count <= 0:
        raise ValueError(
            "piece_countは1以上で指定してください。"
        )

    if not (0 <= shape_variation <= 100):
        raise ValueError(
            "shape_variationは0～100で指定してください。"
        )

    width, height = image.size

    seeds = create_seeds(
        width=width,
        height=height,
        piece_count=piece_count,
        shape_variation=shape_variation,
    )

    yy, xx = np.ogrid[:height, :width]

    distances = (
        (xx[:, :, np.newaxis] - seeds[:, 0]) ** 2
        + (yy[:, :, np.newaxis] - seeds[:, 1]) ** 2
    )

    labels = np.argmin(
        distances,
        axis=-1,
    )

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

        cropped_image = image_array[
            top:bottom,
            left:right,
        ].copy()

        cropped_mask = mask[
            top:bottom,
            left:right,
        ]

        cropped_image[
            ~cropped_mask,
            3,
        ] = 0

        piece_image = Image.fromarray(
            cropped_image,
            mode="RGBA",
        )

        pieces.append(
            Piece(
                id=int(piece_id),
                image=piece_image,
                bbox=(
                    left,
                    top,
                    right,
                    bottom,
                ),
            )
        )

    return pieces