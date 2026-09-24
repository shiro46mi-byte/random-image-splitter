"""
画像片の配置・合成を担当するモジュール。
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.piece import Piece


MIN_ROTATION_ANGLE = 0.0
MAX_ROTATION_ANGLE = 360.0
MAX_PLACEMENT_ATTEMPTS = 1000
ALPHA_THRESHOLD = 0


class PlacementError(Exception):
    """画像片の配置に失敗した場合の例外。"""
    pass


@dataclass
class PlacedPiece:
    """
    キャンバス上に配置された画像片を表すクラス。

    Attributes:
        piece:
            配置元の画像片。

        image:
            回転後のRGBA画像。

        x:
            キャンバス上のX座標。

        y:
            キャンバス上のY座標。

        angle:
            画像片に適用した回転角度。
    """

    piece: Piece
    image: Image.Image
    x: int
    y: int
    angle: float


def arrange_pieces(
    pieces: list[Piece],
    canvas_width: int,
    canvas_height: int,
) -> list[PlacedPiece]:
    """
    画像片をランダムに回転し、重ならないようキャンバスへ配置する。

    Args:
        pieces:
            配置する画像片のリスト。

        canvas_width:
            配置先キャンバスの幅。

        canvas_height:
            配置先キャンバスの高さ。

    Returns:
        配置情報を持つPlacedPieceのリスト。

    Raises:
        ValueError:
            キャンバスのサイズが不正な場合。

        PlacementError:
            規定回数試行しても画像片を配置できなかった場合。
    """
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError(
            "キャンバスの幅と高さは1以上で指定してください。"
        )

    rng = np.random.default_rng()

    occupied = np.zeros(
        (canvas_height, canvas_width),
        dtype=bool,
    )

    placed_pieces: list[PlacedPiece] = []

    for piece in pieces:
        # 7.1 Pieceをランダム回転
        angle = float(
            rng.uniform(
                MIN_ROTATION_ANGLE,
                MAX_ROTATION_ANGLE,
            )
        )

        rotated_image = piece.image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
        )

        rotated_width, rotated_height = rotated_image.size

        # 回転後の画像自体がキャンバスより大きければ
        # 何度再試行しても配置できない。
        if (
            rotated_width > canvas_width
            or rotated_height > canvas_height
        ):
            raise PlacementError(
                f"画像片ID {piece.id} の回転後サイズ"
                f"({rotated_width}x{rotated_height})が"
                f"キャンバスサイズ"
                f"({canvas_width}x{canvas_height})を超えています。"
            )

        # 7.2 回転後アルファマスク生成
        alpha = np.asarray(
            rotated_image.getchannel("A")
        )

        piece_mask = alpha > ALPHA_THRESHOLD

        placed = False

        # 7.3～7.6 配置候補生成・衝突判定・再試行
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            max_x = canvas_width - rotated_width
            max_y = canvas_height - rotated_height

            x = int(rng.integers(0, max_x + 1))
            y = int(rng.integers(0, max_y + 1))

            # 7.4 キャンバス境界判定
            if (
                x < 0
                or y < 0
                or x + rotated_width > canvas_width
                or y + rotated_height > canvas_height
            ):
                continue

            occupied_region = occupied[
                y:y + rotated_height,
                x:x + rotated_width,
            ]

            # 7.5 実形状による衝突判定
            if np.any(occupied_region & piece_mask):
                continue

            # 衝突していないので占有領域を登録する。
            occupied_region |= piece_mask

            # 7.7 PlacedPiece生成
            placed_pieces.append(
                PlacedPiece(
                    piece=piece,
                    image=rotated_image,
                    x=x,
                    y=y,
                    angle=angle,
                )
            )

            placed = True
            break

        # 7.6 最大試行回数到達
        if not placed:
            raise PlacementError(
                f"画像片ID {piece.id} を配置できませんでした。"
                f"最大試行回数: {MAX_PLACEMENT_ATTEMPTS}"
            )

    return placed_pieces

def compose_image(
    placed_pieces: list[PlacedPiece],
    canvas_width: int,
    canvas_height: int,
) -> Image.Image:
    """
    配置済みの画像片を透明キャンバス上に合成する。

    Args:
        placed_pieces:
            配置済み画像片のリスト。

        canvas_width:
            キャンバスの幅。

        canvas_height:
            キャンバスの高さ。

    Returns:
        合成後のRGBA画像。

    Raises:
        ValueError:
            キャンバスのサイズが不正な場合。
    """
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError(
            "キャンバスの幅と高さは1以上で指定してください。"
        )

    canvas = Image.new(
        "RGBA",
        (canvas_width, canvas_height),
        (0, 0, 0, 0),
    )

    for placed_piece in placed_pieces:
        canvas.alpha_composite(
            placed_piece.image,
            dest=(
                placed_piece.x,
                placed_piece.y,
            ),
        )

    return canvas