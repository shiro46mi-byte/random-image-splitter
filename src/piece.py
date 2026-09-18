"""
分割された画像片の管理を担当するモジュール。
"""

from dataclasses import dataclass

from PIL import Image


@dataclass
class Piece:
    """
    分割された1つの画像片を表すクラス。

    Attributes:
        id:
            画像片を識別するID。

        image:
            透明領域を含むRGBA形式の画像片。

        bbox:
            元画像上での画像片の領域。
            (left, top, right, bottom) の形式。
    """

    id: int
    image: Image.Image
    bbox: tuple[int, int, int, int]