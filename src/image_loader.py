"""
画像ファイルの読み込みを担当するモジュール。

対応形式の検証、ファイル存在確認、画像読み込み、
RGBA形式への変換を行う。
"""

from pathlib import Path
from PIL import Image


SUPPORTED_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")


class ImageLoadError(Exception):
    """画像読み込みに関する例外。"""

    pass

def test_load_png():
    """PNG画像の読み込みテスト"""
    image = load_image("sample/test.png")
    assert image.mode == "RGBA"


def load_image(path: str) -> Image.Image:
    """
    指定された画像ファイルを読み込み、RGBA形式へ変換して返す。

    Args:
        path: 画像ファイルパス

    Returns:
        RGBA形式のPIL.Image.Image

    Raises:
        ImageLoadError:
            - ファイルが存在しない場合
            - 未対応形式の場合
            - 読み込みに失敗した場合
    """
    image_path = Path(path)

    ext = image_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"対応していない画像形式です(PNG/JPG/JPEGのみ対応): {ext}"
        )

    if not image_path.exists():
        raise ImageLoadError(
            f"画像ファイルが見つかりません: {path}"
        )

    try:
        with Image.open(image_path) as image:
            return image.convert("RGBA")
    except Exception as exc:
        raise ImageLoadError(
            f"画像の読み込みに失敗しました: {path}"
        ) from exc