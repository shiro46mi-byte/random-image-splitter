"""
画像の保存処理を担当するモジュール。
"""

from pathlib import Path

from PIL import Image


def save_image(
    image: Image.Image,
    output_path: str,
) -> None:
    """
    画像を指定されたパスへ保存する。

    保存先ディレクトリが存在しない場合は自動的に作成する。

    Args:
        image:
            保存する画像。

        output_path:
            保存先ファイルパス。
    """
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(path)