"""
アプリケーションのエントリーポイント。

画像ファイルのパスを受け取り、
画像読み込み処理を実行する。
"""

from __future__ import annotations

import sys

from src.image_loader import ImageLoadError, load_image


def get_image_path() -> str:
    """
    コマンドライン引数または標準入力から画像パスを取得する。

    Returns:
        画像ファイルパス
    """
    if len(sys.argv) > 1:
        return sys.argv[1]

    return input("画像ファイルパスを入力してください: ").strip()


def main() -> None:
    """メイン処理。"""
    image_path = get_image_path()

    try:
        image = load_image(image_path)
    except ImageLoadError as exc:
        print(f"エラー: {exc}")
        sys.exit(1)

    width, height = image.size

    print(
        f"画像を読み込みました: {image_path} "
        f"(サイズ: {width}x{height})"
    )


if __name__ == "__main__":
    main()