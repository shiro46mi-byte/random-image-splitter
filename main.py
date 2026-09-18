"""
アプリケーションのエントリーポイント。

画像ファイルのパスを受け取り、
画像読み込み処理を実行する。
"""

import sys

import numpy as np

from src.image_loader import ImageLoadError, load_image
from src.splitter import create_voronoi_map, extract_pieces


PIECE_COUNT = 20


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

    labels = create_voronoi_map(
        image=image,
        piece_count=PIECE_COUNT,
    )

    print("\n=== Voronoi Map 情報 ===")
    print(f"shape: {labels.shape}")
    print(f"dtype: {labels.dtype}")
    print(f"min label: {labels.min()}")
    print(f"max label: {labels.max()}")
    print(f"unique labels: {len(np.unique(labels))}")

    pieces = extract_pieces(
        image=image,
        labels=labels,
    )

    print("\n=== Piece 情報 ===")
    print(f"生成した画像片数: {len(pieces)}")

    for piece in pieces:
        print(
            f"id={piece.id}, "
            f"size={piece.image.size}, "
            f"mode={piece.image.mode},"
            f"bbox={piece.bbox}"
        )

if __name__ == "__main__":
    main()