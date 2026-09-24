"""
アプリケーションのエントリーポイント。

画像ファイルのパスを受け取り、
画像読み込み処理を実行する。
"""

import sys

import numpy as np

from src.compositor import arrange_pieces, compose_image
from src.image_loader import ImageLoadError, load_image
from src.splitter import create_voronoi_map, extract_pieces


PIECE_COUNT = 20
CANVAS_SCALE = 2 


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

    canvas_width = width * CANVAS_SCALE
    canvas_height = height * CANVAS_SCALE

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
            f"mode={piece.image.mode}, "
            f"bbox={piece.bbox}"
        )

    placed_pieces = arrange_pieces(
        pieces=pieces,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )
    

    print("\n=== 配置情報 ===")

    for placed_piece in placed_pieces:
        print(
            f"id={placed_piece.piece.id}, "
            f"x={placed_piece.x}, "
            f"y={placed_piece.y}, "
            f"angle={placed_piece.angle:.2f}, "
            f"size={placed_piece.image.size}"
        )   

    for placed_piece in placed_pieces:
        piece_width, piece_height = placed_piece.image.size

        assert placed_piece.x >= 0
        assert placed_piece.y >= 0
        assert placed_piece.x + piece_width <= canvas_width
        assert placed_piece.y + piece_height <= canvas_height
    
    print("\nすべての画像片がキャンバス内に配置されています。")

    composed_image = compose_image(
        placed_pieces=placed_pieces,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )

    print("\n=== 合成画像情報 ===")
    print(f"size: {composed_image.size}")
    print(f"mode: {composed_image.mode}")

    composed_image.save("output/composed_image.png")
    print("\n合成画像を output/composed_image.png に保存しました。")

if __name__ == "__main__":
    main()