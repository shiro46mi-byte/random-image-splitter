"""
アプリケーションのエントリーポイント。

画像ファイルのパスを受け取り、
画像読み込み処理を実行する。
"""

import sys

import numpy as np

from src.compositor import (
    arrange_pieces,
    compose_image,
    validate_no_overlap,
)
from src.image_loader import ImageLoadError, load_image
from src.splitter import create_voronoi_map, extract_pieces
from src.image_saver import save_image

CANVAS_SCALE = 2 
MIN_PIECE_COUNT = 1
MAX_PIECE_COUNT = 100
MIN_ROTATION_ANGLE = 0
MAX_ROTATION_ANGLE_LIMIT = 360
MIN_SHAPE_VARIATION = 0
MAX_SHAPE_VARIATION = 100

def get_image_path() -> str:
    """
    コマンドライン引数または標準入力から画像パスを取得する。

    Returns:
        画像ファイルパス
    """
    if len(sys.argv) > 1:
        return sys.argv[1]

    return input("画像ファイルパスを入力してください: ").strip()

def get_piece_count() -> int:
    """
    コマンドライン引数または標準入力から
    分割数を取得する。

    Returns:
        分割数
    """
    try:
        if len(sys.argv) > 2:
            piece_count = int(sys.argv[2])
        else:
            piece_count = int(
                input("分割数を入力してください: ").strip()
            )

    except ValueError:
            raise ValueError(
                "分割数は整数で指定してください。"
            )

    if piece_count < MIN_PIECE_COUNT:
            raise ValueError(
                f"分割数は {MIN_PIECE_COUNT} 以上で指定してください。"
            )

    if piece_count > MAX_PIECE_COUNT:
            raise ValueError(
                f"分割数は {MAX_PIECE_COUNT} 以下で指定してください。"
            )

    return piece_count

def get_max_rotation_angle() -> float:
    """
    最大回転角を取得する。

    Returns:
        最大回転角
    """
    try:
        if len(sys.argv) > 3:
            angle = float(sys.argv[3])
        else:
            angle = float(
                input(
                    "最大回転角を入力してください (0～360): "
                ).strip()
            )
    except ValueError:
        raise ValueError(
            "最大回転角は整数で入力してください。"
        )

    if angle < MIN_ROTATION_ANGLE:
        raise ValueError(
            "最大回転角は0以上で指定してください。"
        )

    if angle > MAX_ROTATION_ANGLE_LIMIT:
        raise ValueError(
            "最大回転角は360以下で指定してください。"
        )

    return angle

def get_shape_variation() -> int:
    """
    形状の変化量を取得する。

    Returns:
        形状の変化量
    """
    try:
        if len(sys.argv) > 4:
            variation = int(sys.argv[4])
        else:
            variation = int(
                input(
                    "形状の変化量を入力してください (0～100): "
                ).strip()
            )
    except ValueError:
        raise ValueError(
            "形状の変化量は整数で入力してください。"
        )

    if variation < 0 or variation > 100:
        raise ValueError(
            "形状の変化量は0～100の範囲で指定してください。"
        )

    return variation   

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
            形状の変化量

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

def main() -> None:
    """メイン処理。"""
    image_path = get_image_path()


    try:
        image = load_image(image_path)
    except ImageLoadError as exc:
        print(f"エラー: {exc}")
        sys.exit(1)

    try:
        piece_count = get_piece_count()
    except ValueError as exc:
        print(f"エラー: {exc}")
        sys.exit(1)

    try:
        max_rotation_angle = get_max_rotation_angle()
    except ValueError as exc:
        print(f"エラー：{exc}")
        sys.exit(1)


    shape_variation = get_shape_variation()
    width, height = image.size

    canvas_width = width * CANVAS_SCALE
    canvas_height = height * CANVAS_SCALE

    print(
        f"画像を読み込みました: {image_path} "
        f"(サイズ: {width}x{height})"
    )
    print(f"分割数: {piece_count}")
    print(f"最大回転角：{max_rotation_angle}")

    labels = create_voronoi_map(
        image=image,
        piece_count=piece_count,
        shape_variation=shape_variation,
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
        max_rotation_angle=max_rotation_angle,
    )

    assert len(placed_pieces) == len(pieces)

    print(
        f"\nすべての画像片を配置しました: "
        f"{len(placed_pieces)}/{len(pieces)}"
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

    if not validate_no_overlap(
        placed_pieces=placed_pieces,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    ):
        raise RuntimeError(
            "配置された画像片に重なりが検出されました。"
        )

    print("\n画像片同士に重なりがないことを確認しました。")


    composed_image = compose_image(
        placed_pieces=placed_pieces,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )

    print("\n=== 合成画像情報 ===")
    print(f"size: {composed_image.size}")
    print(f"mode: {composed_image.mode}")

    output_path = "output/composed_image.png"

    save_image(
        image=composed_image,
        output_path=output_path,
    )

    print(
        f"\n合成画像を {output_path} に保存しました。"
    )

if __name__ == "__main__":
    main()