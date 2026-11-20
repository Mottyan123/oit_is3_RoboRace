"""画像からコース上の物体を見つけ、位置や大きさを返す。"""

import cv2
import numpy as np

# HSVは色相（H）、彩度（S）、明度（V）。OpenCVではHが0～179、SとVが0～255。
# (下限, 上限) の組を並べ、どれかに当てはまる画素を残す。
HSV_RANGES = {
    # 赤は色相の両端にまたがる。彩度の高い画素を選び、床の反射を減らす。
    "flag": [
        ((0, 175, 75), (10, 255, 255)),
        ((168, 175, 75), (179, 255, 255)),
    ],
    # 緑の円柱は影でも拾う。青寄りの緑は彩度が高い場合に絞る。
    "obstacle": [
        ((43, 105, 48), (59, 255, 255)),
        ((60, 225, 48), (67, 255, 255)),
    ],
    # コーンの色相は主に111～114。紫寄りの青は彩度が高い場合だけ残す。
    "cone": [
        ((104, 150, 50), (116, 255, 255)),
        ((117, 220, 50), (122, 255, 255)),
    ],
    # 相手機体の前面にある黄色い目印。
    "enemy": [((18, 100, 60), (34, 255, 255))],
}

MIN_AREA_RATIO = 0.0003  # 小さい色ノイズを除く。画像面積に対する割合。

# 画面端でコーンの一部だけが見えるときは、細い断片も拾う。
CONE_EDGE_HSV = ((104, 130, 40), (122, 255, 255))  # 端に残る青のHSV範囲。
CONE_EDGE_BAND_X = 0.12  # 左右の端から画像幅の12%だけを調べる。
CONE_EDGE_START_Y = 0.64  # 衣類など画面上部の青を避け、画像高の64%より下だけを調べる。
CONE_EDGE_TOUCH_X = 0.005  # 画面の外端から許す隙間。320画素幅では約2画素。
CONE_EDGE_MIN_AREA_RATIO = 0.00007  # 青い断片の最小面積。320×240では6画素以上。

# ゴール調整画像1.pngでは、黒い台座が画面下部の横長の塊として映る。
GOAL_BASE_HSV_MAX = (179, 145, 85)  # 色相は問わず、彩度145・明度85以下を黒の候補にする。
GOAL_BASE_FLOOR_Y = 0.60  # 画像高の60%より下にある台座を探す。背景の暗い物を減らす。
GOAL_BASE_MIN_WIDTH = 0.08  # 画像幅の8%未満の小さい影や脚を除く。
GOAL_BASE_ASPECT = (1.8, 7.0)  # 台座候補として残す横幅÷高さの範囲。
GOAL_BASE_MIN_FILL = 0.55  # 外接矩形の55%以上を輪郭が占めるものを残し、細いケーブルなどを除く。


def preprocess(imInput):
    """BGR画像の細かいノイズを抑え、色判定に使うHSV画像へ変換する。"""
    # HSVへ変換してから平均すると、色相の両端にある赤が別の色になることがある。
    smoothed = cv2.GaussianBlur(imInput, (3, 3), 0)
    return cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)


def _color_mask(imInputHSV, kind):
    """指定色に当てはまる画素を255、それ以外を0にした画像を作る。"""
    mask = np.zeros(imInputHSV.shape[:2], dtype=np.uint8)
    for lower, upper in HSV_RANGES[kind]:
        part = cv2.inRange(imInputHSV, np.array(lower), np.array(upper))
        mask = cv2.bitwise_or(mask, part)
    kernel = np.ones((3, 3), dtype=np.uint8)
    # openingで点ノイズを除き、closingで領域内の小さな穴を埋める。
    opening_kernel = kernel
    if kind == "obstacle":
        # 緑の円柱は縦長。横方向に数画素しかない遠方の円柱を残し、
        # 横に伸びた緑ノイズは除く。高さ240画素では縦7画素を使う。
        vertical_pixels = max(5, round(mask.shape[0] * 7 / 240))
        opening_kernel = np.ones((vertical_pixels, 1), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _find_objects(mask, kind):
    """輪郭を面積・形で絞り込み、大きい候補から順に返す。"""
    height, width = mask.shape
    image_area = width * height
    min_area = max(8, image_area * MIN_AREA_RATIO)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        bottom = y + h
        fill = area / (w * h)
        aspect = w / h
        if kind == "flag":
            # ひし形を囲む長方形にどの程度色が入るか、輪郭にいくつ角があるかを見る。
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if not (
                0.4 <= aspect <= 2.3
                and 0.25 <= fill <= 0.85
                and 4 <= len(polygon) <= 8
            ):
                continue
        elif kind == "enemy" and (aspect < 1.5 or fill < 0.5):
            continue  # 相手の目印は黄色い横長方形。
        elif kind == "obstacle" and aspect > 0.8 and bottom < height * 0.9:
            continue  # 横長の緑ノイズを除く。接近して画面下端に切れる場合は残す。
        elif kind == "cone" and aspect > 2.0:
            continue
        # 制御用の座標は0～1。描画や従来の関数に使うbboxは画素単位で残す。
        objects.append({
            "x": (x + w / 2) / width,
            "y": (y + h / 2) / height,
            "left": x / width,
            "right": (x + w) / width,
            "bottom": bottom / height,
            "height": h / height,
            "area": area / image_area,
            "confidence": min(1.0, fill),
            "bbox": (x, y, w, h),
            "image_width": width,
            "image_height": height,
        })
    return sorted(objects, key=lambda obj: obj["area"], reverse=True)


def _legacy_result(objects, mask):
    # 既存関数の戻り値は (水平画素位置, 垂直画素位置, 縦の画素数), マスク。
    if not objects:
        return (-1, -1, -1), mask
    x, y, w, h = objects[0]["bbox"]
    return (int(x + w / 2), int(y + h / 2), h), mask


def locateFlag(imInputHSV):
    """赤いひし形の中心と高さ（画素単位）、赤のマスクを返す。"""
    mask = _color_mask(imInputHSV, "flag")
    return _legacy_result(_find_objects(mask, "flag"), mask)


def locateEnemy(imInputHSV):
    """相手の黄色い目印の中心と高さ（画素単位）、黄のマスクを返す。"""
    mask = _color_mask(imInputHSV, "enemy")
    return _legacy_result(_find_objects(mask, "enemy"), mask)


def _cone_edge_fragments(imInput):
    """画面端にわずかに残る青を、通常のコーン候補とは別に返す。"""
    height, width = imInput.shape[:2]
    band = max(1, round(width * CONE_EDGE_BAND_X))
    start_y = int(height * CONE_EDGE_START_Y)
    tolerance = round(width * CONE_EDGE_TOUCH_X)
    min_pixels = max(3, int(np.ceil(width * height * CONE_EDGE_MIN_AREA_RATIO)))
    sides = []
    # 右側の帯を反転し、左右とも帯のx=0を画像の外端にそろえる。
    strips = (
        ("left", imInput[start_y:, :band]),
        ("right", imInput[start_y:, -band:][:, ::-1]),
    )
    for side, strip in strips:
        # 細い断片を残すため、ここでは平滑化を省く。
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, *CONE_EDGE_HSV)
        _, _, components, _ = cv2.connectedComponentsWithStats(mask)
        # 先頭は背景。外端に触れ、帯の内側に収まる青い塊を探す。
        if any(
            x <= tolerance and x + w < band and h >= 2 and area >= min_pixels
            for x, _, w, h, area in components[1:]
        ):
            sides.append(side)
    return tuple(sides)


def _goal_bases(hsv):
    """画面下部で台座の形に近い暗い候補を集める。"""
    height, width = hsv.shape[:2]
    floor_y = int(height * GOAL_BASE_FLOOR_Y)
    mask = cv2.inRange(hsv, (0, 0, 0), GOAL_BASE_HSV_MAX)
    mask[:floor_y] = 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    bases = []
    for obj in _find_objects(mask, "goal_base"):
        x, y, w, h = obj["bbox"]
        # 探索範囲の上端や画像の左右端で切れた輪郭は、形を判定できない。
        if (
            y > floor_y
            and x > 0
            and x + w < width
            and w / width >= GOAL_BASE_MIN_WIDTH
            and GOAL_BASE_ASPECT[0] <= w / h <= GOAL_BASE_ASPECT[1]
            and obj["confidence"] >= GOAL_BASE_MIN_FILL
        ):
            bases.append(obj)
    return bases


def observe(imInput):
    """1枚のBGR画像から、物体候補と自車の移動判定用の画像を作る。"""
    hsv = preprocess(imInput)
    masks = {kind: _color_mask(hsv, kind) for kind in HSV_RANGES}
    found = {kind: _find_objects(mask, kind) for kind, mask in masks.items()}
    gray = cv2.cvtColor(imInput, cv2.COLOR_BGR2GRAY)
    # 自車の移動は前後の背景画像を比べて推定する。目印とその周囲は比較から外す。
    colored = np.zeros(gray.shape, dtype=np.uint8)
    for mask in masks.values():
        colored = cv2.bitwise_or(colored, mask)
    colored = cv2.dilate(colored, np.ones((7, 7), np.uint8))
    return {
        "valid": bool(gray.std() > 2 and gray.mean() > 3),
        "width": imInput.shape[1], "height": imInput.shape[0],
        "flag": found["flag"][0] if found["flag"] else None,
        "goal_bases": _goal_bases(hsv),
        "obstacles": found["obstacle"],
        "cones": found["cone"],
        "cone_edges": _cone_edge_fragments(imInput),
        "enemies": found["enemy"],
        "masks": masks,
        "motion_gray": cv2.resize(gray, (64, 48)).astype(np.float32) / 255,
        "motion_mask": cv2.resize(
            colored, (64, 48), interpolation=cv2.INTER_NEAREST
        ) == 0,
    }
