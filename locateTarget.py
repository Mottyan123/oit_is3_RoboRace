"""カメラ画像から色付きの目印を見つける。座標の単位に注意すること。"""
import cv2
import numpy as np

# OpenCVのH（色相）は0～179。H/S/Vの順に下限と上限を指定する。
# contextの5地点の実機画像に合わせ、赤・緑・青を調整した。
HSV_RANGES = {
    # 赤いひし形は彩度が高い。色相の両端を拾いつつ、床の反射などを除く。
    "flag": [((0, 175, 75), (10, 255, 255)),
             ((168, 175, 75), (179, 255, 255))],
    # 円柱の影は暗いので明度の下限は低くする。純緑寄りの帯は鮮やかな画素に絞る。
    "obstacle": [((43, 105, 48), (59, 255, 255)),
                 ((60, 225, 48), (67, 255, 255))],
    # コーンの色相は主に111～114。紫寄りの青は彩度が高い場合だけ残す。
    "cone": [((104, 150, 50), (116, 255, 255)),
             ((117, 220, 50), (122, 255, 255))],
    # 提供画像には黄色い相手機体が写っていないため、従来の範囲を使う。
    "enemy": [((18, 100, 60), (34, 255, 255))],
}
MIN_AREA_RATIO = 0.0003  # 小さい色ノイズを除く。画像面積に対する割合。
# 停止画像の青い先端は、通常の面積閾値や平滑化で消える。見切れ専用の条件で拾う。
CONE_EDGE_HSV = ((104, 130, 40), (122, 255, 255))  # 端に残る青のHSV範囲。
CONE_EDGE_BAND_X = 0.12  # 左右の端から画像幅の12%だけを調べる。
CONE_EDGE_START_Y = 0.64  # 衣類など画面上部の青を避け、画像高の64%より下だけを調べる。
CONE_EDGE_TOUCH_X = 0.005  # 画面の外端から許す隙間。320画素幅では約2画素。
CONE_EDGE_MIN_AREA_RATIO = 0.00007  # 青い断片の最小面積。320×240では6画素以上。


def preprocess(imInput):
    # HSVの赤は両端に分かれるため、色相を平均する前にBGR画像で平滑化する。
    imSmooth = cv2.GaussianBlur(imInput, (3, 3), 0)
    return cv2.cvtColor(imSmooth, cv2.COLOR_BGR2HSV)


def _color_mask(imInputHSV, kind):
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
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(8, width * height * MIN_AREA_RATIO):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        fill = area / (w * h)
        aspect = w / h
        if kind == "flag":
            # ひし形は外接矩形の中を完全には埋めない。厳密な正方形は要求しない。
            polygon = cv2.approxPolyDP(
                contour, 0.04 * cv2.arcLength(contour, True), True)
            if not (0.4 <= aspect <= 2.3 and 0.25 <= fill <= 0.85
                    and 4 <= len(polygon) <= 8):
                continue
        elif kind == "enemy" and (aspect < 1.5 or fill < 0.5):
            continue  # 相手の目印は黄色い横長方形。
        elif kind == "obstacle" and aspect > 0.8 and y + h < height * 0.9:
            continue  # 横長の緑ノイズを除く。接近して画面下端に切れる場合は残す。
        elif kind == "cone" and aspect > 2.0:
            continue
        # 制御用は0～1に正規化。解像度を変えても中心は常に0.5になる。
        objects.append({
            "x": (x + w / 2) / width, "y": (y + h / 2) / height,
            "left": x / width, "right": (x + w) / width,
            "bottom": (y + h) / height, "height": h / height,
            "area": area / (width * height), "confidence": min(1.0, fill),
            "bbox": (x, y, w, h), "image_width": width,
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
    mask = _color_mask(imInputHSV, "flag")
    return _legacy_result(_find_objects(mask, "flag"), mask)


def locateEnemy(imInputHSV):
    # 元の未実装部分。緑の円柱ではなく、黄色い横長方形を探す。
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
    strips = (("left", imInput[start_y:, :band]),
              ("right", imInput[start_y:, -band:][:, ::-1]))
    for side, strip in strips:
        # 右側は反転し、どちらも座標0を画像の外端とする。細い断片を残すため平滑化しない。
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, *CONE_EDGE_HSV)
        _, _, components, _ = cv2.connectedComponentsWithStats(mask)
        if any(x <= tolerance and x + w < band and h >= 2 and area >= min_pixels
               for x, _, w, h, area in components[1:]):
            sides.append(side)
    return tuple(sides)


def observe(imInput):
    """1枚のBGR画像をまとめて解析する。既存の3関数の引数は変更しない。"""
    hsv = preprocess(imInput)
    masks = {kind: _color_mask(hsv, kind) for kind in HSV_RANGES}
    found = {kind: _find_objects(mask, kind) for kind, mask in masks.items()}
    gray = cv2.cvtColor(imInput, cv2.COLOR_BGR2GRAY)
    # 相手や目印そのものの動きを、自車の移動と誤解しにくくするため除外する。
    colored = np.zeros(gray.shape, dtype=np.uint8)
    for mask in masks.values():
        colored = cv2.bitwise_or(colored, mask)
    colored = cv2.dilate(colored, np.ones((7, 7), np.uint8))
    return {
        "valid": bool(gray.std() > 2 and gray.mean() > 3),
        "width": imInput.shape[1], "height": imInput.shape[0],
        "flag": found["flag"][0] if found["flag"] else None,
        "obstacles": found["obstacle"], "cones": found["cone"],
        "cone_edges": _cone_edge_fragments(imInput),
        "enemies": found["enemy"], "masks": masks,
        "motion_gray": cv2.resize(gray, (64, 48)).astype(np.float32) / 255,
        "motion_mask": cv2.resize(colored, (64, 48),
                                  interpolation=cv2.INTER_NEAREST) == 0,
    }
