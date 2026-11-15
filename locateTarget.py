"""カメラ画像から色付きの目印を見つける。座標の単位に注意すること。"""
import cv2
import numpy as np

# OpenCVのH（色相）は0～179。照明で変わるので、実機画像で調整する。
HSV_RANGES = {
    "flag": [((0, 100, 50), (10, 255, 255)),
             ((165, 100, 50), (179, 255, 255))],
    "obstacle": [((35, 80, 45), (85, 255, 255))],
    "cone": [((90, 90, 45), (135, 255, 255))],
    "enemy": [((18, 100, 60), (34, 255, 255))],
}
MIN_AREA_RATIO = 0.0003  # 小さい色ノイズを除く。画像面積に対する割合。


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
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
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
        "enemies": found["enemy"], "masks": masks,
        "motion_gray": cv2.resize(gray, (64, 48)).astype(np.float32) / 255,
        "motion_mask": cv2.resize(colored, (64, 48),
                                  interpolation=cv2.INTER_NEAREST) == 0,
    }
