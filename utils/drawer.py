import cv2


def draw_bbox(image, x, y, width, height, line_color=(0, 0, 255), line_width=2):
    """
    画像にbboxを描画(複数個のbboxを付与する場合、複数回この関数を呼び出すこと)
    Args:
        image[np.ndarray]: bboxを描画する画像
        x[int]: bboxの左上の座標
        y[int]: bboxの右上の座標
        height[int]: bboxの高さ
        width[int]: bboxの幅
        line_color[tuple[int]]: bboxを描画する色をGBR形式(≠RGB)で指定
        line_width[int]: bboxの線の太さを指定
    Returns:
        image_with_bbox[np.ndarray(画像の幅x高さx3)]: bboxを描画した画像
    """
    image_with_bbox = cv2.rectangle(image, (x, y), (x + width, y + height), line_color, line_width)
    return image_with_bbox


def draw_marker(image, x, y, color=(255, 0, 0), marker_size=20):
    return cv2.drawMarker(image, (x, y), color, markerSize=marker_size)