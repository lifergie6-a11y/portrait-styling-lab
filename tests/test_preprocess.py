from portrait_styling.preprocess import compute_portrait_crop_box


def test_portrait_crop_preserves_headroom_and_bounds():
    box = compute_portrait_crop_box(
        img_w=1200,
        img_h=1600,
        face_bbox=(450.0, 500.0, 750.0, 820.0),
        target_aspect=3.0 / 4.0,
    )
    left, top, right, bottom = box
    assert 0 <= left < right <= 1200
    assert 0 <= top < bottom <= 1600
    # Generous headroom above face top (y1=500) and shoulders below chin (y2=820)
    assert top < 300
    assert bottom > 1100
    aspect = (right - left) / (bottom - top)
    assert abs(aspect - 0.75) < 0.03


def test_heuristic_crop_without_face_bbox():
    box = compute_portrait_crop_box(img_w=1600, img_h=1200, face_bbox=None, target_aspect=0.75)
    left, top, right, bottom = box
    assert (right - left) == 900
    assert (bottom - top) == 1200
    assert left == 350 and top == 0
