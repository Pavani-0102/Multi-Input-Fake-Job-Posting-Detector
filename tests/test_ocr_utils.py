import io

import pytest
from PIL import Image, ImageDraw, ImageFont

import ocr_utils
from ocr_utils import OCRError, extract_text_from_image, load_image, normalize_ocr_text, preprocess_image

tesseract_ok = ocr_utils.tesseract_status()[0]


def _png_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _text_image(lines, dark=False) -> Image.Image:
    bg, fg = ((20, 20, 20), (240, 240, 240)) if dark else ("white", "black")
    img = Image.new("RGB", (1400, 80 + 70 * len(lines)), bg)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=40)
    for i, line in enumerate(lines):
        draw.text((40, 40 + 70 * i), line, fill=fg, font=font)
    return img


def test_load_image_rejects_bad_input():
    with pytest.raises(OCRError):
        load_image(b"")
    with pytest.raises(OCRError):
        load_image(b"not an image")
    with pytest.raises(OCRError, match="Unsupported"):
        load_image(_png_bytes(Image.new("RGB", (200, 200), "white"), fmt="GIF"))
    with pytest.raises(OCRError, match="too small"):
        load_image(_png_bytes(Image.new("RGB", (10, 10), "white")))


def test_load_image_accepts_png_and_jpeg():
    assert load_image(_png_bytes(Image.new("RGB", (200, 200), "white"))).size == (200, 200)
    assert load_image(_png_bytes(Image.new("RGB", (200, 200), "white"), fmt="JPEG")).size == (200, 200)


def test_preprocess_inverts_dark_images_and_upscales():
    out = preprocess_image(Image.new("RGB", (400, 200), (10, 10, 10)))
    assert out.mode == "L" and out.width == 800
    assert out.getpixel((5, 5)) > 128  # inverted to light background


def test_normalize_ocr_text():
    raw = "Line  one \r\n\n\n  |||  \n---\nSecond\tline\x0c"
    assert normalize_ocr_text(raw) == "Line one\nSecond line"


@pytest.mark.skipif(not tesseract_ok, reason="Tesseract not installed")
@pytest.mark.parametrize("dark", [False, True])
def test_extract_text_roundtrip(dark):
    img = _text_image(["Senior Python Developer", "Registration fee required today"], dark=dark)
    result = extract_text_from_image(_png_bytes(img))
    lowered = result.text.lower()
    assert "python" in lowered and "registration" in lowered
    assert result.word_count >= 5


@pytest.mark.skipif(not tesseract_ok, reason="Tesseract not installed")
def test_blank_image_warns():
    result = extract_text_from_image(_png_bytes(Image.new("RGB", (600, 400), "white")))
    assert result.word_count == 0 and result.warnings
