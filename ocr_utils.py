"""OCR helpers: image validation, preprocessing and text extraction with Tesseract."""
from __future__ import annotations

import io
import os
import re
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from config import ALLOWED_IMAGE_FORMATS, MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS

try:
    import pytesseract
except ImportError:  # pragma: no cover - depends on the user's environment
    pytesseract = None

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

_WINDOWS_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


class OCRError(RuntimeError):
    """Raised for unreadable images or OCR engine problems (message is user-safe)."""


@dataclass
class OCRResult:
    text: str
    mean_confidence: float | None
    word_count: int
    warnings: list[str] = field(default_factory=list)


def configure_tesseract() -> None:
    """Point pytesseract at the Tesseract binary.

    Order: TESSERACT_CMD environment variable, then PATH, then default Windows paths.
    """
    if pytesseract is None:
        return
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd:
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return
    if shutil.which("tesseract"):
        return
    for candidate in _WINDOWS_TESSERACT_PATHS:
        if Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return


def tesseract_status() -> tuple[bool, str]:
    """Return (available, human-readable message)."""
    if pytesseract is None:
        return False, "The 'pytesseract' package is not installed (pip install -r requirements.txt)."
    configure_tesseract()
    try:
        version = pytesseract.get_tesseract_version()
    except Exception:  # pytesseract raises TesseractNotFoundError or OSError
        return False, (
            "Tesseract OCR is not installed or not found. Install it and/or set the "
            "TESSERACT_CMD environment variable (see README)."
        )
    return True, f"Tesseract {version}"


def load_image(data: bytes) -> Image.Image:
    """Validate and open uploaded image bytes (JPEG or PNG only)."""
    if not data:
        raise OCRError("The uploaded file is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise OCRError(f"Image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB. Please upload a smaller screenshot.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            probe = Image.open(io.BytesIO(data))
            fmt = probe.format
            probe.verify()
            img = Image.open(io.BytesIO(data))
            img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise OCRError("The file could not be read as an image. Upload a valid JPG, JPEG or PNG screenshot.") from None
    if fmt not in ALLOWED_IMAGE_FORMATS:
        raise OCRError(f"Unsupported image format '{fmt}'. Upload a JPG, JPEG or PNG file.")
    if img.width < 50 or img.height < 50:
        raise OCRError("The image is too small to contain readable text.")
    return img


def preprocess_image(img: Image.Image) -> Image.Image:
    """Improve OCR accuracy: fix orientation, grayscale, upscale small images,
    invert dark-mode screenshots and stretch contrast."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba).convert("RGB")
    gray = ImageOps.grayscale(img)

    if gray.width < 1200:
        scale = min(2.0, 4000 / gray.width)
        gray = gray.resize((int(gray.width * scale), int(gray.height * scale)), Image.LANCZOS)

    histogram = gray.histogram()
    total = sum(histogram) or 1
    mean_brightness = sum(i * c for i, c in enumerate(histogram)) / total
    if mean_brightness < 110:
        gray = ImageOps.invert(gray)

    return ImageOps.autocontrast(gray, cutoff=1)


def normalize_ocr_text(text: str) -> str:
    """Tidy raw OCR output: unify newlines, drop lines without letters/digits,
    collapse repeated blank lines and spaces."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line and re.search(r"[A-Za-z0-9]", line):
            lines.append(line)
    return "\n".join(lines).strip()


def extract_text_from_image(data: bytes, lang: str = "eng", psm: int = 3) -> OCRResult:
    """Run OCR on uploaded image bytes.

    Raises OCRError with a user-presentable message on any failure.
    """
    available, message = tesseract_status()
    if not available:
        raise OCRError(message)

    img = load_image(data)
    prepared = preprocess_image(img)
    config = f"--oem 3 --psm {int(psm)}"
    try:
        details = pytesseract.image_to_data(prepared, lang=lang, config=config, output_type=pytesseract.Output.DICT)
        raw_text = pytesseract.image_to_string(prepared, lang=lang, config=config)
    except pytesseract.TesseractError as exc:
        raise OCRError(f"Tesseract failed while reading the image: {str(exc).strip()[:200]}") from exc
    except RuntimeError as exc:  # e.g. Tesseract timed out
        raise OCRError(f"OCR failed: {str(exc)[:200]}") from exc

    text = normalize_ocr_text(raw_text)
    confidences = [
        float(c) for c, w in zip(details.get("conf", []), details.get("text", []))
        if str(w).strip() and float(c) >= 0
    ]
    mean_conf = sum(confidences) / len(confidences) if confidences else None
    word_count = len(text.split())

    warnings_list: list[str] = []
    if word_count == 0:
        warnings_list.append("No text was detected. Try a sharper, higher-resolution screenshot.")
    elif word_count < 15:
        warnings_list.append("Very little text was detected; the screenshot may be cropped or low quality.")
    if mean_conf is not None and mean_conf < 60:
        warnings_list.append(f"Low OCR confidence ({mean_conf:.0f}/100). Please review and correct the text below.")

    return OCRResult(text=text, mean_confidence=mean_conf, word_count=word_count, warnings=warnings_list)
