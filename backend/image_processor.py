"""
Image processing utilities for recruitment poster uploads.
Validates file formats, sizes, checks image integrity, and performs preprocessing
for OCR text extraction.
"""

import os
import uuid
from typing import Tuple, Optional
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps, ImageFilter
from config import Config


def is_allowed_file(filename: str) -> bool:
    """Verify that the file has an allowed image extension."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in Config.ALLOWED_EXTENSIONS


def validate_image_file(file_storage) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded file: check filename, extension, size, and image integrity.
    Returns (is_valid, error_message).
    """
    if not file_storage or file_storage.filename == "":
        return False, "No file selected for upload."

    if not is_allowed_file(file_storage.filename):
        allowed = ", ".join(Config.ALLOWED_EXTENSIONS).upper()
        return False, f"Unsupported file type. Only {allowed} images are allowed."

    # Check file size (seek to end then back to 0)
    file_storage.seek(0, os.SEEK_END)
    size_bytes = file_storage.tell()
    file_storage.seek(0)

    if size_bytes > Config.MAX_CONTENT_LENGTH:
        max_mb = Config.MAX_CONTENT_LENGTH // (1024 * 1024)
        return False, f"File size exceeds the {max_mb} MB limit."

    if size_bytes == 0:
        return False, "Uploaded file is empty."

    # Validate that it is a genuine, uncorrupted image
    try:
        img = Image.open(file_storage)
        img.verify()
        file_storage.seek(0)
    except Exception as e:
        return False, f"Invalid or corrupted image file: {str(e)}"

    return True, None


def save_uploaded_image(file_storage) -> Tuple[str, str, int]:
    """
    Save the uploaded image to the static uploads folder with a unique filename.
    Returns (relative_url_path, absolute_file_path, file_size_bytes).
    """
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    orig_filename = secure_filename(file_storage.filename)
    name, ext = os.path.splitext(orig_filename)
    unique_name = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
    absolute_path = os.path.join(Config.UPLOAD_FOLDER, unique_name)

    file_storage.save(absolute_path)
    file_size = os.path.getsize(absolute_path)
    relative_url = f"/static/uploads/{unique_name}"

    return relative_url, absolute_path, file_size


def preprocess_image_for_ocr(image_path: str) -> Image.Image:
    """
    Normalize orientation, upscale small posters, reduce noise, and increase text
    contrast before passing the image to Tesseract.
    """
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    width, height = image.size
    largest_dimension = max(height, width)
    if largest_dimension < 1800:
        scale = min(3.0, 1800 / largest_dimension)
        image = image.resize(
            (int(width * scale), int(height * scale)),
            Image.Resampling.LANCZOS
        )

    grayscale = ImageOps.grayscale(image)
    normalized = ImageOps.autocontrast(grayscale, cutoff=2)
    return normalized.filter(ImageFilter.SHARPEN)
