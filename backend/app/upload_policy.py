"""Bounded JPEG/MPO/PNG header policy before OpenCV allocates decoded pixels."""
from io import BytesIO
import warnings
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_582_912  # 12 MiPixels: includes ordinary 4032x3024 photos
MAX_IMAGE_DIMENSION = 8192


MAX_HEADER_PARTS = 256
MAX_HEADER_BYTES = 256 * 1024


def _bound_metadata(data):
    """Bound Pillow's pre-pixel metadata walk without inflating or copying chunks."""
    def refuse():
        raise HTTPException(status_code=422, detail="invalid or excessive image metadata")

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        offset = 8
        for _ in range(MAX_HEADER_PARTS):
            if offset + 12 > len(data) or offset + 12 > MAX_HEADER_BYTES:
                refuse()
            size = int.from_bytes(data[offset:offset + 4], "big")
            kind = data[offset + 4:offset + 8]
            end = offset + 12 + size
            if end > len(data):
                refuse()
            if kind == b"IDAT":
                return  # compressed pixels are outside the header walk
            if end > MAX_HEADER_BYTES:
                refuse()
            offset = end
        refuse()
    elif data.startswith(b"\xff\xd8"):
        offset = 2
        for _ in range(MAX_HEADER_PARTS):
            if offset + 2 > len(data) or offset + 2 > MAX_HEADER_BYTES or data[offset] != 255:
                refuse()
            offset += 1
            # Fill bytes count against the byte budget, even without a marker.
            while offset < len(data) and data[offset] == 255:
                offset += 1
                if offset >= MAX_HEADER_BYTES:
                    refuse()
            if offset >= len(data):
                refuse()
            kind = data[offset]; offset += 1
            if kind == 0xD9:
                refuse()  # EOI before SOS is malformed; Pillow may scan past it.
            if kind == 0xDA:
                return  # start of entropy-coded pixels; Pillow stops here too
            if kind == 0 or kind == 0xD8:
                refuse()
            if kind == 0x01 or 0xD0 <= kind <= 0xD7:
                continue
            if offset + 2 > len(data):
                refuse()
            size = int.from_bytes(data[offset:offset + 2], "big")
            if size < 2 or offset + size > len(data) or offset + size > MAX_HEADER_BYTES:
                refuse()
            offset += size
        refuse()
    else:
        raise HTTPException(status_code=422, detail="unsupported image format (JPEG/MPO/PNG only)")


def check_image_header(data):
    _bound_metadata(data)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as header:
                if header.format not in ("JPEG", "MPO", "PNG"):
                    raise HTTPException(status_code=422, detail="unsupported image format (JPEG/MPO/PNG only)")
                width, height = header.size  # no load(), decode or image conversion here
                if width <= 0 or height <= 0:
                    raise HTTPException(status_code=422, detail="invalid image dimensions")
                if max(width, height) > MAX_IMAGE_DIMENSION or width * height > MAX_IMAGE_PIXELS:
                    raise HTTPException(status_code=413, detail="decoded image exceeds pixel/dimension limit")
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise HTTPException(status_code=413, detail="decoded image exceeds pixel/dimension limit")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise HTTPException(status_code=422, detail="undecodable image (not JPEG/PNG)")
