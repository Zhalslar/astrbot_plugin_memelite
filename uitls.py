
import io
from PIL import Image


def compress_image(image: bytes, max_size: int = 512) -> bytes | None:
    """压缩静态图片或GIF到max_size大小"""
    try:
        img = Image.open(io.BytesIO(image))
        output = io.BytesIO()

        if img.format == "GIF":
            return
        else:
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            img.save(output, format=img.format)

        return output.getvalue()

    except Exception as e:
        raise ValueError(f"图片压缩失败: {e}")


