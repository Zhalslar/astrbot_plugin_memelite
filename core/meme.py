import asyncio
import io
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_package_version
from typing import Any, Literal, Protocol

from astrbot import logger
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.platform.astr_message_event import AstrMessageEvent

from .param import ParamsCollector


def _parse_version(version: str) -> tuple[int, int, int]:
    parts = [int(part) for part in re.findall(r"\d+", version)[:3]]
    major, minor, patch = (parts + [0, 0, 0])[:3]
    return major, minor, patch


def _resolve_version(module: Any) -> str:
    try:
        from meme_generator.version import __version__ as legacy_version  # type: ignore

        return str(legacy_version)
    except ImportError:
        pass

    get_version = getattr(module, "get_version", None)
    if callable(get_version):
        try:
            return str(get_version())
        except Exception:
            pass

    try:
        return get_package_version("meme_generator")
    except PackageNotFoundError:
        return "0.0.0"


IMPORT_ERROR: str | None = None


class MemeLike(Protocol):
    key: str
    params_type: Any

    def generate_preview(self) -> Any:
        """Generate a meme preview."""

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Generate a meme image."""


try:
    import meme_generator as meme_generator_module
    from meme_generator import get_memes

    __version__ = _resolve_version(meme_generator_module)
    MEME_GENERATOR_AVAILABLE = True
except ImportError as exc:
    meme_generator_module = None
    MEME_GENERATOR_AVAILABLE = False
    IMPORT_ERROR = str(exc)
    __version__ = "0.0.0"

    def get_memes(*args: Any, **kwargs: Any) -> list[Any]:
        return []


@dataclass
class LegacyMemeProperties:
    disabled: bool = False
    labels: list[Literal["new", "hot"]] = field(default_factory=list)


class MemeManager:
    is_py_version = _parse_version(__version__) < (0, 2, 0)

    def __init__(self, config: AstrBotConfig, collect: ParamsCollector):
        self.conf = config
        self.collect = collect
        self.memes: list[MemeLike] = []
        self.meme_keywords: list[str] = []

        self.render_meme_list_func: Callable[..., Any] | None = None
        self.check_resources_func: Callable[..., Any] | None = None
        self.run_sync: Callable[[Any], Callable[..., Any]] | None = None
        self.MemeImage: type[Any] | None = None
        self.MemePropertiesType: Any = LegacyMemeProperties
        self.MemeSortBy: Any = None

        if not MEME_GENERATOR_AVAILABLE:
            logger.error(f"meme-generator import failed: {IMPORT_ERROR}")
            return

        if self.is_py_version:
            from meme_generator.download import check_resources  # type: ignore
            from meme_generator.utils import render_meme_list, run_sync  # type: ignore

            self.render_meme_list_func = render_meme_list
            self.check_resources_func = check_resources
            self.run_sync = run_sync
            return

        from meme_generator import Image as MemeImage
        from meme_generator.tools import (
            MemeProperties as RustMemeProperties,
        )
        from meme_generator.tools import (
            MemeSortBy,
            render_meme_list,
        )

        try:
            from meme_generator.resources import check_resources
        except ImportError:
            from meme_generator.resources import (
                check_resources_in_background as check_resources,
            )

        self.render_meme_list_func = render_meme_list
        self.check_resources_func = check_resources
        self.MemeImage = MemeImage
        self.MemePropertiesType = RustMemeProperties
        self.MemeSortBy = MemeSortBy

    def _load_memes(self) -> None:
        try:
            self.memes = list(get_memes())
        except Exception as exc:
            logger.error(f"failed to load memes: {exc}")
            self.memes = []
            self.meme_keywords = []
            return

        self.meme_keywords = [
            keyword for meme in self.memes for keyword in self._get_keywords(meme)
        ]

    def _ensure_memes_loaded(self) -> bool:
        if not self.memes:
            self._load_memes()
        return bool(self.memes)

    @staticmethod
    def _get_info(meme: MemeLike) -> Any | None:
        return getattr(meme, "info", None)

    def _get_keywords(self, meme: MemeLike) -> list[str]:
        info = self._get_info(meme)
        if info is not None and hasattr(info, "keywords"):
            return list(info.keywords)
        return list(getattr(meme, "keywords", []))

    def _get_params(self, meme: MemeLike) -> Any:
        info = self._get_info(meme)
        if info is not None and hasattr(info, "params"):
            return info.params
        return meme.params_type

    def _get_tags(self, meme: MemeLike) -> list[str]:
        info = self._get_info(meme)
        tags = (
            info.tags
            if info is not None and hasattr(info, "tags")
            else getattr(meme, "tags", [])
        )
        return list(tags)

    @staticmethod
    def _unwrap_bytes(result: Any, action: str) -> bytes:
        if isinstance(result, io.BytesIO):
            return result.getvalue()
        if isinstance(result, (bytes, bytearray, memoryview)):
            return bytes(result)

        detail = None
        for attr in ("feedback", "error", "path"):
            value = getattr(result, attr, None)
            if value:
                detail = str(value)
                break

        error_message = detail or repr(result)
        raise RuntimeError(f"{action} failed: {error_message}")

    async def check_resources(self):
        if not MEME_GENERATOR_AVAILABLE or not self.check_resources_func:
            return

        if not self.conf["is_check_resources"]:
            self._load_memes()
            return

        logger.info("checking meme resources...")
        try:
            if self.is_py_version:
                await self.check_resources_func()
            else:
                await asyncio.to_thread(self.check_resources_func)
        except Exception as exc:
            logger.warning(f"resource check failed: {exc}")

        self._load_memes()

    def find_meme(self, keyword: str) -> MemeLike | None:
        if not self._ensure_memes_loaded():
            return None

        for meme in self.memes:
            keywords = self._get_keywords(meme)
            if keyword == meme.key or keyword in keywords:
                return meme

        return None

    def is_meme_keyword(self, meme_name: str) -> bool:
        if not self._ensure_memes_loaded():
            return False
        return meme_name in self.meme_keywords

    def match_meme_keyword(self, text: str, fuzzy_match: bool) -> str | None:
        if not self._ensure_memes_loaded():
            return None

        if fuzzy_match:
            return next(
                (keyword for keyword in self.meme_keywords if keyword in text), None
            )

        parts = text.split()
        first_word = parts[0] if parts else ""
        return next(
            (keyword for keyword in self.meme_keywords if keyword == first_word), None
        )

    async def render_meme_list_image(self) -> bytes | None:
        if not self._ensure_memes_loaded():
            return None

        if self.is_py_version:
            render_meme_list = self.render_meme_list_func
            if not render_meme_list:
                return None

            meme_list = [(meme, LegacyMemeProperties(labels=[])) for meme in self.memes]
            rendered = render_meme_list(
                meme_list=meme_list,
                text_template="{index}.{keywords}",
                add_category_icon=True,
            )
            return self._unwrap_bytes(rendered, "render meme list")

        render_meme_list = self.render_meme_list_func
        meme_properties_type = self.MemePropertiesType
        meme_sort_by = self.MemeSortBy
        if not render_meme_list or meme_sort_by is None:
            return None

        meme_props = {meme.key: meme_properties_type() for meme in self.memes}
        rendered = await asyncio.to_thread(
            render_meme_list,
            meme_properties=meme_props,
            exclude_memes=[],
            sort_by=meme_sort_by.KeywordsPinyin,
            sort_reverse=False,
            text_template="{index}. {keywords}",
            add_category_icon=True,
        )
        return self._unwrap_bytes(rendered, "render meme list")

    def get_meme_info(self, keyword: str) -> tuple[str, bytes] | None:
        meme = self.find_meme(keyword)
        if not meme:
            return None

        params = self._get_params(meme)
        keywords = self._get_keywords(meme)
        tags = self._get_tags(meme)

        lines: list[str] = []
        if meme.key:
            lines.append(f"名称：{meme.key}")
        if keywords:
            lines.append(f"别名：{keywords}")
        if params.max_images > 0:
            lines.append(
                f"所需图片：{params.min_images}张"
                if params.min_images == params.max_images
                else f"所需图片：{params.min_images}~{params.max_images}张"
            )
        if params.max_texts > 0:
            lines.append(
                f"所需文本：{params.min_texts}段"
                if params.min_texts == params.max_texts
                else f"所需文本：{params.min_texts}~{params.max_texts}段"
            )
        if params.default_texts:
            lines.append(f"默认文本：{params.default_texts}")
        if tags:
            lines.append(f"标签：{tags}")

        meme_info = "\n".join(lines)
        if meme_info:
            meme_info += "\n"
        preview = self._unwrap_bytes(
            meme.generate_preview(), f"generate preview for {meme.key}"
        )
        return meme_info, preview

    async def generate_meme(
        self, event: AstrMessageEvent, keyword: str
    ) -> bytes | None:
        meme = self.find_meme(keyword)
        if not meme:
            return None

        params = self._get_params(meme)
        images, texts, options = await self.collect.collect_params(event, params)

        if self.is_py_version:
            run_sync = self.run_sync
            if not run_sync:
                return None

            meme_images = [image for _, image in images]
            result = await run_sync(meme)(images=meme_images, texts=texts, args=options)
            return self._unwrap_bytes(result, f"generate meme {meme.key}")

        meme_image_type = self.MemeImage
        if not meme_image_type:
            return None

        meme_images = [
            meme_image_type(name=str(name), data=data) for name, data in images
        ]
        result = await asyncio.to_thread(meme.generate, meme_images, texts, options)
        return self._unwrap_bytes(result, f"generate meme {meme.key}")
