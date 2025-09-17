import asyncio
from typing import Literal
from astrbot import logger
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from .param import ParamsCollector
from dataclasses import dataclass, field

from meme_generator import Meme, get_memes
from meme_generator.version import __version__


@dataclass
class MemeProperties:
    disabled: bool = False
    labels: list[Literal["new", "hot"]] = field(default_factory=list)


class MemeManager:
    is_py_version = tuple(map(int, __version__.split("."))) < (0, 2, 0)
    def __init__(self, config: AstrBotConfig, collect: ParamsCollector):
        self.conf = config
        self.collect = collect

        if self.is_py_version:
            from meme_generator.download import check_resources
            from meme_generator.utils import run_sync, render_meme_list
            self.render_meme_list = render_meme_list
            self.check_resources_func = check_resources
            self.run_sync = run_sync
        else:
            from meme_generator.tools import MemeProperties, MemeSortBy, render_meme_list
            from meme_generator.resources import check_resources_in_background
            from meme_generator import Image as MemeImage
            self.render_meme_list = render_meme_list
            self.check_resources_func = check_resources_in_background
            self.MemeImage = MemeImage
        self.memes: list[Meme] = get_memes()
        self.meme_keywords = [
            k
            for m in self.memes
            for k in (m.keywords if self.is_py_version else m.info.keywords)
        ]

    async def check_resources(self):
        if not self.conf["is_check_resources"]:
            return
        logger.info("开始检查memes资源...")
        if self.is_py_version:
            asyncio.create_task(self.check_resources_func())
        else:
            asyncio.create_task(asyncio.to_thread(self.check_resources_func))

    def find_meme(self, keyword: str) -> Meme | None:
        for meme in self.memes:
            keywords = meme.keywords if self.is_py_version else meme.info.keywords
            if keyword == meme.key or keyword in keywords:
                return meme

    def is_meme_keyword(self, meme_name: str) -> bool:
        return meme_name in self.meme_keywords

    def match_meme_keyword(self, text: str, fuzzy_match: bool) -> str | None:
        if fuzzy_match:
            # 模糊匹配：检查关键词是否在消息字符串中
            keyword = next((k for k in self.meme_keywords if k in text), None)
        else:
            # 精确匹配：检查关键词是否等于消息字符串的第一个单词
            keyword = next(
                (k for k in self.meme_keywords if k == text.split()[0]), None
            )
        return keyword

    async def render_meme_list_image(self) -> bytes | None:
        if self.is_py_version:
            meme_list = [(m, MemeProperties(labels=[])) for m in self.memes]
            return self.render_meme_list(
                meme_list=meme_list,  # type: ignore
                text_template="{index}.{keywords}",
                add_category_icon=True,
            ).getvalue()
        else:
            meme_props = {m.key: MemeProperties() for m in self.memes}
            return await asyncio.to_thread(
                self.render_meme_list,
                meme_properties=meme_props,
                exclude_memes=[],
                sort_by=MemeSortBy.KeywordsPinyin,
                sort_reverse=False,
                text_template="{index}. {keywords}",
                add_category_icon=True,
            )

    def get_meme_info(self, keyword: str) -> tuple[str, bytes] | None:
        """
        根据关键词返回 meme 的详情
        返回 (描述文本, 预览图bytes)
        如果未找到，返回 None
        """
        meme = self.find_meme(keyword)
        if not meme:
            return None

        if self.is_py_version:
            p = meme.params_type
            keywords = meme.keywords
            tags = meme.tags
        else:
            p = meme.info.params
            keywords = meme.info.keywords
            tags = meme.info.tags

        # 组装信息字符串
        meme_info = ""
        if meme.key:
            meme_info += f"名称：{meme.key}\n"
        if keywords:
            meme_info += f"别名：{keywords}\n"
        if p.max_images > 0:
            meme_info += (
                f"所需图片：{p.min_images}张\n"
                if p.min_images == p.max_images
                else f"所需图片：{p.min_images}~{p.max_images}张\n"
            )
        if p.max_texts > 0:
            meme_info += (
                f"所需文本：{p.min_texts}段\n"
                if p.min_texts == p.max_texts
                else f"所需文本：{p.min_texts}~{p.max_texts}段\n"
            )
        if p.default_texts:
            meme_info += f"默认文本：{p.default_texts}\n"
        if tags:
            meme_info += f"标签：{list(tags)}\n"

        preview: bytes = meme.generate_preview()  # type: ignore
        return meme_info, preview

    async def generate_meme(
        self, event: AstrMessageEvent, keyword: str
    ) -> bytes | None:
        # 匹配meme
        meme = self.find_meme(keyword)
        if not meme:
            return
        # 收集参数
        params = meme.params_type if self.is_py_version else meme.info.params
        images, texts, options = await self.collect.collect_params(event, params)

        if self.is_py_version:
            meme_images = [i[1] for i in images]
            return (
                await self.run_sync(meme)(images=meme_images, texts=texts, args=options)
            ).getvalue()
        else:
            meme_images = [self.MemeImage(name=str(i[0]), data=i[1]) for i in images]
            return await asyncio.to_thread(meme.generate, meme_images, texts, options)
