import asyncio
from pathlib import Path
import aiohttp
from meme_generator import (
    DeserializeError,
    ImageAssetMissing,
    ImageDecodeError,
    ImageEncodeError,
    ImageNumberMismatch,
    MemeFeedback,
    TextNumberMismatch,
    TextOverLength,
)
from meme_generator import Meme, get_memes
from meme_generator import Image as MemeImage
from meme_generator.resources import check_resources_in_background
#from meme_generator.tools import MemeProperties, MemeSortBy, render_meme_list

from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register
from astrbot.core import AstrBotConfig
from astrbot.core.platform import AstrMessageEvent

import io
from typing import List, Union
from PIL import Image

import astrbot.core.message.components as Comp
from astrbot.core.star.filter.event_message_type import EventMessageType


@register(
    "astrbot_plugin_memelite",
    "Zhalslar",
    "表情包生成器，制作各种沙雕表情（本地部署，但轻量化）",
    "1.0.8",
    "https://github.com/Zhalslar/astrbot_plugin_memelite",
)
class MemePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.memes_disabled_list: list[str] = config.get("ban_memes", [])
        self.sort_by_str: str = config.get("ban_memes", "key")
        self.memes: list[Meme] = get_memes()
        self.meme_keywords: list[str] = [
            keyword for meme in self.memes for keyword in meme.info.keywords
        ]
        self.meme_keywords_list = [
            keyword for keyword in self.meme_keywords if keyword not in self.memes_disabled_list
        ]
        self.meme_keywords_set = set(self.meme_keywords_list)

        self.prefix: str = config.get("prefix", "")
        self.fuzzy_match: int = config.get("fuzzy_match", True)
        self.is_compress_image: bool = config.get("is_compress_image", True)

        self.is_check_resources: bool = config.get("is_check_resources", True)
        if self.is_check_resources:
            logger.info("正在检查memes资源文件...")
            check_resources_in_background()

    @filter.command("meme帮助", alias={"表情帮助"})
    async def memes_help(self, event: AstrMessageEvent):
        "查看有哪些关键词可以触发meme"
        image_path = Path(__file__).parent / "memes_help.jpg"
        yield event.image_result(str(image_path))

    @filter.command("meme详情", alias={"表情详情"})
    async def meme_details_show(
        self, event: AstrMessageEvent, keyword: str | int | None = None
    ):
        "查看指定meme需要的参数"
        if not keyword:
            yield event.plain_result("未指定要查看的meme")
            return
        keyword = str(keyword)
        target_keyword = next((k for k in self.meme_keywords_set if k == keyword), None)
        if target_keyword is None:
            yield event.plain_result("未支持的meme关键词")
            return

        # 匹配meme
        meme = self._find_meme(keyword)
        if not meme:
            yield event.plain_result("未找到相关meme")
            return

        # 提取meme的所有参数
        name = meme.key
        info = meme.info
        params = info.params
        keywords = info.keywords
        min_images = params.min_images
        max_images = params.max_images
        min_texts = params.min_texts
        max_texts = params.max_texts
        default_texts = params.default_texts
        tags = info.tags

        meme_info = ""
        if name:
            meme_info += f"名称：{name}\n"

        if keywords:
            meme_info += f"别名：{keywords}\n"

        if max_images > 0:
            meme_info += (
                f"所需图片：{min_images}张\n"
                if min_images == max_images
                else f"所需图片：{min_images}~{max_images}张\n"
            )

        if max_texts > 0:
            meme_info += (
                f"所需文本：{min_texts}段\n"
                if min_texts == max_texts
                else f"所需文本：{min_texts}~{max_texts}段\n"
            )

        if default_texts:
            meme_info += f"默认文本：{default_texts}\n"

        if tags:
            meme_info += f"标签：{list(tags)}\n"

        preview: bytes = meme.generate_preview()  # type: ignore
        chain = [
            Comp.Plain(meme_info),
            Comp.Image.fromBytes(preview),
        ]
        yield event.chain_result(chain)

    @filter.event_message_type(EventMessageType.ALL)
    async def meme_handle(self, event: AstrMessageEvent):
        """
        处理 meme 生成的主流程。

        功能描述：
        - 支持匹配所有 meme 关键词。
        - 支持从原始消息中提取参数, 空格隔开参数。
        - 支持引用消息传参 。
        - 自动获取消息发送者、被 @ 的用户以及 bot 自身的相关参数。
        """
        message_str = event.get_message_str()
        if not message_str or "详情" in message_str:
            return
        message_list = message_str.split()

        # 前缀模式
        if self.prefix:
            if message_list[0].startswith(self.prefix):
                message_list[0] = message_list[0][len(self.prefix) :]
            else:
                return

        # 精准/模糊匹配
        keyword = next(
            (
                k
                for k in self.meme_keywords_set
                if k in (message_str if self.fuzzy_match else message_list)
            ),
            None,
        )
        if not keyword:
            return

        # 匹配meme
        meme = self._find_meme(keyword)
        if not meme:
            yield event.plain_result("未找到相关meme")
            return

        # 收集参数
        meme_images, texts, options = await self._get_parms(event, keyword, meme)

        # 合成表情
        image: bytes = await self._meme_generate(meme, meme_images, texts, options)

        # 压缩图片
        if self.is_compress_image:
            try:
                image = self.compress_image(image) or image
            except:  # noqa: E722
                pass

        # 发送图片
        chain = [Comp.Image.fromBytes(image)]
        yield event.chain_result(chain)  # type: ignore

    def _find_meme(self, keyword: str) -> Meme | None:
        """根据关键词寻找meme"""
        for meme in self.memes:
            if keyword == meme.key or any(k == keyword for k in meme.info.keywords):
                return meme

    async def _get_parms(self, event: AstrMessageEvent, keyword: str, meme: Meme):
        """收集参数"""
        meme_images: list[MemeImage] = []
        texts: List[str] = []
        options: dict[str, Union[bool, str, int, float]] = {}

        params = meme.info.params
        max_images: int = params.max_images
        max_texts: int = params.max_texts
        default_texts: list[str] = params.default_texts

        messages = event.get_messages()
        send_id: str = event.get_sender_id()
        self_id: str = event.get_self_id()
        sender_name: str = event.get_sender_name()

        target_ids: list[str] = []
        target_names: list[str] = []

        async def _process_segment(_seg, name):
            """从消息段中获取参数"""
            if isinstance(_seg, Comp.Image):
                if img_url := _seg.url:
                    if msg_image := await self.download_image(img_url):
                        meme_images.append(MemeImage(name, msg_image))

            elif isinstance(_seg, Comp.At):
                seg_qq = str(_seg.qq)
                if seg_qq != self_id:
                    target_ids.append(seg_qq)
                    at_avatar = await self.get_avatar(seg_qq)
                    # 从消息平台获取At者的额外参数
                    if (result := await self._get_extra(event, target_id=seg_qq)):
                        nickname, sex = result
                        options["name"], options["gender"] = nickname, sex
                        target_names.append(nickname)
                        meme_images.append(MemeImage(nickname, at_avatar))

            elif isinstance(_seg, Comp.Plain):
                plains: list[str] = _seg.text.strip().split()
                for text in plains:
                    if text != keyword and text != self.prefix:
                        texts.append(text)

        # 如果有引用消息，也遍历之
        reply_seg = next((seg for seg in messages if isinstance(seg, Comp.Reply)), None)
        if reply_seg and reply_seg.chain:
            for seg in reply_seg.chain:
                await _process_segment(seg, "这家伙")

        # 遍历原始消息段落
        for seg in messages:
            await _process_segment(seg, sender_name)

        # 从消息平台获取发送者的额外参数
        if not target_ids:
            if (result := await self._get_extra(event, target_id=send_id)):
                nickname, sex = result
                options["name"], options["gender"] = nickname, sex
                target_names.append(nickname)

        if not target_names:
            target_names.append(sender_name)

        # 确保图片数量在min_images到max_images之间
        if len(meme_images) < max_images:
            use_avatar = await self.get_avatar(send_id)
            meme_images.insert(0, MemeImage(sender_name, use_avatar))
        if len(meme_images) < max_images:
            bot_avatar = await self.get_avatar(self_id)
            meme_images.append(MemeImage("我", bot_avatar))
        meme_images = meme_images[:max_images]

        # 确保文本数量在min_texts到max_texts之间
        texts.extend(target_names)
        texts.extend(default_texts)
        texts = texts[:max_texts]

        return meme_images, texts, options

    @staticmethod
    async def _meme_generate(
        meme: Meme, meme_images: list[MemeImage], texts: list[str], options
    ) -> bytes:
        """向meme生成器发出请求，返回生成的图片"""

        # 将同步函数运行在默认的线程池中
        result = await asyncio.to_thread(meme.generate, meme_images, texts, options)

        if result is None:
            logger.error("返回内容为空")
        elif isinstance(result, ImageDecodeError):
            logger.error(f"图片解码出错：{result.error}")
        elif isinstance(result, ImageEncodeError):
            logger.error(f"图片编码出错：{result.error}")
        elif isinstance(result, ImageAssetMissing):
            logger.error(f"缺少图片资源：{result.path}")
        elif isinstance(result, DeserializeError):
            logger.error(f"表情选项解析出错：{result.error}")
        elif isinstance(result, ImageNumberMismatch):
            num = (
                f"{result.min} ~ {result.max}"
                if result.min != result.max
                else str(result.min)
            )
            logger.error(f"图片数量不符，应为 {num}，实际传入 {result.actual}")
        elif isinstance(result, TextNumberMismatch):
            num = (
                f"{result.min} ~ {result.max}"
                if result.min != result.max
                else str(result.min)
            )
            logger.error(f"文字数量不符，应为 {num}，实际传入 {result.actual}")
        elif isinstance(result, TextOverLength):
            text = result.text
            repr = text if len(text) <= 10 else (text[:10] + "...")
            logger.error(f"文字过长：{repr}")
        elif isinstance(result, MemeFeedback):
            logger.error(result.feedback)

        if not isinstance(result, bytes):
            raise NotImplementedError

        return result

    @staticmethod
    async def _get_extra(event: AstrMessageEvent, target_id: str):
        """从消息平台获取参数"""
        if event.get_platform_name() == "aiocqhttp":
            from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
            assert isinstance(event, AiocqhttpMessageEvent)
            client = event.bot
            user_info = await client.get_stranger_info(user_id=int(target_id))
            nickname = user_info.get("nickname")
            sex = user_info.get("sex")
            return nickname, sex
        # TODO 适配更多消息平台

    @staticmethod
    def compress_image(image: bytes, max_size: int = 512) -> bytes | None:
        """压缩静态图片到max_size大小"""
        try:
            img = Image.open(io.BytesIO(image))
            if img.format == "GIF":
                return None
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                output = io.BytesIO()
                img.save(output, format=img.format)
                return output.getvalue()

        except Exception as e:
            raise ValueError(f"图片压缩失败: {e}")

    @staticmethod
    async def download_image(url: str) -> bytes | None:
        """下载图片"""
        url = url.replace("https://", "http://")
        try:
            async with aiohttp.ClientSession() as client:
                response = await client.get(url)
                img_bytes = await response.read()
                return img_bytes
        except Exception as e:
            logger.error(f"图片下载失败: {e}")

    @staticmethod
    async def get_avatar(user_id: str) -> bytes:
        """下载头像"""
        avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        try:
            async with aiohttp.ClientSession() as client:
                response = await client.get(avatar_url, timeout=10)
                response.raise_for_status()
                return await response.read()
        except Exception as e:
            logger.error(f"下载头像失败: {e}")
            return b""
