import asyncio
import base64
from pathlib import Path
import random
import aiohttp
from meme_generator import Meme, get_memes
from meme_generator.download import check_resources
from meme_generator.exception import MemeGeneratorException
from meme_generator.utils import run_sync

from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register
from astrbot.core import AstrBotConfig
from astrbot.core.platform import AstrMessageEvent

import io
from typing import Any, List
from PIL import Image
import astrbot.core.message.components as Comp
from astrbot.core.star.filter.event_message_type import EventMessageType


@register(
    "astrbot_plugin_memelite",
    "Zhalslar",
    "表情包生成器，制作各种沙雕表情（本地部署，但轻量化）",
    "1.0.2",
    "https://github.com/Zhalslar/astrbot_plugin_memelite",
)
class MemePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.memes_disabled_list: list[str] = config.get("memes_disabled_list", [])

        self.memes: list[Meme] = get_memes()
        self.meme_keywords: list = [
            keyword for meme in self.memes for keyword in meme.keywords
        ]

        self.prefix: str = config.get("prefix", "")

        self.fuzzy_match: bool = config.get("fuzzy_match", False)
        self.is_compress_image: bool = config.get("is_compress_image", True)

        self.is_check_resources: bool = config.get("is_check_resources", True)
        if self.is_check_resources:
            logger.info("正在检查memes资源文件...")
            asyncio.create_task(check_resources())

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
        target_keyword = next((k for k in self.meme_keywords if k == keyword), None)
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
        params_type = meme.params_type
        keywords = meme.keywords
        min_images = params_type.min_images
        max_images = params_type.max_images
        min_texts = params_type.min_texts
        max_texts = params_type.max_texts
        default_texts = params_type.default_texts
        tags = meme.tags

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

        preview: bytes = meme.generate_preview().getvalue()  # type: ignore
        chain = [
            Comp.Plain(meme_info),
            Comp.Image.fromBytes(preview),
        ]
        yield event.chain_result(chain)

    @filter.command("禁用meme")
    async def add_supervisor(
        self, event: AstrMessageEvent, meme_name: str | None = None
    ):
        """禁用meme"""
        if not meme_name:
            yield event.plain_result("未指定要禁用的meme")
            return
        if meme_name not in self.meme_keywords:
            yield event.plain_result(f"meme: {meme_name} 不存在")
            return
        if meme_name in self.memes_disabled_list:
            yield event.plain_result(f"meme: {meme_name} 已被禁用")
            return
        self.memes_disabled_list.append(meme_name)
        self.config.save_config(replace_config=self.config)
        yield event.plain_result(f"已禁用meme: {meme_name}")
        logger.info(f"当前禁用meme: {self.config['memes_disabled_list']}")

    @filter.command("启用meme")
    async def remove_supervisor(
        self, event: AstrMessageEvent, meme_name: str | None = None
    ):
        """启用meme"""
        if not meme_name:
            yield event.plain_result("未指定要禁用的meme")
            return
        if meme_name not in self.meme_keywords:
            yield event.plain_result(f"meme: {meme_name} 不存在")
            return
        if meme_name not in self.memes_disabled_list:
            yield event.plain_result(f"meme: {meme_name} 未被禁用")
            return
        self.memes_disabled_list.remove(meme_name)
        self.config.save_config(replace_config=self.config)
        yield event.plain_result(f"已禁用meme: {meme_name}")

    @filter.command("meme黑名单")
    async def list_supervisors(self, event: AstrMessageEvent):
        """查看禁用的meme"""
        yield event.plain_result(f"当前禁用的meme: {self.memes_disabled_list}")

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

        # 前缀模式
        if self.prefix:
            chain = event.get_messages()
            if not chain:
                return
            first_seg = chain[0]
            # 前缀触发
            if isinstance(first_seg, Comp.Plain):
                if not first_seg.text.startswith(self.prefix):
                    return
            # @bot触发
            elif isinstance(first_seg, Comp.At):
                if str(first_seg.qq) != str(event.get_self_id()):
                    return
            else:
                return

        message_str = event.get_message_str().removeprefix(self.prefix)
        if not message_str:
            return

        if self.fuzzy_match:
            # 模糊匹配：检查关键词是否在消息字符串中
            keyword = next((k for k in self.meme_keywords if k in message_str), None)
        else:
            # 精确匹配：检查关键词是否等于消息字符串的第一个单词
            keyword = next(
                (k for k in self.meme_keywords if k == message_str.split()[0]), None
            )

        if not keyword or keyword in self.memes_disabled_list:
            return

        # 匹配meme
        meme = self._find_meme(keyword)
        if not meme:
            yield event.plain_result("未找到相关meme")
            return

        # 收集参数
        images, texts, options = await self._get_parms(event, keyword, meme)

        # 合成表情
        try:
            image_io = await run_sync(meme)(images=images, texts=texts, args=options)

        except MemeGeneratorException as e:
            logger.error(e.message)
            return

        # 压缩图片
        if self.is_compress_image:
            try:
                image = self.compress_image(image_io) or image_io
            except:  # noqa: E722
                pass

        # 发送图片
        chain = [Comp.Image.fromBytes(image.getvalue())]
        yield event.chain_result(chain)  # type: ignore

    def _find_meme(self, keyword: str) -> Meme | None:
        """根据关键词寻找meme"""
        for meme in self.memes:
            if keyword == meme.key or any(k == keyword for k in meme.keywords):
                return meme

    async def _get_parms(self, event: AstrMessageEvent, keyword: str, meme: Meme):
        """收集参数"""
        images: list[bytes] = []
        texts: List[str] = []
        options: dict[str, Any] = {}

        params_type = meme.params_type
        max_images = params_type.max_images
        max_texts = params_type.max_texts
        default_texts = params_type.default_texts

        messages = event.get_messages()
        send_id: str = event.get_sender_id()
        self_id: str = event.get_self_id()
        sender_name: str = event.get_sender_name()

        target_ids: list[str] = []
        target_names: list[str] = []

        async def _process_segment(_seg):
            """从消息段中获取参数"""
            if isinstance(_seg, Comp.Image):
                if hasattr(_seg, "url") and _seg.url:
                    img_url = _seg.url
                    # 如果是有效的本地路径，则直接读取文件
                    if Path(img_url).is_file():
                        with open(img_url, "rb") as f:
                            images.append(f.read())
                    else:  # 否则尝试作为URL下载
                        if msg_image := await self.download_image(img_url):
                            images.append(msg_image)

                elif hasattr(_seg, "file"):
                    file_content = _seg.file
                    if isinstance(file_content, str):
                        # 如果是有效的本地路径，则直接读取文件
                        if Path(file_content).is_file():
                            with open(file_content, "rb") as f:
                                images.append(f.read())
                        else:  # 否则尝试作为Base64编码解析
                            if file_content.startswith("base64://"):
                                file_content = file_content[len("base64://") :]
                            file_content = base64.b64decode(file_content)
                    if isinstance(file_content, bytes):
                        images.append(file_content)

            elif isinstance(_seg, Comp.At):
                seg_qq = str(_seg.qq)
                if seg_qq != self_id:
                    target_ids.append(seg_qq)
                    if at_avatar := await self.get_avatar(event, seg_qq):
                        images.append(at_avatar)
                    # 从消息平台获取At者的额外参数
                    if result := await self._get_extra(event, target_id=seg_qq):
                        nickname, sex = result
                        options["user_infos"] = [{"name": nickname, "gender": sex}]
                        target_names.append(nickname)

            elif isinstance(_seg, Comp.Plain):
                plains: list[str] = _seg.text.strip().split()
                for text in plains:
                    if text not in self.prefix and text != self.prefix + keyword:
                        texts.append(text)

        # 如果有引用消息，也遍历之
        reply_seg = next((seg for seg in messages if isinstance(seg, Comp.Reply)), None)
        if reply_seg and reply_seg.chain:
            for seg in reply_seg.chain:
                await _process_segment(seg)

        # 遍历原始消息段落
        for seg in messages:
            await _process_segment(seg)

        # 从消息平台获取发送者的额外参数
        if not target_ids:
            if result := await self._get_extra(event, target_id=send_id):
                nickname, sex = result
                options["user_infos"] = [{"name": nickname, "gender": sex}]
                target_names.append(nickname)

        # 确保图片数量在min_imag

        if not target_names:
            target_names.append(sender_name)

        # 确保图片数量在min_images到max_images之间
        if len(images) < max_images:
            if use_avatar := await self.get_avatar(event, send_id):
                images.insert(0, use_avatar)
        if len(images) < max_images:
            if bot_avatar := await self.get_avatar(event, self_id):
                images.append(bot_avatar)
        meme_images = images[:max_images]

        # 确保文本数量在min_texts到max_texts之间
        texts.extend(target_names)
        texts.extend(default_texts)
        texts = texts[:max_texts]

        return meme_images, texts, options

    @staticmethod
    async def _get_extra(event: AstrMessageEvent, target_id: str):
        """从消息平台获取参数"""
        if event.get_platform_name() == "aiocqhttp":
            from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
                AiocqhttpMessageEvent,
            )

            assert isinstance(event, AiocqhttpMessageEvent)
            client = event.bot
            user_info = await client.get_stranger_info(user_id=int(target_id))
            nickname = user_info.get("nickname")
            sex = user_info.get("sex")
            return nickname, sex
        # TODO 适配更多消息平台

    @staticmethod
    def compress_image(image_io: io.BytesIO, max_size: int = 512) -> io.BytesIO | None:
        """压缩静态图片或GIF到max_size大小"""
        try:
            # 将输入的bytes加载为图片
            img = Image.open(image_io)
            output = io.BytesIO()

            if img.format == "GIF":
                return
            else:
                # 如果是静态图片，检查尺寸并压缩
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                # 保存处理后的图片到内存中的BytesIO对象
                img.save(output, format=img.format)

            # 返回处理后的图片数据（bytes）
            return output

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
    async def get_avatar(event: AstrMessageEvent, user_id: str) -> bytes | None:
        """下载头像"""
        # if event.get_platform_name() == "aiocqhttp":
        if not user_id.isdigit():
            user_id = "".join(random.choices("0123456789", k=9))
        avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        try:
            async with aiohttp.ClientSession() as client:
                response = await client.get(avatar_url, timeout=10)
                response.raise_for_status()
                return await response.read()
        except Exception as e:
            logger.error(f"下载头像失败: {e}")
            return None
