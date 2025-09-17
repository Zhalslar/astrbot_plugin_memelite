from random import random
import base64
from pathlib import Path
from typing import Optional, Union
import aiohttp
from astrbot.api import logger
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.message.components import At, Image, Plain, Reply
from astrbot.core.platform.astr_message_event import AstrMessageEvent


class ParamsCollector:
    """
    参数收集类
    """

    def __init__(self, config: AstrBotConfig):
        self.conf = config
        self.session = aiohttp.ClientSession()

    async def _download_image(self, url: str, http: bool = True) -> bytes | None:
        """下载图片"""
        if http:
            url = url.replace("https://", "http://")
        try:
            async with self.session.get(url) as resp:
                return await resp.read()
        except Exception as e:
            logger.error(f"图片下载失败: {e}")
            return None

    async def get_avatar(self, user_id: str) -> bytes | None:
        """根据 QQ 号下载头像"""
        if not user_id.isdigit():
            user_id = f"{random.randint(10_000_000, 999_999_999)}"
        avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        return await self._download_image(avatar_url)

    async def _decode_image(self, src: str) -> bytes | None:
        """统一把 src 转成 bytes"""
        raw: Optional[bytes] = None
        # 1. 本地文件
        if Path(src).is_file():
            raw = Path(src).read_bytes()
        # 2. URL
        elif src.startswith("http"):
            raw = await self._download_image(src)
        # 3. Base64
        elif src.startswith("base64://"):
            return base64.b64decode(src[9:])
        if not raw:
            return None
        return raw

    async def get_extra(self, event: AstrMessageEvent, target_id: str):
        """从消息平台获取参数"""
        if event.get_platform_name() == "aiocqhttp":
            from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
                AiocqhttpMessageEvent,
            )

            assert isinstance(event, AiocqhttpMessageEvent)
            user_info = await event.bot.get_stranger_info(user_id=int(target_id))
            raw_nickname = user_info.get("nickname")
            nickname = str(raw_nickname if raw_nickname is not None else "Unknown")
            sex = user_info.get("sex")
            return nickname, sex
        # TODO 适配更多消息平台

    async def _append_id_info(
        self,
        event: AstrMessageEvent,
        target_id: str,
        images: list,
        options: dict,
    ):
        """补齐昵称、性别、头像信息"""
        if result := await self.get_extra(event, target_id):
            nickname, sex = result
            options["name"], options["gender"] = nickname, sex
            if avatar := await self.get_avatar(target_id):
                images.append((nickname, avatar))

    async def collect_params(self, event: AstrMessageEvent, params):
        """收集参数，返回 (images, texts, options)"""
        images: list[tuple[str, bytes]] = []
        texts: list[str] = []
        options: dict[str, Union[bool, str, int, float]] = {}

        chain = event.get_messages()
        send_id: str = event.get_sender_id()
        self_id: str = event.get_self_id()
        sender_name: str = str(event.get_sender_name())

        async def _process_segment(seg, name: str):
            if isinstance(seg, Image):
                if src := seg.url or seg.file:
                    if image := await self._decode_image(src):
                        images.append((name, image))
            elif isinstance(seg, At) and seg != chain[0]:
                await self._append_id_info(event, str(seg.qq), images, options)
            elif isinstance(seg, Plain):
                plains: list[str] = seg.text.strip().split(" ")
                if len(plains) > 1:
                    for text in plains[1:]:
                        # 解析其他参数
                        if "=" in text:
                            k, v = text.split("=", 1)
                            options[k] = v
                        #  解析@qq
                        elif text.startswith("@"):
                            target_id = text[1:]
                            if target_id.isdigit():
                                await self._append_id_info(
                                    event, target_id, images, options
                                )
                        elif text:
                            texts.append(text)

        reply_seg = next((seg for seg in chain if isinstance(seg, Reply)), None)
        if reply_seg and reply_seg.chain:
            name = str(reply_seg.sender_nickname or reply_seg.sender_id)
            for seg in reply_seg.chain:
                await _process_segment(seg, name)
        for seg in chain:
            await _process_segment(seg, sender_name)

        # 确保图片数量在min_images到max_images之间(参数足够即可)
        if len(images) < params.min_images:
            if sender_avatar := await self.get_avatar(send_id):
                images.insert(0, (sender_name, sender_avatar))
        if len(images) < params.min_images:
            if bot_avatar := await self.get_avatar(self_id):
                images.insert(0, ("bot", bot_avatar))
        images = images[: params.max_images]

        # 确保文本数量在min_texts到max_texts之间(参数足够即可)
        if len(texts) < params.min_texts and params.default_texts:
            texts.extend(params.default_texts)
        texts = texts[: params.max_texts]

        return images, texts, options

    async def close(self):
        if hasattr(self, "session"):
            await self.session.close()
