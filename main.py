
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

import asyncio
import io
from typing import List, Any
from PIL import Image

import astrbot.core.message.components as Comp
from astrbot.core.star.filter.event_message_type import EventMessageType


memes: list[Meme] = get_memes()
meme_keywords_list = [keyword.lower() for meme in memes for keyword in meme.keywords] # 有序列表
meme_keywords_set = set(meme_keywords_list) # 无序集合

@register("astrbot_plugin_memelite", "Zhalslar", "表情包生成器，制作各种沙雕表情（本地部署，但轻量化）", "1.0.8")
class MemePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.memes: list[Meme] = memes
        self.meme_keywords_list: list = meme_keywords_list

        self.prefix: str = config.get('prefix', '')
        self.fuzzy_match: int = config.get('fuzzy_match', True)
        self.is_compress_image: bool = config.get('is_compress_image', True)
        self.save_temp_image: bool = config.get('save_temp_image', False)

        self.is_check_resources: bool = config.get('is_check_resources', True)
        if self.is_check_resources:
            logger.info("正在检查memes资源文件...")
            asyncio.create_task(check_resources())


    @filter.command("表情列表", desc="查看有哪些关键词可以触发meme")
    async def meme_keyword_show(self, event: AstrMessageEvent):
        meme_keywords_str = "、".join(self.meme_keywords_list)
        url = await self.text_to_image(meme_keywords_str, return_url=True)
        yield event.image_result(url)

    @filter.command("表情详情", desc="查看指定meme需要的参数")
    async def meme_detail_show(self, event: AstrMessageEvent, keyword: str=None):
        if not keyword:
            yield event.plain_result("未指定表情")
            return
        target_keyword = next((k for k in meme_keywords_set if k in event.get_message_str()), None)
        meme = self.find_meme(target_keyword)

        meme_info = (
            f"名称：{meme.key}：\n"
            f"别名：{meme.keywords}：\n"
            f"最小图片数：{meme.params_type.min_images}\n"
            f"最大图片数：{meme.params_type.max_images}\n"
            f"最小文本数：{meme.params_type.min_texts}\n"
            f"最大文本数：{meme.params_type.max_texts}\n"
            f"默认文本：{meme.params_type.default_texts}\n"
            f"标签：{list(meme.tags)}：\n"
            f"预览图片：\n"
        )

        preview: bytes = meme.generate_preview().getvalue()
        chain = [
            Comp.Plain(meme_info),
            Comp.Image.fromBytes(preview),
        ]
        yield event.chain_result(chain)


    @filter.event_message_type(EventMessageType.ALL)
    async def meme_handle(self, event: AstrMessageEvent) -> None:
        """
         处理 meme 生成的核心逻辑。

         功能描述：
         - 支持匹配所有 meme 关键词。
         - 支持从原始消息中提取参数, 空格隔开参数。
         - 支持引用消息传参 。
         - 自动获取消息发送者、被 @ 的用户以及 bot 自身的相关参数。
         """
        message_str = event.get_message_str()
        if not message_str:  # 过滤非文本消息
            return
        message_list = message_str.split()
        first_arg = message_list[0]

        if self.prefix: # 前缀模式
            if self.prefix not in first_arg:
                return
            first_arg = first_arg.replace(self.prefix, '', 1)  # 去除前缀

        if self.fuzzy_match:
            keyword = next((k for k in meme_keywords_set if k in message_str), None) # 模糊匹配模式
        else:
            keyword = next((k for k in meme_keywords_set if k in message_list), None) # 精准匹配模式
        if not keyword:
            return

        images: List[bytes] = []
        texts: List[str] = []
        args: dict[str, Any] = {}

        meme: Meme = self.find_meme(keyword)
        min_images: int = meme.params_type.min_images
        max_images: int = meme.params_type.max_images
        min_texts: int = meme.params_type.min_texts
        max_texts: int = meme.params_type.max_texts
        default_texts: list[str] = meme.params_type.default_texts

        messages = event.get_messages()
        send_id = event.get_sender_id()
        sender_name = event.get_sender_name()
        self_id = event.get_self_id()

        # 获取目标用户的参数
        seg = [seg for seg in messages if (isinstance(seg, Comp.At)) and str(seg.qq) != self_id]
        target_id = seg[0].qq if seg else send_id
        target_name = seg[0].name if seg else sender_name

        # aiocqhttp消息平台可调用Onebot接口“get_stranger_info”获取额外参数
        if event.get_platform_name() == "aiocqhttp":
            from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
            assert isinstance(event, AiocqhttpMessageEvent)
            client = event.bot
            user_info = await client.get_stranger_info(user_id=int(target_id))
            name = user_info.get('nickname')
            gender = user_info.get('sex')
            args["user_infos"] = [{"name": name, "gender": gender}]

        async def _process_segment(_seg):
            """Process a single message segment."""
            if isinstance(_seg, Comp.Image):
                img_url = seg.url
                msg_image = await self.download_image(img_url)
                images.append(msg_image)
            elif isinstance(_seg, Comp.At):
                seg_qq = _seg.qq
                if str(seg_qq) != self_id:
                    at_avatar = await self.get_avatar(str(seg_qq))
                    images.append(at_avatar)
            elif isinstance(_seg, Comp.Plain):
                text: str = _seg.text.strip()
                arg_text = text.lstrip(self.prefix).strip()
                if arg_text != keyword:
                    texts.append(arg_text)


        # 如果有引用消息，也遍历之
        reply_seg = next((seg for seg in messages if isinstance(seg, Comp.Reply)), None)
        if reply_seg:
            for seg in reply_seg.chain:
                await _process_segment(seg)

        # 遍历原始消息段落
        for seg in messages:
            await _process_segment(seg)

        # 确保图片数量在min_images到max_images之间
        if len(images) < max_images:
            use_avatar = await self.get_avatar(send_id)
            images.insert(0, use_avatar)
            if len(images) < max_images:
                bot_avatar = await self.get_avatar(self_id)
                images.append(bot_avatar)
        images = images[:max_images]

        # 确保文本数量在min_texts到max_texts之间
        if len(texts) < max_texts:
            texts.extend([target_name])
            if len(texts) < max_texts:
                texts.extend(default_texts[:max_texts])
        texts = texts[:max_texts]

        try:
            image_io = await run_sync(meme)(
                images=images,
                texts=texts,
                args=args
            )

        except MemeGeneratorException as e:
            logger.error(e.message)
            return
        if self.is_compress_image and Image.open(image_io).format != "GIF":
            image_io = self.compress_image(image_io)
        image_bytes = image_io.getvalue()
        chain = [Comp.Image.fromBytes(image_bytes)]
        yield event.chain_result(chain)



    def find_meme(self, meme_name: str) -> Meme | None:
        """根据给定的meme名字查找并返回符合条件的第一个表情包对象"""
        search_name = meme_name.lower()
        for meme in self.memes:
            # 检查meme的关键字和别名是否包含目标名称
            if (search_name == meme.key.lower() or
                    any(keyword.lower() == search_name for keyword in meme.keywords)):
                return meme
        return None

    @staticmethod
    def compress_image(image_io, max_size: int = 512) -> io.BytesIO | None:
        """压缩图片到max_size大小，gif不处理"""
        try:
            img = Image.open(image_io)
            if img.format == "GIF":
                return
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            output_buffer = io.BytesIO()
            img.save(output_buffer, format=img.format)
            output_buffer.seek(0)
            return output_buffer

        except Exception as e:
            raise ValueError(f"图片压缩失败: {e}")


    @staticmethod
    async def download_image(url: str) -> bytes:
        url = url.replace("https://", "http://")
        try:
            async with aiohttp.ClientSession() as client:
                response = await client.get(url)
                img_bytes = await response.read()
                return img_bytes
        except Exception as e:
            logger.error(f"图片下载失败: {e}")

    @staticmethod
    async def get_events_on_history(month: str) -> str:
        try:
            async with aiohttp.ClientSession() as client:
                url = f"https://baike.baidu.com/cms/home/eventsOnHistory/{month}.json"
                response = await client.get(url)
                response.encoding = "utf-8"
                return await response.text()
        except Exception as e:
            logger.error(f"任务处理失败: {e}")
            return ""

    @staticmethod
    async def get_avatar(user_id: str) -> bytes:
        avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        try:
            async with aiohttp.ClientSession() as client:
                response = await client.get(avatar_url, timeout=10)
                response.raise_for_status()
                return await response.read()
        except Exception as e:
            logger.error(f"下载头像失败: {e}")
            return b''



