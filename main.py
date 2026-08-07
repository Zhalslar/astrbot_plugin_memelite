import asyncio
from contextlib import suppress

import astrbot.core.message.components as Comp
from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star
from astrbot.core import AstrBotConfig
from astrbot.core.platform import AstrMessageEvent
from astrbot.core.star.filter.event_message_type import EventMessageType

from .core.meme import MemeManager
from .core.param import ParamsCollector
from .utils import compress_image


class MemePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.collector = ParamsCollector(config)
        self.manager = MemeManager(config, self.collector)
        self._resource_task: asyncio.Task | None = None

    async def initialize(self):
        self._resource_task = asyncio.create_task(self.manager.check_resources())

    async def terminate(self):
        if self._resource_task and not self._resource_task.done():
            self._resource_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._resource_task
        await self.collector.close()

    @filter.command("meme列表", alias={"表情帮助", "meme菜单", "meme帮助"})
    async def memes_help(self, event):
        """显示meme列表"""
        if output := await self.manager.render_meme_list_image():
            yield event.chain_result([Comp.Image.fromBytes(output)])
        else:
            yield event.plain_result("meme列表图生成失败")

    @filter.command("meme详情", alias={"表情详情", "meme信息"})
    async def meme_details_show(
        self, event: AstrMessageEvent, keyword: str | int | None = None
    ):
        """meme详情 <名称>"""
        if not keyword:
            yield event.plain_result("未指定要查看的meme")
            return
        keyword = str(keyword)

        result = self.manager.get_meme_info(keyword)
        if not result:
            yield event.plain_result("未找到相关meme")
            return

        meme_info, preview = result
        chain = [
            Comp.Plain(meme_info),
            Comp.Image.fromBytes(preview),
        ]
        yield event.chain_result(chain)

    @filter.command("禁用meme")
    async def add_supervisor(
        self, event: AstrMessageEvent, meme_name: str | None = None
    ):
        """禁用meme <名称>"""
        if not meme_name:
            yield event.plain_result("未指定要禁用的meme")
            return
        if not self.manager.is_meme_keyword(meme_name):
            yield event.plain_result(f"meme: {meme_name} 不存在")
            return
        if meme_name in self.conf["memes_disabled_list"]:
            yield event.plain_result(f"meme: {meme_name} 已被禁用")
            return
        self.conf["memes_disabled_list"].append(meme_name)
        self.conf.save_config()
        yield event.plain_result(f"已禁用meme: {meme_name}")
        logger.info(f"当前禁用meme: {self.conf['memes_disabled_list']}")

    @filter.command("启用meme")
    async def remove_supervisor(
        self, event: AstrMessageEvent, meme_name: str | None = None
    ):
        """启用meme <名称>"""
        if not meme_name:
            yield event.plain_result("未指定要禁用的meme")
            return
        if not self.manager.is_meme_keyword(meme_name):
            yield event.plain_result(f"meme: {meme_name} 不存在")
            return
        if meme_name not in self.conf["memes_disabled_list"]:
            yield event.plain_result(f"meme: {meme_name} 未被禁用")
            return
        self.conf["memes_disabled_list"].remove(meme_name)
        self.conf.save_config()
        yield event.plain_result(f"已禁用meme: {meme_name}")

    @filter.command("meme黑名单")
    async def list_supervisors(self, event: AstrMessageEvent):
        """查看禁用的meme"""
        yield event.plain_result(f"当前禁用的meme: {self.conf['memes_disabled_list']}")

    @filter.event_message_type(EventMessageType.ALL)
    async def meme_handle(self, event: AstrMessageEvent):
        """
        处理 meme 生成的主流程
        """
        if self.conf["need_prefix"] and not event.is_at_or_wake_command:
            return
        if self.conf["extra_prefix"] and not event.message_str.startswith(
            self.conf["extra_prefix"]
        ):
            return

        param = event.message_str.removeprefix(self.conf["extra_prefix"])
        if not param:
            return
        # 匹配 meme
        keyword = self.manager.match_meme_keyword(
            text=param, fuzzy_match=self.conf["fuzzy_match"]
        )
        if not keyword or keyword in self.conf["memes_disabled_list"]:
            return

        # 合成表情
        try:
            image = await asyncio.wait_for(
                self.manager.generate_meme(event, keyword),
                timeout=self.conf["meme_timeout"],
            )
        except asyncio.TimeoutError:
            logger.warning(f"meme生成超时: {keyword}")
            yield event.plain_result("meme生成超时")
            return
        except Exception as e:
            logger.error(f"meme生成异常: {e}")
            return

        if image and self.conf["is_compress_image"]:
            try:
                image = compress_image(image) or image
            except Exception:
                pass

        if image:
            yield event.chain_result([Comp.Image.fromBytes(image)])  # type: ignore

