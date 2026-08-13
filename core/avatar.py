import random
from collections import OrderedDict

import aiohttp

from astrbot.api import logger


class AvatarManager:
    def __init__(self, max_cache_size: int = 20):
        self.max_cache_size = max(1, max_cache_size)
        self.session = aiohttp.ClientSession()
        self._cache: OrderedDict[str, bytes] = OrderedDict()

    async def download_image(self, url: str, http: bool = True) -> bytes | None:
        if http:
            url = url.replace("https://", "http://")
        try:
            async with self.session.get(url) as resp:
                return await resp.read()
        except Exception as e:
            logger.error(f"图片下载失败: {e}")
            return None

    async def get_qq_avatar(self, user_id: str) -> bytes | None:
        if not user_id.isdigit():
            user_id = f"{random.randint(10_000_000, 999_999_999)}"

        if cached_avatar := self._cache.get(user_id):
            self._cache.move_to_end(user_id)
            return cached_avatar

        avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        avatar = await self.download_image(avatar_url)
        if avatar:
            self._cache[user_id] = avatar
            self._cache.move_to_end(user_id)
            while len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)
        return avatar

    async def get_qq_official_avatar(self, appid: str, openid: str) -> bytes | None:
        if not appid or not openid:
            return None

        cache_key = f"qq_official:{appid}:{openid}"
        if cached_avatar := self._cache.get(cache_key):
            self._cache.move_to_end(cache_key)
            return cached_avatar

        avatar_url = f"https://q.qlogo.cn/qqapp/{appid}/{openid}/0"
        avatar = await self.download_image(avatar_url)
        if avatar:
            self._cache[cache_key] = avatar
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)
        return avatar

    async def close(self):
        if hasattr(self, "session"):
            await self.session.close()
        self._cache.clear()
