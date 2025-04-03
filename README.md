<div align="center">


# astrbot_plugin_memelite

_✨ [astrbot](https://github.com/nonebot/nonebot2) 表情包制作插件 ✨_
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![AstrBot](https://img.shields.io/badge/AstrBot-3.4%2B-orange.svg)](https://github.com/Soulter/AstrBot)
</div>


> [!NOTE]  本插件负责处理聊天机器人与表情包生成器的对接。  
> 具体表情包制作相关资源文件和代码在 [表情包生成器 meme-generator](https://github.com/MeetWq/meme-generator) 中   
> 本插件使用本地部署的meme-generator。同时尽量保持插件的轻量化，表情包生成快，性能要求低  
> 




## 📦 安装

- 可以直接在astrbot的插件市场搜索astrbot_plugin_memelite，点击安装，耐心等待安装完成即可  

- 或者可以直接克隆源码到插件文件夹：
```bash
# 克隆仓库到插件目录
cd /AstrBot/data/plugins
git clone https://github.com/Zhalslar/astrbot_plugin_memelite

# 控制台重启AstrBot
```

- 首次启动插件会触发资源下载（如果没触发，请重载一下插件），然后会自动下载两千多张图片，下载速度取决于网速，
  ![图片](https://github.com/user-attachments/assets/8d6c2fb6-3b79-49b0-ba85-eca1d128ca64)


## ⚙️ 配置
请在astrbot面板配置，插件管理 -> astrbot_plugin_memelite -> 操作 -> 插件配置
![图片](https://github.com/user-attachments/assets/fe3c6adf-f210-4d93-9d8c-a06216507f10)





## ⌨️ 命令

|     命令      |                    说明                    |
|:-------------:|:-----------------------------------------------:|
| /表情列表      | 查看所有能触发meme合成的关键词  |
| /表情详情 xxx  | 具体查看某个meme的参数         |
|   {关键词}     |   触发meme合成            |


, 关键词包括：
```plaintext
['5000兆', '戒导', '逆转裁判气泡', '二次元入口', '上瘾', '毒瘾发作', '添乱', '给社会添乱', '一样', '支付宝支付', '一直', '我永远喜欢', '防诱拐',
 '阿尼亚喜欢', '鼓掌', '阿罗娜扔', '升天', '问问', '亚托莉枕头', '继续干活', '打工人', '悲报', 'ba说', '拍头', '揍', '啃', '真寻挨骂', '高血压',
 '蔚蓝档案标题', 'batitle', '波奇手稿', '布洛妮娅举牌', '大鸭鸭举牌', '奶茶', '遇到困难请拨打', '咖波画', '咖波指', '咖波撕', '咖波蹭', '咖波贴'
, '咖波说', '咖波炖', '咖波撞', '咖波头槌', '舰长', '这个引起的', '奖状', '证书', '字符画', '追列车', '追火车', '国旗', '鼠鼠搓', '小丑', '小丑 
面具', '迷惑', '兑换券', '捂脸', '爬', '群青', '白天黑夜', '白天晚上', '像样的亲亲', '入典', '典中典', '黑白草图', '恐龙', '小恐龙', '注意力涣散
, '离婚协议', '离婚申请', '狗都不玩', '管人痴', '不要靠近', '不要按', '别碰', 'douyin', '吃', '皇帝龙图', '意若思镜', '灰飞烟灭', '狂爱', '狂粉粉
, '闭嘴', '我爸爸', '击剑', '🤺', '我打宿傩', '我打宿傩吗', '满脑子', '整点薯条', '流萤举牌', '闪瞎', '红温', '关注', '芙莉莲拿', '哈哈镜', '垃垃
', '垃圾桶', '原神吃', '原神启动', '王境泽', '为所欲为', '馋身子', '切格瓦拉', '谁反对', '假面骑士', '乌鸦哥', '曾小贤', '压力大爷', '你好骚啊啊
, '食屎啦你', '五年怎么过的', '麦克阿瑟说', '喜报', 'google', '谷歌验证码', '猩猩扔', '鬼畜', '手枪', '锤', '凉宫春日举', '高低情商', '低高情商商
, '打穿', '打穿屏幕', '记仇', '抱紧', '抱', '抱抱', '抱大腿', '胡桃啃', '坐牢', '不文明', 'inside', '采访', '杰瑞盯', '急急国王', '汐汐', '今汐汐
, '啾啾', '跳', '万花筒', '万花镜', '凯露指', '远离', '压岁钱不要交给', '踢球', '卡比锤', '卡比重锤', '亲', '亲亲', '可莉吃', '敲', '心奈印章',,
 '泉此方看', '偷学', '左右横跳', '让我进去', '舔糖', '舔棒棒糖', '等价无穷小', '听音乐', '小天使', '加载中', '看扁', '看图标', '循环', '寻狗启事
, '永远爱你', '洛天依要', '天依要', '洛天依说', '天依说', '罗永浩说', '鲁迅说', '鲁迅说过', '真寻看书', '旅行伙伴觉醒', '旅行伙伴加入', '交个朋朋
', '结婚申请', '结婚登记', '流星', '米哈游', '上香', '低语', '我朋友说', '我的意见如下', '我的意见是', '我老婆', '这是我老婆', '纳西妲啃', '草草
啃', '亚文化取名机', '亚名', '需要', '你可能需要', '猫羽雫举牌', '猫猫举牌', '伊地知虹夏举牌', '虹夏举牌', '诺基亚', '有内鬼', '请假条', '不喊喊
', '无响应', '我推的网友', 'osu', 'out', '加班', '女神异闻录5预告信', 'p5预告信', '这像画吗', '小画家', '熊猫龙图', '推锅', '甩锅', '拍', '佩   
佩举', '完美', '摸', '摸摸', '摸头', 'rua', '捏', '捏脸', '像素化', 'pjsk', '世界计划', '普拉娜吃', '普拉娜舔', '顶', '玩', '玩游戏', '一起玩', 
'出警', '警察', 'ph', 'pornhub', '土豆', '捣', '打印', '舔', '舔屏', 'prpr', '可达鸭', '打拳', '四棱锥', '金字塔', '举', '举牌', '看书', '复读',
 '撕', '怒撕', '诈尸', '秽土转生', '滚', '三维旋转', '贴', '贴贴', '蹭', '蹭蹭', '快跑', '快逃', '安全感', '催眠app', '刮刮乐', '挠头', '滚屏', 
'世界第一可爱', '晃脑', '白子舔', '震惊', '别说了', '坐得住', '坐的住', '一巴掌', '口号', '砸', '卖掉了', '无语', '盯着你', 'steam消息', '踩', '
炖', '科目三', '吸', '嗦', '精神支柱', '回旋转', '旋风转', '对称', '唐可可举牌', '嘲讽', '讲课', '敲黑板', '拿捏', '戏弄', '望远镜', '体温枪', '
想什么', '这是鸡', '🐔', '丢', '扔', '抛', '掷', '捶', '捶爆', '爆捶', '紧贴', '紧紧贴着', '该走了', '一起', '上坟', '坟前比耶', '汤姆嘲笑', '顶
', '恍惚', '转', '搓', '万能表情', '空白表情', '反了', '震动', '好起来了', '墙纸', '胡桃平板', '胡桃放大', '洗衣机', '波纹', '微信支付', '最想想
的东西', '我想上的', '为什么@我', '为什么要有手', '风车转', '许愿失败', '木鱼', '膜', '膜拜', '吴京中国', '椰树椰汁', '你的跨年', 'yt', 'youtu ube', 
'致电', '你应该致电']

```
## 🐔 使用说明
- 本插件支持从原始消息中提取参数，请用空格隔开参数，如 “喜报 nmsl”
- 本插件支持从引用消息中提取参数，如“[引用的消息] 喜报”
- 提供的参数不够时，插件自动获取消息发送者、被 @ 的用户以及 bot 自身的相关参数来补充。
示例
![b421d15916a8db6109bb36c002ba2e5](https://github.com/user-attachments/assets/ec15b5f7-eec2-4552-814d-60dcc4196713)



## 📌 注意事项
1. 想第一时间得到反馈的可以来作者的插件反馈群（QQ群）：460973561
2. 感觉本插件做得还不错的话，点个star呗（右上角的星星）
3. 一些会引起不适的meme（如'射','撅'）需要自己去添加：前往[meme-generator 额外表情仓库](https://github.com/MemeCrafters/meme-generator-contrib),
 将仓库中memes文件夹里的文件添加到astrbot目录下的路径：./venv/Lib/site-packages/meme_generator/memes，然后重启astrbot即可。


## 📜 开源协议
本项目采用 [MIT License](LICENSE)
