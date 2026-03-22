import json
from pathlib import Path

payload = {
  "mode": "single_session",
  "conversation": [
    {"role": "user", "content": "救命，真的受够了，宿舍那帮人真的有病吧！！！"},
    {"role": "assistant", "content": "听起来你现在心情非常糟糕，而且在宿舍遇到了很不开心的事情。能跟我说说发生了什么吗？我会一直在这里听你说。"},
    {"role": "user", "content": "就那个A，之前天天跟我吐槽B，说B有多恶心不想跟她玩，结果现在呢？"},
    {"role": "user", "content": "她居然转头就跟B好上了，还把之前她自己说的话全赖在我头上！说是我在背后骂B！真的绝了，怎么有这么不要脸的人啊？"},
    {"role": "assistant", "content": "这种被原本信任的朋友背叛，还要替对方背黑锅的感觉确实太让人委屈和愤怒了。换做是谁可能都很难冷静下来。她们现在在寝室里是怎么对你的？"},
    {"role": "user", "content": "她们现在合伙孤立我啊，在寝室里阴阳怪气的，故意当着我面说我坏话，我一进去她们就闭嘴然后冷笑，那种眼神你懂吗？"},
    {"role": "user", "content": "我真的不想呆了，一回宿舍就想哭，感觉心跳特别快，手都在抖，我真的想明天就回家，再也不想回这个破学校了。"},
    {"role": "assistant", "content": "这种被排挤的冷暴力环境确实非常压抑，你现在的生理反应说明你已经承受了太大的压力。逃避虽然是一种本能，但我们也可以想想有没有什么办法能让你稍微舒服一点？比如找老师调换宿舍，或者跟父母说说你的处境？"},
    {"role": "user", "content": "没用的，告老师只会让她们觉得我是在装可怜告黑状，到时候只会更排挤我。我刚才看着寝室阳台，真的想一了百了算了，反正也没人真的在乎我，活着真的好累啊。"}
  ],
  "sessions": [],
  "context": {
    "time_features": {},
    "retrieved_cases": [],
    "prior_profile": {},
    "raw_time_hint": "2025-11-20 周四 23:25",
    "channel": "",
    "locale": ""
  },
  "meta": {
    "user_id": "social_pos_001674_gemini-3-flash-preview_1675",
    "request_id": "social_pos_001674_gemini-3-flash-preview_1675",
    "source": "codex_skill_runner"
  }
}

Path("payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
