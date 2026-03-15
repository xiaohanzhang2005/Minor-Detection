"""
长期记忆模块
跨会话存储用户画像，支持渐进式画像更新
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict, field
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import USER_MEMORY_DB_PATH
from src.models import SkillOutput


@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    
    # 累积判定
    is_minor: bool = True                    # 综合判定
    minor_confidence: float = 0.5            # 综合置信度
    total_sessions: int = 0                  # 总会话数
    
    # 画像信息
    estimated_age_range: str = "未知"
    education_stage: str = "未知"
    identity_markers: List[str] = field(default_factory=list)
    
    # ICBO 累积特征（存储最近几次的特征用于分析）
    recent_intentions: List[str] = field(default_factory=list)
    recent_cognitions: List[str] = field(default_factory=list)
    recent_behaviors: List[str] = field(default_factory=list)
    
    # 元数据
    first_seen: str = ""
    last_seen: str = ""
    
    def to_context_string(self) -> str:
        """转换为可注入 prompt 的上下文字符串"""
        parts = ["## 用户历史画像\n"]
        parts.append(f"- 历史会话数: {self.total_sessions}")
        parts.append(f"- 累积判定: {'未成年人' if self.is_minor else '成年人'} (置信度: {self.minor_confidence:.2f})")
        parts.append(f"- 推测年龄段: {self.estimated_age_range}")
        parts.append(f"- 教育阶段: {self.education_stage}")
        
        if self.identity_markers:
            parts.append(f"- 身份标记: {', '.join(self.identity_markers[:5])}")
        
        if self.recent_intentions:
            parts.append(f"- 近期意图: {'; '.join(self.recent_intentions[-3:])}")
        
        return "\n".join(parts)


class UserMemory:
    """
    用户长期记忆管理器
    使用 SQLite 存储跨会话的用户画像
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        初始化记忆管理器
        
        Args:
            db_path: 数据库路径，默认使用配置中的路径
        """
        self.db_path = Path(db_path) if db_path else USER_MEMORY_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    is_minor INTEGER DEFAULT 1,
                    minor_confidence REAL DEFAULT 0.5,
                    total_sessions INTEGER DEFAULT 0,
                    estimated_age_range TEXT DEFAULT '未知',
                    education_stage TEXT DEFAULT '未知',
                    identity_markers TEXT DEFAULT '[]',
                    recent_intentions TEXT DEFAULT '[]',
                    recent_cognitions TEXT DEFAULT '[]',
                    recent_behaviors TEXT DEFAULT '[]',
                    first_seen TEXT,
                    last_seen TEXT
                )
            """)
            
            # 会话历史表（用于追溯）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    session_time TEXT,
                    is_minor INTEGER,
                    minor_confidence REAL,
                    icbo_features TEXT,
                    user_persona TEXT,
                    reasoning TEXT,
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                )
            """)
            
            conn.commit()
    
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        获取用户画像
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户画像，如果不存在返回 None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return UserProfile(
                user_id=row["user_id"],
                is_minor=bool(row["is_minor"]),
                minor_confidence=row["minor_confidence"],
                total_sessions=row["total_sessions"],
                estimated_age_range=row["estimated_age_range"],
                education_stage=row["education_stage"],
                identity_markers=json.loads(row["identity_markers"]),
                recent_intentions=json.loads(row["recent_intentions"]),
                recent_cognitions=json.loads(row["recent_cognitions"]),
                recent_behaviors=json.loads(row["recent_behaviors"]),
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
            )
    
    def update_profile(
        self,
        user_id: str,
        skill_output: SkillOutput,
        decay_factor: float = 0.8,
    ) -> UserProfile:
        """
        根据新的分析结果更新用户画像
        
        使用指数加权平均融合新旧信息：
        new_confidence = decay * old_confidence + (1 - decay) * new_confidence
        
        Args:
            user_id: 用户ID
            skill_output: 本次分析结果
            decay_factor: 历史衰减因子，越大表示越信任历史
            
        Returns:
            更新后的用户画像
        """
        now = datetime.now().isoformat()
        existing = self.get_profile(user_id)
        
        if existing is None:
            # 新用户：直接使用本次结果
            profile = UserProfile(
                user_id=user_id,
                is_minor=skill_output.is_minor,
                minor_confidence=skill_output.minor_confidence,
                total_sessions=1,
                estimated_age_range=skill_output.user_persona.age_range,
                education_stage=skill_output.user_persona.education_stage,
                identity_markers=skill_output.user_persona.identity_markers[:10],
                recent_intentions=[skill_output.icbo_features.intention],
                recent_cognitions=[skill_output.icbo_features.cognition],
                recent_behaviors=[skill_output.icbo_features.behavior_style],
                first_seen=now,
                last_seen=now,
            )
        else:
            # 老用户：融合更新
            # 置信度加权融合
            new_confidence = (
                decay_factor * existing.minor_confidence +
                (1 - decay_factor) * skill_output.minor_confidence
            )
            
            # 判定结果更新（基于融合后的置信度）
            is_minor = new_confidence >= 0.5
            
            # 年龄/教育阶段：如果新结果更具体则更新
            age_range = skill_output.user_persona.age_range
            if age_range == "未知" or age_range == "":
                age_range = existing.estimated_age_range
            
            edu_stage = skill_output.user_persona.education_stage
            if edu_stage == "未知" or edu_stage == "":
                edu_stage = existing.education_stage
            
            # 身份标记：合并去重
            markers = list(set(
                existing.identity_markers +
                skill_output.user_persona.identity_markers
            ))[:15]  # 最多保留15个
            
            # ICBO 特征：追加并保留最近 5 条
            recent_intentions = (
                existing.recent_intentions +
                [skill_output.icbo_features.intention]
            )[-5:]
            recent_cognitions = (
                existing.recent_cognitions +
                [skill_output.icbo_features.cognition]
            )[-5:]
            recent_behaviors = (
                existing.recent_behaviors +
                [skill_output.icbo_features.behavior_style]
            )[-5:]
            
            profile = UserProfile(
                user_id=user_id,
                is_minor=is_minor,
                minor_confidence=new_confidence,
                total_sessions=existing.total_sessions + 1,
                estimated_age_range=age_range,
                education_stage=edu_stage,
                identity_markers=markers,
                recent_intentions=recent_intentions,
                recent_cognitions=recent_cognitions,
                recent_behaviors=recent_behaviors,
                first_seen=existing.first_seen,
                last_seen=now,
            )
        
        # 保存画像
        self._save_profile(profile)
        
        # 记录会话历史
        self._save_session_history(user_id, skill_output, now)
        
        return profile
    
    def _save_profile(self, profile: UserProfile):
        """保存用户画像"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_profiles
                (user_id, is_minor, minor_confidence, total_sessions,
                 estimated_age_range, education_stage, identity_markers,
                 recent_intentions, recent_cognitions, recent_behaviors,
                 first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.user_id,
                int(profile.is_minor),
                profile.minor_confidence,
                profile.total_sessions,
                profile.estimated_age_range,
                profile.education_stage,
                json.dumps(profile.identity_markers, ensure_ascii=False),
                json.dumps(profile.recent_intentions, ensure_ascii=False),
                json.dumps(profile.recent_cognitions, ensure_ascii=False),
                json.dumps(profile.recent_behaviors, ensure_ascii=False),
                profile.first_seen,
                profile.last_seen,
            ))
            conn.commit()
    
    def _save_session_history(
        self,
        user_id: str,
        skill_output: SkillOutput,
        session_time: str,
    ):
        """保存会话历史"""
        # Pydantic 模型使用 model_dump() 而不是 asdict()
        icbo_dict = skill_output.icbo_features.model_dump() if hasattr(skill_output.icbo_features, 'model_dump') else asdict(skill_output.icbo_features)
        persona_dict = skill_output.user_persona.model_dump() if hasattr(skill_output.user_persona, 'model_dump') else asdict(skill_output.user_persona)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO session_history
                (user_id, session_time, is_minor, minor_confidence,
                 icbo_features, user_persona, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                session_time,
                int(skill_output.is_minor),
                skill_output.minor_confidence,
                json.dumps(icbo_dict, ensure_ascii=False),
                json.dumps(persona_dict, ensure_ascii=False),
                skill_output.reasoning,
            ))
            conn.commit()
    
    def get_session_history(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        获取用户会话历史
        
        Args:
            user_id: 用户ID
            limit: 返回的最大记录数
            
        Returns:
            会话历史列表
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM session_history
                WHERE user_id = ?
                ORDER BY session_time DESC
                LIMIT ?
            """, (user_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_all_users(self) -> List[str]:
        """获取所有用户ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT user_id FROM user_profiles")
            return [row[0] for row in cursor.fetchall()]
    
    def delete_user(self, user_id: str):
        """删除用户数据"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM session_history WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
            conn.commit()
    
    def clear_all(self):
        """清空所有数据（危险操作）"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM session_history")
            conn.execute("DELETE FROM user_profiles")
            conn.commit()
        print(f"⚠️ 已清空所有用户记忆数据")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            total_users = conn.execute(
                "SELECT COUNT(*) FROM user_profiles"
            ).fetchone()[0]
            
            minor_users = conn.execute(
                "SELECT COUNT(*) FROM user_profiles WHERE is_minor = 1"
            ).fetchone()[0]
            
            total_sessions = conn.execute(
                "SELECT SUM(total_sessions) FROM user_profiles"
            ).fetchone()[0] or 0
            
            return {
                "total_users": total_users,
                "minor_users": minor_users,
                "adult_users": total_users - minor_users,
                "total_sessions": total_sessions,
            }


def get_user_context(user_id: str, memory: Optional[UserMemory] = None) -> str:
    """
    获取用户上下文字符串（用于注入 prompt）
    
    Args:
        user_id: 用户ID
        memory: UserMemory 实例
        
    Returns:
        可注入 prompt 的上下文字符串
    """
    if memory is None:
        memory = UserMemory()
    
    profile = memory.get_profile(user_id)
    if profile is None:
        return ""
    
    return profile.to_context_string()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="用户记忆管理")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")
    parser.add_argument("--clear", action="store_true", help="清空所有数据")
    parser.add_argument("--test", action="store_true", help="运行测试")
    
    args = parser.parse_args()
    
    memory = UserMemory()
    
    if args.stats:
        stats = memory.get_statistics()
        print("📊 用户记忆统计:")
        for k, v in stats.items():
            print(f"  - {k}: {v}")
    
    if args.clear:
        confirm = input("确定要清空所有用户数据吗？(yes/no): ")
        if confirm.lower() == "yes":
            memory.clear_all()
    
    if args.test:
        print("🧪 测试用户记忆模块...")
        
        # 模拟 SkillOutput
        from src.models import SkillOutput, ICBOFeatures, UserPersona, RiskLevel
        
        mock_output = SkillOutput(
            is_minor=True,
            minor_confidence=0.9,
            risk_level=RiskLevel.LOW,
            icbo_features=ICBOFeatures(
                intention="寻求学习帮助",
                cognition="对数学有畏难情绪",
                behavior_style="语气较为焦虑",
                opportunity_time="晚间",
            ),
            user_persona=UserPersona(
                age=15,
                age_range="14-16岁",
                gender="女",
                education_stage="初中",
                identity_markers=["学生", "初三"],
            ),
            reasoning="用户提到数学考试和作业，符合初中生特征",
            key_evidence=["明天考数学", "作业太多"],
        )
        
        # 测试更新
        profile = memory.update_profile("test_user_001", mock_output)
        print(f"✅ 创建用户画像: {profile.user_id}")
        print(f"   is_minor: {profile.is_minor}")
        print(f"   confidence: {profile.minor_confidence:.2f}")
        print(f"   sessions: {profile.total_sessions}")
        
        # 模拟第二次会话
        mock_output_2 = SkillOutput(
            is_minor=True,
            minor_confidence=0.85,
            risk_level=RiskLevel.LOW,
            icbo_features=ICBOFeatures(
                intention="分享心事",
                cognition="对同学关系有困惑",
                behavior_style="表达较为随意",
                opportunity_time="下午",
            ),
            user_persona=UserPersona(
                age_range="14-16岁",
                education_stage="初中",
                identity_markers=["学生"],
            ),
            reasoning="继续表现出初中生特征",
            key_evidence=["班上同学"],
        )
        
        profile = memory.update_profile("test_user_001", mock_output_2)
        print(f"\n✅ 更新用户画像:")
        print(f"   sessions: {profile.total_sessions}")
        print(f"   confidence: {profile.minor_confidence:.2f}")
        print(f"   markers: {profile.identity_markers}")
        
        # 测试上下文生成
        context = profile.to_context_string()
        print(f"\n📝 上下文字符串:\n{context}")
        
        # 清理测试数据
        memory.delete_user("test_user_001")
        print("\n🗑️ 已清理测试数据")
