"""按时间索引的用户历史库。

ProfileStore 的任务是保证画像构建不偷看未来：
给定一个 CommentBlock 的 t0，只能取该用户在 `comment_time < t0` 的历史评论。
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from datetime import datetime

from data.schema import CommentBlock, flatten_replies
from profiles.user_profile import (
    UserHistoryRecord,
    UserProfile,
    build_user_profile,
)


class ProfileStore:
    """索引用户历史，并按 block.t0 生成时间安全画像。"""

    def __init__(self, histories: dict[str, list[UserHistoryRecord]] | None = None):
        self._histories: dict[str, list[UserHistoryRecord]] = histories or {}
        self._timestamps: dict[str, list[datetime]] = {
            author: [record.timestamp for record in records]
            for author, records in self._histories.items()
        }

    @classmethod
    def from_blocks(cls, blocks: list[CommentBlock]) -> "ProfileStore":
        """从全部 CommentBlock 中收集用户历史。

        注意：这里先把所有历史收集起来没有问题，因为真正取画像时会用
        `get_history_before(author, t0)` 做时间截断。
        """
        histories: dict[str, list[UserHistoryRecord]] = defaultdict(list)
        for block in blocks:
            root = block.root_comment
            if root.author and root.post_time is not None:
                # 根评论有价格窗口标签，因此可用于历史 stance / reaction consistency。
                histories[root.author].append(
                    UserHistoryRecord(
                        author=root.author,
                        text=root.text,
                        timestamp=root.post_time,
                        label=block.label,
                        product=block.product,
                        p0=block.p0,
                        p1=block.p1,
                        reply_count=len(flatten_replies(block.replies)),
                    )
                )

            for reply in flatten_replies(block.replies):
                if reply.author and reply.post_time is not None:
                    # reply 通常没有价格窗口标签，但仍可用于 activity / influence 等画像特征。
                    histories[reply.author].append(
                        UserHistoryRecord(
                            author=reply.author,
                            text=reply.text,
                            timestamp=reply.post_time,
                            label=reply.label,
                            product=block.product,
                            reply_count=len(reply.replies),
                        )
                    )

        sorted_histories = {
            author: sorted(records, key=lambda item: item.timestamp)
            for author, records in histories.items()
        }
        return cls(sorted_histories)

    def get_history_before(self, author: str, t0) -> list[UserHistoryRecord]:
        """返回某用户在 t0 之前的历史记录，不包含 t0 当刻及之后。"""
        records = self._histories.get(author, [])
        timestamps = self._timestamps.get(author, [])
        # bisect_left 会把 timestamp == t0 的记录排除在外，符合严格 `t < t0`。
        end = bisect_left(timestamps, t0)
        return records[:end]

    def get_profile(self, author: str, t0) -> UserProfile:
        """构建单个用户在某个时间点可见的画像。"""
        profile = build_user_profile(author, self.get_history_before(author, t0), as_of=t0)
        if profile.built_until is not None and profile.built_until > t0:
            raise AssertionError("UserProfile.built_until must be <= block.t0")
        return profile

    def get_profiles_for_block(self, block: CommentBlock) -> dict[str, UserProfile]:
        """为一个 CommentBlock 中出现的所有作者生成画像。"""
        profiles = {
            author: self.get_profile(author, block.t0)
            for author in block.participants()
        }
        for profile in profiles.values():
            if profile.built_until is not None and profile.built_until > block.t0:
                raise AssertionError("UserProfile.built_until must be <= block.t0")
        return profiles

