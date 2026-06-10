from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from pathlib import Path

from data import build_comment_blocks, load_posts, temporal_split_blocks
from data.schema import CommentBlock, PostRecord, RawComment
from profiles import ProfileStore


class StageOnePipelineTest(unittest.TestCase):
    def test_jsonl_single_line_loads_post_record(self):
        path = Path(__file__).parent / "fixtures" / "sample_post.jsonl"
        posts = load_posts(path)

        self.assertEqual(len(posts), 1)
        self.assertIsInstance(posts[0], PostRecord)
        self.assertEqual(posts[0].post_id, "p1")
        self.assertEqual(posts[0].comments[0].t0, datetime(2026, 5, 30, 16, 0, 0))

    def test_multiple_root_comments_build_multiple_blocks(self):
        post = PostRecord.from_dict(
            _post_dict(
                "p1",
                comments=[
                    _comment_dict("c1", text="bull", label=1),
                    _comment_dict("c2", text="bear", label=-1, minutes=5),
                ],
            )
        )

        blocks, issues = build_comment_blocks([post])

        self.assertEqual(len(issues), 0)
        self.assertEqual([block.block_id for block in blocks], ["p1:c1", "p1:c2"])
        self.assertEqual([block.label for block in blocks], [1, -1])

    def test_filter_issues_are_recorded(self):
        posts = [
            PostRecord.from_dict(_post_dict("bad-post", label_error="bad label")),
            PostRecord.from_dict(
                _post_dict(
                    "bad-comment",
                    comments=[
                        _comment_dict("c1", text="", label=1),
                        _comment_dict("c2", text="missing", label=None),
                        _comment_dict("c3", text="error", label=1, comment_error="bad comment"),
                    ],
                )
            ),
        ]

        blocks, issues = build_comment_blocks(posts)

        self.assertEqual(len(blocks), 0)
        reasons = [issue.reason for issue in issues]
        self.assertIn("label_error", reasons)
        self.assertIn("empty_text", reasons)
        self.assertIn("missing_required_fields", reasons)
        self.assertIn("comment_error", reasons)

    def test_temporal_split_is_sorted_by_t0(self):
        base = datetime(2026, 1, 1, 0, 0, 0)
        blocks = [_make_block(f"b{i}", base + timedelta(minutes=i)) for i in reversed(range(10))]

        split = temporal_split_blocks(blocks)

        self.assertEqual(len(split.train), 7)
        self.assertEqual(len(split.val), 1)
        self.assertEqual(len(split.test), 2)
        self.assertLessEqual(max(block.t0 for block in split.train), min(block.t0 for block in split.val))
        self.assertLessEqual(max(block.t0 for block in split.val), min(block.t0 for block in split.test))

    def test_profile_uses_only_history_before_t0(self):
        old_time = datetime(2026, 1, 1, 10, 0, 0)
        target_time = datetime(2026, 1, 1, 11, 0, 0)
        future_time = datetime(2026, 1, 1, 12, 0, 0)
        blocks = [
            _make_block("old", old_time, author="alice", label=1),
            _make_block("target", target_time, author="alice", label=-1),
            _make_block("future", future_time, author="alice", label=-1),
        ]
        store = ProfileStore.from_blocks(blocks)

        target_profile = store.get_profiles_for_block(blocks[1])["alice"]

        self.assertEqual(target_profile.history_count, 1)
        self.assertEqual(target_profile.stance_bias, 1.0)
        self.assertLess(target_profile.built_until, blocks[1].t0)

    def test_cold_start_profile(self):
        block = _make_block("cold", datetime(2026, 1, 1, 11, 0, 0), author="new-user")
        store = ProfileStore.from_blocks([])

        profile = store.get_profiles_for_block(block)["new-user"]

        self.assertEqual(profile.history_count, 0)
        self.assertEqual(profile.stance_bias, 0.0)
        self.assertEqual(profile.built_until, block.t0)


def _post_dict(post_id: str, comments: list[dict] | None = None, label_error: str = "") -> dict:
    return {
        "post_id": post_id,
        "post_content": "news body",
        "post_time": "2026-05-30 15:00:00",
        "products": ["BNB"],
        "first_product": "BNB",
        "market_type": "spot",
        "comments": comments or [_comment_dict("c1")],
        "label_error": label_error,
    }


def _comment_dict(
    comment_id: str,
    text: str = "鐗涙潵浜嗭紵",
    label: int | None = 1,
    minutes: int = 0,
    comment_error: str = "",
) -> dict:
    t0 = datetime(2026, 5, 30, 16, 0, 0) + timedelta(minutes=minutes)
    data = {
        "comment_id": comment_id,
        "original_comment_id": f"orig-{comment_id}",
        "author": "alice",
        "text": text,
        "post_time": t0.strftime("%Y-%m-%d %H:%M:%S"),
        "replies": [],
        "t0": t0.strftime("%Y-%m-%d %H:%M:%S"),
        "t_window": "24h",
        "p0": 100.0,
        "p1": 110.0 if label == 1 else 90.0,
        "comment_error": comment_error,
    }
    if label is not None:
        data["label"] = label
    return data


def _make_block(
    block_id: str,
    t0: datetime,
    author: str = "alice",
    label: int = 1,
) -> CommentBlock:
    root = RawComment(
        comment_id=block_id,
        original_comment_id=f"orig-{block_id}",
        author=author,
        text=f"text-{block_id}",
        post_time=t0,
        t0=t0,
        t_window="24h",
        p0=100.0,
        p1=110.0 if label == 1 else 90.0,
        label=label,
    )
    return CommentBlock(
        block_id=f"post:{block_id}",
        post_id="post",
        post_content="news body",
        products=["BNB"],
        root_comment=root,
        replies=[],
        t0=t0,
        t_window="24h",
        p0=100.0,
        p1=110.0 if label == 1 else 90.0,
        label=label,
        product="BNB",
        market_type="spot",
        post_time=t0 - timedelta(minutes=5),
    )


if __name__ == "__main__":
    unittest.main()


