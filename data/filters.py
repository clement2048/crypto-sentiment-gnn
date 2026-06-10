"""Filtering rules for v2 comment-block construction."""

from __future__ import annotations
from data.schema import FilterIssue, PostRecord, RawComment


REQUIRED_ROOT_FIELDS = ("t0", "t_window", "p0", "p1", "label")

# 已检查

# 如果出现了label_error，当前策略是整条帖子都不做样本构建了
def post_level_issue(post: PostRecord) -> FilterIssue | None:
    if post.label_error.strip():
        return FilterIssue(
            post_id=post.post_id,
            comment_id=None,
            reason="label_error",
            detail=post.label_error,
            source_file=post.source_file,
        )
    return None


# 校验根评论是否满足构建 CommentBlock 的要求，不满足的话记录具体问题。
# 例如出现了 comment_error、文本内容为空、缺少必要字段等问题。
def validate_root_comment(post: PostRecord, comment: RawComment) -> list[FilterIssue]:
    issues: list[FilterIssue] = []
    comment_id = comment.comment_id or None

    if comment.comment_error.strip():
        issues.append(
            FilterIssue(
                post_id=post.post_id,
                comment_id=comment_id,
                reason="comment_error",
                detail=comment.comment_error,
                source_file=post.source_file,
            )
        )

    if not comment.text.strip():
        issues.append(
            FilterIssue(
                post_id=post.post_id,
                comment_id=comment_id,
                reason="empty_text",
                source_file=post.source_file,
            )
        )

    missing = []
    if comment.t0 is None:
        missing.append("t0")
    if comment.t_window is None:
        missing.append("t_window")
    if comment.p0 is None:
        missing.append("p0")
    if comment.p1 is None:
        missing.append("p1")
    if comment.label is None:
        missing.append("label")
    if missing:
        issues.append(
            FilterIssue(
                post_id=post.post_id,
                comment_id=comment_id,
                reason="missing_required_fields",
                detail=",".join(missing),
                source_file=post.source_file,
            )
        )

    return issues



