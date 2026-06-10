"""把原始帖子拆成 CommentBlock 样本。

一个 CommentBlock = 一个根评论 + 它下面的所有 replies。
这是 v2 设计里的核心样本粒度：后续画像、辩论、图模型、训练标签都围绕它展开。
"""

from __future__ import annotations

from data.filters import post_level_issue, validate_root_comment
from data.schema import CommentBlock, FilterIssue, PostRecord

# ✅已检查

# 构建评论块
def build_comment_blocks(posts: list[PostRecord]) -> tuple[list[CommentBlock], list[FilterIssue]]:
    """将 PostRecord 列表转换成 CommentBlock 列表。

    返回两个列表：
    - blocks：可用样本，每个样本的 label 只来自根评论自己的 `label`
    - issues：被过滤掉的帖子/评论及原因，方便后续检查数据质量
    """
    blocks: list[CommentBlock] = []
    issues: list[FilterIssue] = []

    for post in posts:
        # 帖子级 label_error 表示整条帖子标注有问题，当前策略是整帖跳过。
        issue = post_level_issue(post)
        if issue is not None:
            issues.append(issue)
            continue

        for root_comment in post.comments:
            # 只校验根评论，因为根评论的 t0/p0/p1/label 决定这个 block 的监督标签。
            root_issues = validate_root_comment(post, root_comment)
            if root_issues:
                issues.extend(root_issues)
                continue

            # 经过 validate_root_comment 后，这些字段必须存在。
            # assert 既是给类型检查看的，也是防止未来改过滤逻辑时悄悄放进坏样本。
            assert root_comment.t0 is not None
            assert root_comment.t_window is not None
            assert root_comment.p0 is not None
            assert root_comment.p1 is not None
            assert root_comment.label is not None

            # block_id 固定为 post_id:comment_id，后续图、辩论、输出文件都用这个 ID 对齐。
            block_id = f"{post.post_id}:{root_comment.comment_id}"
            product = post.first_product or (post.products[0] if post.products else None)
            blocks.append(
                CommentBlock(
                    block_id=block_id,
                    post_id=post.post_id,
                    post_content=post.post_content,
                    products=post.products,
                    root_comment=root_comment,
                    replies=root_comment.replies,
                    t0=root_comment.t0,
                    t_window=root_comment.t_window,
                    p0=root_comment.p0,
                    p1=root_comment.p1,
                    label=root_comment.label,
                    product=product,
                    market_type=post.market_type,
                    post_time=post.post_time,
                )
            )

    return blocks, issues



