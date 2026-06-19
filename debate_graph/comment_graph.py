"""Build comment nodes for the v3 single-relation graph.

Reply structure is stored in comment node attrs["parent_id"] instead of reply edges.
"""

from __future__ import annotations

from data.schema import CommentBlock, RawComment
from debate_graph.schema import GraphEdge, GraphNode, comment_node_id


def build_comment_graph(block: CommentBlock) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Build comment nodes; no reply edges are emitted in v3."""
    nodes: list[GraphNode] = []

    root = block.root_comment
    nodes.append(_comment_node(root, depth=0, is_root=True, parent_id=None))
    _walk_replies(root.replies, parent=root, depth=1, nodes=nodes)
    return nodes, []


def _walk_replies(
    replies: list[RawComment],
    parent: RawComment,
    depth: int,
    nodes: list[GraphNode],
) -> None:
    for reply in replies:
        nodes.append(_comment_node(reply, depth=depth, is_root=False, parent_id=parent.comment_id))
        _walk_replies(reply.replies, parent=reply, depth=depth + 1, nodes=nodes)


def _comment_node(comment: RawComment, depth: int, is_root: bool, parent_id: str | None) -> GraphNode:
    return GraphNode(
        node_id=comment_node_id(comment.comment_id),
        node_type="comment",
        ref_id=comment.comment_id,
        text=comment.text,
        attrs={
            "author": comment.author,
            "depth": depth,
            "is_root": is_root,
            "parent_id": parent_id,
            "post_time": comment.post_time.isoformat(sep=" ") if comment.post_time else None,
        },
    )



