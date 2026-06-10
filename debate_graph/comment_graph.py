"""Build reply graphs from CommentBlock trees."""

from __future__ import annotations

from data.schema import CommentBlock, RawComment
from debate_graph.schema import GraphEdge, GraphNode, comment_node_id


def build_comment_graph(block: CommentBlock) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Build comment nodes and reply edges for one CommentBlock."""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    root = block.root_comment
    nodes.append(_comment_node(root, depth=0, is_root=True))
    _walk_replies(root.replies, parent=root, depth=1, nodes=nodes, edges=edges)
    return nodes, edges


def _walk_replies(
    replies: list[RawComment],
    parent: RawComment,
    depth: int,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> None:
    for reply in replies:
        nodes.append(_comment_node(reply, depth=depth, is_root=False))
        edges.append(
            GraphEdge(
                source=comment_node_id(reply.comment_id),
                target=comment_node_id(parent.comment_id),
                relation="reply",
                attrs={"depth": depth},
            )
        )
        _walk_replies(reply.replies, parent=reply, depth=depth + 1, nodes=nodes, edges=edges)


def _comment_node(comment: RawComment, depth: int, is_root: bool) -> GraphNode:
    return GraphNode(
        node_id=comment_node_id(comment.comment_id),
        node_type="comment",
        ref_id=comment.comment_id,
        text=comment.text,
        attrs={
            "author": comment.author,
            "depth": depth,
            "is_root": is_root,
            "post_time": comment.post_time.isoformat(sep=" ") if comment.post_time else None,
        },
    )



