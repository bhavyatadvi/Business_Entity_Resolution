
import pandas as pd


REQUIRED_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
]


def audit_candidate_components(
    candidates: pd.DataFrame,
) -> dict:
    """
    Audit connected components in the bipartite candidate graph.

    Each Source 1 entity and each target entity is a node.
    Every candidate pair creates an edge between those nodes.

    A component-aware split can keep both sides of the matching
    graph separate across train, validation, and test.
    """

    missing = set(REQUIRED_COLUMNS) - set(candidates.columns)

    if missing:
        raise ValueError(
            f"Candidates are missing columns: {sorted(missing)}"
        )

    if candidates.empty:
        raise ValueError("Candidates dataframe is empty")

    parent = {}
    size = {}

    def find(node):
        parent.setdefault(node, node)
        size.setdefault(node, 1)

        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]

        return node

    def union(left, right):
        root_left = find(left)
        root_right = find(right)

        if root_left == root_right:
            return

        # Union by size keeps the tree shallow.
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left

        parent[root_right] = root_left
        size[root_left] += size[root_right]

    for row in candidates[REQUIRED_COLUMNS].itertuples(
        index=False,
        name=None,
    ):
        source1_id, target_id, target_source = row

        source1_node = ("source1", str(source1_id))
        target_node = (str(target_source), str(target_id))

        union(source1_node, target_node)

    component_sizes = {}

    for node in parent:
        root = find(node)
        component_sizes[root] = component_sizes.get(root, 0) + 1

    sizes = sorted(component_sizes.values(), reverse=True)
    total_nodes = sum(sizes)

    return {
        "total_nodes": total_nodes,
        "total_components": len(sizes),
        "largest_component_nodes": sizes[0],
        "largest_component_fraction": sizes[0] / total_nodes,
        "components_with_at_least_2_nodes": sum(
            component_size >= 2 for component_size in sizes
        ),
        "largest_10_component_sizes": sizes[:10],
    }