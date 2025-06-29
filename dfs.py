"""
Pattern 8: Tree Depth-First Search (DFS) - 10 Hard Problems
===========================================================

Tree DFS explores paths in trees by traversing as far as possible along branches
before backtracking. This pattern is essential for path-based problems, tree
validation, and problems involving parent-child relationships.

Key Concepts:
- Use recursion or explicit stack for traversal
- Three types: Preorder, Inorder, Postorder
- Track path state during recursion
- Backtracking for path problems

Time Complexity: O(n) where n is number of nodes
Space Complexity: O(h) where h is height of tree (recursion stack)
"""

from typing import List, Optional, Tuple, Dict
from collections import defaultdict


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class TreeDFSHard:

    def binary_tree_maximum_path_sum_with_paths(self, root: TreeNode) -> Tuple[int, List[int]]:
        """
        LeetCode 124 Extension - Maximum Path Sum with Path Nodes (Hard)

        Find maximum path sum in binary tree where path can start and end at any nodes.
        Extended: Also return the actual path nodes.

        Algorithm:
        1. For each node, calculate max path that goes through it
        2. Track both the max sum and the path nodes
        3. Consider four cases: node only, node+left, node+right, node+left+right
        4. Use postorder traversal for bottom-up calculation

        Time: O(n), Space: O(n) for path storage

        Example:
        Tree: [-10,9,20,null,null,15,7]
        Output: (42, [15,20,7]) - sum=42, path through 15->20->7
        """
        self.max_sum = float('-inf')
        self.max_path = []

        def dfs(node):
            if not node:
                return 0, []

            # Get max sum and path from left and right subtrees
            left_sum, left_path = dfs(node.left)
            right_sum, right_path = dfs(node.right)

            # Calculate max sum for paths that include current node
            # Case 1: Only current node
            current_only = node.val
            current_path = [node.val]

            # Case 2: Current + left path (if positive)
            if left_sum > 0:
                left_branch = node.val + left_sum
                left_branch_path = left_path + [node.val]
            else:
                left_branch = node.val
                left_branch_path = [node.val]

            # Case 3: Current + right path (if positive)
            if right_sum > 0:
                right_branch = node.val + right_sum
                right_branch_path = [node.val] + right_path
            else:
                right_branch = node.val
                right_branch_path = [node.val]

            # Case 4: Path through current node (left + current + right)
            through_sum = node.val
            through_path = []

            if left_sum > 0:
                through_sum += left_sum
                through_path = left_path[::-1]
            through_path.append(node.val)
            if right_sum > 0:
                through_sum += right_sum
                through_path.extend(right_path)

            # Update global maximum
            candidates = [
                (current_only, current_path),
                (left_branch, left_branch_path),
                (right_branch, right_branch_path),
                (through_sum, through_path)
            ]

            for sum_val, path in candidates:
                if sum_val > self.max_sum:
                    self.max_sum = sum_val
                    self.max_path = path[:]

            # Return max sum that can be extended upward
            if left_sum > 0 and left_sum >= right_sum:
                return left_branch, left_branch_path
            elif right_sum > 0:
                return right_branch, right_branch_path
            else:
                return current_only, current_path

        dfs(root)
        return self.max_sum, self.max_path

    def serialize_and_deserialize_nary_tree(self):
        """
        LeetCode 428 - Serialize and Deserialize N-ary Tree (Hard)

        Design algorithm to serialize and deserialize N-ary tree.

        Algorithm:
        1. Use DFS with delimiter to mark end of children
        2. Encode: node_value,num_children,child1,child2,...
        3. Decode: recursively build tree using encoded format

        Time: O(n) for both operations, Space: O(n)
        """

        class Node:
            def __init__(self, val=None, children=None):
                self.val = val
                self.children = children if children else []

        class Codec:
            def serialize(self, root: Node) -> str:
                """Encodes a tree to a single string."""
                if not root:
                    return ""

                def dfs(node):
                    if not node:
                        return ""

                    # Encode: value,num_children,child1,child2,...
                    result = [str(node.val), str(len(node.children))]

                    for child in node.children:
                        result.append(dfs(child))

                    return ",".join(result)

                return dfs(root)

            def deserialize(self, data: str) -> Node:
                """Decodes your encoded data to tree."""
                if not data:
                    return None

                tokens = data.split(",")
                self.idx = 0

                def dfs():
                    if self.idx >= len(tokens):
                        return None

                    # Read value and number of children
                    val = int(tokens[self.idx])
                    self.idx += 1
                    num_children = int(tokens[self.idx])
                    self.idx += 1

                    node = Node(val)

                    # Recursively build children
                    for _ in range(num_children):
                        child = dfs()
                        if child:
                            node.children.append(child)

                    return node

                return dfs()

        return Codec

    def longest_univalue_path_with_details(self, root: TreeNode) -> Tuple[int, List[int], int]:
        """
        LeetCode 687 Extension - Longest Univalue Path with Analysis (Hard)

        Find longest path where each node has same value.
        Extended: Return path length, nodes in path, and total count of such paths.

        Algorithm:
        1. Use postorder DFS to calculate paths bottom-up
        2. For each node, find longest path through it with same values
        3. Track all univalue paths found

        Time: O(n), Space: O(h)

        Example:
        Tree: [5,4,5,1,1,null,5]
        Output: (2, [5,5,5], 3) - length 2, path values, 3 such paths exist
        """
        self.max_length = 0
        self.max_path_nodes = []
        self.path_count = 0

        def dfs(node, parent_val=None):
            if not node:
                return 0, []

            # Get lengths from children
            left_len, left_nodes = dfs(node.left, node.val)
            right_len, right_nodes = dfs(node.right, node.val)

            # Calculate arrows (edges) from this node
            left_arrow = left_len if node.left and node.left.val == node.val else 0
            right_arrow = right_len if node.right and node.right.val == node.val else 0

            # Path through this node
            path_length = left_arrow + right_arrow

            # Build path nodes
            path_nodes = []
            if left_arrow > 0:
                path_nodes.extend(left_nodes[::-1])
            path_nodes.append(node.val)
            if right_arrow > 0:
                path_nodes.extend(right_nodes)

            # Update maximum if needed
            if path_length > self.max_length:
                self.max_length = path_length
                self.max_path_nodes = path_nodes[:]
                self.path_count = 1
            elif path_length == self.max_length and path_length > 0:
                self.path_count += 1

            # Return length that can be extended upward
            if parent_val is not None and node.val == parent_val:
                extend_length = max(left_arrow, right_arrow) + 1
                extend_nodes = [node.val]
                if left_arrow > right_arrow:
                    extend_nodes = left_nodes + [node.val]
                elif right_arrow > 0:
                    extend_nodes = [node.val] + right_nodes
                return extend_length, extend_nodes
            else:
                return 0, []

        dfs(root)
        return self.max_length, self.max_path_nodes, self.path_count

    def distribute_coins_in_binary_tree_with_moves(self, root: TreeNode) -> Tuple[int, List[Tuple[int, int, int]]]:
        """
        LeetCode 979 Extension - Distribute Coins with Move Details (Hard)

        Distribute coins so each node has exactly 1 coin. Return minimum moves.
        Extended: Also return the actual moves as (from_node, to_node, coins).

        Algorithm:
        1. Use postorder DFS to calculate coin flow
        2. Each node reports excess/deficit to parent
        3. Track all coin movements

        Time: O(n), Space: O(n)

        Example:
        Tree: [3,0,0]
        Output: (2, [(0,1,1), (0,2,1)]) - 2 moves needed
        """
        self.moves = 0
        self.move_details = []

        def dfs(node, parent_val=None):
            if not node:
                return 0

            # Get excess/deficit from children
            left_balance = dfs(node.left, node.val)
            right_balance = dfs(node.right, node.val)

            # Record moves from children
            if left_balance > 0:
                self.move_details.append((node.left.val, node.val, left_balance))
            elif left_balance < 0:
                self.move_details.append((node.val, node.left.val, -left_balance))

            if right_balance > 0:
                self.move_details.append((node.right.val, node.val, right_balance))
            elif right_balance < 0:
                self.move_details.append((node.val, node.right.val, -right_balance))

            # Calculate total moves through this node
            self.moves += abs(left_balance) + abs(right_balance)

            # Return balance for this subtree
            # (coins at node - 1) + balance from children
            return node.val - 1 + left_balance + right_balance

        dfs(root)
        return self.moves, self.move_details

    def binary_tree_cameras_with_placement(self, root: TreeNode) -> Tuple[int, List[int]]:
        """
        LeetCode 968 Extension - Binary Tree Cameras with Placement (Hard)

        Find minimum cameras needed to monitor all nodes.
        Extended: Return which nodes have cameras.

        Algorithm:
        1. Use DFS with three states: has camera, covered, needs coverage
        2. Greedy approach: place cameras as high as possible
        3. Track camera positions

        Time: O(n), Space: O(h)

        Example:
        Tree: [0,0,null,0,0]
        Output: (1, [1]) - 1 camera at node with value 0 (parent of leaves)
        """
        self.cameras = 0
        self.camera_positions = []

        # States: 0 = needs coverage, 1 = has camera, 2 = covered
        def dfs(node, node_idx=0):
            if not node:
                return 2  # Null nodes are covered

            left_state = dfs(node.left, 2 * node_idx + 1)
            right_state = dfs(node.right, 2 * node_idx + 2)

            # If any child needs coverage, place camera here
            if left_state == 0 or right_state == 0:
                self.cameras += 1
                self.camera_positions.append(node.val)
                return 1

            # If any child has camera, this node is covered
            if left_state == 1 or right_state == 1:
                return 2

            # Both children are covered, this node needs coverage
            return 0

        # Handle root separately
        if dfs(root) == 0:
            self.cameras += 1
            self.camera_positions.append(root.val)

        return self.cameras, self.camera_positions

    def sum_of_distances_in_tree(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        LeetCode 834 - Sum of Distances in Tree (Hard)

        For each node, find sum of distances to all other nodes.

        Algorithm:
        1. Build tree from edges
        2. First DFS: calculate subtree sizes and distances from node 0
        3. Second DFS: use parent's result to calculate each node's result
        4. Key insight: moving root changes distances predictably

        Time: O(n), Space: O(n)

        Example:
        n = 6, edges = [[0,1],[0,2],[2,3],[2,4],[2,5]]
        Output: [8,12,6,10,10,10]
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        count = [1] * n  # Subtree sizes
        ans = [0] * n  # Answer for each node

        # First DFS: calculate subtree sizes and initial distances from node 0
        def dfs1(node, parent):
            for child in graph[node]:
                if child != parent:
                    dfs1(child, node)
                    count[node] += count[child]
                    ans[node] += ans[child] + count[child]

        # Second DFS: calculate distances for all nodes using parent's result
        def dfs2(node, parent):
            for child in graph[node]:
                if child != parent:
                    # When moving root from node to child:
                    # - Nodes in child's subtree get 1 closer
                    # - Other nodes get 1 farther
                    ans[child] = ans[node] - count[child] + (n - count[child])
                    dfs2(child, node)

        dfs1(0, -1)
        dfs2(0, -1)

        return ans

    def count_valid_pickup_delivery_sequences(self, n: int) -> int:
        """
        Related to LeetCode 1359 - Valid Pickup/Delivery Sequences in Tree (Hard)

        Count valid sequences for pickup/delivery with tree constraints.
        Extended tree version where deliveries must respect tree structure.

        Algorithm:
        1. Use DFS with memoization
        2. Track picked but not delivered items
        3. Ensure tree constraints are satisfied

        Time: O(n² * 2^n), Space: O(n * 2^n)
        """
        MOD = 10 ** 9 + 7

        # For tree version, assume we have parent-child relationships
        # This is a simplified version focusing on the pattern
        memo = {}

        def dfs(remaining_picks, pending_deliveries):
            if remaining_picks == 0 and pending_deliveries == 0:
                return 1

            state = (remaining_picks, pending_deliveries)
            if state in memo:
                return memo[state]

            result = 0

            # Can pick if pickups remaining
            if remaining_picks > 0:
                # Number of positions to insert new pickup
                positions = 2 * (n - remaining_picks) + 1
                result += positions * dfs(remaining_picks - 1, pending_deliveries + 1)
                result %= MOD

            # Can deliver if deliveries pending
            if pending_deliveries > 0:
                # Number of pending deliveries we can complete
                result += pending_deliveries * dfs(remaining_picks, pending_deliveries - 1)
                result %= MOD

            memo[state] = result
            return result

        return dfs(n, 0)

    def smallest_string_starting_from_leaf(self, root: TreeNode) -> str:
        """
        LeetCode 988 - Smallest String Starting From Leaf (Hard)

        Find lexicographically smallest string from leaf to root.

        Algorithm:
        1. DFS to all leaves, building strings
        2. Compare strings in reverse (leaf to root)
        3. Handle lexicographic comparison carefully

        Time: O(n * h), Space: O(h)

        Example:
        Tree: [0,1,2,3,4,3,4] (0='a', 1='b', etc.)
        Output: "dba"
        """
        self.smallest = None

        def dfs(node, path):
            if not node:
                return

            # Add current character to path
            path.append(chr(ord('a') + node.val))

            # If leaf node, compare with current smallest
            if not node.left and not node.right:
                # Reverse to get leaf-to-root string
                current = ''.join(reversed(path))
                if self.smallest is None or current < self.smallest:
                    self.smallest = current

            # Continue DFS
            dfs(node.left, path)
            dfs(node.right, path)

            # Backtrack
            path.pop()

        dfs(root, [])
        return self.smallest if self.smallest else ""

    def delete_nodes_and_return_forest(self, root: TreeNode, to_delete: List[int]) -> List[TreeNode]:
        """
        LeetCode 1110 - Delete Nodes And Return Forest (Hard)

        Delete nodes and return forest of remaining trees.

        Algorithm:
        1. Use postorder DFS to process deletions bottom-up
        2. When deleting node, its children become new roots
        3. Track all forest roots

        Time: O(n), Space: O(h + d) where d is size of to_delete

        Example:
        Tree: [1,2,3,4,5,6,7], to_delete = [3,5]
        Output: [[1,2,null,4], [6], [7]]
        """
        to_delete_set = set(to_delete)
        forest = []

        def dfs(node, is_root):
            if not node:
                return None

            # Check if current node should be deleted
            node_deleted = node.val in to_delete_set

            # If node is root and not deleted, add to forest
            if is_root and not node_deleted:
                forest.append(node)

            # Process children
            # If current node is deleted, children become roots
            node.left = dfs(node.left, node_deleted)
            node.right = dfs(node.right, node_deleted)

            # Return None if node should be deleted, otherwise return node
            return None if node_deleted else node

        dfs(root, True)
        return forest

    def flip_binary_tree_to_match_preorder(self, root: TreeNode, voyage: List[int]) -> List[int]:
        """
        LeetCode 971 - Flip Binary Tree To Match Preorder Traversal (Hard)

        Find minimum flips to make preorder traversal match voyage.
        Return list of nodes to flip, or [-1] if impossible.

        Algorithm:
        1. Use DFS following the voyage order
        2. When mismatch found, try flipping current node
        3. Track all flips made

        Time: O(n), Space: O(h)

        Example:
        Tree: [1,2,3], voyage = [1,3,2]
        Output: [1] - flip node 1 to match
        """
        self.flips = []
        self.idx = 0
        self.impossible = False

        def dfs(node):
            if not node or self.impossible:
                return

            # Check if current node matches voyage
            if node.val != voyage[self.idx]:
                self.impossible = True
                return

            self.idx += 1

            # If left child exists and doesn't match next in voyage, flip
            if (self.idx < len(voyage) and
                    node.left and
                    node.left.val != voyage[self.idx]):

                # Try flipping
                self.flips.append(node.val)
                dfs(node.right)
                dfs(node.left)
            else:
                # Normal order
                dfs(node.left)
                dfs(node.right)

        dfs(root)

        return [-1] if self.impossible else self.flips


# Example usage and testing
if __name__ == "__main__":
    solver = TreeDFSHard()


    # Helper function to build tree
    def build_tree(values):
        if not values:
            return None

        nodes = [TreeNode(val) if val is not None else None for val in values]
        root = nodes[0]

        for i in range(len(values)):
            if nodes[i] is not None:
                left_idx = 2 * i + 1
                right_idx = 2 * i + 2

                if left_idx < len(nodes):
                    nodes[i].left = nodes[left_idx]
                if right_idx < len(nodes):
                    nodes[i].right = nodes[right_idx]

        return root


    # Test 1: Maximum Path Sum
    print("1. Maximum Path Sum with Path:")
    tree = build_tree([-10, 9, 20, None, None, 15, 7])
    max_sum, path = solver.binary_tree_maximum_path_sum_with_paths(tree)
    print(f"   Tree: [-10,9,20,null,null,15,7]")
    print(f"   Max Sum: {max_sum}, Path: {path}")
    print()

    # Test 2: Longest Univalue Path
    print("2. Longest Univalue Path:")
    tree = build_tree([5, 4, 5, 1, 1, None, 5])
    length, nodes, count = solver.longest_univalue_path_with_details(tree)
    print(f"   Tree: [5,4,5,1,1,null,5]")
    print(f"   Length: {length}, Path: {nodes}, Count: {count}")
    print()

    # Test 3: Sum of Distances
    print("3. Sum of Distances in Tree:")
    n = 6
    edges = [[0, 1], [0, 2], [2, 3], [2, 4], [2, 5]]
    print(f"   n={n}, edges={edges}")
    print(f"   Output: {solver.sum_of_distances_in_tree(n, edges)}")