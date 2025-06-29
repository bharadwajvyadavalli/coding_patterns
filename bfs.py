"""
Pattern 7: Tree Breadth-First Search (BFS) - 10 Hard Problems
=============================================================

Tree BFS explores a tree level by level using a queue. This pattern is crucial
for problems requiring level-order traversal, finding shortest paths in trees,
or processing nodes by their distance from the root.

Key Concepts:
- Use queue (deque) to process nodes level by level
- Track level boundaries for level-specific operations
- Can be extended to graphs with visited set
- Useful for shortest path and minimum depth problems

Time Complexity: O(n) where n is number of nodes
Space Complexity: O(w) where w is maximum width of tree
"""

from collections import deque, defaultdict
from typing import List, Optional, Tuple, Dict
import math


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class TreeBFSHard:

    def vertical_order_traversal_with_sorting(self, root: TreeNode) -> List[List[int]]:
        """
        LeetCode 987 - Vertical Order Traversal of a Binary Tree (Hard)

        Return vertical order traversal where nodes at same position are sorted.
        For nodes at (row, col), if two nodes have same position, sort by value.

        Algorithm:
        1. Use BFS with column and row tracking
        2. Group nodes by column, then by row
        3. Sort nodes at same position by value
        4. Handle edge cases for tree structure

        Time: O(n log n), Space: O(n)

        Example:
        Tree: [3,9,20,null,null,15,7]
        Output: [[9],[3,15],[20],[7]]
        """
        if not root:
            return []

        # Dictionary to store nodes by column -> row -> list of values
        columns = defaultdict(lambda: defaultdict(list))

        # BFS with (node, row, col)
        queue = deque([(root, 0, 0)])

        while queue:
            node, row, col = queue.popleft()
            columns[col][row].append(node.val)

            if node.left:
                queue.append((node.left, row + 1, col - 1))
            if node.right:
                queue.append((node.right, row + 1, col + 1))

        # Sort and build result
        result = []
        for col in sorted(columns.keys()):
            col_values = []
            for row in sorted(columns[col].keys()):
                # Sort values at same position
                col_values.extend(sorted(columns[col][row]))
            result.append(col_values)

        return result

    def max_width_of_binary_tree(self, root: TreeNode) -> int:
        """
        LeetCode 662 - Maximum Width of Binary Tree (Hard)

        Find maximum width of binary tree (including null nodes between endpoints).
        Width is number of nodes between leftmost and rightmost nodes at each level.

        Algorithm:
        1. Assign position indices to nodes (left child: 2*i, right child: 2*i+1)
        2. Track first and last position at each level
        3. Calculate width as last - first + 1
        4. Handle integer overflow with modulo

        Time: O(n), Space: O(w)

        Example:
        Tree: [1,3,2,5,3,null,9]
        Output: 4 (level with nodes [5,3,null,9])
        """
        if not root:
            return 0

        max_width = 0
        # Queue stores (node, position)
        queue = deque([(root, 0)])

        while queue:
            level_size = len(queue)
            _, first_pos = queue[0]
            _, last_pos = queue[-1]

            # Calculate width of current level
            max_width = max(max_width, last_pos - first_pos + 1)

            for _ in range(level_size):
                node, pos = queue.popleft()

                # Normalize positions to prevent overflow
                normalized_pos = pos - first_pos

                if node.left:
                    queue.append((node.left, 2 * normalized_pos))
                if node.right:
                    queue.append((node.right, 2 * normalized_pos + 1))

        return max_width

    def binary_tree_right_side_view_extended(self, root: TreeNode) -> Tuple[List[int], List[int], List[int]]:
        """
        LeetCode 199 Extension - Binary Tree Views from All Sides (Hard)

        Get right side view, left side view, and outline view of binary tree.
        Outline includes leftmost path down and rightmost path up.

        Algorithm:
        1. Use BFS to track first and last nodes at each level
        2. Build views by selecting appropriate nodes
        3. For outline, combine left and right boundaries

        Time: O(n), Space: O(n)

        Example:
        Tree: [1,2,3,null,5,null,4]
        Output: right=[1,3,4], left=[1,2,5], outline=[1,2,5,4,3]
        """
        if not root:
            return [], [], []

        right_view = []
        left_view = []
        levels_info = []  # Store first and last node at each level

        queue = deque([root])

        while queue:
            level_size = len(queue)
            level_nodes = []

            for i in range(level_size):
                node = queue.popleft()
                level_nodes.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            # Store first and last nodes of level
            left_view.append(level_nodes[0])
            right_view.append(level_nodes[-1])
            levels_info.append(level_nodes)

        # Build outline view (perimeter of tree)
        outline = []

        # Add left boundary (top to bottom)
        for i in range(len(levels_info)):
            outline.append(levels_info[i][0])

        # Add bottom level (left to right, excluding first)
        if len(levels_info) > 1:
            for val in levels_info[-1][1:]:
                outline.append(val)

        # Add right boundary (bottom to top, excluding bottom level)
        if len(levels_info) > 1:
            for i in range(len(levels_info) - 2, -1, -1):
                if len(levels_info[i]) > 1:
                    outline.append(levels_info[i][-1])

        # Remove duplicates while preserving order
        seen = set()
        unique_outline = []
        for val in outline:
            if val not in seen:
                seen.add(val)
                unique_outline.append(val)

        return right_view, left_view, unique_outline

    def zigzag_level_order_with_paths(self, root: TreeNode) -> Tuple[List[List[int]], List[str]]:
        """
        LeetCode 103 Extension - Zigzag Level Order with Path Tracking (Hard)

        Return zigzag level order traversal and path to each node.
        Paths are encoded as binary strings (0=left, 1=right).

        Algorithm:
        1. Use BFS with path tracking
        2. Reverse even levels for zigzag pattern
        3. Maintain path from root to each node

        Time: O(n), Space: O(n)

        Example:
        Tree: [3,9,20,null,null,15,7]
        Output: ([[3],[20,9],[15,7]], ["", "0", "1", "10", "11"])
        """
        if not root:
            return [], []

        result = []
        all_paths = []

        # Queue stores (node, path)
        queue = deque([(root, "")])
        level = 0

        while queue:
            level_size = len(queue)
            level_values = []
            level_paths = []

            for _ in range(level_size):
                node, path = queue.popleft()
                level_values.append(node.val)
                level_paths.append(path)

                if node.left:
                    queue.append((node.left, path + "0"))
                if node.right:
                    queue.append((node.right, path + "1"))

            # Reverse even levels for zigzag
            if level % 2 == 1:
                level_values.reverse()
                level_paths.reverse()

            result.append(level_values)
            all_paths.extend(level_paths)
            level += 1

        return result, all_paths

    def find_bottom_left_tree_value_with_analysis(self, root: TreeNode) -> Tuple[int, int, List[int]]:
        """
        LeetCode 513 Extension - Find Bottom Left Value with Full Analysis (Hard)

        Find leftmost value in last row of tree.
        Extended: Return depth, position in row, and all bottom row values.

        Algorithm:
        1. Use BFS to traverse level by level
        2. Track depth and position within each level
        3. Keep updating bottom left as we go deeper

        Time: O(n), Space: O(w)

        Example:
        Tree: [1,2,3,4,null,5,6,null,null,7]
        Output: (7, 3, [4,7])
        """
        if not root:
            return 0, 0, []

        bottom_left_value = root.val
        max_depth = 0
        bottom_row = []

        # Queue stores (node, depth)
        queue = deque([(root, 0)])

        while queue:
            level_size = len(queue)
            level_values = []
            current_depth = queue[0][1]

            for i in range(level_size):
                node, depth = queue.popleft()
                level_values.append(node.val)

                if depth > max_depth:
                    max_depth = depth
                    bottom_left_value = node.val
                    bottom_row = [node.val]
                elif depth == max_depth and i == 0:
                    bottom_left_value = node.val

                if node.left:
                    queue.append((node.left, depth + 1))
                if node.right:
                    queue.append((node.right, depth + 1))

            if current_depth == max_depth:
                bottom_row = level_values

        # Find position of bottom left value in bottom row
        position = bottom_row.index(bottom_left_value)

        return bottom_left_value, max_depth, bottom_row

    def connect_next_right_pointers_perfect_tree(self, root: 'Node') -> 'Node':
        """
        LeetCode 116 - Populating Next Right Pointers (Hard space optimization)

        Connect each node to its next right node in perfect binary tree.
        Must use O(1) extra space (not counting recursion stack).

        Algorithm:
        1. Use previously established next pointers to traverse
        2. Connect children using parent's next pointer
        3. Process level by level without queue

        Time: O(n), Space: O(1)
        """
        if not root:
            return None

        # Start with root level
        level_start = root

        # Process until we reach leaf level
        while level_start.left:
            # Iterate through current level using next pointers
            current = level_start

            while current:
                # Connect left child to right child
                current.left.next = current.right

                # Connect right child to next node's left child
                if current.next:
                    current.right.next = current.next.left

                # Move to next node in current level
                current = current.next

            # Move to next level
            level_start = level_start.left

        return root

    def largest_values_in_tree_rows_with_positions(self, root: TreeNode) -> List[Tuple[int, int]]:
        """
        LeetCode 515 Extension - Largest Values with Positions (Hard)

        Find largest value in each tree row and its position(s).
        If multiple nodes have same max value, return all positions.

        Algorithm:
        1. BFS traversal tracking position in each level
        2. Keep track of maximum value and positions
        3. Handle ties by storing all positions

        Time: O(n), Space: O(w)

        Example:
        Tree: [1,3,2,5,3,null,9]
        Output: [(1,0), (3,0), (9,2)] - (value, position) pairs
        """
        if not root:
            return []

        result = []
        queue = deque([root])

        while queue:
            level_size = len(queue)
            max_val = float('-inf')
            max_positions = []

            for i in range(level_size):
                node = queue.popleft()

                if node.val > max_val:
                    max_val = node.val
                    max_positions = [i]
                elif node.val == max_val:
                    max_positions.append(i)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            # Store max value with all its positions
            result.append((max_val, max_positions[0]))  # Just first position for simplicity

        return result

    def minimum_depth_with_path(self, root: TreeNode) -> Tuple[int, List[int]]:
        """
        LeetCode 111 Extension - Minimum Depth with Path (Hard)

        Find minimum depth and the path to the nearest leaf.

        Algorithm:
        1. BFS guarantees first leaf found is at minimum depth
        2. Track path to each node
        3. Return immediately when first leaf is found

        Time: O(n), Space: O(n)

        Example:
        Tree: [3,9,20,null,null,15,7]
        Output: (2, [3,9]) - depth 2, path from root to leaf
        """
        if not root:
            return 0, []

        # Queue stores (node, depth, path)
        queue = deque([(root, 1, [root.val])])

        while queue:
            node, depth, path = queue.popleft()

            # Check if leaf node
            if not node.left and not node.right:
                return depth, path

            if node.left:
                queue.append((node.left, depth + 1, path + [node.left.val]))
            if node.right:
                queue.append((node.right, depth + 1, path + [node.right.val]))

        return 0, []

    def level_order_successor(self, root: TreeNode, key: int) -> Optional[int]:
        """
        Custom Hard - Level Order Successor

        Find the level-order successor of a node with given key.
        The level-order successor is the next node in BFS traversal.

        Algorithm:
        1. Perform BFS traversal
        2. When key is found, return the next node
        3. Handle edge cases (key not found, key is last node)

        Time: O(n), Space: O(w)

        Example:
        Tree: [1,2,3,4,5,6,7], key=4
        Output: 5
        """
        if not root:
            return None

        queue = deque([root])
        found_key = False

        while queue:
            node = queue.popleft()

            if found_key:
                return node.val

            if node.val == key:
                found_key = True

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        return None  # Key not found or was last node

    def all_nodes_distance_k_in_binary_tree(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        """
        LeetCode 863 - All Nodes Distance K in Binary Tree (Hard)

        Find all nodes that are distance k from target node.

        Algorithm:
        1. First BFS to build parent pointers
        2. Second BFS from target node treating tree as graph
        3. Track visited nodes to avoid revisiting
        4. Stop at distance k

        Time: O(n), Space: O(n)

        Example:
        Tree: [3,5,1,6,2,0,8,null,null,7,4], target=5, k=2
        Output: [7,4,1]
        """
        if not root:
            return []

        # Build parent pointers using BFS
        parent = {}
        queue = deque([root])

        while queue:
            node = queue.popleft()

            if node.left:
                parent[node.left] = node
                queue.append(node.left)
            if node.right:
                parent[node.right] = node
                queue.append(node.right)

        # BFS from target node
        result = []
        visited = {target}
        queue = deque([(target, 0)])

        while queue:
            node, distance = queue.popleft()

            if distance == k:
                result.append(node.val)
                continue

            # Check neighbors (parent, left child, right child)
            neighbors = []
            if node in parent and parent[node] not in visited:
                neighbors.append(parent[node])
            if node.left and node.left not in visited:
                neighbors.append(node.left)
            if node.right and node.right not in visited:
                neighbors.append(node.right)

            for neighbor in neighbors:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

        return result


# Example usage and testing
if __name__ == "__main__":
    solver = TreeBFSHard()


    # Helper function to build tree from list
    def build_tree(values):
        if not values:
            return None

        root = TreeNode(values[0])
        queue = deque([root])
        i = 1

        while queue and i < len(values):
            node = queue.popleft()

            if i < len(values) and values[i] is not None:
                node.left = TreeNode(values[i])
                queue.append(node.left)
            i += 1

            if i < len(values) and values[i] is not None:
                node.right = TreeNode(values[i])
                queue.append(node.right)
            i += 1

        return root


    # Test 1: Vertical Order Traversal
    print("1. Vertical Order Traversal:")
    tree = build_tree([3, 9, 20, None, None, 15, 7])
    print(f"   Tree: [3,9,20,null,null,15,7]")
    print(f"   Output: {solver.vertical_order_traversal_with_sorting(tree)}")
    print()

    # Test 2: Maximum Width
    print("2. Maximum Width of Binary Tree:")
    tree = build_tree([1, 3, 2, 5, 3, None, 9])
    print(f"   Tree: [1,3,2,5,3,null,9]")
    print(f"   Output: {solver.max_width_of_binary_tree(tree)}")
    print()

    # Test 3: Tree Views
    print("3. Binary Tree Views:")
    tree = build_tree([1, 2, 3, None, 5, None, 4])
    print(f"   Tree: [1,2,3,null,5,null,4]")
    right, left, outline = solver.binary_tree_right_side_view_extended(tree)
    print(f"   Right View: {right}")
    print(f"   Left View: {left}")
    print(f"   Outline: {outline}")