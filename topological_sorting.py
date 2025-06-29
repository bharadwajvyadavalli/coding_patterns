"""
Pattern 14: Topological Sort - 10 Hard Problems
==============================================

Topological Sort creates an ordering of vertices in a directed graph such that
for each directed edge u->v, u comes before v in the ordering. This pattern is
essential for dependency resolution, task scheduling, and course prerequisites.

Key Concepts:
- Only works on Directed Acyclic Graphs (DAGs)
- Two main approaches: Kahn's algorithm (BFS) and DFS
- Track in-degrees for Kahn's algorithm
- Detect cycles to verify DAG property

Time Complexity: O(V + E) where V is vertices and E is edges
Space Complexity: O(V) for various data structures
"""

from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict, deque
import heapq


class TopologicalSortHard:

    def alien_dictionary_with_analysis(self, words: List[str]) -> Tuple[str, List[Tuple[str, str]], bool]:
        """
        LeetCode 269 Extension - Alien Dictionary with Full Analysis (Hard)

        Derive alien language order from sorted word list.
        Extended: Return order, all ordering rules found, and validity.

        Algorithm:
        1. Build graph from adjacent word comparisons
        2. Find first differing character between adjacent words
        3. Use topological sort to determine character order
        4. Detect invalid cases (cycles, contradictions)

        Time: O(C) where C is total length of all words
        Space: O(1) - at most 26 characters

        Example:
        words = ["wrt","wrf","er","ett","rftt"]
        Output: ("wertf", [(w,e),(r,t),(t,f),(e,r)], True)
        """
        # Build graph
        graph = defaultdict(set)
        in_degree = {}
        rules = []

        # Initialize all characters
        for word in words:
            for char in word:
                in_degree[char] = 0

        # Compare adjacent words
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]

            # Find first different character
            min_len = min(len(word1), len(word2))
            for j in range(min_len):
                if word1[j] != word2[j]:
                    if word2[j] not in graph[word1[j]]:
                        graph[word1[j]].add(word2[j])
                        in_degree[word2[j]] += 1
                        rules.append((word1[j], word2[j]))
                    break
            else:
                # No difference found, check if word1 is longer
                if len(word1) > len(word2):
                    return "", rules, False  # Invalid ordering

        # Topological sort using Kahn's algorithm
        queue = deque([char for char in in_degree if in_degree[char] == 0])
        result = []

        while queue:
            char = queue.popleft()
            result.append(char)

            for neighbor in graph[char]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check if all characters are included (no cycle)
        if len(result) != len(in_degree):
            return "", rules, False

        return "".join(result), rules, True

    def parallel_courses_iii_with_schedule(self, n: int, relations: List[List[int]], time: List[int]) -> Tuple[
        int, Dict[int, int]]:
        """
        LeetCode 2050 Extension - Parallel Courses III with Schedule (Hard)

        Find minimum time to complete all courses with prerequisites.
        Extended: Return start time for each course.

        Algorithm:
        1. Build dependency graph
        2. Use topological sort with time tracking
        3. Each course starts after all prerequisites complete
        4. Parallel execution when possible

        Time: O(V + E), Space: O(V + E)

        Example:
        n = 3, relations = [[1,3],[2,3]], time = [3,2,5]
        Output: (8, {1:0, 2:0, 3:3}) - total time 8, course times
        """
        # Build graph and calculate in-degrees
        graph = defaultdict(list)
        in_degree = [0] * (n + 1)

        for prereq, course in relations:
            graph[prereq].append(course)
            in_degree[course] += 1

        # Find courses with no prerequisites
        queue = deque()
        start_time = {}

        for i in range(1, n + 1):
            if in_degree[i] == 0:
                queue.append(i)
                start_time[i] = 0

        max_time = 0

        # Process courses in topological order
        while queue:
            course = queue.popleft()
            course_end_time = start_time[course] + time[course - 1]
            max_time = max(max_time, course_end_time)

            # Update dependent courses
            for next_course in graph[course]:
                # Next course starts after this one ends
                if next_course not in start_time:
                    start_time[next_course] = 0
                start_time[next_course] = max(start_time[next_course], course_end_time)

                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)

        return max_time, start_time

    def find_all_topological_orders(self, numCourses: int, prerequisites: List[List[int]]) -> List[List[int]]:
        """
        Extension of LeetCode 210 - All Possible Topological Orders (Hard)

        Find all valid course orderings satisfying prerequisites.

        Algorithm:
        1. Use backtracking with in-degree tracking
        2. At each step, try all courses with in-degree 0
        3. Temporarily reduce in-degrees when selecting course
        4. Restore state when backtracking

        Time: O(V! * E) worst case, Space: O(V)

        Example:
        numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
        Output: [[0,1,2,3], [0,2,1,3]]
        """
        # Build graph
        graph = defaultdict(list)
        in_degree = [0] * numCourses

        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1

        all_orders = []

        def backtrack(path: List[int]):
            if len(path) == numCourses:
                all_orders.append(path[:])
                return

            # Try all courses with in-degree 0
            for course in range(numCourses):
                if in_degree[course] == 0 and course not in path:
                    # Select this course
                    path.append(course)

                    # Temporarily reduce in-degrees
                    for neighbor in graph[course]:
                        in_degree[neighbor] -= 1

                    backtrack(path)

                    # Restore state
                    path.pop()
                    for neighbor in graph[course]:
                        in_degree[neighbor] += 1

        backtrack([])
        return all_orders

    def minimum_semesters_with_course_limit(self, n: int, relations: List[List[int]], k: int) -> int:
        """
        LeetCode 1136 Extension - Minimum Semesters with Course Limit (Hard)

        Find minimum semesters to complete all courses with at most k courses per semester.

        Algorithm:
        1. Layer courses by dependency depth using BFS
        2. Greedily schedule courses respecting k limit
        3. Prioritize courses that unlock most future courses

        Time: O(V + E), Space: O(V + E)

        Example:
        n = 4, relations = [[1,3],[2,3],[3,4]], k = 2
        Output: 3 (Sem1: [1,2], Sem2: [3], Sem3: [4])
        """
        # Build graph and in-degrees
        graph = defaultdict(list)
        in_degree = [0] * (n + 1)
        out_degree = [0] * (n + 1)

        for prereq, course in relations:
            graph[prereq].append(course)
            in_degree[course] += 1
            out_degree[prereq] += 1

        # Find initial available courses
        available = []
        for i in range(1, n + 1):
            if in_degree[i] == 0:
                # Priority: courses that unlock more courses
                heapq.heappush(available, (-out_degree[i], i))

        semesters = 0
        courses_taken = 0

        while courses_taken < n:
            # Take up to k courses this semester
            next_available = []
            semester_courses = []

            for _ in range(min(k, len(available))):
                if not available:
                    break
                _, course = heapq.heappop(available)
                semester_courses.append(course)
                courses_taken += 1

                # Update available courses
                for next_course in graph[course]:
                    in_degree[next_course] -= 1
                    if in_degree[next_course] == 0:
                        next_available.append((-out_degree[next_course], next_course))

            # Add newly available courses
            for item in next_available:
                heapq.heappush(available, item)

            semesters += 1

        return semesters

    def sequence_reconstruction(self, nums: List[int], sequences: List[List[int]]) -> bool:
        """
        LeetCode 444 - Sequence Reconstruction (Hard)

        Check if sequences uniquely reconstruct the given sequence.

        Algorithm:
        1. Build graph from all subsequence constraints
        2. Perform topological sort
        3. At each step, must have exactly one choice
        4. Result must match given sequence

        Time: O(S) where S is sum of sequence lengths
        Space: O(n)

        Example:
        nums = [1,2,3], sequences = [[1,2],[1,3],[2,3]]
        Output: True
        """
        n = len(nums)
        if n == 0:
            return len(sequences) == 0

        # Build graph from sequences
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        nodes = set()

        for seq in sequences:
            for num in seq:
                nodes.add(num)

            for i in range(len(seq) - 1):
                if seq[i + 1] not in graph[seq[i]]:
                    graph[seq[i]].add(seq[i + 1])
                    in_degree[seq[i + 1]] += 1

        # Check if all numbers are covered
        if nodes != set(nums):
            return False

        # Initialize in-degrees for all nodes
        for node in nodes:
            if node not in in_degree:
                in_degree[node] = 0

        # Topological sort
        queue = deque([node for node in nodes if in_degree[node] == 0])
        result = []

        while queue:
            # Must have unique choice at each step
            if len(queue) != 1:
                return False

            node = queue.popleft()
            result.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result == nums

    def largest_color_value_in_directed_graph(self, colors: str, edges: List[List[int]]) -> int:
        """
        LeetCode 1857 - Largest Color Value in a Directed Graph (Hard)

        Find largest value of color along any valid path.
        Value = number of nodes with same color in path.

        Algorithm:
        1. Topological sort with DP
        2. dp[node][color] = max count of color ending at node
        3. Detect cycles (return -1)
        4. Track maximum across all nodes and colors

        Time: O(n + m), Space: O(n * 26)

        Example:
        colors = "abaca", edges = [[0,1],[0,2],[2,3],[3,4]]
        Output: 3 (path 0->2->3->4 has 3 'a's)
        """
        n = len(colors)

        # Build graph
        graph = defaultdict(list)
        in_degree = [0] * n

        for u, v in edges:
            graph[u].append(v)
            in_degree[v] += 1

        # Find nodes with no incoming edges
        queue = deque([i for i in range(n) if in_degree[i] == 0])

        # dp[node][color] = max count of color in any path ending at node
        dp = [[0] * 26 for _ in range(n)]

        # Initialize with node's own color
        for i in range(n):
            color_idx = ord(colors[i]) - ord('a')
            dp[i][color_idx] = 1

        processed = 0
        max_value = 0

        while queue:
            node = queue.popleft()
            processed += 1

            # Update maximum value seen so far
            max_value = max(max_value, max(dp[node]))

            for neighbor in graph[node]:
                # Update DP values for neighbor
                for color in range(26):
                    if color == ord(colors[neighbor]) - ord('a'):
                        dp[neighbor][color] = max(dp[neighbor][color], dp[node][color] + 1)
                    else:
                        dp[neighbor][color] = max(dp[neighbor][color], dp[node][color])

                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycle
        if processed != n:
            return -1

        return max_value

    def build_matrix_with_conditions(self, k: int, rowConditions: List[List[int]],
                                     colConditions: List[List[int]]) -> List[List[int]]:
        """
        LeetCode 2392 - Build a Matrix With Conditions (Hard)

        Build k×k matrix where numbers 1 to k appear exactly once.
        Must satisfy row and column ordering constraints.

        Algorithm:
        1. Topological sort for row positions
        2. Topological sort for column positions
        3. Check for contradictions
        4. Place numbers at intersection of row/col positions

        Time: O(k + E), Space: O(k²)

        Example:
        k = 3, rowConditions = [[1,2],[3,2]], colConditions = [[2,1],[3,2]]
        Output: [[3,0,0],[0,0,1],[0,2,0]]
        """

        def topological_sort(conditions: List[List[int]], k: int) -> Optional[List[int]]:
            """Get topological ordering from conditions."""
            graph = defaultdict(list)
            in_degree = [0] * (k + 1)

            for u, v in conditions:
                graph[u].append(v)
                in_degree[v] += 1

            queue = deque([i for i in range(1, k + 1) if in_degree[i] == 0])
            order = []

            while queue:
                node = queue.popleft()
                order.append(node)

                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            return order if len(order) == k else None

        # Get row and column orderings
        row_order = topological_sort(rowConditions, k)
        col_order = topological_sort(colConditions, k)

        if not row_order or not col_order:
            return []  # Contradiction exists

        # Create position mappings
        row_pos = {num: i for i, num in enumerate(row_order)}
        col_pos = {num: i for i, num in enumerate(col_order)}

        # Build matrix
        matrix = [[0] * k for _ in range(k)]

        for num in range(1, k + 1):
            matrix[row_pos[num]][col_pos[num]] = num

        return matrix

    def sort_items_by_groups(self, n: int, m: int, group: List[int],
                             beforeItems: List[List[int]]) -> List[int]:
        """
        LeetCode 1203 - Sort Items by Groups with Respect to Dependencies (Hard)

        Sort items respecting both item dependencies and group constraints.
        Items in same group must be together.

        Algorithm:
        1. Create virtual groups for ungrouped items
        2. Build item dependency graph
        3. Build group dependency graph
        4. Topological sort on both graphs
        5. Combine results respecting both orderings

        Time: O(n + E), Space: O(n + m)

        Example:
        n = 8, m = 2, group = [-1,-1,1,0,0,1,0,-1]
        beforeItems = [[],[6],[5],[6],[3,6],[],[],[]]
        Output: [6,3,4,5,2,0,1,7]
        """
        # Assign groups to ungrouped items
        for i in range(n):
            if group[i] == -1:
                group[i] = m
                m += 1

        # Build item graph
        item_graph = defaultdict(list)
        item_in_degree = [0] * n

        # Build group graph
        group_graph = defaultdict(set)
        group_in_degree = [0] * m

        for i in range(n):
            for j in beforeItems[i]:
                item_graph[j].append(i)
                item_in_degree[i] += 1

                if group[i] != group[j]:
                    if group[i] not in group_graph[group[j]]:
                        group_graph[group[j]].add(group[i])
                        group_in_degree[group[i]] += 1

        # Topological sort on groups
        def topological_sort_groups():
            queue = deque([g for g in range(m) if group_in_degree[g] == 0])
            result = []

            while queue:
                g = queue.popleft()
                result.append(g)

                for next_g in group_graph[g]:
                    group_in_degree[next_g] -= 1
                    if group_in_degree[next_g] == 0:
                        queue.append(next_g)

            return result if len(result) == m else None

        # Topological sort items within each group
        def topological_sort_items_in_group(g):
            items = [i for i in range(n) if group[i] == g]
            queue = deque([i for i in items if item_in_degree[i] == 0])
            result = []

            while queue:
                item = queue.popleft()
                result.append(item)

                for next_item in item_graph[item]:
                    if group[next_item] == g:
                        item_in_degree[next_item] -= 1
                        if item_in_degree[next_item] == 0:
                            queue.append(next_item)

            return result if len(result) == len(items) else None

        # Get group order
        group_order = topological_sort_groups()
        if not group_order:
            return []

        # Sort items within each group and combine
        result = []
        for g in group_order:
            items_in_group = topological_sort_items_in_group(g)
            if not items_in_group:
                return []
            result.extend(items_in_group)

        return result

    def count_visited_nodes_in_directed_graph(self, edges: List[int]) -> List[int]:
        """
        LeetCode 2876 - Count Visited Nodes in a Directed Graph (Hard)

        For each node, count nodes visited in infinite walk.
        Each node has exactly one outgoing edge.

        Algorithm:
        1. Find all cycles using DFS
        2. For nodes in cycles, answer is cycle length
        3. For other nodes, find distance to cycle
        4. Answer = distance to cycle + cycle length

        Time: O(n), Space: O(n)

        Example:
        edges = [1,2,0,0]
        Output: [3,3,3,1] (cycle 0->1->2->0 has length 3)
        """
        n = len(edges)
        visited = [False] * n
        in_cycle = [False] * n
        cycle_id = [-1] * n
        cycle_lengths = []
        answer = [0] * n

        # Find all cycles
        def find_cycle(start):
            path = []
            node = start
            positions = {}

            while node not in positions and not visited[node]:
                positions[node] = len(path)
                path.append(node)
                node = edges[node]

            if not visited[node]:
                # Found new cycle
                cycle_start_pos = positions[node]
                cycle = path[cycle_start_pos:]
                cycle_len = len(cycle)
                cycle_lengths.append(cycle_len)

                # Mark cycle nodes
                for i in range(len(cycle)):
                    cycle_node = cycle[i]
                    in_cycle[cycle_node] = True
                    cycle_id[cycle_node] = len(cycle_lengths) - 1
                    answer[cycle_node] = cycle_len
                    visited[cycle_node] = True

            # Mark path nodes as visited
            for node in path:
                visited[node] = True

        # Find all cycles
        for i in range(n):
            if not visited[i]:
                find_cycle(i)

        # Calculate answers for non-cycle nodes
        def get_distance_to_cycle(node):
            if in_cycle[node]:
                return 0

            distance = 0
            current = node

            while not in_cycle[current]:
                distance += 1
                current = edges[current]

            return distance, cycle_id[current]

        for i in range(n):
            if not in_cycle[i]:
                dist, cid = get_distance_to_cycle(i)
                answer[i] = dist + cycle_lengths[cid]

        return answer


# Example usage and testing
if __name__ == "__main__":
    solver = TopologicalSortHard()

    # Test 1: Alien Dictionary
    print("1. Alien Dictionary with Analysis:")
    words = ["wrt", "wrf", "er", "ett", "rftt"]
    order, rules, valid = solver.alien_dictionary_with_analysis(words)
    print(f"   Words: {words}")
    print(f"   Order: {order}")
    print(f"   Rules: {rules}")
    print(f"   Valid: {valid}")
    print()

    # Test 2: Parallel Courses III
    print("2. Parallel Courses III with Schedule:")
    n = 3
    relations = [[1, 3], [2, 3]]
    time = [3, 2, 5]
    total_time, schedule = solver.parallel_courses_iii_with_schedule(n, relations, time)
    print(f"   n={n}, relations={relations}, time={time}")
    print(f"   Total time: {total_time}")
    print(f"   Schedule: {schedule}")
    print()

    # Test 3: Sequence Reconstruction
    print("3. Sequence Reconstruction:")
    nums = [1, 2, 3]
    sequences = [[1, 2], [1, 3], [2, 3]]
    print(f"   nums={nums}, sequences={sequences}")
    print(f"   Can reconstruct: {solver.sequence_reconstruction(nums, sequences)}")