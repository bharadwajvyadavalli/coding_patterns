"""
Pattern 3: Fast & Slow Pointers - 10 Hard Problems
==================================================

The Fast & Slow Pointers pattern (also known as Floyd's Tortoise and Hare algorithm)
uses two pointers moving at different speeds to detect cycles, find middle elements,
or solve problems involving circular arrays or linked lists.

Key Concepts:
- Fast pointer moves 2x (or more) speed of slow pointer
- Used for cycle detection (they meet if cycle exists)
- Finding middle element (when fast reaches end, slow is at middle)
- Can be extended to find cycle start, length, etc.

Time Complexity: Usually O(n) for cycle detection, O(n) for finding middle
Space Complexity: O(1) - constant space
"""

from typing import List, Optional, Tuple


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class FastSlowPointersHard:

    def find_duplicate_with_modifications(self, nums: List[int]) -> Tuple[int, List[int]]:
        """
        LeetCode 287 Extension - Find Duplicate Number with Cycle Analysis (Hard)

        Find duplicate in array containing n+1 integers between 1 and n.
        Extended: Also find all numbers in the cycle and cycle length.
        Array is read-only and must use O(1) space.

        Algorithm:
        1. Treat array as linked list where nums[i] points to index nums[i]
        2. Use Floyd's algorithm to find cycle
        3. Find cycle entry point (the duplicate)
        4. Traverse cycle to get all elements and length

        Time: O(n), Space: O(1) for finding, O(k) for storing cycle elements

        Example:
        nums = [1,3,4,2,2]
        Output: (2, [2]) - duplicate is 2, cycle contains only [2]
        """
        # Phase 1: Find intersection point in cycle using Floyd's algorithm
        slow = fast = nums[0]

        # Move until they meet
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break

        # Phase 2: Find cycle entry point (duplicate number)
        slow = nums[0]
        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]

        duplicate = slow

        # Phase 3: Find all elements in cycle and cycle length
        cycle_elements = []
        current = nums[duplicate]
        cycle_elements.append(duplicate)

        while current != duplicate:
            cycle_elements.append(current)
            current = nums[current]

        return duplicate, cycle_elements

    def linked_list_cycle_ii_extended(self, head: ListNode) -> Optional[Tuple[ListNode, int, List[int]]]:
        """
        LeetCode 142 Extension - Linked List Cycle II with Full Analysis (Hard)

        Find where cycle begins in linked list.
        Extended: Also return cycle length and all values in cycle.

        Algorithm:
        1. Detect cycle using fast/slow pointers
        2. Find cycle start using mathematical property
        3. Traverse cycle to get length and values
        4. Handle edge cases (no cycle, single node cycle)

        Time: O(n), Space: O(k) for storing cycle values

        Example:
        3 -> 2 -> 0 -> -4
             ^          |
             |__________|
        Output: (node with value 2, length 3, values [2, 0, -4])
        """
        if not head or not head.next:
            return None

        # Phase 1: Detect cycle
        slow = fast = head
        has_cycle = False

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                has_cycle = True
                break

        if not has_cycle:
            return None

        # Phase 2: Find cycle start
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next

        cycle_start = slow

        # Phase 3: Find cycle length and values
        cycle_values = []
        current = cycle_start
        cycle_length = 0

        while True:
            cycle_values.append(current.val)
            cycle_length += 1
            current = current.next
            if current == cycle_start:
                break

        return cycle_start, cycle_length, cycle_values

    def happy_number_extended(self, n: int) -> Tuple[bool, List[int], Optional[List[int]]]:
        """
        LeetCode 202 Extension - Happy Number with Full Path Analysis (Hard)

        Determine if number is happy (reaches 1) or loops endlessly.
        Extended: Return full path and cycle if exists.

        Algorithm:
        1. Use fast/slow pointers on sum of squares transformation
        2. Track full path for analysis
        3. If cycle detected, extract cycle elements
        4. Optimize using digit square precomputation

        Time: O(log n), Space: O(log n) for path storage

        Example:
        n = 19
        Output: (True, [19, 82, 68, 100, 1], None) - happy number
        n = 2
        Output: (False, [2, 4, 16, 37, 58, 89, 145, 42, 20], [4, 16, 37, 58, 89, 145, 42, 20])
        """

        def sum_of_squares(num: int) -> int:
            total = 0
            while num > 0:
                digit = num % 10
                total += digit * digit
                num //= 10
            return total

        # Track path
        path = [n]
        slow = fast = n

        # Phase 1: Detect cycle or reaching 1
        while True:
            slow = sum_of_squares(slow)
            fast = sum_of_squares(sum_of_squares(fast))

            if fast == 1:
                # Complete path to 1
                current = slow
                while current != 1:
                    path.append(current)
                    current = sum_of_squares(current)
                path.append(1)
                return True, path, None

            if slow == fast:
                break

        # Phase 2: Find cycle start and extract cycle
        slow = n
        while slow != fast:
            slow = sum_of_squares(slow)
            fast = sum_of_squares(fast)

        # Build complete path including cycle
        current = n
        seen = set()
        full_path = []

        while current not in seen:
            seen.add(current)
            full_path.append(current)
            current = sum_of_squares(current)

        # Extract cycle
        cycle_start_idx = full_path.index(current)
        cycle = full_path[cycle_start_idx:]

        return False, full_path, cycle

    def circular_array_loop_extended(self, nums: List[int]) -> Tuple[bool, Optional[List[int]]]:
        """
        LeetCode 457 Extension - Circular Array Loop with Path (Hard)

        Detect if circular array has a loop. Movement must be in single direction.
        Extended: Return the loop path if exists.

        Algorithm:
        1. Try each position as potential loop start
        2. Use fast/slow pointers with modular arithmetic
        3. Ensure single direction (all positive or all negative)
        4. Mark visited positions to avoid reprocessing

        Time: O(n), Space: O(n) for visited marking

        Example:
        nums = [2,-1,1,2,2]
        Output: (True, [0, 2, 3]) - loop exists with these indices
        """
        n = len(nums)

        def next_index(i: int) -> int:
            return (i + nums[i]) % n

        for start in range(n):
            if nums[start] == 0:
                continue

            slow = fast = start

            # Check if all movements are in same direction
            while True:
                slow = next_index(slow)
                fast = next_index(next_index(fast))

                # Check single direction constraint
                if nums[slow] * nums[start] < 0 or nums[fast] * nums[start] < 0:
                    break

                if slow == fast:
                    # Found potential cycle, verify it's valid (length > 1)
                    if slow == next_index(slow):
                        break

                    # Extract cycle path
                    cycle_path = [slow]
                    current = next_index(slow)

                    while current != slow:
                        cycle_path.append(current)
                        current = next_index(current)

                    return True, cycle_path

            # Mark all elements in this path as visited
            current = start
            while nums[current] * nums[start] > 0:
                next_pos = next_index(current)
                nums[current] = 0
                current = next_pos

        return False, None

    def find_middle_of_circular_list(self, head: ListNode) -> Tuple[Optional[ListNode], bool, int]:
        """
        Custom Hard - Find Middle of Potentially Circular Linked List

        Find middle node even if list contains a cycle.
        If cycle exists, find middle of the entire traversal.

        Algorithm:
        1. First detect if cycle exists and find its start
        2. Calculate total unique nodes including cycle
        3. Find middle considering full traversal
        4. Handle odd/even lengths appropriately

        Time: O(n), Space: O(1)

        Example:
        1 -> 2 -> 3 -> 4 -> 5
                  ^         |
                  |_________|
        Output: (node 4, True, 5) - middle of 5 unique nodes
        """
        if not head:
            return None, False, 0

        # Phase 1: Detect cycle
        slow = fast = head
        has_cycle = False

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                has_cycle = True
                break

        if not has_cycle:
            # Regular linked list - find middle normally
            slow = fast = head
            length = 0

            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
                length += 2

            if fast:
                length += 1
            else:
                length += 0

            return slow, False, length

        # Phase 2: Find cycle start and lengths
        cycle_start = head
        path_to_cycle = 0

        while cycle_start != slow:
            cycle_start = cycle_start.next
            slow = slow.next
            path_to_cycle += 1

        # Find cycle length
        cycle_length = 1
        current = cycle_start.next
        while current != cycle_start:
            cycle_length += 1
            current = current.next

        total_unique_nodes = path_to_cycle + cycle_length

        # Phase 3: Find middle node
        middle_pos = total_unique_nodes // 2
        current = head

        for _ in range(middle_pos):
            current = current.next

        return current, True, total_unique_nodes

    def reorder_list_with_analysis(self, head: ListNode) -> Tuple[ListNode, List[int]]:
        """
        LeetCode 143 Extension - Reorder List with Step Analysis (Hard)

        Reorder list: L0→L1→...→Ln-1→Ln to L0→Ln→L1→Ln-1→L2→Ln-2→...
        Extended: Track all pointer movements and operations.

        Algorithm:
        1. Find middle using fast/slow pointers
        2. Reverse second half
        3. Merge two halves alternately
        4. Track each operation for analysis

        Time: O(n), Space: O(n) for operation tracking

        Example:
        1->2->3->4->5
        Output: 1->5->2->4->3, operations logged
        """
        if not head or not head.next:
            return head, []

        operations = []

        # Step 1: Find middle using fast/slow pointers
        slow = fast = head
        prev = None

        while fast and fast.next:
            operations.append(f"Slow at {slow.val}, Fast at {fast.val}")
            prev = slow
            slow = slow.next
            fast = fast.next.next

        operations.append(f"Middle found at {slow.val}")

        # Disconnect first and second halves
        prev.next = None

        # Step 2: Reverse second half
        second = slow
        prev = None

        while second:
            operations.append(f"Reversing node {second.val}")
            next_node = second.next
            second.next = prev
            prev = second
            second = next_node

        second = prev

        # Step 3: Merge two halves
        first = head

        while second:
            operations.append(f"Merging {first.val} with {second.val}")
            tmp1 = first.next
            tmp2 = second.next

            first.next = second
            second.next = tmp1

            first = tmp1
            second = tmp2

        return head, operations

    def nth_node_from_end_circular(self, head: ListNode, n: int) -> Optional[ListNode]:
        """
        Custom Hard - Find Nth Node from End in Potentially Circular List

        Find nth node from end, handling both regular and circular lists.
        For circular lists, consider the traversal order.

        Algorithm:
        1. Detect if list has cycle
        2. For regular list: use two pointers with n gap
        3. For circular list: find total length and calculate position
        4. Handle edge cases (n > length, etc.)

        Time: O(n), Space: O(1)

        Example:
        1->2->3->4->5 (regular), n=2
        Output: node 4
        """
        if not head:
            return None

        # First, detect if there's a cycle
        slow = fast = head
        has_cycle = False

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                has_cycle = True
                break

        if not has_cycle:
            # Regular linked list
            first = second = head

            # Move first pointer n steps ahead
            for _ in range(n):
                if not first:
                    return None  # n is greater than list length
                first = first.next

            # Move both pointers until first reaches end
            while first:
                first = first.next
                second = second.next

            return second

        else:
            # Circular linked list
            # Find cycle start and length
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next

            cycle_start = slow

            # Count total unique nodes
            total_nodes = 0
            current = head
            visited = set()

            while current not in visited:
                visited.add(current)
                total_nodes += 1
                current = current.next

            # Calculate target position
            if n > total_nodes:
                return None

            target_pos = total_nodes - n

            # Traverse to target position
            current = head
            for _ in range(target_pos):
                current = current.next

            return current

    def palindrome_linked_list_with_reconstruction(self, head: ListNode) -> Tuple[bool, List[int], List[int]]:
        """
        LeetCode 234 Extension - Palindrome Linked List with Full Analysis (Hard)

        Check if linked list is palindrome.
        Extended: Return first half, second half (reversed back), and comparison results.
        Must restore original list structure.

        Algorithm:
        1. Find middle using fast/slow pointers
        2. Reverse second half
        3. Compare both halves
        4. Restore original structure
        5. Track all comparisons

        Time: O(n), Space: O(n) for tracking values

        Example:
        1->2->2->1
        Output: (True, [1,2], [2,1], all comparisons match)
        """
        if not head or not head.next:
            return True, [head.val] if head else [], [], []

        # Find middle
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next

        # Split into two halves
        second_half_start = slow.next
        slow.next = None

        # Store first half values
        first_half = []
        current = head
        while current:
            first_half.append(current.val)
            current = current.next

        # Reverse second half
        prev = None
        current = second_half_start
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node

        reversed_second = prev

        # Store second half values (in reversed order)
        second_half = []
        current = reversed_second
        while current:
            second_half.append(current.val)
            current = current.next

        # Compare both halves
        is_palindrome = True
        comparisons = []
        p1, p2 = head, reversed_second

        while p2:  # p2 is shorter or equal in length
            match = p1.val == p2.val
            comparisons.append((p1.val, p2.val, match))
            if not match:
                is_palindrome = False
            p1 = p1.next
            p2 = p2.next

        # Restore original list structure
        prev = None
        current = reversed_second
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node

        # Reconnect the two halves
        current = head
        while current.next:
            current = current.next
        current.next = prev

        return is_palindrome, first_half, second_half, comparisons

    def detect_and_remove_loop(self, head: ListNode) -> Tuple[bool, Optional[ListNode], int]:
        """
        Custom Hard - Detect and Remove Loop in Linked List

        Detect loop, find the node where loop should be broken, and return loop length.
        The loop should be removed at the last node that points back to loop start.

        Algorithm:
        1. Detect loop using Floyd's algorithm
        2. Find loop start point
        3. Find the last node in loop (that points to loop start)
        4. Break the loop and return information

        Time: O(n), Space: O(1)

        Example:
        1->2->3->4->5
              ^     |
              |_____|
        Output: (True, node 5, 3) - loop removed at node 5, loop length was 3
        """
        if not head or not head.next:
            return False, None, 0

        # Detect loop
        slow = fast = head
        has_loop = False

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                has_loop = True
                break

        if not has_loop:
            return False, None, 0

        # Find loop start
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next

        loop_start = slow

        # Find loop length and last node
        loop_length = 1
        current = loop_start.next
        prev = loop_start

        while current != loop_start:
            loop_length += 1
            prev = current
            current = current.next

        # Remove loop
        prev.next = None

        return True, prev, loop_length

    def find_longest_cycle_in_directed_graph(self, edges: List[int]) -> Tuple[int, List[int]]:
        """
        LeetCode 2360 Extension - Longest Cycle in Directed Graph (Hard)

        Find longest cycle in a directed graph where each node has at most one outgoing edge.
        Extended: Return the actual cycle path.

        Algorithm:
        1. Use fast/slow pointers for each unvisited node
        2. Track visited nodes globally to avoid reprocessing
        3. For each cycle found, calculate its length
        4. Keep track of the longest cycle

        Time: O(n), Space: O(n)

        Example:
        edges = [3,3,4,2,3]
        Output: (3, [3,2,4]) - longest cycle has length 3
        """
        n = len(edges)
        visited = [False] * n
        longest_cycle_length = -1
        longest_cycle = []

        for start in range(n):
            if visited[start] or edges[start] == -1:
                continue

            # Use fast/slow pointers to detect cycle
            slow = fast = start

            # Move pointers
            while edges[fast] != -1 and edges[edges[fast]] != -1:
                slow = edges[slow]
                fast = edges[edges[fast]]

                if slow == fast:
                    # Found cycle, calculate length
                    cycle_length = 1
                    current = edges[slow]
                    cycle_nodes = [slow]

                    while current != slow:
                        cycle_nodes.append(current)
                        cycle_length += 1
                        current = edges[current]

                    if cycle_length > longest_cycle_length:
                        longest_cycle_length = cycle_length
                        longest_cycle = cycle_nodes

                    break

            # Mark all nodes in this path as visited
            current = start
            while current != -1 and not visited[current]:
                visited[current] = True
                current = edges[current]

        return longest_cycle_length, longest_cycle


# Example usage and testing
if __name__ == "__main__":
    solver = FastSlowPointersHard()

    # Test 1: Find Duplicate with Cycle Analysis
    print("1. Find Duplicate with Cycle Analysis:")
    print(f"   Input: nums=[1,3,4,2,2]")
    duplicate, cycle = solver.find_duplicate_with_modifications([1, 3, 4, 2, 2])
    print(f"   Output: Duplicate={duplicate}, Cycle={cycle}")
    print()

    # Test 2: Happy Number Extended
    print("2. Happy Number Extended:")
    print(f"   Input: n=19")
    is_happy, path, cycle = solver.happy_number_extended(19)
    print(f"   Output: Happy={is_happy}, Path={path}, Cycle={cycle}")
    print()

    # Test 3: Circular Array Loop
    print("3. Circular Array Loop:")
    print(f"   Input: nums=[2,-1,1,2,2]")
    has_loop, loop_path = solver.circular_array_loop_extended([2, -1, 1, 2, 2])
    print(f"   Output: Has Loop={has_loop}, Path={loop_path}")
    print()

    # Test 4: Longest Cycle in Directed Graph
    print("4. Longest Cycle in Directed Graph:")
    print(f"   Input: edges=[3,3,4,2,3]")
    length, cycle = solver.find_longest_cycle_in_directed_graph([3, 3, 4, 2, 3])
    print(f"   Output: Length={length}, Cycle={cycle}")