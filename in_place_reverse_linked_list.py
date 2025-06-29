"""
Pattern 6: In-place Reversal of Linked List - 10 Hard Problems
==============================================================

The In-place Reversal pattern deals with reversing linked lists or portions
of linked lists without using extra space. This pattern is crucial for many
linked list manipulation problems.

Key Concepts:
- Use three pointers: previous, current, and next
- Reverse links while traversing
- Handle partial reversals carefully
- Combine with other patterns (fast/slow pointers, recursion)

Time Complexity: Usually O(n) for single pass
Space Complexity: O(1) for iterative, O(n) for recursive (call stack)
"""

from typing import Optional, List, Tuple


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class InPlaceReversalHard:

    def reverse_nodes_in_k_group_extended(self, head: ListNode, k: int) -> Tuple[ListNode, int, List[int]]:
        """
        LeetCode 25 Extension - Reverse Nodes in k-Group with Analysis (Hard)

        Reverse every k consecutive nodes. If remaining nodes < k, keep them as is.
        Extended: Return number of groups reversed and node values after reversal.

        Algorithm:
        1. Count total nodes to determine number of complete groups
        2. For each complete group, reverse k nodes
        3. Connect reversed groups properly
        4. Track all operations for analysis

        Time: O(n), Space: O(1) for reversal, O(n) for tracking

        Example:
        1->2->3->4->5, k=2
        Output: 2->1->4->3->5, 2 groups reversed
        """
        # Count total nodes
        count = 0
        node = head
        while node:
            count += 1
            node = node.next

        groups_reversed = count // k

        # Dummy node to simplify edge cases
        dummy = ListNode(0)
        dummy.next = head
        prev_group = dummy

        for _ in range(groups_reversed):
            # Reverse k nodes
            group_start = prev_group.next
            current = group_start
            prev = None

            for _ in range(k):
                next_node = current.next
                current.next = prev
                prev = current
                current = next_node

            # Connect with previous group
            prev_group.next = prev
            group_start.next = current
            prev_group = group_start

        # Collect final values
        result_values = []
        node = dummy.next
        while node:
            result_values.append(node.val)
            node = node.next

        return dummy.next, groups_reversed, result_values


    def reverse_linked_list_ii_extended(self, head: ListNode, left: int, right: int) -> Tuple[ListNode, List[Tuple[int, int]]]:
        """
        LeetCode 92 Extension - Reverse Linked List II with Operations Log (Hard)

        Reverse nodes from position left to right (1-indexed).
        Extended: Track all pointer operations during reversal.

        Algorithm:
        1. Find the node before position left
        2. Reverse nodes from left to right
        3. Reconnect the reversed portion
        4. Log all operations

        Time: O(n), Space: O(n) for operations log

        Example:
        1->2->3->4->5, left=2, right=4
        Output: 1->4->3->2->5, operations logged
        """
        if not head or left == right:
            return head, []

        operations = []
        dummy = ListNode(0)
        dummy.next = head

        # Find the node before the reversal starts
        prev = dummy
        for i in range(left - 1):
            operations.append(("traverse", prev.val, prev.next.val))
            prev = prev.next

        # Start reversing
        start = prev.next
        then = start.next

        # Reverse nodes between left and right
        for i in range(right - left):
            operations.append(("reverse", start.val, then.val))

            start.next = then.next
            then.next = prev.next
            prev.next = then
            then = start.next

        # Collect final state
        final_values = []
        node = dummy.next
        while node:
            final_values.append(node.val)
            node = node.next

        return dummy.next, operations


    def reverse_alternating_k_element_sublist(self, head: ListNode, k: int) -> ListNode:
        """
        Custom Hard - Reverse Alternating K-Element Sub-list

        Reverse first k nodes, skip next k nodes, reverse next k nodes, and so on.
        Handle cases where remaining nodes < k.

        Algorithm:
        1. Use a flag to alternate between reversing and skipping
        2. For each group, either reverse or skip k nodes
        3. Handle edge cases for last group

        Time: O(n), Space: O(1)

        Example:
        1->2->3->4->5->6->7->8, k=2
        Output: 2->1->3->4->6->5->7->8
        """
        if not head or k <= 1:
            return head

        dummy = ListNode(0)
        dummy.next = head
        prev_group = dummy
        should_reverse = True

        while True:
            # Check if we have k nodes remaining
            kth_node = prev_group
            for i in range(k):
                kth_node = kth_node.next
                if not kth_node:
                    return dummy.next

            if should_reverse:
                # Reverse k nodes
                group_start = prev_group.next
                current = group_start
                prev = kth_node.next

                for _ in range(k):
                    next_node = current.next
                    current.next = prev
                    prev = current
                    current = next_node

                prev_group.next = prev
                prev_group = group_start
            else:
                # Skip k nodes
                for _ in range(k):
                    prev_group = prev_group.next

            should_reverse = not should_reverse


    def swap_nodes_in_pairs_recursive_and_iterative(self, head: ListNode) -> Tuple[ListNode, ListNode]:
        """
        LeetCode 24 Extension - Swap Nodes in Pairs (Both Methods) (Hard)

        Swap every two adjacent nodes using both recursive and iterative methods.
        Compare both approaches and return results.

        Algorithm:
        1. Iterative: Use dummy node and adjust pointers
        2. Recursive: Swap first pair and recurse on remaining
        3. Both achieve O(1) space (excluding recursion stack)

        Time: O(n) for both, Space: O(1) iterative, O(n) recursive

        Example:
        1->2->3->4
        Output: 2->1->4->3 (both methods)
        """
        # Method 1: Iterative
        def swap_iterative(head: ListNode) -> ListNode:
            dummy = ListNode(0)
            dummy.next = head
            prev = dummy

            while prev.next and prev.next.next:
                # Nodes to be swapped
                first = prev.next
                second = prev.next.next

                # Swapping
                first.next = second.next
                second.next = first
                prev.next = second

                # Move to next pair
                prev = first

            return dummy.next

        # Method 2: Recursive
        def swap_recursive(head: ListNode) -> ListNode:
            if not head or not head.next:
                return head

            # Nodes to be swapped
            first = head
            second = head.next

            # Swapping
            first.next = swap_recursive(second.next)
            second.next = first

            return second

        # Create a copy for recursive method (to preserve original)
        def copy_list(node: ListNode) -> ListNode:
            if not node:
                return None
            new_node = ListNode(node.val)
            new_node.next = copy_list(node.next)
            return new_node

        head_copy = copy_list(head)

        iterative_result = swap_iterative(head)
        recursive_result = swap_recursive(head_copy)

        return iterative_result, recursive_result


    def reverse_between_patterns(self, head: ListNode, pattern: str) -> ListNode:
        """
        Custom Hard - Reverse Based on Pattern String

        Pattern string contains 'R' (reverse) and 'K' (keep).
        Apply pattern cyclically to the linked list.

        Algorithm:
        1. Parse pattern to get reverse/keep segments
        2. Apply pattern cyclically to entire list
        3. Handle pattern shorter than list length

        Time: O(n), Space: O(1)

        Example:
        1->2->3->4->5->6, pattern="RRK"
        Output: 2->1->3->5->4->6
        """
        if not head or not pattern:
            return head

        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        pattern_idx = 0

        # Count nodes
        count = 0
        node = head
        while node:
            count += 1
            node = node.next

        # Process pattern segments
        current = head
        i = 0

        while i < count:
            action = pattern[pattern_idx % len(pattern)]

            if action == 'R':
                # Find the end of reverse segment
                segment_len = 1
                j = pattern_idx + 1
                while j < pattern_idx + len(pattern) and pattern[j % len(pattern)] == 'R':
                    segment_len += 1
                    j += 1

                # Check how many nodes we can actually reverse
                actual_len = min(segment_len, count - i)

                # Reverse actual_len nodes
                if actual_len > 1:
                    group_start = current
                    prev_node = None

                    for _ in range(actual_len):
                        if not current:
                            break
                        next_node = current.next
                        current.next = prev_node
                        prev_node = current
                        current = next_node

                    # Connect reversed segment
                    prev.next = prev_node
                    group_start.next = current
                    prev = group_start
                else:
                    # Single node, just move forward
                    if current:
                        prev = current
                        current = current.next

                i += actual_len
                pattern_idx += segment_len
            else:  # 'K' - keep
                # Move forward without reversing
                if current:
                    prev = current
                    current = current.next
                i += 1
                pattern_idx += 1

        return dummy.next


    def plus_one_linked_list(self, head: ListNode) -> ListNode:
        """
        LeetCode 369 - Plus One Linked List (Hard)

        Add one to a number represented as linked list (MSB first).
        Must handle carries and potential new digit.

        Algorithm:
        1. Reverse list to add from LSB
        2. Add one and handle carries
        3. Reverse back to get result
        4. Optimize: find rightmost non-9 digit

        Time: O(n), Space: O(1)

        Example:
        1->2->9
        Output: 1->3->0
        """
        # Find the rightmost digit that is not 9
        sentinel = ListNode(0)
        sentinel.next = head
        not_nine = sentinel

        # Find rightmost non-9 digit
        node = head
        while node:
            if node.val != 9:
                not_nine = node
            node = node.next

        # Add 1 to the rightmost non-9 digit
        not_nine.val += 1

        # Set all following digits to 0
        node = not_nine.next
        while node:
            node.val = 0
            node = node.next

        # Return new head if needed (when all digits were 9)
        return sentinel if sentinel.val == 1 else sentinel.next


    def odd_even_linked_list_extended(self, head: ListNode) -> Tuple[ListNode, int, int]:
        """
        LeetCode 328 Extension - Odd Even Linked List with Stats (Hard)

        Group odd-indexed nodes followed by even-indexed nodes.
        Extended: Count odd/even nodes and maintain relative order.

        Algorithm:
        1. Separate odd and even nodes into two lists
        2. Connect odd list to even list
        3. Track counts for analysis

        Time: O(n), Space: O(1)

        Example:
        1->2->3->4->5
        Output: 1->3->5->2->4, odd_count=3, even_count=2
        """
        if not head or not head.next:
            return head, 1 if head else 0, 0

        odd = head
        even = head.next
        even_head = even
        odd_count = 1
        even_count = 1

        while even and even.next:
            odd.next = even.next
            odd = odd.next
            odd_count += 1

            even.next = odd.next
            even = even.next
            if even:
                even_count += 1

        # Connect odd list to even list
        odd.next = even_head

        return head, odd_count, even_count


    def rotate_list_with_optimization(self, head: ListNode, k: int) -> Tuple[ListNode, int]:
        """
        LeetCode 61 Extension - Rotate List with Optimization (Hard)

        Rotate list to the right by k places.
        Extended: Optimize for large k and return actual rotation count.

        Algorithm:
        1. Convert to circular list temporarily
        2. Find new head after rotation
        3. Break circle at appropriate position
        4. Handle k > length efficiently

        Time: O(n), Space: O(1)

        Example:
        1->2->3->4->5, k=2
        Output: 4->5->1->2->3, actual_rotation=2
        """
        if not head or not head.next or k == 0:
            return head, 0

        # Find length and last node
        length = 1
        last = head
        while last.next:
            length += 1
            last = last.next

        # Optimize k
        k = k % length
        if k == 0:
            return head, 0

        # Make circular
        last.next = head

        # Find new last node (length - k - 1 steps from old head)
        steps_to_new_last = length - k
        new_last = head
        for _ in range(steps_to_new_last - 1):
            new_last = new_last.next

        # Break circle
        new_head = new_last.next
        new_last.next = None

        return new_head, k


    def remove_zero_sum_consecutive_nodes(self, head: ListNode) -> ListNode:
        """
        LeetCode 1171 - Remove Zero Sum Consecutive Nodes (Hard)

        Remove all consecutive sequences of nodes that sum to 0.

        Algorithm:
        1. Use prefix sum with hash map
        2. If same prefix sum appears twice, nodes between sum to 0
        3. Remove zero-sum sequences
        4. Handle multiple passes if needed

        Time: O(n), Space: O(n)

        Example:
        1->2->-3->3->1
        Output: 3->1 (removed [1,2,-3])
        """
        dummy = ListNode(0)
        dummy.next = head

        # First pass: calculate prefix sums
        prefix_sum = 0
        sum_to_node = {0: dummy}
        node = head

        while node:
            prefix_sum += node.val
            sum_to_node[prefix_sum] = node
            node = node.next

        # Second pass: remove zero-sum sequences
        prefix_sum = 0
        node = dummy

        while node:
            prefix_sum += node.val
            if prefix_sum in sum_to_node:
                node.next = sum_to_node[prefix_sum].next
            node = node.next

        return dummy.next


    def merge_in_between_linked_lists(self, list1: ListNode, a: int, b: int,
                                      list2: ListNode) -> ListNode:
        """
        LeetCode 1669 - Merge In Between Linked Lists (Hard)

        Remove nodes from position a to b in list1 and insert list2 in their place.

        Algorithm:
        1. Find node before position a
        2. Find node after position b
        3. Find last node of list2
        4. Connect: before_a -> list2 -> after_b

        Time: O(m + n), Space: O(1)

        Example:
        list1 = 0->1->2->3->4->5, a=3, b=4
        list2 = 100->101->102
        Output: 0->1->2->100->101->102->5
        """
        # Find node at position a-1
        before_a = list1
        for _ in range(a - 1):
            before_a = before_a.next

        # Find node at position b+1
        after_b = before_a
        for _ in range(b - a + 2):
            after_b = after_b.next

        # Find last node of list2
        last_of_list2 = list2
        while last_of_list2.next:
            last_of_list2 = last_of_list2.next

        # Connect the lists
        before_a.next = list2
        last_of_list2.next = after_b

        return list1


# Example usage and testing
if __name__ == "__main__":
    solver = InPlaceReversalHard()

    # Helper function to create linked list from array
    def create_list(arr):
        if not arr:
            return None
        head = ListNode(arr[0])
        current = head
        for val in arr[1:]:
            current.next = ListNode(val)
            current = current.next
        return head

    # Helper function to convert linked list to array
    def list_to_array(head):
        arr = []
        while head:
            arr.append(head.val)
            head = head.next
        return arr

    # Test 1: Reverse Nodes in k-Group
    print("1. Reverse Nodes in k-Group:")
    head = create_list([1, 2, 3, 4, 5])
    print(f"   Input: [1,2,3,4,5], k=2")
    result, groups, values = solver.reverse_nodes_in_k_group_extended(head, 2)
    print(f"   Output: {values}, Groups reversed: {groups}")
    print()

    # Test 2: Reverse Alternating K-Element
    print("2. Reverse Alternating K-Element Sub-list:")
    head = create_list([1, 2, 3, 4, 5, 6, 7, 8])
    print(f"   Input: [1,2,3,4,5,6,7,8], k=2")
    result = solver.reverse_alternating_k_element_sublist(head, 2)
    print(f"   Output: {list_to_array(result)}")
    print()

    # Test 3: Plus One Linked List
    print("3. Plus One Linked List:")
    head = create_list([1, 2, 9])
    print(f"   Input: [1,2,9]")
    result = solver.plus_one_linked_list(head)
    print(f"   Output: {list_to_array(result)}")
    print()

    # Test 4: Remove Zero Sum Nodes
    print("4. Remove Zero Sum Consecutive Nodes:")
    head = create_list([1, 2, -3, 3, 1])
    print(f"   Input: [1,2,-3,3,1]")
    result = solver.remove_zero_sum_consecutive_nodes(head)
    print(f"   Output: {list_to_array(result)}")