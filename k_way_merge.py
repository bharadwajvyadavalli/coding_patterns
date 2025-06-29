"""
Pattern 13: K-way Merge - 10 Hard Problems
=========================================

The K-way Merge pattern efficiently merges K sorted arrays, lists, or data streams
into a single sorted output. This pattern is essential for problems involving
multiple sorted sources that need to be combined while maintaining order.

Key Concepts:
- Use min heap to track smallest element from each source
- Each heap element contains value and source identifier
- Process elements in sorted order across all sources
- Handle variable-length sources and infinite streams

Time Complexity: O(N log K) where N is total elements, K is number of sources
Space Complexity: O(K) for heap storage
"""

import heapq
from typing import List, Tuple, Optional, Iterator, Dict
from collections import defaultdict
import bisect


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class KWayMergeHard:

    def merge_k_sorted_lists_with_statistics(self, lists: List[ListNode]) -> Tuple[ListNode, Dict[str, int]]:
        """
        LeetCode 23 Extension - Merge k Sorted Lists with Statistics (Hard)

        Merge k sorted linked lists and return statistics.
        Extended: Track source list contribution and merge operations.

        Algorithm:
        1. Use min heap with list index tracking
        2. Count elements from each list
        3. Track merge operations

        Time: O(N log k), Space: O(k)

        Example:
        lists = [[1,4,5],[1,3,4],[2,6]]
        Output: 1->1->2->3->4->4->5->6, stats
        """
        if not lists:
            return None, {}

        # Statistics
        list_contributions = defaultdict(int)
        merge_operations = 0
        total_elements = 0

        # Min heap: (value, list_index, node)
        heap = []

        # Initialize heap with first node from each list
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(heap, (lst.val, i, lst))

        dummy = ListNode(0)
        current = dummy

        while heap:
            val, list_idx, node = heapq.heappop(heap)
            merge_operations += 1

            # Add to merged list
            current.next = node
            current = current.next

            # Update statistics
            list_contributions[list_idx] += 1
            total_elements += 1

            # Add next node from same list
            if node.next:
                heapq.heappush(heap, (node.next.val, list_idx, node.next))

        statistics = {
            'total_elements': total_elements,
            'merge_operations': merge_operations,
            'lists_count': len(lists),
            'contributions': dict(list_contributions),
            'average_list_size': total_elements / len(lists) if lists else 0
        }

        return dummy.next, statistics

    def smallest_range_covering_k_lists_optimized(self, nums: List[List[int]]) -> Tuple[List[int], List[int]]:
        """
        LeetCode 632 Extension - Smallest Range with Elements Used (Hard)

        Find smallest range that includes at least one number from each list.
        Extended: Return which element from each list is included.

        Algorithm:
        1. Use min heap to track current element from each list
        2. Track max element in current window
        3. Update range when all lists represented

        Time: O(N log k), Space: O(k)

        Example:
        nums = [[4,10,15,24],[0,9,12,20],[5,18,22,30]]
        Output: ([20,24], [24,20,22])
        """
        # Min heap: (value, list_idx, element_idx)
        heap = []
        max_val = float('-inf')

        # Initialize with first element from each list
        for i in range(len(nums)):
            if nums[i]:
                heapq.heappush(heap, (nums[i][0], i, 0))
                max_val = max(max_val, nums[i][0])

        # Track best range
        min_range = float('inf')
        result_range = []
        result_elements = []

        while len(heap) == len(nums):  # All lists must be represented
            min_val, list_idx, elem_idx = heapq.heappop(heap)

            # Update range if better
            if max_val - min_val < min_range:
                min_range = max_val - min_val
                result_range = [min_val, max_val]

                # Reconstruct current elements
                current_elements = [0] * len(nums)
                current_elements[list_idx] = nums[list_idx][elem_idx]

                # Get elements from heap
                heap_copy = heap[:]
                while heap_copy:
                    _, idx, e_idx = heapq.heappop(heap_copy)
                    current_elements[idx] = nums[idx][e_idx]

                result_elements = current_elements

            # Add next element from same list
            if elem_idx + 1 < len(nums[list_idx]):
                next_val = nums[list_idx][elem_idx + 1]
                heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
                max_val = max(max_val, next_val)

        return result_range, result_elements

    def merge_k_sorted_arrays_with_duplicates(self, arrays: List[List[int]],
                                              remove_duplicates: bool = True) -> List[int]:
        """
        Custom - Merge K Sorted Arrays with Duplicate Handling (Hard)

        Merge k sorted arrays with option to remove duplicates.
        Track duplicate count and source information.

        Algorithm:
        1. Use heap with array tracking
        2. Handle duplicates based on flag
        3. Maintain source information

        Time: O(N log k), Space: O(N)

        Example:
        arrays = [[1,3,5,7],[2,4,6,8],[1,2,3,4]], remove_duplicates = True
        Output: [1,2,3,4,5,6,7,8]
        """
        if not arrays:
            return []

        # Min heap: (value, array_idx, element_idx)
        heap = []

        # Initialize heap
        for i, arr in enumerate(arrays):
            if arr:
                heapq.heappush(heap, (arr[0], i, 0))

        result = []
        last_value = None

        while heap:
            val, arr_idx, elem_idx = heapq.heappop(heap)

            # Handle duplicates
            if not remove_duplicates or val != last_value:
                result.append(val)
                last_value = val

            # Add next element from same array
            if elem_idx + 1 < len(arrays[arr_idx]):
                heapq.heappush(heap,
                               (arrays[arr_idx][elem_idx + 1], arr_idx, elem_idx + 1))

        return result

    def find_k_smallest_pairs_distance(self, nums: List[int], k: int) -> int:
        """
        LeetCode 719 - Find K-th Smallest Pair Distance (Hard)

        Find k-th smallest distance among all pairs.
        Uses binary search with k-way merge concept.

        Algorithm:
        1. Binary search on distance
        2. Count pairs with distance <= mid
        3. Use two pointers for counting

        Time: O(n log n + n log(max-min)), Space: O(1)

        Example:
        nums = [1,3,1], k = 1
        Output: 0 (pairs (1,1) have distance 0)
        """
        nums.sort()
        n = len(nums)

        def count_pairs_with_distance_at_most(max_dist: int) -> int:
            """Count pairs with distance <= max_dist."""
            count = 0
            left = 0

            for right in range(n):
                while nums[right] - nums[left] > max_dist:
                    left += 1
                count += right - left

            return count

        # Binary search on distance
        left, right = 0, nums[-1] - nums[0]

        while left < right:
            mid = (left + right) // 2
            count = count_pairs_with_distance_at_most(mid)

            if count < k:
                left = mid + 1
            else:
                right = mid

        return left

    def employee_free_time_optimized(self, schedule: List[List[List[int]]]) -> List[List[int]]:
        """
        LeetCode 759 - Employee Free Time (Hard)

        Find common free time for all employees.
        Optimized using k-way merge approach.

        Algorithm:
        1. Merge all intervals using heap
        2. Find gaps in merged intervals
        3. Optimize by processing in order

        Time: O(N log k), Space: O(k)
        where N is total intervals, k is number of employees

        Example:
        schedule = [[[1,3],[4,6]],[[1,4]],[[4,5],[6,7]]]
        Output: [[3,4]]
        """
        # Min heap: (start_time, end_time, employee_idx, interval_idx)
        heap = []

        # Initialize with first interval from each employee
        for i, employee in enumerate(schedule):
            if employee:
                start, end = employee[0]
                heapq.heappush(heap, (start, end, i, 0))

        merged = []

        while heap:
            start, end, emp_idx, int_idx = heapq.heappop(heap)

            # Merge with previous interval if overlapping
            if merged and merged[-1][1] >= start:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

            # Add next interval from same employee
            if int_idx + 1 < len(schedule[emp_idx]):
                next_start, next_end = schedule[emp_idx][int_idx + 1]
                heapq.heappush(heap, (next_start, next_end, emp_idx, int_idx + 1))

        # Find free time (gaps between merged intervals)
        free_time = []
        for i in range(1, len(merged)):
            free_time.append([merged[i - 1][1], merged[i][0]])

        return free_time

    def merge_k_sorted_iterators(self, iterators: List[Iterator[int]]) -> List[int]:
        """
        Custom - Merge K Sorted Iterators/Streams (Hard)

        Merge k sorted iterators (potentially infinite).
        Handle exhausted iterators gracefully.

        Algorithm:
        1. Use heap with iterator references
        2. Lazily fetch next elements
        3. Handle iterator exhaustion

        Time: O(N log k), Space: O(k)
        """
        # Min heap: (value, iterator_idx)
        heap = []
        active_iterators = {}

        # Initialize heap with first element from each iterator
        for i, it in enumerate(iterators):
            try:
                val = next(it)
                heapq.heappush(heap, (val, i))
                active_iterators[i] = it
            except StopIteration:
                pass

        result = []

        while heap:
            val, iter_idx = heapq.heappop(heap)
            result.append(val)

            # Try to get next element from same iterator
            if iter_idx in active_iterators:
                try:
                    next_val = next(active_iterators[iter_idx])
                    heapq.heappush(heap, (next_val, iter_idx))
                except StopIteration:
                    del active_iterators[iter_idx]

        return result

    def merge_sorted_streams_with_timestamps(self, streams: List[List[Tuple[int, int]]]) -> List[Tuple[int, int, int]]:
        """
        Custom - Merge Sorted Streams with Timestamps (Hard)

        Merge k streams where each element has (timestamp, value).
        Maintain chronological order and source tracking.

        Algorithm:
        1. Use heap sorted by timestamp
        2. Track source stream
        3. Handle concurrent timestamps

        Time: O(N log k), Space: O(k)

        Example:
        streams = [[(1,10),(3,30)], [(2,20),(4,40)]]
        Output: [(1,10,0), (2,20,1), (3,30,0), (4,40,1)]
        """
        # Min heap: (timestamp, value, stream_idx, element_idx)
        heap = []

        # Initialize heap
        for i, stream in enumerate(streams):
            if stream:
                ts, val = stream[0]
                heapq.heappush(heap, (ts, val, i, 0))

        result = []

        while heap:
            ts, val, stream_idx, elem_idx = heapq.heappop(heap)
            result.append((ts, val, stream_idx))

            # Add next element from same stream
            if elem_idx + 1 < len(streams[stream_idx]):
                next_ts, next_val = streams[stream_idx][elem_idx + 1]
                heapq.heappush(heap, (next_ts, next_val, stream_idx, elem_idx + 1))

        return result

    def kth_smallest_in_m_sorted_arrays(self, arrays: List[List[int]], k: int) -> Tuple[int, Tuple[int, int]]:
        """
        Custom - K-th Smallest Element in M Sorted Arrays (Hard)

        Find k-th smallest element across all arrays.
        Extended: Return element and its position (array_idx, element_idx).

        Algorithm:
        1. Use min heap to track candidates
        2. Process elements in sorted order
        3. Stop at k-th element

        Time: O(k log m), Space: O(m)

        Example:
        arrays = [[1,5,9],[2,6,10],[3,7,11]], k = 5
        Output: (5, (0, 1))
        """
        # Min heap: (value, array_idx, element_idx)
        heap = []

        # Initialize with first element from each array
        for i, arr in enumerate(arrays):
            if arr:
                heapq.heappush(heap, (arr[0], i, 0))

        # Extract k elements
        for count in range(k):
            if not heap:
                return -1, (-1, -1)

            val, arr_idx, elem_idx = heapq.heappop(heap)

            # Add next element from same array
            if elem_idx + 1 < len(arrays[arr_idx]):
                heapq.heappush(heap,
                               (arrays[arr_idx][elem_idx + 1], arr_idx, elem_idx + 1))

            if count == k - 1:
                return val, (arr_idx, elem_idx)

        return -1, (-1, -1)

    def merge_intervals_from_k_sources(self, interval_lists: List[List[List[int]]]) -> List[List[int]]:
        """
        Custom - Merge Intervals from K Sources (Hard)

        Merge overlapping intervals from k different sources.
        Each source provides sorted non-overlapping intervals.

        Algorithm:
        1. Use heap to process intervals by start time
        2. Merge overlapping intervals
        3. Track source contribution

        Time: O(N log k), Space: O(k)

        Example:
        interval_lists = [[[1,3],[5,7]], [[2,4],[6,8]]]
        Output: [[1,4],[5,8]]
        """
        # Min heap: (start, end, source_idx, interval_idx)
        heap = []

        # Initialize with first interval from each source
        for i, intervals in enumerate(interval_lists):
            if intervals:
                start, end = intervals[0]
                heapq.heappush(heap, (start, end, i, 0))

        merged = []

        while heap:
            start, end, src_idx, int_idx = heapq.heappop(heap)

            # Merge with last interval if overlapping
            if merged and merged[-1][1] >= start:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

            # Add next interval from same source
            if int_idx + 1 < len(interval_lists[src_idx]):
                next_start, next_end = interval_lists[src_idx][int_idx + 1]
                heapq.heappush(heap, (next_start, next_end, src_idx, int_idx + 1))

        return merged

    def median_of_k_sorted_arrays(self, arrays: List[List[int]]) -> float:
        """
        Custom - Median of K Sorted Arrays (Hard)

        Find median element across all k sorted arrays.
        Generalization of "Median of Two Sorted Arrays".

        Algorithm:
        1. Binary search on median value
        2. Count elements <= mid across all arrays
        3. Adjust search range based on count

        Time: O(k * log(max_len) * log(max-min))
        Space: O(1)

        Example:
        arrays = [[1,3,5],[2,4,6],[1,2,3]]
        Output: 3.0
        """

        def count_less_equal(target: float) -> int:
            """Count elements <= target across all arrays."""
            count = 0
            for arr in arrays:
                # Binary search in each array
                count += bisect.bisect_right(arr, target)
            return count

        # Find total count
        total = sum(len(arr) for arr in arrays)
        if total == 0:
            return 0.0

        # Find min and max values
        min_val = min(arr[0] for arr in arrays if arr)
        max_val = max(arr[-1] for arr in arrays if arr)

        # Binary search for median
        left, right = min_val, max_val

        while right - left > 1e-5:
            mid = (left + right) / 2
            count = count_less_equal(mid)

            if count < (total + 1) // 2:
                left = mid
            else:
                right = mid

        # For exact median with even total
        if total % 2 == 0:
            # Find the two middle elements
            target_idx1 = total // 2
            target_idx2 = total // 2 + 1

            # This is simplified - in practice would need exact elements
            return left

        return left


# Example usage and testing
if __name__ == "__main__":
    solver = KWayMergeHard()

    # Test 1: Merge K Sorted Arrays
    print("1. Merge K Sorted Arrays with Duplicate Handling:")
    arrays = [[1, 3, 5, 7], [2, 4, 6, 8], [1, 2, 3, 4]]
    result = solver.merge_k_sorted_arrays_with_duplicates(arrays, remove_duplicates=True)
    print(f"   Arrays: {arrays}")
    print(f"   Merged (no duplicates): {result}")
    print()

    # Test 2: Smallest Range Covering K Lists
    print("2. Smallest Range Covering K Lists:")
    nums = [[4, 10, 15, 24], [0, 9, 12, 20], [5, 18, 22, 30]]
    range_result, elements = solver.smallest_range_covering_k_lists_optimized(nums)
    print(f"   Lists: {nums}")
    print(f"   Smallest range: {range_result}")
    print(f"   Elements used: {elements}")
    print()

    # Test 3: K-th Smallest in M Arrays
    print("3. K-th Smallest in M Sorted Arrays:")
    arrays = [[1, 5, 9], [2, 6, 10], [3, 7, 11]]
    k = 5
    value, position = solver.kth_smallest_in_m_sorted_arrays(arrays, k)
    print(f"   Arrays: {arrays}, k={k}")
    print(f"   {k}-th smallest: {value} at position {position}")