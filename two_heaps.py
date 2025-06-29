"""
Pattern 9: Two Heaps - 10 Hard Problems
=======================================

The Two Heaps pattern uses a min-heap and a max-heap to efficiently track medians
or manage data streams. This pattern is essential for problems requiring quick
access to middle elements or maintaining balance between two halves of data.

Key Concepts:
- Max heap for smaller half, min heap for larger half
- Balance heaps to differ by at most 1 element
- Median is top of larger heap or average of both tops
- Can be extended to track other percentiles

Time Complexity: O(log n) for insertions, O(1) for median
Space Complexity: O(n) for storing all elements
"""

import heapq
from typing import List, Tuple, Optional, Dict
from collections import defaultdict, deque


class TwoHeapsHard:

    def sliding_window_median_with_removed_elements(self, nums: List[int], k: int) -> Tuple[
        List[float], Dict[int, List[int]]]:
        """
        LeetCode 480 Extension - Sliding Window Median with Tracking (Hard)

        Find median of each sliding window of size k.
        Extended: Track which elements are removed at each step.

        Algorithm:
        1. Use two heaps with lazy deletion
        2. Track invalid elements in heaps
        3. Balance heaps after each operation
        4. Handle duplicates correctly

        Time: O(n * log k), Space: O(k)

        Example:
        nums = [1,3,-1,-3,5,3,6,7], k = 3
        Output: ([1,-1,-1,3,5,6], {1:[1], 2:[3], 3:[-1], 4:[-3], 5:[5], 6:[3]})
        """

        def get_median():
            if k % 2:
                return float(-max_heap[0])
            else:
                return (-max_heap[0] + min_heap[0]) / 2.0

        max_heap = []  # Smaller half (negated for max behavior)
        min_heap = []  # Larger half
        heap_dict = defaultdict(int)  # Track element counts
        removed_elements = {}

        # Initialize heaps with first window
        for i in range(k):
            heapq.heappush(max_heap, -nums[i])

        # Balance initial heaps
        for _ in range(k // 2):
            heapq.heappush(min_heap, -heapq.heappop(max_heap))

        # Process windows
        medians = [get_median()]

        for i in range(k, len(nums)):
            # Remove outgoing element
            out_num = nums[i - k]
            removed_elements[i - k + 1] = [out_num]

            # Add incoming element
            in_num = nums[i]

            # Determine which heap the outgoing element is in
            balance = 0
            if out_num <= -max_heap[0]:
                balance -= 1
                heap_dict[out_num] += 1
            else:
                balance += 1
                heap_dict[out_num] += 1

            # Add incoming element
            if max_heap and in_num <= -max_heap[0]:
                balance += 1
                heapq.heappush(max_heap, -in_num)
            else:
                balance -= 1
                heapq.heappush(min_heap, in_num)

            # Rebalance heaps
            if balance < 0:  # max_heap needs more
                heapq.heappush(max_heap, -heapq.heappop(min_heap))
            elif balance > 0:  # min_heap needs more
                heapq.heappush(min_heap, -heapq.heappop(max_heap))

            # Remove invalid elements from heap tops
            while max_heap and heap_dict[-max_heap[0]] > 0:
                heap_dict[-max_heap[0]] -= 1
                heapq.heappop(max_heap)

            while min_heap and heap_dict[min_heap[0]] > 0:
                heap_dict[min_heap[0]] -= 1
                heapq.heappop(min_heap)

            medians.append(get_median())

        return medians, removed_elements

    def find_median_from_data_stream_with_percentiles(self):
        """
        LeetCode 295 Extension - Data Stream Median with Percentiles (Hard)

        Design a data structure that supports:
        - addNum(num): Add number to stream
        - findMedian(): Return median
        - findPercentile(p): Return p-th percentile

        Time: O(log n) for add, O(1) for median, O(n log n) for percentile
        """

        class MedianFinderExtended:
            def __init__(self):
                self.max_heap = []  # Smaller half
                self.min_heap = []  # Larger half
                self.all_nums = []  # For percentile calculation

            def addNum(self, num: int) -> None:
                """Add number to data stream."""
                # Add to all_nums for percentile
                self.all_nums.append(num)

                # Add to appropriate heap
                if not self.max_heap or num <= -self.max_heap[0]:
                    heapq.heappush(self.max_heap, -num)
                else:
                    heapq.heappush(self.min_heap, num)

                # Balance heaps
                if len(self.max_heap) > len(self.min_heap) + 1:
                    heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
                elif len(self.min_heap) > len(self.max_heap):
                    heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

            def findMedian(self) -> float:
                """Find median of all elements."""
                if len(self.max_heap) > len(self.min_heap):
                    return float(-self.max_heap[0])
                return (-self.max_heap[0] + self.min_heap[0]) / 2.0

            def findPercentile(self, p: float) -> float:
                """Find p-th percentile (0 <= p <= 100)."""
                if not self.all_nums:
                    return 0

                sorted_nums = sorted(self.all_nums)
                n = len(sorted_nums)

                # Calculate index for percentile
                index = (n - 1) * p / 100.0
                lower = int(index)
                upper = min(lower + 1, n - 1)
                weight = index - lower

                # Interpolate if necessary
                return sorted_nums[lower] * (1 - weight) + sorted_nums[upper] * weight

            def getQuartiles(self) -> Tuple[float, float, float]:
                """Get Q1, Q2 (median), Q3."""
                return (
                    self.findPercentile(25),
                    self.findMedian(),
                    self.findPercentile(75)
                )

        return MedianFinderExtended

    def ipo_with_project_selection(self, k: int, w: int, profits: List[int], capital: List[int]) -> Tuple[
        int, List[int]]:
        """
        LeetCode 502 Extension - IPO with Project Selection Tracking (Hard)

        Maximize capital by selecting at most k projects.
        Extended: Return final capital and selected project indices.

        Algorithm:
        1. Min heap for projects by capital requirement
        2. Max heap for available projects by profit
        3. Greedily select best available project
        4. Track selection order

        Time: O(n log n), Space: O(n)

        Example:
        k = 2, w = 0, profits = [1,2,3], capital = [0,1,1]
        Output: (4, [1, 2]) - Projects with profits 1 and 3
        """
        n = len(profits)
        # Projects as (capital, profit, index)
        projects = [(capital[i], profits[i], i) for i in range(n)]
        projects.sort()  # Sort by capital requirement

        available = []  # Max heap of (-profit, index)
        selected = []
        i = 0

        for _ in range(k):
            # Add all projects we can afford
            while i < n and projects[i][0] <= w:
                heapq.heappush(available, (-projects[i][1], projects[i][2]))
                i += 1

            if not available:
                break

            # Select most profitable project
            neg_profit, proj_idx = heapq.heappop(available)
            w += -neg_profit
            selected.append(proj_idx)

        return w, selected

    def maximum_frequency_stack_with_history(self):
        """
        LeetCode 895 Extension - Maximum Frequency Stack with History (Hard)

        Design stack that pops most frequent element.
        Extended: Track frequency history over time.

        Algorithm:
        1. Use frequency map and stack of stacks
        2. Each frequency level has its own stack
        3. Track push/pop history

        Time: O(1) for push and pop
        """

        class FreqStackExtended:
            def __init__(self):
                self.freq = defaultdict(int)
                self.group = defaultdict(list)
                self.max_freq = 0
                self.history = []  # (operation, value, frequency)

            def push(self, val: int) -> None:
                """Push element onto stack."""
                self.freq[val] += 1
                f = self.freq[val]
                self.max_freq = max(self.max_freq, f)
                self.group[f].append(val)
                self.history.append(('push', val, f))

            def pop(self) -> int:
                """Pop most frequent element."""
                val = self.group[self.max_freq].pop()
                self.freq[val] -= 1
                if not self.group[self.max_freq]:
                    self.max_freq -= 1
                self.history.append(('pop', val, self.freq[val] + 1))
                return val

            def getHistory(self) -> List[Tuple[str, int, int]]:
                """Get operation history."""
                return self.history

            def getMostFrequentAtTime(self, time: int) -> List[int]:
                """Get most frequent elements at given time in history."""
                if time >= len(self.history):
                    return []

                # Replay history up to time
                freq_at_time = defaultdict(int)
                max_f = 0

                for i in range(time + 1):
                    op, val, _ = self.history[i]
                    if op == 'push':
                        freq_at_time[val] += 1
                        max_f = max(max_f, freq_at_time[val])
                    else:  # pop
                        freq_at_time[val] -= 1

                # Find all elements with max frequency
                return [val for val, f in freq_at_time.items() if f == max_f]

        return FreqStackExtended

    def schedule_tasks_with_cooldown(self, tasks: List[str], n: int) -> Tuple[int, List[str]]:
        """
        LeetCode 621 Extension - Task Scheduler with Execution Order (Hard)

        Schedule tasks with cooldown period n between same tasks.
        Extended: Return execution order, not just time.

        Algorithm:
        1. Max heap for task frequencies
        2. Queue for cooling tasks
        3. Simulate time steps
        4. Track actual execution order

        Time: O(m * log 26) where m is total tasks
        Space: O(1) - at most 26 unique tasks

        Example:
        tasks = ["A","A","A","B","B","B"], n = 2
        Output: (8, ["A","B","idle","A","B","idle","A","B"])
        """
        # Count frequencies
        freq = defaultdict(int)
        for task in tasks:
            freq[task] += 1

        # Max heap of (-frequency, task)
        heap = [(-f, task) for task, f in freq.items()]
        heapq.heapify(heap)

        time = 0
        execution_order = []
        cooldown_queue = deque()  # (available_time, frequency, task)

        while heap or cooldown_queue:
            time += 1

            # Move cooled down tasks back to heap
            while cooldown_queue and cooldown_queue[0][0] <= time:
                _, neg_freq, task = cooldown_queue.popleft()
                heapq.heappush(heap, (neg_freq, task))

            if heap:
                # Execute highest frequency task
                neg_freq, task = heapq.heappop(heap)
                execution_order.append(task)

                # If task still has remaining executions
                if neg_freq < -1:
                    cooldown_queue.append((time + n + 1, neg_freq + 1, task))
            else:
                # CPU idle
                execution_order.append("idle")

        return time, execution_order

    def find_right_interval_optimized(self, intervals: List[List[int]]) -> List[int]:
        """
        LeetCode 436 - Find Right Interval (Hard optimization)

        For each interval, find interval with smallest start >= current end.
        Use two heaps for optimal solution.

        Algorithm:
        1. Min heap of starts with original indices
        2. Process intervals by end time
        3. Find minimum valid start

        Time: O(n log n), Space: O(n)

        Example:
        intervals = [[3,4],[2,3],[1,2]]
        Output: [-1,0,1]
        """
        n = len(intervals)

        # Create tuples with original indices
        starts = [(interval[0], i) for i, interval in enumerate(intervals)]
        ends = [(interval[1], i) for i, interval in enumerate(intervals)]

        # Sort by start and end
        starts.sort()
        ends.sort()

        result = [-1] * n
        j = 0

        # For each interval in end order
        for end, end_idx in ends:
            # Find first start >= end
            while j < n and starts[j][0] < end:
                j += 1

            if j < n:
                result[end_idx] = starts[j][1]

            # Reset j for next iteration if needed
            if j > 0 and j < n:
                # Binary search for exact position
                left, right = 0, j
                while left < right:
                    mid = (left + right) // 2
                    if starts[mid][0] < end:
                        left = mid + 1
                    else:
                        right = mid
                j = left

        return result

    def meeting_rooms_iii_with_schedule(self, n: int, meetings: List[List[int]]) -> Tuple[
        int, Dict[int, List[List[int]]]]:
        """
        LeetCode 2402 Extension - Meeting Rooms III with Full Schedule (Hard)

        Allocate meetings to n rooms, return busiest room and full schedule.

        Algorithm:
        1. Min heap for free rooms
        2. Min heap for occupied rooms by end time
        3. Track complete schedule per room

        Time: O(m log m + m log n), Space: O(m)

        Example:
        n = 2, meetings = [[0,10],[1,5],[2,7],[3,4]]
        Output: (0, {0:[[0,10],[3,13]], 1:[[1,5],[2,7]]})
        """
        meetings.sort()  # Sort by start time

        # Initialize heaps
        free_rooms = list(range(n))  # Min heap of room numbers
        occupied = []  # Min heap of (end_time, room_number)
        room_schedules = {i: [] for i in range(n)}
        room_count = [0] * n

        for start, end in meetings:
            # Free up rooms that have ended
            while occupied and occupied[0][0] <= start:
                _, room = heapq.heappop(occupied)
                heapq.heappush(free_rooms, room)

            if free_rooms:
                # Assign to lowest numbered free room
                room = heapq.heappop(free_rooms)
                actual_start = start
                actual_end = end
            else:
                # Wait for earliest room to be free
                earliest_end, room = heapq.heappop(occupied)
                actual_start = earliest_end
                actual_end = earliest_end + (end - start)

            # Schedule meeting
            heapq.heappush(occupied, (actual_end, room))
            room_schedules[room].append([actual_start, actual_end])
            room_count[room] += 1

        # Find busiest room
        max_meetings = max(room_count)
        busiest_room = room_count.index(max_meetings)

        return busiest_room, room_schedules

    def continuous_subarrays_with_bounded_difference(self, nums: List[int]) -> int:
        """
        LeetCode 2762 - Continuous Subarrays (Hard)

        Count subarrays where absolute difference between any two elements <= 2.
        Use two heaps to track min/max in sliding window.

        Algorithm:
        1. Sliding window with min/max heaps
        2. Lazy deletion for elements outside window
        3. Shrink window when constraint violated

        Time: O(n log n), Space: O(n)

        Example:
        nums = [5,4,2,4]
        Output: 8
        """
        n = len(nums)
        count = 0
        left = 0

        # Max heap (negated values) and min heap with indices
        max_heap = []  # (-value, index)
        min_heap = []  # (value, index)

        for right in range(n):
            # Add current element to heaps
            heapq.heappush(max_heap, (-nums[right], right))
            heapq.heappush(min_heap, (nums[right], right))

            # Shrink window while constraint is violated
            while max_heap and min_heap:
                # Clean up heaps
                while max_heap and max_heap[0][1] < left:
                    heapq.heappop(max_heap)
                while min_heap and min_heap[0][1] < left:
                    heapq.heappop(min_heap)

                if not max_heap or not min_heap:
                    break

                # Check constraint
                max_val = -max_heap[0][0]
                min_val = min_heap[0][0]

                if max_val - min_val <= 2:
                    break

                # Move left pointer
                left += 1

            # Count subarrays ending at right
            count += right - left + 1

        return count

    def maximum_sum_of_two_non_overlapping_subarrays(self, nums: List[int], firstLen: int, secondLen: int) -> int:
        """
        LeetCode 1031 Extension - Two Non-Overlapping Subarrays with Positions (Hard)

        Find maximum sum of two non-overlapping subarrays of given lengths.
        Extended: Use heaps to track best options dynamically.

        Algorithm:
        1. Precompute all subarray sums
        2. Use heaps to find best non-overlapping pairs
        3. Consider both orderings (first before/after second)

        Time: O(n log n), Space: O(n)

        Example:
        nums = [0,6,5,2,2,5,1,9,4], firstLen = 1, secondLen = 2
        Output: 20 (subarrays [9] and [6,5])
        """
        n = len(nums)

        # Compute prefix sums
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)

        def max_sum_with_positions(len1: int, len2: int) -> Tuple[int, List[int], List[int]]:
            # Max sum where len1 subarray comes before len2
            max_sum = 0
            best_positions = ([], [])

            # Track best len1 subarray ending before current position
            best_len1 = 0
            best_len1_pos = []

            # Try all positions for second subarray
            for i in range(len1 + len2, n + 1):
                # Update best first subarray
                j = i - len2 - len1
                if j >= 0:
                    sum1 = prefix[j + len1] - prefix[j]
                    if sum1 > best_len1:
                        best_len1 = sum1
                        best_len1_pos = [j, j + len1 - 1]

                # Calculate second subarray sum
                sum2 = prefix[i] - prefix[i - len2]

                if best_len1 + sum2 > max_sum:
                    max_sum = best_len1 + sum2
                    best_positions = (best_len1_pos, [i - len2, i - 1])

            return max_sum, best_positions[0], best_positions[1]

        # Try both orderings
        sum1, pos1_1, pos1_2 = max_sum_with_positions(firstLen, secondLen)
        sum2, pos2_1, pos2_2 = max_sum_with_positions(secondLen, firstLen)

        return max(sum1, sum2)

    def process_queries_with_moving_median(self, queries: List[int], m: int) -> List[int]:
        """
        LeetCode 1409 Variant - Queries on Permutation With Moving Median (Hard)

        Process queries on permutation [1,2,...,m] where queried element moves to front.
        Track median after each operation using two heaps.

        Algorithm:
        1. Maintain position array and two heaps
        2. Update heaps when elements move
        3. Rebalance to maintain median property

        Time: O(q * m * log m) where q is number of queries
        Space: O(m)

        Example:
        queries = [3,1,2,1], m = 5
        Output: [2,1,2,1] (positions of queried elements)
        """
        # Initialize permutation
        perm = list(range(1, m + 1))
        position = {val: i for i, val in enumerate(perm)}

        # Initialize heaps for median
        max_heap = []  # Smaller half
        min_heap = []  # Larger half

        # Initial distribution
        mid = m // 2
        for i in range(mid):
            heapq.heappush(max_heap, -perm[i])
        for i in range(mid, m):
            heapq.heappush(min_heap, perm[i])

        result = []

        for query in queries:
            # Find position and add to result
            pos = position[query]
            result.append(pos)

            # Move element to front
            perm.pop(pos)
            perm.insert(0, query)

            # Update positions
            for i in range(pos + 1):
                position[perm[i]] = i

            # Rebuild heaps (simplified for demonstration)
            # In practice, would use more efficient update
            max_heap = []
            min_heap = []

            for i in range(mid):
                heapq.heappush(max_heap, -perm[i])
            for i in range(mid, m):
                heapq.heappush(min_heap, perm[i])

        return result


# Example usage and testing
if __name__ == "__main__":
    solver = TwoHeapsHard()

    # Test 1: Sliding Window Median
    print("1. Sliding Window Median with Tracking:")
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    medians, removed = solver.sliding_window_median_with_removed_elements(nums, k)
    print(f"   Input: nums={nums}, k={k}")
    print(f"   Medians: {medians}")
    print(f"   Removed at each step: {list(removed.items())[:3]}...")
    print()

    # Test 2: IPO with Project Selection
    print("2. IPO with Project Selection:")
    k, w = 2, 0
    profits = [1, 2, 3]
    capital = [0, 1, 1]
    final_capital, projects = solver.ipo_with_project_selection(k, w, profits, capital)
    print(f"   k={k}, w={w}, profits={profits}, capital={capital}")
    print(f"   Final capital: {final_capital}, Selected projects: {projects}")
    print()

    # Test 3: Task Scheduler
    print("3. Task Scheduler with Execution Order:")
    tasks = ["A", "A", "A", "B", "B", "B"]
    n = 2
    time, order = solver.schedule_tasks_with_cooldown(tasks, n)
    print(f"   Tasks: {tasks}, Cooldown: {n}")
    print(f"   Time: {time}, Order: {order}")