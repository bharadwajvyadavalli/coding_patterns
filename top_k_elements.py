"""
Pattern 12: Top K Elements - 10 Hard Problems
============================================

The Top K Elements pattern finds the K largest/smallest elements in a collection
using heaps or similar data structures. This pattern is crucial for problems
involving rankings, frequent elements, or maintaining top K statistics.

Key Concepts:
- Use min heap of size K for K largest elements
- Use max heap of size K for K smallest elements
- Quick select for O(n) average time complexity
- Handle streaming data and updates efficiently

Time Complexity: O(n log k) with heap, O(n) average with quick select
Space Complexity: O(k) for storing K elements
"""

import heapq
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict, Counter
import random


class TopKElementsHard:

    def k_closest_points_with_equal_distance_handling(self, points: List[List[int]], k: int) -> Tuple[
        List[List[int]], Dict[float, List[List[int]]]]:
        """
        LeetCode 973 Extension - K Closest Points with Distance Groups (Hard)

        Find K closest points to origin, handling equal distances.
        Extended: Group all points by distance and handle ties.

        Algorithm:
        1. Use max heap to track K closest
        2. Group points by distance
        3. Handle ties at boundary distance

        Time: O(n log k), Space: O(n)

        Example:
        points = [[1,3],[-2,2],[2,1]], k = 2
        Output: ([[-2,2],[2,1]], {5.0: [[2,1],[-2,2]], 10.0: [[1,3]]})
        """
        # Calculate distances and group by distance
        distance_groups = defaultdict(list)

        for point in points:
            dist = point[0] ** 2 + point[1] ** 2
            distance_groups[dist].append(point)

        # Sort distances
        sorted_distances = sorted(distance_groups.keys())

        # Select K closest points
        result = []
        remaining = k

        for dist in sorted_distances:
            points_at_dist = distance_groups[dist]

            if len(points_at_dist) <= remaining:
                result.extend(points_at_dist)
                remaining -= len(points_at_dist)
            else:
                # Handle tie at boundary
                # Could use additional criteria or random selection
                result.extend(points_at_dist[:remaining])
                break

        # Convert distances to float for output
        float_groups = {float(d): pts for d, pts in distance_groups.items()}

        return result, float_groups

    def top_k_frequent_words_with_trends(self, words: List[str], k: int) -> Tuple[List[str], Dict[str, List[int]]]:
        """
        LeetCode 692 Extension - Top K Frequent Words with Position Trends (Hard)

        Find K most frequent words with lexicographical ordering for ties.
        Extended: Track position trends for each word.

        Algorithm:
        1. Count frequencies and track positions
        2. Use heap with custom comparator
        3. Analyze position trends

        Time: O(n log k), Space: O(n)

        Example:
        words = ["i","love","leetcode","i","love","coding"], k = 2
        Output: (["i","love"], {"i": [0,3], "love": [1,4], ...})
        """
        # Count frequencies and track positions
        word_count = Counter(words)
        word_positions = defaultdict(list)

        for i, word in enumerate(words):
            word_positions[word].append(i)

        # Use min heap with custom ordering
        # Heap elements: (-frequency, word)
        heap = []

        for word, freq in word_count.items():
            heapq.heappush(heap, (-freq, word))

        # Extract top K
        result = []
        for _ in range(min(k, len(heap))):
            _, word = heapq.heappop(heap)
            result.append(word)

        return result, dict(word_positions)

    def find_k_pairs_with_smallest_sums_optimized(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        """
        LeetCode 373 - Find K Pairs with Smallest Sums (Hard Optimization)

        Find K pairs (u,v) with smallest sums where u from nums1, v from nums2.
        Optimized to handle large arrays efficiently.

        Algorithm:
        1. Use min heap with smart initialization
        2. Only explore necessary pairs
        3. Track visited pairs to avoid duplicates

        Time: O(k log k), Space: O(k)

        Example:
        nums1 = [1,7,11], nums2 = [2,4,6], k = 3
        Output: [[1,2],[1,4],[1,6]]
        """
        if not nums1 or not nums2:
            return []

        # Min heap: (sum, i, j)
        heap = [(nums1[0] + nums2[0], 0, 0)]
        visited = {(0, 0)}
        result = []

        while heap and len(result) < k:
            curr_sum, i, j = heapq.heappop(heap)
            result.append([nums1[i], nums2[j]])

            # Add next candidates
            if i + 1 < len(nums1) and (i + 1, j) not in visited:
                heapq.heappush(heap, (nums1[i + 1] + nums2[j], i + 1, j))
                visited.add((i + 1, j))

            if j + 1 < len(nums2) and (i, j + 1) not in visited:
                heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
                visited.add((i, j + 1))

        return result

    def kth_smallest_prime_fraction(self, arr: List[int], k: int) -> List[int]:
        """
        LeetCode 786 - K-th Smallest Prime Fraction (Hard)

        Find K-th smallest fraction from array elements.
        Array is sorted with unique primes.

        Algorithm:
        1. Binary search on fraction value
        2. Count fractions less than mid
        3. Track the largest fraction < mid

        Time: O(n log(max/min)), Space: O(1)

        Example:
        arr = [1,2,3,5], k = 3
        Output: [2,5] (fraction 2/5)
        """

        def count_fractions_less_than(target: float) -> Tuple[int, List[int]]:
            """Count fractions < target and find largest such fraction."""
            count = 0
            max_fraction = [0, 1]

            j = 1
            for i in range(len(arr) - 1):
                # Find largest j where arr[i]/arr[j] < target
                while j < len(arr) and arr[i] < target * arr[j]:
                    j += 1

                # All fractions arr[i]/arr[j:] are < target
                count += len(arr) - j

                # Track largest fraction
                if j < len(arr) and arr[i] * max_fraction[1] > arr[j] * max_fraction[0]:
                    max_fraction = [arr[i], arr[j]]

            return count, max_fraction

        # Binary search on fraction value
        left, right = 0.0, 1.0

        while left < right:
            mid = (left + right) / 2
            count, fraction = count_fractions_less_than(mid)

            if count < k:
                left = mid
            else:
                right = mid
                result = fraction

        # Final check
        _, result = count_fractions_less_than(right)
        return result

    def super_ugly_number_with_factors(self, n: int, primes: List[int]) -> Tuple[int, List[Tuple[int, List[int]]]]:
        """
        LeetCode 313 Extension - Super Ugly Number with Factorization (Hard)

        Find n-th ugly number using given primes.
        Extended: Track prime factorization of each ugly number.

        Algorithm:
        1. Use heap to generate ugly numbers in order
        2. Track which primes were used
        3. Avoid duplicates with set

        Time: O(n * len(primes)), Space: O(n)

        Example:
        n = 12, primes = [2,7,13,19]
        Output: (32, [(1,[]), (2,[2]), (4,[2,2]), ..., (32,[2,2,2,2,2])])
        """
        # Min heap: (value, factorization as list of primes)
        heap = [(1, [])]
        seen = {1}
        ugly_numbers = []

        for _ in range(n):
            val, factors = heapq.heappop(heap)
            ugly_numbers.append((val, factors[:]))

            # Generate next ugly numbers
            for prime in primes:
                new_val = val * prime
                if new_val not in seen:
                    seen.add(new_val)
                    new_factors = factors + [prime]
                    heapq.heappush(heap, (new_val, new_factors))

        return ugly_numbers[-1][0], ugly_numbers[:min(20, n)]  # Limit output

    def reorganize_string_with_frequency_limit(self, s: str, k: int) -> str:
        """
        Extension of LeetCode 767 - Reorganize String with K-Distance (Hard)

        Rearrange string so same characters are at least k distance apart.
        Return empty string if impossible.

        Algorithm:
        1. Count frequencies
        2. Use max heap to greedily place most frequent
        3. Use queue to track cooling characters

        Time: O(n log 26) = O(n), Space: O(1)

        Example:
        s = "aabbcc", k = 3
        Output: "abcabc"
        """
        # Count frequencies
        freq = Counter(s)

        # Max heap of (-frequency, char)
        heap = [(-f, ch) for ch, f in freq.items()]
        heapq.heapify(heap)

        # Queue for cooling characters: (available_position, freq, char)
        cooling = deque()
        result = []

        while heap or cooling:
            # Move cooled characters back to heap
            while cooling and cooling[0][0] <= len(result):
                _, neg_freq, char = cooling.popleft()
                heapq.heappush(heap, (neg_freq, char))

            if not heap:
                # No available characters
                return ""

            # Place most frequent available character
            neg_freq, char = heapq.heappop(heap)
            result.append(char)

            # Add to cooling queue if more instances remain
            if neg_freq < -1:
                cooling.append((len(result) + k - 1, neg_freq + 1, char))

        return ''.join(result)

    def k_smallest_in_multiplication_table(self, m: int, n: int, k: int) -> int:
        """
        LeetCode 668 - Kth Smallest Number in Multiplication Table (Hard)

        Find k-th smallest number in m x n multiplication table.

        Algorithm:
        1. Binary search on value
        2. Count elements <= mid value
        3. Optimize counting using division

        Time: O(m * log(m*n)), Space: O(1)

        Example:
        m = 3, n = 3, k = 5
        Output: 3 (table: [[1,2,3],[2,4,6],[3,6,9]])
        """

        def count_less_equal(x: int) -> int:
            """Count numbers <= x in multiplication table."""
            count = 0
            for i in range(1, m + 1):
                # In row i, elements are i, 2i, 3i, ..., ni
                # Count of elements <= x is min(x // i, n)
                count += min(x // i, n)
            return count

        left, right = 1, m * n

        while left < right:
            mid = (left + right) // 2
            if count_less_equal(mid) < k:
                left = mid + 1
            else:
                right = mid

        return left

    def max_sum_of_k_subsequences(self, nums: List[int], k: int) -> List[int]:
        """
        Custom Hard - Maximum Sum of K Non-Overlapping Subsequences

        Partition array into k subsequences to maximize total sum.
        Each element must belong to exactly one subsequence.

        Algorithm:
        1. Use DP with heap optimization
        2. Track k best partial solutions
        3. Reconstruct partition

        Time: O(n * k * log k), Space: O(n * k)

        Example:
        nums = [1,2,3,4,5,6], k = 3
        Output: [11, 7, 3] (subsequences: [5,6], [3,4], [1,2])
        """
        n = len(nums)
        if k > n:
            return []

        # dp[i][j] = max sum using first i elements in j subsequences
        # Store top solutions with heap
        dp = [[[] for _ in range(k + 1)] for _ in range(n + 1)]

        # Initialize
        for i in range(n + 1):
            dp[i][0] = [(0, [])]

        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                candidates = []

                # Option 1: Start new subsequence with nums[i-1]
                for sum_val, partition in dp[i - 1][j - 1]:
                    new_partition = partition + [[nums[i - 1]]]
                    candidates.append((sum_val + nums[i - 1], new_partition))

                # Option 2: Add to existing subsequence
                if j <= i - 1:
                    for sum_val, partition in dp[i - 1][j]:
                        if partition:
                            # Try adding to each existing subsequence
                            for idx in range(len(partition)):
                                new_partition = [p[:] for p in partition]
                                new_partition[idx].append(nums[i - 1])
                                candidates.append((sum_val + nums[i - 1], new_partition))

                # Keep top solutions
                candidates.sort(reverse=True)
                dp[i][j] = candidates[:5]  # Keep top 5 for efficiency

        # Extract best solution
        if dp[n][k]:
            _, best_partition = dp[n][k][0]
            return [sum(subseq) for subseq in best_partition]

        return []

    def skyline_with_k_buildings(self, buildings: List[List[int]], k: int) -> List[List[int]]:
        """
        Extension of LeetCode 218 - Skyline with K Tallest Buildings (Hard)

        Find skyline considering only k tallest buildings at each position.

        Algorithm:
        1. Create events for building start/end
        2. Use heap to track k tallest at each position
        3. Output key points where height changes

        Time: O(n log n log k), Space: O(n)

        Example:
        buildings = [[2,9,10],[3,7,15],[5,12,12]], k = 1
        Output: [[2,10],[3,15],[7,12],[12,0]]
        """
        # Create events
        events = []
        for left, right, height in buildings:
            events.append((left, 'start', height))
            events.append((right, 'end', height))

        # Sort events
        events.sort(key=lambda x: (x[0], x[1] == 'end', -x[2] if x[1] == 'start' else x[2]))

        result = []
        # Min heap of k tallest buildings (to easily remove smallest)
        active_heights = []
        height_count = defaultdict(int)

        i = 0
        while i < len(events):
            curr_x = events[i][0]

            # Process all events at current x
            while i < len(events) and events[i][0] == curr_x:
                x, event_type, height = events[i]

                if event_type == 'start':
                    height_count[height] += 1

                    if len(active_heights) < k:
                        heapq.heappush(active_heights, height)
                    elif height > active_heights[0]:
                        removed = heapq.heappushpop(active_heights, height)
                        if height_count[removed] == 0:
                            # Need to find replacement
                            for h in sorted(height_count.keys(), reverse=True):
                                if height_count[h] > 0 and h not in active_heights:
                                    heapq.heappush(active_heights, h)
                                    break
                else:  # end
                    height_count[height] -= 1
                    if height_count[height] == 0:
                        del height_count[height]

                    if height in active_heights:
                        active_heights.remove(height)
                        heapq.heapify(active_heights)

                        # Find replacement
                        for h in sorted(height_count.keys(), reverse=True):
                            if height_count[h] > 0 and h not in active_heights:
                                heapq.heappush(active_heights, h)
                                break

                i += 1

            # Determine current max height
            max_height = max(active_heights) if active_heights else 0

            # Add key point if height changed
            if not result or result[-1][1] != max_height:
                result.append([curr_x, max_height])

        return result

    def kth_largest_in_stream_with_statistics(self):
        """
        LeetCode 703 Extension - Kth Largest with Running Statistics (Hard)

        Design class that tracks kth largest and provides statistics.
        Extended: Track mean, median, and range of top k elements.

        Time: O(log k) for add, O(k) for statistics
        """

        class KthLargestExtended:
            def __init__(self, k: int, nums: List[int]):
                self.k = k
                self.min_heap = []
                self.all_top_k = []  # For statistics

                for num in nums:
                    self.add(num)

            def add(self, val: int) -> int:
                heapq.heappush(self.min_heap, val)

                if len(self.min_heap) > self.k:
                    heapq.heappop(self.min_heap)

                # Update statistics list
                self.all_top_k = sorted(self.min_heap, reverse=True)

                return self.min_heap[0] if len(self.min_heap) == self.k else -1

            def get_mean(self) -> float:
                """Get mean of top k elements."""
                if not self.all_top_k:
                    return 0
                return sum(self.all_top_k) / len(self.all_top_k)

            def get_median(self) -> float:
                """Get median of top k elements."""
                if not self.all_top_k:
                    return 0

                n = len(self.all_top_k)
                if n % 2 == 1:
                    return self.all_top_k[n // 2]
                else:
                    return (self.all_top_k[n // 2 - 1] + self.all_top_k[n // 2]) / 2

            def get_range(self) -> Tuple[int, int]:
                """Get range (min, max) of top k elements."""
                if not self.all_top_k:
                    return (0, 0)
                return (self.all_top_k[-1], self.all_top_k[0])

            def get_statistics(self) -> Dict[str, float]:
                """Get all statistics."""
                return {
                    'kth_largest': self.min_heap[0] if len(self.min_heap) == self.k else -1,
                    'mean': self.get_mean(),
                    'median': self.get_median(),
                    'min': self.get_range()[0],
                    'max': self.get_range()[1]
                }

        return KthLargestExtended


# Example usage and testing
if __name__ == "__main__":
    solver = TopKElementsHard()

    # Test 1: K Closest Points
    print("1. K Closest Points with Distance Groups:")
    points = [[1, 3], [-2, 2], [2, 1]]
    k = 2
    closest, groups = solver.k_closest_points_with_equal_distance_handling(points, k)
    print(f"   Points: {points}, k={k}")
    print(f"   K closest: {closest}")
    print(f"   Distance groups: {groups}")
    print()

    # Test 2: Find K Pairs with Smallest Sums
    print("2. K Pairs with Smallest Sums:")
    nums1 = [1, 7, 11]
    nums2 = [2, 4, 6]
    k = 3
    pairs = solver.find_k_pairs_with_smallest_sums_optimized(nums1, nums2, k)
    print(f"   nums1={nums1}, nums2={nums2}, k={k}")
    print(f"   K smallest pairs: {pairs}")
    print()

    # Test 3: Reorganize String with K Distance
    print("3. Reorganize String with K-Distance:")
    s = "aabbcc"
    k = 3
    result = solver.reorganize_string_with_frequency_limit(s, k)
    print(f"   String: '{s}', k={k}")
    print(f"   Reorganized: '{result}'")
    print()

    # Test 4: Kth Smallest Prime Fraction
    print("4. Kth Smallest Prime Fraction:")
    arr = [1, 2, 3, 5]
    k = 3
    fraction = solver.kth_smallest_prime_fraction(arr, k)
    print(f"   Array: {arr}, k={k}")
    print(f"   {k}th smallest fraction: {fraction[0]}/{fraction[1]}")