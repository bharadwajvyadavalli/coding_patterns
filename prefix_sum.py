"""
Pattern 20: Prefix Sums - 10 Hard Problems
==========================================

The Prefix Sums pattern precomputes cumulative sums to efficiently calculate
range sums or find subarrays with specific properties. This pattern is essential
for range query problems and subarray sum problems.

Key Concepts:
- Prefix sum array where prefix[i] = sum of elements from 0 to i-1
- Range sum from i to j = prefix[j+1] - prefix[i]
- Often combined with hash maps for subarray problems
- Can be extended to 2D arrays and other operations

Time Complexity: O(1) for queries after O(n) preprocessing
Space Complexity: O(n) for storing prefix sums
"""

from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import bisect


class PrefixSumsHard:

    def subarray_sum_equals_k_with_all_subarrays(self, nums: List[int], k: int) -> Tuple[int, List[Tuple[int, int]]]:
        """
        LeetCode 560 Extension - Subarray Sum Equals K with All Subarrays (Hard)

        Count and find all subarrays with sum equal to k.
        Handle negative numbers and zeros.

        Algorithm:
        1. Use prefix sum with hash map
        2. Track all subarrays with sum k
        3. Handle duplicates and overlapping subarrays

        Time: O(n), Space: O(n)

        Example:
        nums = [1,2,1,2,1], k = 3
        Output: (4, [(0,1), (1,2), (2,4), (3,4)])
        """
        count = 0
        subarrays = []
        prefix_sum = 0
        # Map from prefix sum to list of indices
        sum_indices = defaultdict(list)
        sum_indices[0].append(-1)

        for i in range(len(nums)):
            prefix_sum += nums[i]

            # Check if there's a prefix sum such that current - prefix = k
            target = prefix_sum - k
            if target in sum_indices:
                count += len(sum_indices[target])
                # Add all valid subarrays
                for j in sum_indices[target]:
                    subarrays.append((j + 1, i))

            sum_indices[prefix_sum].append(i)

        return count, subarrays[:20]  # Limit output

    def maximum_size_subarray_sum_equals_k_with_constraints(self, nums: List[int], k: int, min_size: int) -> Tuple[
        int, Tuple[int, int]]:
        """
        LeetCode 325 Extension - Maximum Size Subarray Sum Equals k with Constraints (Hard)

        Find maximum length subarray with sum k and length >= min_size.

        Algorithm:
        1. Use prefix sum with earliest occurrence tracking
        2. Ensure minimum size constraint
        3. Track the actual subarray

        Time: O(n), Space: O(n)

        Example:
        nums = [1,-1,5,-2,3], k = 3, min_size = 2
        Output: (4, (1,4)) - subarray [-1,5,-2,3]
        """
        max_length = 0
        best_subarray = (-1, -1)
        prefix_sum = 0
        # Map from sum to earliest index
        sum_index = {0: -1}

        for i in range(len(nums)):
            prefix_sum += nums[i]

            # Check if we can form subarray with sum k
            target = prefix_sum - k
            if target in sum_index:
                length = i - sum_index[target]
                if length >= min_size and length > max_length:
                    max_length = length
                    best_subarray = (sum_index[target] + 1, i)

            # Store earliest occurrence
            if prefix_sum not in sum_index:
                sum_index[prefix_sum] = i

        return max_length, best_subarray

    def continuous_subarray_sum_multiple_of_k(self, nums: List[int], k: int) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        LeetCode 523 Extension - Continuous Subarray Sum with All Valid Subarrays (Hard)

        Find all subarrays of size >= 2 with sum multiple of k.

        Algorithm:
        1. Use modulo with prefix sums
        2. Track all valid subarrays
        3. Handle k = 0 case

        Time: O(n), Space: O(n)

        Example:
        nums = [23,2,4,6,7], k = 6
        Output: (True, [(1,3), (0,4)])
        """
        if len(nums) < 2:
            return False, []

        valid_subarrays = []
        prefix_sum = 0
        # Map from remainder to list of indices
        remainder_indices = defaultdict(list)
        remainder_indices[0].append(-1)

        for i in range(len(nums)):
            prefix_sum += nums[i]

            if k != 0:
                remainder = prefix_sum % k
            else:
                remainder = prefix_sum

            # Check if we've seen this remainder before
            if remainder in remainder_indices:
                for j in remainder_indices[remainder]:
                    if i - j >= 2:  # Size >= 2
                        valid_subarrays.append((j + 1, i))

            # Store current index for this remainder
            # For size constraint, we store at the current position
            if i > 0:  # Can only form size >= 2 from next position
                if k != 0:
                    prev_remainder = (prefix_sum - nums[i]) % k
                else:
                    prev_remainder = prefix_sum - nums[i]
                remainder_indices[prev_remainder].append(i - 1)

        return len(valid_subarrays) > 0, valid_subarrays

    def maximum_subarray_sum_with_one_deletion(self, nums: List[int]) -> Tuple[int, Tuple[int, int, int]]:
        """
        LeetCode 1186 Extension - Maximum Subarray Sum with One Deletion Details (Hard)

        Find maximum sum of subarray with at most one element deletion.
        Extended: Return the subarray bounds and deleted index (if any).

        Algorithm:
        1. Calculate max subarray ending at each position
        2. Calculate max subarray starting at each position
        3. Try deleting each element

        Time: O(n), Space: O(n)

        Example:
        nums = [1,-2,0,3]
        Output: (4, (0, 3, 1)) - delete index 1
        """
        n = len(nums)
        if n == 1:
            return nums[0], (0, 0, -1)

        # fw[i] = maximum subarray sum ending at i
        fw = [float('-inf')] * n
        fw[0] = nums[0]

        # bw[i] = maximum subarray sum starting at i
        bw = [float('-inf')] * n
        bw[n - 1] = nums[n - 1]

        # Forward pass
        for i in range(1, n):
            fw[i] = max(nums[i], fw[i - 1] + nums[i])

        # Backward pass
        for i in range(n - 2, -1, -1):
            bw[i] = max(nums[i], bw[i + 1] + nums[i])

        # Case 1: No deletion
        max_sum = max(fw)
        no_del_idx = fw.index(max_sum)

        # Find start of subarray for no deletion case
        start = no_del_idx
        temp_sum = nums[no_del_idx]
        while start > 0 and temp_sum < fw[no_del_idx]:
            start -= 1
            temp_sum += nums[start]

        result = (max_sum, (start, no_del_idx, -1))

        # Case 2: Delete one element
        for i in range(1, n - 1):
            # Delete element at i, connect fw[i-1] and bw[i+1]
            sum_with_deletion = fw[i - 1] + bw[i + 1]
            if sum_with_deletion > max_sum:
                max_sum = sum_with_deletion

                # Find actual bounds
                # Find start of fw[i-1]
                start = i - 1
                temp_sum = nums[i - 1]
                while start > 0 and temp_sum < fw[i - 1]:
                    start -= 1
                    temp_sum += nums[start]

                # Find end of bw[i+1]
                end = i + 1
                temp_sum = nums[i + 1]
                while end < n - 1 and temp_sum < bw[i + 1]:
                    end += 1
                    temp_sum += nums[end]

                result = (max_sum, (start, end, i))

        return result

    def make_sum_divisible_by_p(self, nums: List[int], p: int) -> Tuple[int, Tuple[int, int]]:
        """
        LeetCode 1590 Extension - Make Sum Divisible by P with Subarray (Hard)

        Remove shortest subarray to make remaining sum divisible by p.
        Extended: Return the actual subarray to remove.

        Algorithm:
        1. Calculate total sum modulo p
        2. Find shortest subarray with sum ≡ total_sum (mod p)
        3. Use prefix sum with modulo

        Time: O(n), Space: O(n)

        Example:
        nums = [6,3,5,2], p = 9
        Output: (2, (2,3)) - remove subarray [5,2]
        """
        n = len(nums)
        total_sum = sum(nums)
        remainder = total_sum % p

        if remainder == 0:
            return 0, (-1, -1)

        min_length = n
        best_subarray = (-1, -1)
        prefix_sum = 0
        # Map from remainder to most recent index
        remainder_index = {0: -1}

        for i in range(n):
            prefix_sum = (prefix_sum + nums[i]) % p

            # We want subarray sum ≡ remainder (mod p)
            # So prefix[j] - prefix[i] ≡ remainder (mod p)
            # prefix[i] ≡ prefix[j] - remainder (mod p)
            target = (prefix_sum - remainder + p) % p

            if target in remainder_index:
                length = i - remainder_index[target]
                if length < min_length:
                    min_length = length
                    best_subarray = (remainder_index[target] + 1, i)

            remainder_index[prefix_sum] = i

        if min_length == n:
            return -1, (-1, -1)

        return min_length, best_subarray

    def count_range_sum_with_ranges(self, nums: List[int], lower: int, upper: int) -> Tuple[int, List[Tuple[int, int]]]:
        """
        LeetCode 327 Extension - Count of Range Sum with Sample Ranges (Hard)

        Count subarrays with sum in [lower, upper].
        Extended: Return sample valid ranges.

        Algorithm:
        1. Use merge sort with counting
        2. During merge, count valid ranges
        3. Track sample ranges

        Time: O(n log n), Space: O(n)

        Example:
        nums = [-2,5,-1], lower = -2, upper = 2
        Output: (3, [(0,0), (2,2), (0,2)])
        """

        def merge_sort_count(lo: int, hi: int) -> int:
            if lo >= hi:
                return 0

            mid = (lo + hi) // 2
            count = merge_sort_count(lo, mid) + merge_sort_count(mid + 1, hi)

            # Count ranges crossing mid
            j = k = mid + 1
            for i in range(lo, mid + 1):
                while j <= hi and prefix[j] - prefix[i] < lower:
                    j += 1
                while k <= hi and prefix[k] - prefix[i] <= upper:
                    k += 1

                count += k - j

                # Track sample ranges
                if len(sample_ranges) < 50:
                    for idx in range(j, k):
                        if idx <= hi:
                            sample_ranges.append((i, idx - 1))

            # Merge
            prefix[lo:hi + 1] = sorted(prefix[lo:hi + 1])

            return count

        # Build prefix sum
        n = len(nums)
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)

        sample_ranges = []
        count = merge_sort_count(0, n)

        # Convert to actual indices (not prefix indices)
        actual_ranges = []
        for i, j in sample_ranges[:10]:  # Limit output
            if i < len(nums) and j < len(nums):
                actual_ranges.append((i, j))

        return count, actual_ranges

    def subarray_sums_divisible_by_k(self, nums: List[int], k: int) -> Tuple[int, Dict[int, List[Tuple[int, int]]]]:
        """
        LeetCode 974 Extension - Subarray Sums Divisible by K with Grouping (Hard)

        Count subarrays with sum divisible by k.
        Extended: Group subarrays by their sum modulo k.

        Algorithm:
        1. Use prefix sum with modulo
        2. Count pairs with same remainder
        3. Group by actual sum values

        Time: O(n), Space: O(n)

        Example:
        nums = [4,5,0,-2,-3,1], k = 5
        Output: (7, {0: [(0,2), (1,4), ...]})
        """
        count = 0
        grouped_subarrays = defaultdict(list)
        prefix_sum = 0
        # Map from remainder to list of indices
        remainder_indices = defaultdict(list)
        remainder_indices[0].append(-1)

        for i in range(len(nums)):
            prefix_sum += nums[i]
            remainder = ((prefix_sum % k) + k) % k  # Handle negative numbers

            # Count subarrays ending at i
            count += len(remainder_indices[remainder])

            # Add subarrays to groups
            for j in remainder_indices[remainder]:
                start = j + 1
                # Calculate actual sum
                subarray_sum = prefix_sum - (0 if j == -1 else sum(nums[:start]))
                sum_mod_k = ((subarray_sum % k) + k) % k

                if len(grouped_subarrays[sum_mod_k]) < 10:  # Limit per group
                    grouped_subarrays[sum_mod_k].append((start, i))

            remainder_indices[remainder].append(i)

        return count, dict(grouped_subarrays)

    def max_subarray_sum_circular_with_details(self, nums: List[int]) -> Tuple[int, Tuple[int, int], str]:
        """
        LeetCode 918 Extension - Maximum Sum Circular Subarray with Details (Hard)

        Find maximum sum subarray in circular array.
        Extended: Return subarray bounds and type (normal/circular).

        Algorithm:
        1. Case 1: Max subarray doesn't wrap (Kadane's)
        2. Case 2: Max subarray wraps (total - min subarray)
        3. Handle all negative case

        Time: O(n), Space: O(1)

        Example:
        nums = [5,-3,5]
        Output: (10, (0,2), "circular") - wraps around
        """
        n = len(nums)

        # Case 1: Maximum subarray (no wrap)
        max_kadane = float('-inf')
        max_start = max_end = 0
        current_max = 0
        temp_start = 0

        for i in range(n):
            current_max += nums[i]

            if current_max > max_kadane:
                max_kadane = current_max
                max_start = temp_start
                max_end = i

            if current_max < 0:
                current_max = 0
                temp_start = i + 1

        # Case 2: Maximum subarray (with wrap)
        # Find minimum subarray
        min_kadane = float('inf')
        min_start = min_end = 0
        current_min = 0
        temp_start = 0

        for i in range(n):
            current_min += nums[i]

            if current_min < min_kadane:
                min_kadane = current_min
                min_start = temp_start
                min_end = i

            if current_min > 0:
                current_min = 0
                temp_start = i + 1

        total_sum = sum(nums)

        # Handle all negative case
        if total_sum == min_kadane:
            return max_kadane, (max_start, max_end), "normal"

        # Compare cases
        if max_kadane >= total_sum - min_kadane:
            return max_kadane, (max_start, max_end), "normal"
        else:
            # Circular case: from min_end+1 to min_start-1
            if min_end + 1 <= min_start - 1:
                return total_sum - min_kadane, (min_end + 1, min_start - 1), "circular"
            else:
                # Wraps around
                return total_sum - min_kadane, (0, n - 1), "circular (full wrap)"

    def minimum_operations_to_reduce_x_to_zero(self, nums: List[int], x: int) -> Tuple[int, List[str]]:
        """
        LeetCode 1658 Extension - Minimum Operations with Strategy (Hard)

        Remove elements from left/right to make sum equal x.
        Extended: Return the removal strategy.

        Algorithm:
        1. Convert to finding longest subarray with sum = total - x
        2. Use sliding window
        3. Track operations

        Time: O(n), Space: O(1)

        Example:
        nums = [3,2,20,1,1,3], x = 10
        Output: (5, ["left", "left", "right", "right", "right"])
        """
        total = sum(nums)
        target = total - x

        if target < 0:
            return -1, []

        if target == 0:
            return len(nums), ["left"] * len(nums)

        # Find longest subarray with sum = target
        left = 0
        current_sum = 0
        max_length = -1
        best_window = (-1, -1)

        for right in range(len(nums)):
            current_sum += nums[right]

            while current_sum > target and left <= right:
                current_sum -= nums[left]
                left += 1

            if current_sum == target:
                if right - left + 1 > max_length:
                    max_length = right - left + 1
                    best_window = (left, right)

        if max_length == -1:
            return -1, []

        # Build operations
        operations = []
        left_ops = best_window[0]
        right_ops = len(nums) - best_window[1] - 1

        operations.extend(["left"] * left_ops)
        operations.extend(["right"] * right_ops)

        return left_ops + right_ops, operations

    def find_pivot_index_all(self, nums: List[int]) -> List[Tuple[int, int, int]]:
        """
        LeetCode 724 Extension - Find All Pivot Indices with Sums (Hard)

        Find all indices where left sum equals right sum.
        Extended: Return (index, left_sum, right_sum) for each pivot.

        Algorithm:
        1. Calculate total sum
        2. Iterate and check balance at each position
        3. Track all valid pivots

        Time: O(n), Space: O(1)

        Example:
        nums = [1,7,3,6,5,6]
        Output: [(3, 11, 11)]
        """
        total = sum(nums)
        pivots = []
        left_sum = 0

        for i in range(len(nums)):
            right_sum = total - left_sum - nums[i]

            if left_sum == right_sum:
                pivots.append((i, left_sum, right_sum))

            left_sum += nums[i]

        return pivots


# Example usage and testing
if __name__ == "__main__":
    solver = PrefixSumsHard()

    # Test 1: Subarray Sum Equals K
    print("1. Subarray Sum Equals K with All Subarrays:")
    nums = [1, 2, 1, 2, 1]
    k = 3
    count, subarrays = solver.subarray_sum_equals_k_with_all_subarrays(nums, k)
    print(f"   Array: {nums}, k={k}")
    print(f"   Count: {count}, Sample subarrays: {subarrays[:5]}")
    print()

    # Test 2: Continuous Subarray Sum
    print("2. Continuous Subarray Sum Multiple of K:")
    nums = [23, 2, 4, 6, 7]
    k = 6
    has_valid, subarrays = solver.continuous_subarray_sum_multiple_of_k(nums, k)
    print(f"   Array: {nums}, k={k}")
    print(f"   Has valid subarray: {has_valid}")
    print(f"   Valid subarrays: {subarrays}")
    print()

    # Test 3: Maximum Subarray with One Deletion
    print("3. Maximum Subarray Sum with One Deletion:")
    nums = [1, -2, 0, 3]
    max_sum, details = solver.maximum_subarray_sum_with_one_deletion(nums)
    print(f"   Array: {nums}")
    print(f"   Max sum: {max_sum}")
    print(f"   Details (start, end, deleted): {details}")
    print()

    # Test 4: Circular Maximum Subarray
    print("4. Maximum Sum Circular Subarray:")
    nums = [5, -3, 5]
    max_sum, bounds, type_str = solver.max_subarray_sum_circular_with_details(nums)
    print(f"   Array: {nums}")
    print(f"   Max sum: {max_sum}, Bounds: {bounds}, Type: {type_str}")