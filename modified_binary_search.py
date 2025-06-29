"""
Pattern 11: Modified Binary Search - 10 Hard Problems
====================================================

Modified Binary Search adapts the classic binary search algorithm for complex
scenarios beyond simple element lookup. This includes searching in rotated arrays,
finding boundaries, and searching in 2D matrices or with special conditions.

Key Concepts:
- Modify comparison logic based on problem constraints
- Handle edge cases in rotated or modified arrays
- Search for boundaries instead of exact values
- Apply to 2D matrices and other data structures

Time Complexity: Usually O(log n) or O(log m*n) for 2D
Space Complexity: O(1) for iterative implementations
"""

from typing import List, Tuple, Optional
import bisect


class ModifiedBinarySearchHard:

    def median_of_two_sorted_arrays_with_ranges(self, nums1: List[int], nums2: List[int]) -> Tuple[
        float, Tuple[int, int]]:
        """
        LeetCode 4 Extension - Median of Two Sorted Arrays with Index Ranges (Hard)

        Find median and the index ranges contributing to it.
        Extended: Return indices in both arrays that form the median.

        Algorithm:
        1. Binary search on smaller array for partition point
        2. Ensure elements on left <= elements on right
        3. Handle odd/even total length
        4. Track contributing indices

        Time: O(log min(m, n)), Space: O(1)

        Example:
        nums1 = [1,3], nums2 = [2,7]
        Output: (2.5, ((0,1), (0,0))) - median is average of nums1[1]=3 and nums2[0]=2
        """
        # Ensure nums1 is smaller
        if len(nums1) > len(nums2):
            return self.median_of_two_sorted_arrays_with_ranges(nums2, nums1)

        m, n = len(nums1), len(nums2)
        left, right = 0, m

        while left <= right:
            partition1 = (left + right) // 2
            partition2 = (m + n + 1) // 2 - partition1

            # Handle edge cases
            max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
            min_right1 = float('inf') if partition1 == m else nums1[partition1]

            max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
            min_right2 = float('inf') if partition2 == n else nums2[partition2]

            if max_left1 <= min_right2 and max_left2 <= min_right1:
                # Found correct partition
                if (m + n) % 2 == 0:
                    # Even length - median is average
                    left_max = max(max_left1, max_left2)
                    right_min = min(min_right1, min_right2)

                    # Find indices
                    left_idx1 = partition1 - 1 if max_left1 >= max_left2 else -1
                    left_idx2 = partition2 - 1 if max_left2 >= max_left1 else -1
                    right_idx1 = partition1 if min_right1 <= min_right2 else -1
                    right_idx2 = partition2 if min_right2 <= min_right1 else -1

                    return (left_max + right_min) / 2, ((left_idx1, left_idx2), (right_idx1, right_idx2))
                else:
                    # Odd length - median is max of left
                    if max_left1 > max_left2:
                        return max_left1, ((partition1 - 1, -1), (-1, -1))
                    else:
                        return max_left2, ((-1, partition2 - 1), (-1, -1))

            elif max_left1 > min_right2:
                right = partition1 - 1
            else:
                left = partition1 + 1

        return 0.0, ((-1, -1), (-1, -1))

    def search_in_rotated_sorted_array_ii_with_rotation_point(self, nums: List[int], target: int) -> Tuple[int, int]:
        """
        LeetCode 81 Extension - Search in Rotated Array II with Rotation Point (Hard)

        Search in rotated sorted array with duplicates.
        Extended: Also find the rotation point.

        Algorithm:
        1. Handle duplicates by skipping equal elements
        2. Determine which half is sorted
        3. Binary search in appropriate half
        4. Find rotation point separately

        Time: O(n) worst case (all duplicates), O(log n) average
        Space: O(1)

        Example:
        nums = [2,5,6,0,0,1,2], target = 0
        Output: (3, 3) - target at index 3, rotation at index 3
        """

        def find_target():
            left, right = 0, len(nums) - 1

            while left <= right:
                mid = (left + right) // 2

                if nums[mid] == target:
                    return mid

                # Handle duplicates
                while left < mid and nums[left] == nums[mid]:
                    left += 1
                while right > mid and nums[right] == nums[mid]:
                    right -= 1

                # Check which half is sorted
                if nums[left] <= nums[mid]:
                    # Left half is sorted
                    if nums[left] <= target < nums[mid]:
                        right = mid - 1
                    else:
                        left = mid + 1
                else:
                    # Right half is sorted
                    if nums[mid] < target <= nums[right]:
                        left = mid + 1
                    else:
                        right = mid - 1

            return -1

        def find_rotation_point():
            """Find index where rotation occurs."""
            left, right = 0, len(nums) - 1

            # Handle no rotation case
            if nums[left] < nums[right]:
                return 0

            while left < right:
                # Skip duplicates from both ends
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1

                mid = (left + right) // 2

                # Check if mid is the rotation point
                if mid > 0 and nums[mid] < nums[mid - 1]:
                    return mid
                if mid < len(nums) - 1 and nums[mid] > nums[mid + 1]:
                    return mid + 1

                # Decide which half to search
                if nums[mid] > nums[right]:
                    left = mid + 1
                else:
                    right = mid

            return left

        target_index = find_target()
        rotation_point = find_rotation_point()

        return target_index, rotation_point

    def find_peak_element_with_all_peaks(self, nums: List[int]) -> Tuple[int, List[int]]:
        """
        LeetCode 162 Extension - Find All Peak Elements (Hard)

        Find a peak element and all peak positions.
        A peak is greater than its neighbors.

        Algorithm:
        1. Binary search for one peak
        2. Recursively search both sides for more peaks
        3. Handle edge cases

        Time: O(n) for all peaks, O(log n) for one peak
        Space: O(log n) for recursion

        Example:
        nums = [1,2,1,3,5,6,4]
        Output: (5, [1, 5]) - one peak at index 5, all peaks at indices 1 and 5
        """

        def find_one_peak(left: int, right: int) -> int:
            """Find any peak in range [left, right]."""
            if left == right:
                return left

            mid = (left + right) // 2

            if nums[mid] > nums[mid + 1]:
                return find_one_peak(left, mid)
            else:
                return find_one_peak(mid + 1, right)

        def find_all_peaks_in_range(left: int, right: int, peaks: List[int]):
            """Find all peaks in range [left, right]."""
            if left > right:
                return

            if left == right:
                # Check if it's a peak
                is_peak = True
                if left > 0 and nums[left] <= nums[left - 1]:
                    is_peak = False
                if left < len(nums) - 1 and nums[left] <= nums[left + 1]:
                    is_peak = False
                if is_peak:
                    peaks.append(left)
                return

            # Find one peak
            peak = find_one_peak(left, right)
            peaks.append(peak)

            # Search both sides for more peaks
            if peak > left:
                find_all_peaks_in_range(left, peak - 1, peaks)
            if peak < right:
                find_all_peaks_in_range(peak + 1, right, peaks)

        # Find one peak using binary search
        one_peak = find_one_peak(0, len(nums) - 1)

        # Find all peaks
        all_peaks = []
        find_all_peaks_in_range(0, len(nums) - 1, all_peaks)
        all_peaks.sort()

        return one_peak, all_peaks

    def kth_smallest_element_in_sorted_matrix_with_count(self, matrix: List[List[int]], k: int) -> Tuple[int, int]:
        """
        LeetCode 378 Extension - Kth Smallest in Sorted Matrix with Count (Hard)

        Find kth smallest element in row-wise and column-wise sorted matrix.
        Extended: Count elements less than or equal to result.

        Algorithm:
        1. Binary search on value range
        2. Count elements <= mid value
        3. Adjust search range based on count

        Time: O(n * log(max-min)), Space: O(1)

        Example:
        matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
        Output: (13, 8) - 8th smallest is 13, with 8 elements <= 13
        """
        n = len(matrix)

        def count_less_equal(target: int) -> int:
            """Count elements <= target using staircase search."""
            count = 0
            row = n - 1
            col = 0

            while row >= 0 and col < n:
                if matrix[row][col] <= target:
                    count += row + 1
                    col += 1
                else:
                    row -= 1

            return count

        left, right = matrix[0][0], matrix[n - 1][n - 1]

        while left < right:
            mid = (left + right) // 2
            count = count_less_equal(mid)

            if count < k:
                left = mid + 1
            else:
                right = mid

        # Count exact number of elements <= result
        final_count = count_less_equal(left)

        return left, final_count

    def smallest_rectangle_enclosing_black_pixels(self, image: List[List[str]], x: int, y: int) -> int:
        """
        LeetCode 302 - Smallest Rectangle Enclosing Black Pixels (Hard)

        Find smallest rectangle area enclosing all black pixels.
        Given one black pixel position, use binary search.

        Algorithm:
        1. Binary search for left, right, top, bottom boundaries
        2. Check if row/column contains black pixel
        3. Calculate rectangle area

        Time: O(m log n + n log m), Space: O(1)

        Example:
        image = [["0","0","1","0"],
                 ["0","1","1","0"],
                 ["0","1","0","0"]], x=0, y=2
        Output: 6 (rectangle from (0,1) to (2,2))
        """
        if not image or not image[0]:
            return 0

        m, n = len(image), len(image[0])

        def search_columns(i: int, j: int, opt: bool) -> int:
            """Binary search for leftmost/rightmost black column."""
            while i != j:
                mid = (i + j) // 2
                if any(image[k][mid] == '1' for k in range(m)) == opt:
                    j = mid
                else:
                    i = mid + 1
            return i

        def search_rows(i: int, j: int, opt: bool) -> int:
            """Binary search for topmost/bottommost black row."""
            while i != j:
                mid = (i + j) // 2
                if any(image[mid][k] == '1' for k in range(n)) == opt:
                    j = mid
                else:
                    i = mid + 1
            return i

        # Find boundaries
        left = search_columns(0, y, True)
        right = search_columns(y + 1, n, False)
        top = search_rows(0, x, True)
        bottom = search_rows(x + 1, m, False)

        return (right - left) * (bottom - top)

    def split_array_largest_sum_with_splits(self, nums: List[int], m: int) -> Tuple[int, List[List[int]]]:
        """
        LeetCode 410 Extension - Split Array Largest Sum with Split Points (Hard)

        Split array into m subarrays to minimize largest sum.
        Extended: Return the actual splits.

        Algorithm:
        1. Binary search on maximum subarray sum
        2. Check if can split with given max sum
        3. Reconstruct splits for optimal sum

        Time: O(n * log(sum)), Space: O(m)

        Example:
        nums = [7,2,5,10,8], m = 2
        Output: (18, [[7,2,5], [10,8]])
        """

        def can_split(max_sum: int) -> bool:
            """Check if can split into m parts with max sum."""
            count = 1
            current_sum = 0

            for num in nums:
                if current_sum + num > max_sum:
                    count += 1
                    current_sum = num
                else:
                    current_sum += num

            return count <= m

        def get_splits(max_sum: int) -> List[List[int]]:
            """Get actual splits for given max sum."""
            splits = []
            current_split = []
            current_sum = 0

            for num in nums:
                if current_sum + num > max_sum:
                    splits.append(current_split)
                    current_split = [num]
                    current_sum = num
                else:
                    current_split.append(num)
                    current_sum += num

            if current_split:
                splits.append(current_split)

            # Merge splits if we have too many
            while len(splits) > m:
                # Find best merge point
                min_increase = float('inf')
                merge_idx = 0

                for i in range(len(splits) - 1):
                    increase = sum(splits[i]) + sum(splits[i + 1]) - max(sum(splits[i]), sum(splits[i + 1]))
                    if increase < min_increase:
                        min_increase = increase
                        merge_idx = i

                splits[merge_idx].extend(splits[merge_idx + 1])
                splits.pop(merge_idx + 1)

            return splits

        # Binary search on maximum sum
        left = max(nums)
        right = sum(nums)

        while left < right:
            mid = (left + right) // 2
            if can_split(mid):
                right = mid
            else:
                left = mid + 1

        return left, get_splits(left)

    def find_minimum_in_rotated_sorted_array_ii_with_duplicates(self, nums: List[int]) -> Tuple[int, int, int]:
        """
        LeetCode 154 Extension - Find Min in Rotated Array II with Analysis (Hard)

        Find minimum in rotated sorted array with duplicates.
        Extended: Return min value, index, and rotation count.

        Algorithm:
        1. Handle duplicates by comparing with both ends
        2. Binary search with duplicate handling
        3. Count rotations from original position

        Time: O(n) worst case, O(log n) average
        Space: O(1)

        Example:
        nums = [2,2,2,0,1]
        Output: (0, 3, 3) - min value 0 at index 3, rotated 3 positions
        """
        n = len(nums)
        left, right = 0, n - 1

        # Handle duplicates at boundaries
        while left < right and nums[left] == nums[right]:
            if nums[left] > nums[left + 1]:
                return nums[left + 1], left + 1, left + 1
            left += 1

        # Regular binary search
        min_idx = left
        original_left = left

        while left < right:
            mid = (left + right) // 2

            # Check if mid is minimum
            if mid > 0 and nums[mid] < nums[mid - 1]:
                min_idx = mid
                break

            # Handle duplicates
            if nums[mid] == nums[right]:
                right -= 1
            elif nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid

        if left == right:
            min_idx = left

        # Calculate rotation count
        rotation_count = (min_idx - original_left) % n

        return nums[min_idx], min_idx, rotation_count

    def search_2d_matrix_ii_with_path(self, matrix: List[List[int]], target: int) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        LeetCode 240 Extension - Search 2D Matrix II with Path (Hard)

        Search in matrix where rows and columns are sorted.
        Extended: Return search path taken.

        Algorithm:
        1. Start from top-right or bottom-left
        2. Eliminate row or column at each step
        3. Track path taken during search

        Time: O(m + n), Space: O(m + n) for path

        Example:
        matrix = [[1,4,7,11],[2,5,8,12],[3,6,9,16]], target = 5
        Output: (True, [(0,3), (0,2), (0,1), (1,1)])
        """
        if not matrix or not matrix[0]:
            return False, []

        m, n = len(matrix), len(matrix[0])
        row, col = 0, n - 1
        path = []

        while row < m and col >= 0:
            path.append((row, col))

            if matrix[row][col] == target:
                return True, path
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1

        return False, path

    def maximum_average_subarray_ii(self, nums: List[int], k: int) -> float:
        """
        LeetCode 644 - Maximum Average Subarray II (Hard)

        Find maximum average of subarray with length >= k.

        Algorithm:
        1. Binary search on average value
        2. Check if subarray with average >= mid exists
        3. Use prefix sum technique with offset

        Time: O(n * log((max-min)/ε)), Space: O(n)

        Example:
        nums = [1,12,-5,-6,50,3], k = 4
        Output: 12.75 (subarray [12,-5,-6,50])
        """

        def has_average_greater_than(avg: float) -> bool:
            """Check if subarray with average >= avg exists."""
            # Transform problem: subtract avg from each element
            # Now need subarray with sum >= 0
            prefix = 0
            min_prefix = 0
            prev_min = 0

            for i in range(len(nums)):
                prefix += nums[i] - avg

                if i >= k - 1:
                    # Can form subarray of length >= k
                    if prefix - min_prefix >= 0:
                        return True

                    # Update minimum prefix ending at least k positions back
                    prev_min += nums[i - k + 1] - avg
                    min_prefix = min(min_prefix, prev_min)

            return False

        # Binary search on average
        left = min(nums)
        right = max(nums)

        # Binary search with precision
        while right - left > 1e-5:
            mid = (left + right) / 2
            if has_average_greater_than(mid):
                left = mid
            else:
                right = mid

        return left

    def count_of_range_sum_with_indices(self, nums: List[int], lower: int, upper: int) -> Tuple[
        int, List[Tuple[int, int]]]:
        """
        LeetCode 327 Extension - Count of Range Sum with Sample Ranges (Hard)

        Count subarrays with sum in [lower, upper].
        Extended: Return sample valid ranges (limited for efficiency).

        Algorithm:
        1. Use merge sort with modification
        2. Count valid ranges during merge
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
                # Find range [j, k) where sum is in [lower, upper]
                while j <= hi and prefix[j] - prefix[i] < lower:
                    j += 1
                while k <= hi and prefix[k] - prefix[i] <= upper:
                    k += 1

                count += k - j

                # Track sample ranges (limit to prevent memory issues)
                if len(sample_ranges) < 100:
                    for idx in range(j, k):
                        if idx <= hi:
                            sample_ranges.append((i, idx - 1))

            # Merge
            temp = []
            i = lo
            j = mid + 1

            while i <= mid and j <= hi:
                if prefix[i] < prefix[j]:
                    temp.append(prefix[i])
                    i += 1
                else:
                    temp.append(prefix[j])
                    j += 1

            while i <= mid:
                temp.append(prefix[i])
                i += 1
            while j <= hi:
                temp.append(prefix[j])
                j += 1

            for i in range(len(temp)):
                prefix[lo + i] = temp[i]

            return count

        # Build prefix sum array
        n = len(nums)
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)

        sample_ranges = []
        count = merge_sort_count(0, n)

        # Convert to actual ranges (not prefix indices)
        actual_ranges = [(i, j) for i, j in sample_ranges[:10]]  # Limit output

        return count, actual_ranges


# Example usage and testing
if __name__ == "__main__":
    solver = ModifiedBinarySearchHard()

    # Test 1: Median of Two Sorted Arrays
    print("1. Median of Two Sorted Arrays with Ranges:")
    nums1 = [1, 3]
    nums2 = [2, 7]
    median, indices = solver.median_of_two_sorted_arrays_with_ranges(nums1, nums2)
    print(f"   nums1={nums1}, nums2={nums2}")
    print(f"   Median: {median}, Indices: {indices}")
    print()

    # Test 2: Search in Rotated Array II
    print("2. Search in Rotated Sorted Array II:")
    nums = [2, 5, 6, 0, 0, 1, 2]
    target = 0
    idx, rotation = solver.search_in_rotated_sorted_array_ii_with_rotation_point(nums, target)
    print(f"   nums={nums}, target={target}")
    print(f"   Found at: {idx}, Rotation point: {rotation}")
    print()

    # Test 3: Split Array Largest Sum
    print("3. Split Array Largest Sum:")
    nums = [7, 2, 5, 10, 8]
    m = 2
    min_sum, splits = solver.split_array_largest_sum_with_splits(nums, m)
    print(f"   nums={nums}, m={m}")
    print(f"   Min largest sum: {min_sum}, Splits: {splits}")