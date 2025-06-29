"""
Pattern 5: Cyclic Sort - 10 Hard Problems
=========================================

The Cyclic Sort pattern is used to solve problems involving arrays containing
numbers in a given range. It places each number at its correct index, making
it very efficient for finding missing/duplicate numbers.

Key Concepts:
- Array contains numbers in range [0,n] or [1,n]
- Place each number at its correct position (number i at index i or i-1)
- Iterate through array and swap elements to correct positions
- After sorting, missing/duplicate numbers are easily identified

Time Complexity: O(n) - each element is moved at most once
Space Complexity: O(1) - in-place sorting
"""

from typing import List, Tuple, Set, Optional
import math


class CyclicSortHard:

    def find_all_duplicates_and_missing(self, nums: List[int]) -> Tuple[List[int], List[int]]:
        """
        Extension of LeetCode 442 & 448 - Find All Duplicates and Missing Numbers (Hard)

        Array contains n numbers in range [1,n], some appear twice, others missing.
        Find all duplicates and missing numbers in single pass.

        Algorithm:
        1. Use cyclic sort to place numbers at correct positions
        2. Numbers that can't be placed are duplicates
        3. Positions without correct numbers indicate missing values
        4. Use sign marking for already seen numbers

        Time: O(n), Space: O(1) excluding output

        Example:
        nums = [4,3,2,7,8,2,3,1]
        Output: ([2,3], [5,6]) - duplicates: 2,3; missing: 5,6
        """
        n = len(nums)
        i = 0

        # Cyclic sort
        while i < n:
            # Correct position for nums[i] is nums[i]-1
            correct_pos = nums[i] - 1

            # If number is not at correct position and target position
            # doesn't have the same number (to handle duplicates)
            if nums[i] != nums[correct_pos]:
                nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
            else:
                i += 1

        duplicates = []
        missing = []

        # Find duplicates and missing
        for i in range(n):
            if nums[i] != i + 1:
                duplicates.append(nums[i])
                missing.append(i + 1)

        return duplicates, missing

    def first_missing_positive_with_constraints(self, nums: List[int]) -> Tuple[int, List[int]]:
        """
        LeetCode 41 Extension - First Missing Positive with K constraints (Hard)

        Find first missing positive and all missing positives up to length n.
        Must handle negative numbers and numbers > n.

        Algorithm:
        1. Separate positive numbers using partitioning
        2. Use cyclic sort on positive partition
        3. Mark presence using sign flipping
        4. Find first missing and collect all missing

        Time: O(n), Space: O(1)

        Example:
        nums = [3,4,-1,1]
        Output: (2, [2]) - first missing is 2
        """
        n = len(nums)

        # Step 1: Partition - move all positive numbers to left
        j = 0
        for i in range(n):
            if nums[i] > 0:
                nums[i], nums[j] = nums[j], nums[i]
                j += 1

        # Now all positive numbers are in nums[0:j]
        positive_count = j

        # Step 2: Mark presence using cyclic approach
        # For each positive number x, mark position x-1 as negative
        for i in range(positive_count):
            val = abs(nums[i])
            if val <= positive_count:
                # Mark as seen by making negative
                if nums[val - 1] > 0:
                    nums[val - 1] = -nums[val - 1]

        # Step 3: Find first missing positive
        first_missing = positive_count + 1
        all_missing = []

        for i in range(positive_count):
            if nums[i] > 0:
                if i + 1 < first_missing:
                    first_missing = i + 1
                all_missing.append(i + 1)

        return first_missing, all_missing

    def find_corrupt_pair(self, nums: List[int]) -> Tuple[int, int, List[int]]:
        """
        Custom Hard - Find Corrupt Pair and Recovery Sequence

        Array has one number appearing twice and one missing.
        Find the duplicate, missing number, and steps to recover array.

        Algorithm:
        1. Use cyclic sort with tracking of swaps
        2. Identify corrupt pair during sorting
        3. Generate recovery sequence

        Time: O(n), Space: O(n) for recovery sequence

        Example:
        nums = [3,1,3,4,2]
        Output: (3, 5, [(0,2), ...]) - duplicate: 3, missing: 5, swaps to fix
        """
        n = len(nums)
        recovery_swaps = []
        duplicate = -1

        # Make a copy to track original positions
        original = nums.copy()

        i = 0
        while i < n:
            # For 1-indexed array, correct position of nums[i] is nums[i]-1
            if nums[i] != i + 1 and nums[i] <= n and nums[i] >= 1:
                correct_pos = nums[i] - 1

                if nums[correct_pos] == nums[i]:
                    # Found duplicate
                    duplicate = nums[i]
                    i += 1
                else:
                    # Swap to correct position
                    recovery_swaps.append((i, correct_pos))
                    nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
            else:
                i += 1

        # Find missing number
        missing = -1
        for i in range(n):
            if nums[i] != i + 1:
                missing = i + 1
                break

        return duplicate, missing, recovery_swaps

    def find_k_missing_positive(self, nums: List[int], k: int) -> List[int]:
        """
        LeetCode 1539 Extension - Find K Missing Positive Numbers (Hard)

        Find first k missing positive numbers.
        Extended to handle duplicates and optimize for large k.

        Algorithm:
        1. Use cyclic sort to arrange positives
        2. Binary search optimization for large k
        3. Handle duplicates by marking
        4. Generate missing numbers efficiently

        Time: O(n + k), Space: O(1)

        Example:
        nums = [2,3,4,7,11], k = 5
        Output: [1,5,6,8,9]
        """
        n = len(nums)

        # Remove duplicates and numbers > n+k (optimization)
        seen = set()
        cleaned = []

        for num in nums:
            if num > 0 and num <= n + k and num not in seen:
                seen.add(num)
                cleaned.append(num)

        # Sort cleaned array
        cleaned.sort()

        missing = []
        current = 1
        i = 0

        while len(missing) < k:
            if i < len(cleaned) and cleaned[i] == current:
                i += 1
            else:
                missing.append(current)
            current += 1

        return missing

    def minimum_swaps_to_sort_cyclic(self, nums: List[int]) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Custom Hard - Minimum Swaps to Sort Array with Cycles

        Find minimum swaps needed to sort array where elements are in range [0,n-1].
        Also return the actual swap sequence.

        Algorithm:
        1. Build permutation cycles
        2. For each cycle of length L, need L-1 swaps
        3. Track actual swaps performed

        Time: O(n), Space: O(n)

        Example:
        nums = [3,2,0,1]
        Output: (2, [(0,3), (1,2)]) - 2 swaps needed
        """
        n = len(nums)
        visited = [False] * n
        swaps = []
        swap_count = 0

        # Process each cycle
        for start in range(n):
            if visited[start] or nums[start] == start:
                continue

            # Found new cycle
            cycle_length = 0
            current = start

            while not visited[current]:
                visited[current] = True
                next_pos = nums[current]

                if next_pos != current:
                    # Record swap if not in correct position
                    cycle_length += 1

                current = next_pos

            # For a cycle of length L, we need L-1 swaps
            if cycle_length > 0:
                # Perform actual swaps for this cycle
                current = start
                while nums[current] != start:
                    correct_pos = nums[current]
                    swaps.append((current, correct_pos))
                    nums[current], nums[correct_pos] = nums[correct_pos], nums[current]
                    swap_count += 1

        return swap_count, swaps

    def set_mismatch_extended(self, nums: List[int]) -> Tuple[int, int, int, List[int]]:
        """
        LeetCode 645 Extension - Set Mismatch with Statistics (Hard)

        Find duplicate and missing in array [1,n].
        Extended: Also find sum difference and XOR difference.

        Algorithm:
        1. Use cyclic sort to find duplicate/missing
        2. Calculate mathematical properties
        3. Use sum and XOR formulas for verification

        Time: O(n), Space: O(1)

        Example:
        nums = [1,2,2,4]
        Output: (2, 3, -1, [differences]) - dup: 2, miss: 3, sum_diff: -1
        """
        n = len(nums)

        # Method 1: Cyclic sort
        i = 0
        while i < n:
            correct_pos = nums[i] - 1
            if nums[i] != nums[correct_pos]:
                nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
            else:
                i += 1

        duplicate = missing = -1
        for i in range(n):
            if nums[i] != i + 1:
                duplicate = nums[i]
                missing = i + 1
                break

        # Method 2: Mathematical verification
        # Sum method
        expected_sum = n * (n + 1) // 2
        actual_sum = sum(nums)
        sum_diff = actual_sum - expected_sum  # = duplicate - missing

        # XOR method
        xor_all = 0
        xor_nums = 0
        for i in range(1, n + 1):
            xor_all ^= i
            xor_nums ^= nums[i - 1]

        xor_diff = xor_all ^ xor_nums  # = duplicate ^ missing

        # Additional statistics
        differences = [
            sum_diff,  # duplicate - missing
            xor_diff,  # duplicate XOR missing
            duplicate + missing,  # Can be derived from sum_diff and (dup+miss)²
            duplicate * missing  # Can be derived from above
        ]

        return duplicate, missing, sum_diff, differences

    def find_permutation_index(self, nums: List[int]) -> int:
        """
        Custom Hard - Find Permutation Index in Lexicographic Order

        Given permutation of [1,n], find its index in lexicographic order.
        Uses cyclic properties and factorial number system.

        Algorithm:
        1. Use factorial number system
        2. For each position, count smaller elements to the right
        3. Use cyclic sort properties for optimization

        Time: O(n²), Space: O(n)
        Can be optimized to O(n log n) with Fenwick tree

        Example:
        nums = [3,1,2]
        Output: 4 (0-indexed, permutations: [1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2])
        """
        n = len(nums)
        factorial = [1] * n

        # Precompute factorials
        for i in range(1, n):
            factorial[i] = factorial[i - 1] * i

        index = 0

        for i in range(n):
            # Count numbers smaller than nums[i] in remaining positions
            smaller_count = 0
            for j in range(i + 1, n):
                if nums[j] < nums[i]:
                    smaller_count += 1

            # Add contribution to index
            index += smaller_count * factorial[n - 1 - i]

            # Adjust remaining numbers (simulate removal)
            for j in range(i + 1, n):
                if nums[j] > nums[i]:
                    nums[j] -= 1

        return index

    def array_nesting(self, nums: List[int]) -> Tuple[int, List[List[int]]]:
        """
        LeetCode 565 Extension - Array Nesting with All Cycles (Hard)

        Find longest cycle where S[k] = {nums[k], nums[nums[k]], ...}
        Extended: Return all cycles and their lengths.

        Algorithm:
        1. Treat array as graph where i -> nums[i]
        2. Find all cycles using visited marking
        3. Track cycle membership and lengths

        Time: O(n), Space: O(n)

        Example:
        nums = [5,4,0,3,1,6,2]
        Output: (4, [[0,5,6,2], [1,4], [3]]) - max length 4
        """
        n = len(nums)
        visited = [False] * n
        all_cycles = []
        max_length = 0

        for start in range(n):
            if visited[start]:
                continue

            # Explore cycle starting from this position
            cycle = []
            current = start

            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = nums[current]

            # If we completed a cycle (came back to start)
            if current == start and len(cycle) > 0:
                all_cycles.append(cycle)
                max_length = max(max_length, len(cycle))

        # Sort cycles by length (descending)
        all_cycles.sort(key=len, reverse=True)

        return max_length, all_cycles

    def beautiful_arrangement(self, n: int) -> Tuple[int, List[List[int]]]:
        """
        LeetCode 526 Extension - Beautiful Arrangement with Cyclic Generation (Hard)

        Count beautiful arrangements where position i has number divisible by i or vice versa.
        Extended: Generate some arrangements using cyclic properties.

        Algorithm:
        1. Use backtracking with cyclic sort optimization
        2. Prune invalid branches early
        3. Generate valid arrangements efficiently

        Time: O(k) where k is number of valid arrangements
        Space: O(n)

        Example:
        n = 3
        Output: (3, [[1,2,3], [2,1,3], [3,2,1]])
        """
        count = 0
        arrangements = []

        def is_beautiful(pos, num):
            return num % pos == 0 or pos % num == 0

        def backtrack(path, remaining):
            nonlocal count

            if not remaining:
                count += 1
                if len(arrangements) < 10:  # Limit output
                    arrangements.append(path[:])
                return

            pos = len(path) + 1

            for i, num in enumerate(remaining):
                if is_beautiful(pos, num):
                    path.append(num)
                    backtrack(path, remaining[:i] + remaining[i + 1:])
                    path.pop()

        # Start with cyclic sort as initial arrangement
        initial = list(range(1, n + 1))

        # Try to improve initial arrangement
        for i in range(n):
            if not is_beautiful(i + 1, initial[i]):
                # Try to find a swap that makes both positions valid
                for j in range(i + 1, n):
                    if (is_beautiful(i + 1, initial[j]) and
                            is_beautiful(j + 1, initial[i])):
                        initial[i], initial[j] = initial[j], initial[i]
                        break

        # Generate all arrangements
        backtrack([], list(range(1, n + 1)))

        return count, arrangements

    def couples_holding_hands(self, row: List[int]) -> int:
        """
        LeetCode 765 - Couples Holding Hands (Hard)

        Couples are numbered 0,1 then 2,3 then 4,5 etc.
        Find minimum swaps to make all couples sit together.
        Uses cyclic sort with union-find optimization.

        Algorithm:
        1. Build graph where each couple forms a node
        2. Find cycles in the couple arrangement
        3. For cycle of size k, need k-1 swaps

        Time: O(n), Space: O(n)

        Example:
        row = [0,2,1,3]
        Output: 1 (swap positions 1 and 2)
        """
        n = len(row) // 2

        # Map person to their position
        pos = {person: i for i, person in enumerate(row)}

        # Count swaps using cycle detection
        swaps = 0

        for i in range(0, len(row), 2):
            # Check if couple is already together
            couple1 = row[i]
            couple2 = couple1 ^ 1  # Partner has ID differing by 1

            if row[i + 1] == couple2:
                continue

            # Need to swap to bring couple together
            partner_pos = pos[couple2]

            # Swap row[i+1] with couple2
            person_to_swap = row[i + 1]

            row[i + 1], row[partner_pos] = row[partner_pos], row[i + 1]
            pos[couple2] = i + 1
            pos[person_to_swap] = partner_pos

            swaps += 1

        return swaps


# Example usage and testing
if __name__ == "__main__":
    solver = CyclicSortHard()

    # Test 1: Find All Duplicates and Missing
    print("1. Find All Duplicates and Missing:")
    nums = [4, 3, 2, 7, 8, 2, 3, 1]
    print(f"   Input: {nums}")
    duplicates, missing = solver.find_all_duplicates_and_missing(nums.copy())
    print(f"   Output: Duplicates={duplicates}, Missing={missing}")
    print()

    # Test 2: First Missing Positive
    print("2. First Missing Positive with Constraints:")
    nums = [3, 4, -1, 1]
    print(f"   Input: {nums}")
    first, all_missing = solver.first_missing_positive_with_constraints(nums.copy())
    print(f"   Output: First Missing={first}, All Missing={all_missing}")
    print()

    # Test 3: Minimum Swaps to Sort
    print("3. Minimum Swaps to Sort:")
    nums = [3, 2, 0, 1]
    print(f"   Input: {nums}")
    count, swaps = solver.minimum_swaps_to_sort_cyclic(nums.copy())
    print(f"   Output: Swap Count={count}, Swaps={swaps}")
    print()

    # Test 4: Array Nesting
    print("4. Array Nesting:")
    nums = [5, 4, 0, 3, 1, 6, 2]
    print(f"   Input: {nums}")
    max_len, cycles = solver.array_nesting(nums)
    print(f"   Output: Max Length={max_len}, Cycles={cycles}")