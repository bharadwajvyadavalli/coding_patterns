"""
Pattern 15: 0/1 Knapsack (Dynamic Programming) - 10 Hard Problems
================================================================

The 0/1 Knapsack pattern solves optimization problems where items must be either
fully included or excluded (no fractional items). This pattern is fundamental
for subset selection problems with constraints.

Key Concepts:
- Each item can be taken 0 or 1 time
- Build solution using previously computed subproblems
- State typically includes current item index and remaining capacity
- Can often optimize space from O(n*capacity) to O(capacity)

Time Complexity: Usually O(n * capacity) or O(n * target)
Space Complexity: O(capacity) with optimization, O(n * capacity) without
"""

from typing import List, Tuple, Dict, Set
import bisect


class ZeroOneKnapsackHard:

    def partition_to_k_equal_sum_subsets(self, nums: List[int], k: int) -> bool:
        """
        LeetCode 698 - Partition to K Equal Sum Subsets (Hard)

        Determine if array can be partitioned into k subsets with equal sum.

        Algorithm:
        1. Check if total sum is divisible by k
        2. Use bitmask DP where state represents which elements are used
        3. For each state, try to form complete groups of target sum
        4. Optimize with memoization and pruning

        Time: O(n * 2^n), Space: O(2^n)

        Example:
        nums = [4,3,2,3,5,2,1], k = 4
        Output: True (can form 4 subsets of sum 5 each)
        """
        total_sum = sum(nums)
        if total_sum % k != 0:
            return False

        target = total_sum // k
        nums.sort(reverse=True)  # Optimization: try larger numbers first

        # Early termination
        if nums[0] > target:
            return False

        n = len(nums)
        memo = {}

        def can_partition(mask: int, current_sum: int) -> bool:
            if mask == (1 << n) - 1:  # All elements used
                return True

            if (mask, current_sum) in memo:
                return memo[(mask, current_sum)]

            # If current group is complete, start new group
            if current_sum == target:
                result = can_partition(mask, 0)
                memo[(mask, current_sum)] = result
                return result

            # Try adding each unused element to current group
            for i in range(n):
                if mask & (1 << i):  # Element already used
                    continue

                if current_sum + nums[i] > target:  # Would exceed target
                    continue

                # Try including this element
                if can_partition(mask | (1 << i), current_sum + nums[i]):
                    memo[(mask, current_sum)] = True
                    return True

            memo[(mask, current_sum)] = False
            return False

        return can_partition(0, 0)

    def target_sum_with_paths(self, nums: List[int], target: int) -> Tuple[int, List[List[int]]]:
        """
        LeetCode 494 Extension - Target Sum with All Paths (Hard)

        Find number of ways to assign + or - to make sum equal target.
        Extended: Return sample paths showing the assignments.

        Algorithm:
        1. Convert to subset sum: find subset with sum = (total + target) / 2
        2. Use DP to count ways
        3. Backtrack to find actual paths

        Time: O(n * sum), Space: O(n * sum)

        Example:
        nums = [1,1,1,1,1], target = 3
        Output: (5, [[+1,+1,+1,-1,-1], ...]) - 5 ways total
        """
        total = sum(nums)

        # Check if target is achievable
        if abs(target) > total or (total + target) % 2 != 0:
            return 0, []

        # Convert to subset sum problem
        subset_sum = (total + target) // 2

        # DP to count ways
        n = len(nums)
        dp = [[0] * (subset_sum + 1) for _ in range(n + 1)]
        dp[0][0] = 1

        for i in range(1, n + 1):
            for j in range(subset_sum + 1):
                # Don't include current number
                dp[i][j] = dp[i - 1][j]

                # Include current number if possible
                if j >= nums[i - 1]:
                    dp[i][j] += dp[i - 1][j - nums[i - 1]]

        count = dp[n][subset_sum]

        # Backtrack to find paths (limit to 10 for efficiency)
        paths = []

        def backtrack(i: int, current_sum: int, path: List[str]):
            if len(paths) >= 10:  # Limit number of paths
                return

            if i == 0:
                if current_sum == 0:
                    paths.append(path[:])
                return

            # Try not including in positive subset (use -)
            if dp[i - 1][current_sum] > 0:
                path.append(f"-{nums[i - 1]}")
                backtrack(i - 1, current_sum, path)
                path.pop()

            # Try including in positive subset (use +)
            if current_sum >= nums[i - 1] and dp[i - 1][current_sum - nums[i - 1]] > 0:
                path.append(f"+{nums[i - 1]}")
                backtrack(i - 1, current_sum - nums[i - 1], path)
                path.pop()

        backtrack(n, subset_sum, [])

        # Reverse paths to match original order
        for path in paths:
            path.reverse()

        return count, paths

    def ones_and_zeroes_with_selection(self, strs: List[str], m: int, n: int) -> Tuple[int, List[str]]:
        """
        LeetCode 474 Extension - Ones and Zeroes with Selected Strings (Hard)

        Find largest subset with at most m 0's and n 1's.
        Extended: Return the actual strings selected.

        Algorithm:
        1. 3D DP: dp[i][j][k] = max strings using first i with j zeros, k ones
        2. Track parent pointers for reconstruction
        3. Backtrack to find selected strings

        Time: O(len * m * n), Space: O(len * m * n)

        Example:
        strs = ["10","0001","111001","1","0"], m = 5, n = 3
        Output: (4, ["10","0001","1","0"])
        """
        # Count 0s and 1s for each string
        counts = []
        for s in strs:
            zeros = s.count('0')
            ones = s.count('1')
            counts.append((zeros, ones))

        # 3D DP with parent tracking
        length = len(strs)
        dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(length + 1)]
        parent = [[[None] * (n + 1) for _ in range(m + 1)] for _ in range(length + 1)]

        for i in range(1, length + 1):
            zeros, ones = counts[i - 1]

            for j in range(m + 1):
                for k in range(n + 1):
                    # Option 1: Don't include current string
                    dp[i][j][k] = dp[i - 1][j][k]
                    parent[i][j][k] = (i - 1, j, k, False)

                    # Option 2: Include current string if possible
                    if j >= zeros and k >= ones:
                        include_val = dp[i - 1][j - zeros][k - ones] + 1
                        if include_val > dp[i][j][k]:
                            dp[i][j][k] = include_val
                            parent[i][j][k] = (i - 1, j - zeros, k - ones, True)

        # Backtrack to find selected strings
        selected = []
        i, j, k = length, m, n

        while i > 0:
            prev_i, prev_j, prev_k, included = parent[i][j][k]
            if included:
                selected.append(strs[i - 1])
            i, j, k = prev_i, prev_j, prev_k

        selected.reverse()
        return dp[length][m][n], selected

    def profitable_schemes(self, n: int, minProfit: int, group: List[int], profit: List[int]) -> int:
        """
        LeetCode 879 - Profitable Schemes (Hard)

        Count schemes with at most n people achieving at least minProfit.

        Algorithm:
        1. 3D DP: dp[i][j][k] = schemes using i crimes, j people, k profit
        2. Optimize by capping profit at minProfit (anything above counts same)
        3. Use modulo arithmetic for large numbers

        Time: O(len * n * minProfit), Space: O(n * minProfit)

        Example:
        n = 5, minProfit = 3, group = [2,2], profit = [2,3]
        Output: 2 (schemes: [0], [1], both achieve profit >= 3)
        """
        MOD = 10 ** 9 + 7

        # dp[i][j] = number of schemes with i people and j profit
        dp = [[0] * (minProfit + 1) for _ in range(n + 1)]
        dp[0][0] = 1  # Empty scheme

        for crime_idx in range(len(group)):
            people_needed = group[crime_idx]
            crime_profit = profit[crime_idx]

            # Traverse in reverse to avoid using same crime multiple times
            for people in range(n, people_needed - 1, -1):
                for curr_profit in range(minProfit, -1, -1):
                    # New profit is capped at minProfit
                    new_profit = min(minProfit, curr_profit + crime_profit)

                    dp[people][new_profit] += dp[people - people_needed][curr_profit]
                    dp[people][new_profit] %= MOD

        # Sum all schemes achieving at least minProfit
        result = 0
        for people in range(n + 1):
            result = (result + dp[people][minProfit]) % MOD

        return result

    def last_stone_weight_ii_with_split(self, stones: List[int]) -> Tuple[int, List[int], List[int]]:
        """
        LeetCode 1049 Extension - Last Stone Weight II with Groups (Hard)

        Minimize final stone weight after smashing.
        Extended: Show how to split stones into two groups.

        Algorithm:
        1. Convert to partition problem: minimize difference between two groups
        2. Use subset sum DP to find closest partition to total/2
        3. Backtrack to find actual partition

        Time: O(n * sum), Space: O(n * sum)

        Example:
        stones = [2,7,4,1,8,1]
        Output: (1, [2,4,1,1], [7,8]) - difference = |8-15| = 1
        """
        total = sum(stones)
        target = total // 2
        n = len(stones)

        # DP to find if sum is achievable
        dp = [[False] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = True

        for i in range(1, n + 1):
            for j in range(target + 1):
                # Don't include current stone
                dp[i][j] = dp[i - 1][j]

                # Include current stone if possible
                if j >= stones[i - 1]:
                    dp[i][j] |= dp[i - 1][j - stones[i - 1]]

        # Find largest sum <= target that's achievable
        closest_sum = 0
        for s in range(target, -1, -1):
            if dp[n][s]:
                closest_sum = s
                break

        # Backtrack to find partition
        group1 = []
        group2 = []
        i, current_sum = n, closest_sum

        for idx in range(n - 1, -1, -1):
            # Check if current stone was included
            if current_sum >= stones[idx] and dp[idx][current_sum - stones[idx]]:
                group1.append(stones[idx])
                current_sum -= stones[idx]
            else:
                group2.append(stones[idx])

        # Final weight is absolute difference
        sum1 = sum(group1)
        sum2 = sum(group2)

        return abs(sum1 - sum2), sorted(group1), sorted(group2)

    def tallest_billboard(self, rods: List[int]) -> int:
        """
        LeetCode 956 - Tallest Billboard (Hard)

        Build two equal-height billboards using given rods.
        Return maximum possible height.

        Algorithm:
        1. DP where state is difference between two billboard heights
        2. dp[diff] = max height of taller billboard with given difference
        3. For each rod, three choices: add to left, add to right, or skip

        Time: O(n * sum), Space: O(sum)

        Example:
        rods = [1,2,3,6]
        Output: 6 (use [1,2,3] and [6] for both sides)
        """
        # dp[diff] = maximum height of taller billboard with difference = diff
        dp = {0: 0}

        for rod in rods:
            new_dp = dp.copy()

            for diff, taller in dp.items():
                shorter = taller - diff

                # Add rod to taller side
                new_diff = diff + rod
                if new_diff not in new_dp or new_dp[new_diff] < taller + rod:
                    new_dp[new_diff] = taller + rod

                # Add rod to shorter side
                new_taller = max(taller, shorter + rod)
                new_diff = abs(taller - (shorter + rod))
                if new_diff not in new_dp or new_dp[new_diff] < new_taller:
                    new_dp[new_diff] = new_taller

            dp = new_dp

        return dp.get(0, 0)

    def minimum_difference_after_partition(self, nums: List[int]) -> int:
        """
        Custom Hard - Minimum Difference with 3-Way Partition

        Partition array into 3 non-empty subsets to minimize max(sum) - min(sum).

        Algorithm:
        1. Try all ways to partition into 3 groups
        2. Use bitmask DP to track assignments
        3. Optimize with meet-in-the-middle approach

        Time: O(3^n), Space: O(3^n)
        Can be optimized to O(n * 3^(n/2))

        Example:
        nums = [3,9,7,3]
        Output: 2 (groups: [3,3], [7], [9] with sums 6,7,9)
        """
        n = len(nums)
        if n < 3:
            return float('inf')

        min_diff = float('inf')

        # For small n, try all partitions
        if n <= 10:
            # Generate all 3-way partitions
            def generate_partitions(index, groups):
                nonlocal min_diff

                if index == n:
                    # Check if all groups are non-empty
                    if all(groups):
                        sums = [sum(g) for g in groups]
                        diff = max(sums) - min(sums)
                        min_diff = min(min_diff, diff)
                    return

                # Try adding current element to each group
                for i in range(3):
                    groups[i].append(nums[index])
                    generate_partitions(index + 1, groups)
                    groups[i].pop()

            generate_partitions(0, [[], [], []])

        else:
            # For larger n, use meet-in-the-middle
            mid = n // 2
            first_half = nums[:mid]
            second_half = nums[mid:]

            # Generate all possible (sum1, sum2) for first half
            first_sums = {}
            for mask in range(3 ** mid):
                sums = [0,