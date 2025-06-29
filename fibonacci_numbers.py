"""
Pattern 17: Fibonacci Numbers (Dynamic Programming) - 10 Hard Problems
=====================================================================

The Fibonacci Numbers pattern applies to problems with recursive relationships
where each state depends on one or more previous states. This includes classic
Fibonacci, climbing stairs, and other sequence-based problems.

Key Concepts:
- Current state depends on previous states (usually 1-2 steps back)
- Can optimize space from O(n) to O(1) using rolling variables
- Often involves counting ways or finding optimal solutions
- May include variations with constraints or modifications

Time Complexity: Usually O(n) or O(n²) for complex variants
Space Complexity: O(1) to O(n) depending on optimization
"""

from typing import List, Tuple, Dict
from functools import lru_cache
import math


class FibonacciNumbersHard:

    def climb_stairs_with_variable_steps(self, n: int, steps: List[int]) -> Tuple[int, List[List[int]]]:
        """
        Extension of LeetCode 70 - Climbing Stairs with Variable Steps (Hard)

        Climb n stairs with variable step sizes allowed.
        Return number of ways and sample paths.

        Algorithm:
        1. DP where dp[i] = ways to reach stair i
        2. For each stair, try all allowed steps
        3. Generate sample paths using backtracking

        Time: O(n * len(steps)), Space: O(n)

        Example:
        n = 4, steps = [1, 2, 3]
        Output: (7, [[1,1,1,1], [1,1,2], [1,2,1], [2,1,1], [2,2], [1,3], [3,1]])
        """
        dp = [0] * (n + 1)
        dp[0] = 1
        parent = [[] for _ in range(n + 1)]

        # Calculate number of ways
        for i in range(1, n + 1):
            for step in steps:
                if i - step >= 0:
                    dp[i] += dp[i - step]
                    if dp[i - step] > 0:
                        parent[i].append(step)

        # Generate sample paths (limit to prevent memory issues)
        paths = []

        def generate_paths(remaining: int, current_path: List[int]):
            if len(paths) >= 20:  # Limit paths
                return

            if remaining == 0:
                paths.append(current_path[:])
                return

            for step in steps:
                if step <= remaining:
                    current_path.append(step)
                    generate_paths(remaining - step, current_path)
                    current_path.pop()

        generate_paths(n, [])

        return dp[n], paths[:10]  # Return first 10 paths

    def house_robber_iii_with_strategy(self, root: 'TreeNode') -> Tuple[int, List[int], str]:
        """
        LeetCode 337 Extension - House Robber III with Strategy (Hard)

        Rob houses in binary tree without robbing adjacent houses.
        Extended: Return robbed houses and strategy explanation.

        Algorithm:
        1. Tree DP with two states per node (rob/not rob)
        2. Track which houses are robbed
        3. Generate strategy explanation

        Time: O(n), Space: O(h) where h is tree height

        Example:
        Tree: [3,2,3,null,3,null,1]
        Output: (7, [0,3,5], "Rob root and leaves")
        """

        class TreeNode:
            def __init__(self, val=0, left=None, right=None, idx=0):
                self.val = val
                self.left = left
                self.right = right
                self.idx = idx

        if not root:
            return 0, [], "No houses to rob"

        robbed_houses = []

        def dfs(node):
            if not node:
                return 0, 0, []

            # Get results from children
            rob_left, not_rob_left, houses_left = dfs(node.left)
            rob_right, not_rob_right, houses_right = dfs(node.right)

            # If we rob this house
            rob_current = node.val + not_rob_left + not_rob_right
            houses_if_rob = [node.idx] + houses_left + houses_right

            # If we don't rob this house
            not_rob_current = max(rob_left, not_rob_left) + max(rob_right, not_rob_right)
            houses_if_not_rob = []

            if max(rob_left, not_rob_left) == rob_left:
                houses_if_not_rob.extend(houses_left)
            if max(rob_right, not_rob_right) == rob_right:
                houses_if_not_rob.extend(houses_right)

            # Return best option
            if rob_current > not_rob_current:
                return rob_current, not_rob_current, houses_if_rob
            else:
                return not_rob_current, not_rob_current, houses_if_not_rob

        max_money, _, robbed_houses = dfs(root)

        # Generate strategy
        if len(robbed_houses) == 1:
            strategy = "Rob only the root"
        elif all(h > 2 for h in robbed_houses):  # Assuming indices represent levels
            strategy = "Rob leaves and deep nodes"
        else:
            strategy = f"Rob {len(robbed_houses)} non-adjacent houses"

        return max_money, sorted(robbed_houses), strategy

    def decode_ways_with_mappings(self, s: str) -> Tuple[int, List[List[str]]]:
        """
        LeetCode 91 Extension - Decode Ways with All Mappings (Hard)

        Decode string where 'A'=1, 'B'=2, ..., 'Z'=26.
        Extended: Return all possible decodings (limited).

        Algorithm:
        1. DP with state tracking
        2. Generate actual decodings
        3. Handle edge cases (0s, invalid codes)

        Time: O(n), Space: O(n) for DP, O(2^n) for all decodings

        Example:
        s = "226"
        Output: (3, [['B','B','F'], ['B','Z'], ['V','F']])
        """
        if not s or s[0] == '0':
            return 0, []

        n = len(s)
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1

        # Calculate number of decodings
        for i in range(2, n + 1):
            # Single digit
            if s[i - 1] != '0':
                dp[i] += dp[i - 1]

            # Two digits
            two_digit = int(s[i - 2:i])
            if 10 <= two_digit <= 26:
                dp[i] += dp[i - 2]

        # Generate actual decodings (limit to prevent memory issues)
        decodings = []

        def generate_decodings(idx: int, current: List[str]):
            if len(decodings) >= 100:  # Limit
                return

            if idx == n:
                decodings.append(current[:])
                return

            # Single digit
            if s[idx] != '0':
                digit = int(s[idx])
                current.append(chr(ord('A') + digit - 1))
                generate_decodings(idx + 1, current)
                current.pop()

            # Two digits
            if idx + 1 < n:
                two_digit = int(s[idx:idx + 2])
                if 10 <= two_digit <= 26:
                    current.append(chr(ord('A') + two_digit - 1))
                    generate_decodings(idx + 2, current)
                    current.pop()

        generate_decodings(0, [])

        return dp[n], decodings[:10]  # Return first 10 decodings

    def knight_dialer_with_paths(self, n: int) -> Tuple[int, Dict[int, int]]:
        """
        LeetCode 935 Extension - Knight Dialer with Path Analysis (Hard)

        Count knight's distinct phone numbers of length n.
        Extended: Analyze distribution of ending positions.

        Algorithm:
        1. DP with state = current position
        2. Use adjacency list for knight moves
        3. Track ending position distribution

        Time: O(n), Space: O(1)

        Example:
        n = 3
        Output: (46, {0:6, 1:6, 2:6, 3:6, 4:0, 5:0, 6:6, 7:6, 8:6, 9:4})
        """
        MOD = 10 ** 9 + 7

        # Knight move adjacency list for phone pad
        moves = {
            0: [4, 6],
            1: [6, 8],
            2: [7, 9],
            3: [4, 8],
            4: [0, 3, 9],
            5: [],  # No valid moves
            6: [0, 1, 7],
            7: [2, 6],
            8: [1, 3],
            9: [2, 4]
        }

        # dp[i] = number of paths ending at digit i
        dp = [1] * 10

        for _ in range(n - 1):
            new_dp = [0] * 10
            for i in range(10):
                for j in moves[i]:
                    new_dp[j] = (new_dp[j] + dp[i]) % MOD
            dp = new_dp

        total = sum(dp) % MOD
        distribution = {i: dp[i] for i in range(10)}

        return total, distribution

    def maximum_alternating_subsequence_sum(self, nums: List[int]) -> Tuple[int, List[int]]:
        """
        LeetCode 1911 Extension - Maximum Alternating Subsequence Sum with Sequence (Hard)

        Find maximum alternating sum (even index + , odd index -).
        Extended: Return the actual subsequence.

        Algorithm:
        1. DP with two states (last taken at even/odd position)
        2. Track parent pointers for reconstruction
        3. Handle transitions carefully

        Time: O(n), Space: O(n)

        Example:
        nums = [6,2,1,2,4,5]
        Output: (10, [6,1,5]) - 6-1+5=10
        """
        n = len(nums)
        # dp[i][0] = max sum ending at i with even length
        # dp[i][1] = max sum ending at i with odd length
        dp = [[0, 0] for _ in range(n)]
        parent = [[(-1, -1), (-1, -1)] for _ in range(n)]

        # Base case
        dp[0][1] = nums[0]  # Take first element (odd length)

        for i in range(1, n):
            # Option 1: Don't take current element
            dp[i][0] = dp[i - 1][0]
            dp[i][1] = dp[i - 1][1]
            parent[i][0] = (i - 1, 0)
            parent[i][1] = (i - 1, 1)

            # Option 2: Take current element
            # If previous was odd length, current makes it even (subtract)
            if dp[i - 1][1] - nums[i] > dp[i][0]:
                dp[i][0] = dp[i - 1][1] - nums[i]
                parent[i][0] = (i - 1, 1)

            # If previous was even length (or empty), current makes it odd (add)
            if dp[i - 1][0] + nums[i] > dp[i][1]:
                dp[i][1] = dp[i - 1][0] + nums[i]
                parent[i][1] = (i - 1, 0)

        # Find maximum and reconstruct
        max_sum = max(dp[n - 1][0], dp[n - 1][1])
        state = 0 if dp[n - 1][0] > dp[n - 1][1] else 1

        # Reconstruct subsequence
        subsequence = []
        i = n - 1

        while i >= 0:
            prev_i, prev_state = parent[i][state]

            # Check if current element was taken
            if prev_i == -1 or dp[i][state] != dp[prev_i][prev_state]:
                subsequence.append(nums[i])

            i = prev_i
            state = prev_state

            if i == -1:
                break

        subsequence.reverse()

        return max_sum, subsequence

    def tribonacci_with_matrix_power(self, n: int) -> Tuple[int, List[List[int]]]:
        """
        LeetCode 1137 Extension - Tribonacci with Matrix Exponentiation (Hard)

        Calculate n-th Tribonacci number using matrix exponentiation.
        Extended: Return transformation matrix and intermediate results.

        Algorithm:
        1. Express as matrix multiplication
        2. Use fast matrix exponentiation
        3. Track intermediate matrices

        Time: O(log n), Space: O(1)

        Example:
        n = 10
        Output: (149, transformation_matrix)
        """
        if n == 0:
            return 0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        if n == 1 or n == 2:
            return 1, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        def matrix_multiply(A, B):
            """Multiply two 3x3 matrices."""
            result = [[0] * 3 for _ in range(3)]
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        result[i][j] += A[i][k] * B[k][j]
            return result

        def matrix_power(matrix, p):
            """Compute matrix^p using fast exponentiation."""
            if p == 0:
                return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

            if p == 1:
                return matrix

            if p % 2 == 0:
                half = matrix_power(matrix, p // 2)
                return matrix_multiply(half, half)
            else:
                return matrix_multiply(matrix, matrix_power(matrix, p - 1))

        # Tribonacci transformation matrix
        trans_matrix = [
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 0]
        ]

        # Calculate T(n) = trans_matrix^(n-2) * [T(2), T(1), T(0)]
        result_matrix = matrix_power(trans_matrix, n - 2)

        # Initial values [T(2), T(1), T(0)] = [1, 1, 0]
        tribonacci_n = result_matrix[0][0] + result_matrix[0][1]

        return tribonacci_n, result_matrix

    def minimum_cost_climbing_stairs_with_k_steps(self, cost: List[int], k: int) -> Tuple[int, List[int]]:
        """
        Extension of LeetCode 746 - Min Cost Climbing Stairs with K Steps (Hard)

        Climb stairs with cost, can climb 1 to k steps at a time.
        Extended: Return minimum cost and path taken.

        Algorithm:
        1. DP where dp[i] = min cost to reach stair i
        2. Try all steps from 1 to k
        3. Reconstruct optimal path

        Time: O(n * k), Space: O(n)

        Example:
        cost = [10,15,20,25], k = 3
        Output: (15, [0,3]) - jump from index 0 to 3
        """
        n = len(cost)
        dp = [float('inf')] * (n + 1)
        parent = [-1] * (n + 1)

        # Can start from index 0 or 1
        dp[0] = 0
        dp[1] = 0

        for i in range(2, n + 1):
            # Try all possible steps
            for step in range(1, min(k + 1, i + 1)):
                if i - step < n:  # Landing on a stair with cost
                    new_cost = dp[i - step] + cost[i - step]
                else:  # Landing beyond the stairs
                    new_cost = dp[i - step]

                if new_cost < dp[i]:
                    dp[i] = new_cost
                    parent[i] = i - step

        # Reconstruct path
        path = []
        current = n

        while current > 1:
            if parent[current] != -1:
                path.append(parent[current])
                current = parent[current]
            else:
                break

        path.reverse()

        return dp[n], path

    def domino_and_tromino_tiling_extended(self, n: int) -> Tuple[int, int, int]:
        """
        LeetCode 790 Extension - Domino and Tromino Tiling Analysis (Hard)

        Tile 2×n board with dominoes and L-shaped trominos.
        Extended: Count tilings by number of each piece type.

        Algorithm:
        1. DP with multiple states for partial tilings
        2. Track piece usage
        3. Analyze tiling patterns

        Time: O(n), Space: O(n)

        Example:
        n = 4
        Output: (11, 5, 6) - total tilings, using only dominoes, using trominos
        """
        MOD = 10 ** 9 + 7

        if n <= 2:
            return n, n, 0

        # dp[i][j] represents:
        # j=0: both rows filled up to column i
        # j=1: top row has one extra cell filled
        # j=2: bottom row has one extra cell filled
        dp = [[0] * 3 for _ in range(n + 1)]

        # Base cases
        dp[0][0] = 1
        dp[1][0] = 1

        for i in range(2, n + 1):
            # Both rows filled
            dp[i][0] = (dp[i - 1][0] + dp[i - 2][0] + dp[i - 1][1] + dp[i - 1][2]) % MOD

            # Top row has extra
            dp[i][1] = (dp[i - 2][0] + dp[i - 1][2]) % MOD

            # Bottom row has extra
            dp[i][2] = (dp[i - 2][0] + dp[i - 1][1]) % MOD

        total_tilings = dp[n][0]

        # Count tilings using only dominoes (Fibonacci pattern)
        domino_only = [0] * (n + 1)
        domino_only[0] = 1
        domino_only[1] = 1

        for i in range(2, n + 1):
            domino_only[i] = (domino_only[i - 1] + domino_only[i - 2]) % MOD

        # Tilings with trominos = total - domino_only
        tromino_tilings = (total_tilings - domino_only[n] + MOD) % MOD

        return total_tilings, domino_only[n], tromino_tilings

    def strange_printer_ii_fibonacci(self, s: str) -> int:
        """
        Custom - Strange Printer with Fibonacci Constraints (Hard)

        Printer can print segments of same character. Each print operation
        can cover at most F(k) characters where F is Fibonacci sequence.
        Find minimum operations to print string.

        Algorithm:
        1. DP with Fibonacci constraints
        2. Try all valid segment lengths
        3. Optimize overlapping prints

        Time: O(n²), Space: O(n)

        Example:
        s = "aaabbb"
        Output: 2 (print "aaa" then "bbb", both length 3 = F(4))
        """
        n = len(s)

        # Generate Fibonacci numbers up to n
        fib = [1, 1]
        while fib[-1] < n:
            fib.append(fib[-1] + fib[-2])

        # dp[i] = minimum prints for s[0:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            # Try each Fibonacci length
            for f in fib:
                if f > i:
                    break

                # Check if we can print last f characters as one segment
                start = i - f
                char = s[start]
                can_print = True

                # Allow some different characters (will be overwritten)
                diff_count = 0
                for j in range(start, i):
                    if s[j] != char:
                        diff_count += 1

                # If mostly same character, we can print
                if diff_count <= f // 3:  # Allow up to 1/3 different
                    dp[i] = min(dp[i], dp[start] + 1)

            # Also try printing just the last character
            dp[i] = min(dp[i], dp[i - 1] + 1)

        return dp[n]

    def maximum_sum_circular_subarray_k_segments(self, nums: List[int], k: int) -> int:
        """
        Extension of LeetCode 918 - Maximum Sum Circular Subarray with K Segments (Hard)

        Find maximum sum of exactly k non-overlapping subarrays in circular array.

        Algorithm:
        1. DP considering circular nature
        2. Track k best segments
        3. Handle wrap-around cases

        Time: O(n² * k), Space: O(n * k)

        Example:
        nums = [1,-2,3,-2,5], k = 2
        Output: 9 (segments [3] and [5,1])
        """
        n = len(nums)

        # Handle non-circular case first
        # dp[i][j] = max sum using j segments from first i elements
        dp = [[-float('inf')] * (k + 1) for _ in range(n + 1)]

        # Base case
        for i in range(n + 1):
            dp[i][0] = 0

        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, min(i + 1, k + 1)):
                # Don't include nums[i-1]
                dp[i][j] = dp[i - 1][j]

                # Include nums[i-1] in a segment
                curr_sum = 0
                for start in range(i - 1, max(-1, i - n - 1), -1):
                    curr_sum += nums[start]
                    if start == 0:
                        if j == 1:
                            dp[i][j] = max(dp[i][j], curr_sum)
                    else:
                        dp[i][j] = max(dp[i][j], dp[start][j - 1] + curr_sum)

        result = dp[n][k]

        # Handle circular case
        # Try splitting k segments between prefix and suffix
        prefix_sum = [0]
        for num in nums:
            prefix_sum.append(prefix_sum[-1] + num)

        for prefix_segments in range(k):
            suffix_segments = k - prefix_segments

            if prefix_segments > 0 and suffix_segments > 0:
                # Find best prefix with prefix_segments
                prefix_max = dp[n][prefix_segments]

                # Find best suffix with suffix_segments
                suffix_dp = [[-float('inf')] * (suffix_segments + 1) for _ in range(n + 1)]
                suffix_dp[0][0] = 0

                for i in range(1, n + 1):
                    for j in range(1, min(i + 1, suffix_segments + 1)):
                        suffix_dp[i][j] = suffix_dp[i - 1][j]

                        curr_sum = 0
                        for start in range(i - 1, max(-1, i - n - 1), -1):
                            curr_sum += nums[n - i + start]
                            if start == 0:
                                if j == 1:
                                    suffix_dp[i][j] = max(suffix_dp[i][j], curr_sum)
                            else:
                                suffix_dp[i][j] = max(suffix_dp[i][j],
                                                      suffix_dp[start][j - 1] + curr_sum)

                suffix_max = suffix_dp[n][suffix_segments]

                # Ensure no overlap
                total = prefix_sum[n]
                if prefix_max + suffix_max - total > result:
                    result = prefix_max + suffix_max - total

        return result


# Example usage and testing
if __name__ == "__main__":
    solver = FibonacciNumbersHard()

    # Test 1: Climb Stairs with Variable Steps
    print("1. Climb Stairs with Variable Steps:")
    n = 4
    steps = [1, 2, 3]
    ways, paths = solver.climb_stairs_with_variable_steps(n, steps)
    print(f"   n={n}, steps={steps}")
    print(f"   Ways: {ways}, Sample paths: {paths[:5]}")
    print()

    # Test 2: Decode Ways
    print("2. Decode Ways with Mappings:")
    s = "226"
    ways, decodings = solver.decode_ways_with_mappings(s)
    print(f"   String: '{s}'")
    print(f"   Ways: {ways}, Decodings: {decodings}")
    print()

    # Test 3: Knight Dialer
    print("3. Knight Dialer with Path Analysis:")
    n = 3
    count, distribution = solver.knight_dialer_with_paths(n)
    print(f"   Length: {n}")
    print(f"   Total numbers: {count}")
    print(f"   Ending distribution: {distribution}")
    print()

    # Test 4: Tribonacci with Matrix
    print("4. Tribonacci with Matrix Exponentiation:")
    n = 10
    value, matrix = solver.tribonacci_with_matrix_power(n)
    print(f"   n={n}")
    print(f"   T({n}) = {value}")
    print(f"   Transformation matrix: {matrix[0]}")