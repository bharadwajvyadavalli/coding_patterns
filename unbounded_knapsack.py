"""
Pattern 16: Unbounded Knapsack (Dynamic Programming) - 10 Hard Problems
======================================================================

The Unbounded Knapsack pattern extends the knapsack problem to allow unlimited
use of items. This pattern is essential for optimization problems where resources
can be reused, such as coin change, rod cutting, and ribbon cutting problems.

Key Concepts:
- Each item can be used multiple times
- State typically represents remaining capacity/target
- Build solution using previously computed optimal values
- Often can optimize space to O(capacity) instead of O(n * capacity)

Time Complexity: Usually O(n * capacity) or O(n * target)
Space Complexity: O(capacity) or O(target)
"""

from typing import List, Tuple, Dict, Set
from collections import defaultdict
import math


class UnboundedKnapsackHard:

    def coin_change_with_minimum_coins_used(self, coins: List[int], amount: int) -> Tuple[int, Dict[int, int]]:
        """
        LeetCode 322 Extension - Coin Change with Coin Usage (Hard)

        Find minimum coins needed and track which coins are used.
        Return -1 if amount cannot be made.

        Algorithm:
        1. DP where dp[i] = min coins for amount i
        2. Track which coin was used for each amount
        3. Backtrack to find coin usage

        Time: O(amount * len(coins)), Space: O(amount)

        Example:
        coins = [1,2,5], amount = 11
        Output: (3, {5: 2, 1: 1}) - use two 5s and one 1
        """
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        parent = [-1] * (amount + 1)  # Track which coin was used

        for i in range(1, amount + 1):
            for j, coin in enumerate(coins):
                if coin <= i and dp[i - coin] + 1 < dp[i]:
                    dp[i] = dp[i - coin] + 1
                    parent[i] = j

        if dp[amount] == float('inf'):
            return -1, {}

        # Backtrack to find coin usage
        coin_usage = defaultdict(int)
        current = amount

        while current > 0:
            coin_idx = parent[current]
            coin_value = coins[coin_idx]
            coin_usage[coin_value] += 1
            current -= coin_value

        return dp[amount], dict(coin_usage)

    def integer_break_with_factors(self, n: int) -> Tuple[int, List[int]]:
        """
        LeetCode 343 Extension - Integer Break with Factorization (Hard)

        Break integer into sum of at least two positive integers
        to maximize product. Return product and factors.

        Algorithm:
        1. DP where dp[i] = max product for integer i
        2. Try all possible breaks
        3. Track factorization for reconstruction

        Time: O(n²), Space: O(n)

        Example:
        n = 10
        Output: (36, [3,3,4]) - 3*3*4 = 36
        """
        if n <= 3:
            return n - 1, [1, n - 1]

        dp = [0] * (n + 1)
        parent = [[] for _ in range(n + 1)]

        # Base cases
        dp[1] = 1
        parent[1] = [1]
        dp[2] = 1
        parent[2] = [1, 1]

        for i in range(3, n + 1):
            # Try breaking i into j and (i-j)
            for j in range(1, i):
                # Option 1: Use j as is and break (i-j)
                product1 = j * dp[i - j]
                # Option 2: Use j as is and (i-j) as is
                product2 = j * (i - j)

                if product1 > dp[i]:
                    dp[i] = product1
                    parent[i] = [j] + parent[i - j]

                if product2 > dp[i]:
                    dp[i] = product2
                    parent[i] = [j, i - j]

        # For the original number n, we must break it
        result = 0
        factors = []

        for j in range(1, n):
            if j * dp[n - j] > result:
                result = j * dp[n - j]
                factors = [j] + parent[n - j]

            if j * (n - j) > result:
                result = j * (n - j)
                factors = [j, n - j]

        # Optimize factors (convert to mostly 3s)
        optimized = []
        for f in factors:
            while f > 4:
                optimized.append(3)
                f -= 3
            if f > 0:
                optimized.append(f)

        return result, sorted(optimized)

    def perfect_squares_with_path(self, n: int) -> Tuple[int, List[int]]:
        """
        LeetCode 279 Extension - Perfect Squares with Decomposition (Hard)

        Find minimum perfect squares that sum to n.
        Return count and the actual squares used.

        Algorithm:
        1. DP with unbounded knapsack approach
        2. Track which square was used
        3. Reconstruct decomposition

        Time: O(n * sqrt(n)), Space: O(n)

        Example:
        n = 12
        Output: (3, [4,4,4]) - 4+4+4 = 12
        """
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        parent = [-1] * (n + 1)

        # Generate perfect squares
        squares = []
        i = 1
        while i * i <= n:
            squares.append(i * i)
            i += 1

        for i in range(1, n + 1):
            for j, square in enumerate(squares):
                if square > i:
                    break
                if dp[i - square] + 1 < dp[i]:
                    dp[i] = dp[i - square] + 1
                    parent[i] = j

        # Reconstruct path
        result_squares = []
        current = n

        while current > 0:
            square_idx = parent[current]
            square_value = squares[square_idx]
            result_squares.append(square_value)
            current -= square_value

        return dp[n], sorted(result_squares)

    def word_break_ii_with_optimization(self, s: str, wordDict: List[str]) -> List[str]:
        """
        LeetCode 140 - Word Break II (Hard)

        Return all possible sentences by breaking s using dictionary.
        Optimized with memoization and pruning.

        Algorithm:
        1. Use DP with memoization
        2. Build trie for efficient word matching
        3. Prune impossible branches early

        Time: O(n³) average, Space: O(n³)

        Example:
        s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
        Output: ["cats and dog", "cat sand dog"]
        """

        # Build trie for efficient lookup
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.is_word = False

        root = TrieNode()
        for word in wordDict:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True

        # Check if word break is possible (optimization)
        dp_possible = [False] * (len(s) + 1)
        dp_possible[0] = True

        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp_possible[j] and s[j:i] in wordDict:
                    dp_possible[i] = True
                    break

        if not dp_possible[len(s)]:
            return []

        # Memoized DFS
        memo = {}

        def dfs(start: int) -> List[str]:
            if start in memo:
                return memo[start]

            if start == len(s):
                return [""]

            result = []
            node = root

            for end in range(start, len(s)):
                char = s[end]
                if char not in node.children:
                    break

                node = node.children[char]
                if node.is_word:
                    word = s[start:end + 1]
                    for sentence in dfs(end + 1):
                        if sentence:
                            result.append(word + " " + sentence)
                        else:
                            result.append(word)

            memo[start] = result
            return result

        return dfs(0)

    def combination_sum_iv_with_constraints(self, nums: List[int], target: int,
                                            min_length: int, max_length: int) -> int:
        """
        LeetCode 377 Extension - Combination Sum IV with Length Constraints (Hard)

        Count combinations that sum to target with length constraints.
        Different sequences counted as different combinations.

        Algorithm:
        1. 2D DP: dp[i][j] = combinations of length j summing to i
        2. Ensure length constraints are met
        3. Use modulo for large numbers

        Time: O(target * len(nums) * max_length), Space: O(target * max_length)

        Example:
        nums = [1,2,3], target = 4, min_length = 2, max_length = 3
        Output: 6 (combinations: [1,3], [3,1], [2,2], [1,1,2], [1,2,1], [2,1,1])
        """
        MOD = 10 ** 9 + 7

        # dp[i][j] = number of combinations of length j that sum to i
        dp = [[0] * (max_length + 1) for _ in range(target + 1)]
        dp[0][0] = 1

        for i in range(1, target + 1):
            for length in range(1, min(i, max_length) + 1):
                for num in nums:
                    if num <= i and length >= 1:
                        dp[i][length] = (dp[i][length] + dp[i - num][length - 1]) % MOD

        # Sum combinations of valid lengths
        result = 0
        for length in range(min_length, max_length + 1):
            result = (result + dp[target][length]) % MOD

        return result

    def minimum_cost_for_tickets_with_schedule(self, days: List[int], costs: List[int]) -> Tuple[
        int, List[Tuple[int, int]]]:
        """
        LeetCode 983 Extension - Minimum Cost For Tickets with Schedule (Hard)

        Find minimum cost to cover all travel days.
        Extended: Return which pass to buy on which days.

        Algorithm:
        1. DP where dp[i] = min cost up to day i
        2. Track which pass was bought
        3. Reconstruct purchase schedule

        Time: O(n), Space: O(n) where n is last day

        Example:
        days = [1,4,6,7,8,20], costs = [2,7,15]
        Output: (11, [(1,7), (20,2)]) - buy 7-day pass on day 1, 1-day on day 20
        """
        if not days:
            return 0, []

        last_day = days[-1]
        dp = [0] * (last_day + 1)
        parent = [(-1, -1)] * (last_day + 1)  # (day_bought, pass_type)
        day_set = set(days)

        for i in range(1, last_day + 1):
            if i not in day_set:
                dp[i] = dp[i - 1]
                parent[i] = parent[i - 1]
            else:
                # Option 1: Buy 1-day pass
                cost1 = dp[i - 1] + costs[0]

                # Option 2: Buy 7-day pass
                cost7 = dp[max(0, i - 7)] + costs[1]

                # Option 3: Buy 30-day pass
                cost30 = dp[max(0, i - 30)] + costs[2]

                # Choose minimum
                if cost1 <= cost7 and cost1 <= cost30:
                    dp[i] = cost1
                    parent[i] = (i, 1)
                elif cost7 <= cost30:
                    dp[i] = cost7
                    parent[i] = (i, 7)
                else:
                    dp[i] = cost30
                    parent[i] = (i, 30)

        # Reconstruct schedule
        schedule = []
        i = last_day

        while i > 0:
            if parent[i][0] != -1 and parent[i][0] <= i:
                day_bought, pass_duration = parent[i]
                if i in day_set:
                    schedule.append((day_bought, costs[0] if pass_duration == 1
                    else costs[1] if pass_duration == 7 else costs[2]))
                i = day_bought - pass_duration
            else:
                i -= 1

        schedule.reverse()

        # Merge consecutive purchases on same day
        merged_schedule = []
        for day, cost in schedule:
            if not merged_schedule or merged_schedule[-1][0] != day:
                merged_schedule.append((day, cost))

        return dp[last_day], merged_schedule

    def rod_cutting_with_pattern(self, n: int, prices: List[int]) -> Tuple[int, List[int], float]:
        """
        Custom - Rod Cutting with Pattern Analysis (Hard)

        Cut rod to maximize revenue. Analyze cutting pattern.
        Extended: Return revenue, cuts, and waste percentage.

        Algorithm:
        1. DP for maximum revenue
        2. Track cutting decisions
        3. Analyze pattern for insights

        Time: O(n²), Space: O(n)

        Example:
        n = 8, prices = [1,5,8,9,10,17,17,20]
        Output: (22, [2,6], 0.0) - cut into pieces of length 2 and 6
        """
        # Extend prices if needed
        if len(prices) < n:
            prices = prices + [prices[-1]] * (n - len(prices))

        dp = [0] * (n + 1)
        parent = [-1] * (n + 1)

        for i in range(1, n + 1):
            max_val = 0
            best_cut = -1

            # Try all possible first cuts
            for j in range(1, i + 1):
                if j <= len(prices):
                    val = prices[j - 1] + dp[i - j]
                    if val > max_val:
                        max_val = val
                        best_cut = j

            dp[i] = max_val
            parent[i] = best_cut

        # Reconstruct cuts
        cuts = []
        remaining = n

        while remaining > 0:
            cut = parent[remaining]
            cuts.append(cut)
            remaining -= cut

        # Calculate waste (if any pieces are sold below optimal)
        total_length = sum(cuts)
        waste_percentage = 0.0  # No waste in basic rod cutting

        return dp[n], sorted(cuts), waste_percentage

    def unbounded_knapsack_with_item_limit(self, W: int, weights: List[int],
                                           values: List[int], limits: List[int]) -> Tuple[int, Dict[int, int]]:
        """
        Custom - Unbounded Knapsack with Item Limits (Hard)

        Each item can be used multiple times but with a limit.
        Return maximum value and items used.

        Algorithm:
        1. Modified unbounded knapsack
        2. Track item usage counts
        3. Respect individual item limits

        Time: O(W * n * max_limit), Space: O(W)

        Example:
        W = 10, weights = [2,3,5], values = [1,2,3], limits = [3,2,1]
        Output: (6, {0: 3, 1: 1}) - use item 0 three times, item 1 once
        """
        n = len(weights)
        dp = [0] * (W + 1)
        parent = [[] for _ in range(W + 1)]

        for w in range(1, W + 1):
            for i in range(n):
                if weights[i] <= w:
                    # Try using item i (up to its limit)
                    for count in range(1, min(limits[i], w // weights[i]) + 1):
                        total_weight = count * weights[i]
                        total_value = count * values[i]

                        if total_weight <= w:
                            new_value = dp[w - total_weight] + total_value
                            if new_value > dp[w]:
                                dp[w] = new_value
                                parent[w] = parent[w - total_weight] + [(i, count)]

        # Consolidate item usage
        item_usage = defaultdict(int)
        for item_idx, count in parent[W]:
            item_usage[item_idx] += count

        return dp[W], dict(item_usage)

    def dice_combinations_with_constraints(self, n: int, k: int, target: int) -> Tuple[int, List[List[int]]]:
        """
        Custom - Dice Combinations with Constraints (Hard)

        Roll n dice with k faces (1 to k) to get sum = target.
        Find number of ways and sample combinations.

        Algorithm:
        1. DP with state tracking
        2. Generate sample combinations
        3. Handle large numbers with modulo

        Time: O(n * target * k), Space: O(n * target)

        Example:
        n = 2, k = 6, target = 7
        Output: (6, [[1,6], [2,5], [3,4], [4,3], [5,2], [6,1]])
        """
        MOD = 10 ** 9 + 7

        # dp[i][j] = ways to get sum j using i dice
        dp = [[0] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = 1

        for i in range(1, n + 1):
            for j in range(i, min(i * k, target) + 1):
                for face in range(1, min(k, j) + 1):
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - face]) % MOD

        # Generate sample combinations (limit to prevent memory issues)
        def generate_combinations(dice_left: int, current_sum: int, path: List[int]) -> List[List[int]]:
            if dice_left == 0:
                return [path] if current_sum == 0 else []

            if current_sum <= 0 or len(combinations) >= 100:  # Limit samples
                return []

            result = []
            for face in range(1, min(k + 1, current_sum + 1)):
                result.extend(generate_combinations(dice_left - 1, current_sum - face, path + [face]))

            return result

        combinations = generate_combinations(n, target, [])

        return dp[n][target], combinations[:10]  # Return first 10 combinations

    def painting_fence_with_k_colors(self, n: int, k: int) -> Tuple[int, int, int]:
        """
        LeetCode 276 Extension - Paint Fence With K Colors Analysis (Hard)

        Paint n fence posts with k colors, no more than 2 adjacent same color.
        Extended: Return total ways, ways with max color diversity, min diversity.

        Algorithm:
        1. DP tracking same/different color patterns
        2. Analyze color diversity
        3. Count patterns with specific properties

        Time: O(n), Space: O(n)

        Example:
        n = 3, k = 2
        Output: (6, 2, 4) - total ways, max diversity ways, min diversity ways
        """
        if n == 0:
            return 0, 0, 0
        if n == 1:
            return k, k, 0

        # dp[i] = ways to paint i posts
        # same[i] = ways where post i same color as i-1
        # diff[i] = ways where post i different color from i-1
        same = [0] * n
        diff = [0] * n

        # Base cases
        same[1] = k  # First two posts same color
        diff[1] = k * (k - 1)  # First two posts different colors

        for i in range(2, n):
            same[i] = diff[i - 1]  # Can only have same if previous were different
            diff[i] = (same[i - 1] + diff[i - 1]) * (k - 1)

        total_ways = same[n - 1] + diff[n - 1]

        # Max diversity: minimize consecutive same colors
        max_diversity_ways = diff[n - 1]  # All different is not possible for n > k

        # Min diversity: maximize consecutive same colors
        min_diversity_ways = same[n - 1]

        return total_ways, max_diversity_ways, min_diversity_ways


# Example usage and testing
if __name__ == "__main__":
    solver = UnboundedKnapsackHard()

    # Test 1: Coin Change with Usage
    print("1. Coin Change with Coin Usage:")
    coins = [1, 2, 5]
    amount = 11
    min_coins, usage = solver.coin_change_with_minimum_coins_used(coins, amount)
    print(f"   Coins: {coins}, Amount: {amount}")
    print(f"   Min coins: {min_coins}, Usage: {usage}")
    print()

    # Test 2: Integer Break
    print("2. Integer Break with Factors:")
    n = 10
    product, factors = solver.integer_break_with_factors(n)
    print(f"   n = {n}")
    print(f"   Max product: {product}, Factors: {factors}")
    print()

    # Test 3: Perfect Squares
    print("3. Perfect Squares with Path:")
    n = 12
    count, squares = solver.perfect_squares_with_path(n)
    print(f"   n = {n}")
    print(f"   Min squares: {count}, Squares used: {squares}")
    print()

    # Test 4: Minimum Cost for Tickets
    print("4. Minimum Cost for Tickets with Schedule:")
    days = [1, 4, 6, 7, 8, 20]
    costs = [2, 7, 15]
    min_cost, schedule = solver.minimum_cost_for_tickets_with_schedule(days, costs)
    print(f"   Days: {days}, Costs: {costs}")
    print(f"   Min cost: {min_cost}, Schedule: {schedule}")