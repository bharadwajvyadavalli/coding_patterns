"""
Pattern 10: Subsets - 10 Hard Problems
======================================

The Subsets pattern generates all possible subsets, combinations, or permutations
of a set of elements. This pattern is essential for solving combinatorial problems
and is often implemented using backtracking or bit manipulation.

Key Concepts:
- Use backtracking to explore all possibilities
- Bit manipulation for subset generation
- Handle duplicates with sorting and skipping
- Optimize with pruning and memoization

Time Complexity: Usually O(2^n) for subsets, O(n!) for permutations
Space Complexity: O(n) for recursion depth, plus output space
"""

from typing import List, Set, Tuple
from collections import defaultdict, Counter
import itertools


class SubsetsHard:

    def subsets_with_constraints(self, nums: List[int], min_size: int,
                                 max_size: int, target_sum: int) -> List[List[int]]:
        """
        Custom Hard - Subsets with Multiple Constraints

        Find all subsets that:
        1. Have size between min_size and max_size (inclusive)
        2. Sum equals target_sum
        3. Handle duplicates in input array

        Algorithm:
        1. Sort array to handle duplicates
        2. Use backtracking with pruning
        3. Skip duplicates at same recursion level
        4. Prune based on sum and size constraints

        Time: O(2^n), Space: O(n)

        Example:
        nums = [1,2,2,3,3,4], min_size=2, max_size=4, target_sum=8
        Output: [[1,3,4], [2,2,4], [2,3,3], [1,2,2,3]]
        """
        nums.sort()
        result = []

        def backtrack(start: int, path: List[int], current_sum: int):
            # Check if current subset meets criteria
            if min_size <= len(path) <= max_size and current_sum == target_sum:
                result.append(path[:])

            # Pruning conditions
            if len(path) >= max_size or current_sum >= target_sum:
                return

            for i in range(start, len(nums)):
                # Skip duplicates at same level
                if i > start and nums[i] == nums[i - 1]:
                    continue

                # Further pruning: if adding this number exceeds target
                if current_sum + nums[i] > target_sum:
                    break  # Since array is sorted, all following will also exceed

                path.append(nums[i])
                backtrack(i + 1, path, current_sum + nums[i])
                path.pop()

        backtrack(0, [], 0)
        return result

    def combination_sum_iv_with_path(self, nums: List[int], target: int) -> Tuple[int, List[List[int]]]:
        """
        LeetCode 377 Extension - Combination Sum IV with Paths (Hard)

        Find number of combinations that sum to target.
        Different sequences are counted as different combinations.
        Extended: Also return sample paths (limit to prevent memory issues).

        Algorithm:
        1. Use DP for counting
        2. Use backtracking to generate sample paths
        3. Optimize with memoization

        Time: O(target * n), Space: O(target)

        Example:
        nums = [1,2,3], target = 4
        Output: (7, [[1,1,1,1], [1,1,2], [1,2,1], [2,1,1], [2,2], [1,3], [3,1]])
        """
        # DP for counting
        dp = [0] * (target + 1)
        dp[0] = 1

        for i in range(1, target + 1):
            for num in nums:
                if i >= num:
                    dp[i] += dp[i - num]

        count = dp[target]

        # Generate sample paths (limit to 100 to avoid memory issues)
        paths = []

        def backtrack(remaining: int, path: List[int]):
            if remaining == 0:
                paths.append(path[:])
                return

            if len(paths) >= 100:  # Limit paths
                return

            for num in nums:
                if num <= remaining:
                    path.append(num)
                    backtrack(remaining - num, path)
                    path.pop()

        backtrack(target, [])
        return count, paths

    def letter_case_permutation_extended(self, s: str) -> Tuple[List[str], int, dict]:
        """
        LeetCode 784 Extension - Letter Case Permutation with Analysis (Hard)

        Generate all permutations by changing case of letters.
        Extended: Count permutations by number of uppercase letters.

        Algorithm:
        1. Use bit manipulation or backtracking
        2. Track uppercase count for each permutation
        3. Group results by uppercase count

        Time: O(2^n * n), Space: O(2^n * n)

        Example:
        s = "a1b2"
        Output: (["a1b2","a1B2","A1b2","A1B2"], 4, {0:1, 1:2, 2:1})
        """
        result = []
        uppercase_counts = defaultdict(int)

        def backtrack(index: int, current: List[str], uppercase_count: int):
            if index == len(s):
                permutation = ''.join(current)
                result.append(permutation)
                uppercase_counts[uppercase_count] += 1
                return

            char = s[index]

            if char.isalpha():
                # Try lowercase
                current.append(char.lower())
                backtrack(index + 1, current, uppercase_count)
                current.pop()

                # Try uppercase
                current.append(char.upper())
                backtrack(index + 1, current, uppercase_count + 1)
                current.pop()
            else:
                # Non-alphabetic character
                current.append(char)
                backtrack(index + 1, current, uppercase_count)
                current.pop()

        backtrack(0, [], 0)
        return result, len(result), dict(uppercase_counts)

    def palindrome_partitioning_optimized(self, s: str) -> List[List[str]]:
        """
        LeetCode 131 - Palindrome Partitioning (Hard Optimization)

        Partition string such that every substring is palindrome.
        Optimized with DP preprocessing and pruning.

        Algorithm:
        1. Precompute palindrome table using DP
        2. Use backtracking with palindrome table
        3. Prune invalid branches early

        Time: O(n * 2^n), Space: O(n^2)

        Example:
        s = "aab"
        Output: [["a","a","b"], ["aa","b"]]
        """
        n = len(s)

        # Precompute palindrome table
        is_palindrome = [[False] * n for _ in range(n)]

        # Every single character is palindrome
        for i in range(n):
            is_palindrome[i][i] = True

        # Check 2-character substrings
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                is_palindrome[i][i + 1] = True

        # Check substrings of length 3 or more
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and is_palindrome[i + 1][j - 1]:
                    is_palindrome[i][j] = True

        # Backtracking with palindrome table
        result = []

        def backtrack(start: int, path: List[str]):
            if start == n:
                result.append(path[:])
                return

            for end in range(start, n):
                if is_palindrome[start][end]:
                    path.append(s[start:end + 1])
                    backtrack(end + 1, path)
                    path.pop()

        backtrack(0, [])
        return result

    def word_break_ii_optimized(self, s: str, wordDict: List[str]) -> List[str]:
        """
        LeetCode 140 - Word Break II (Hard)

        Return all possible sentences by breaking s using dictionary words.
        Optimized with memoization and trie.

        Algorithm:
        1. Build trie for efficient word lookup
        2. Use memoization to avoid recomputation
        3. Backtrack with pruning

        Time: O(n^3), Space: O(n^3)

        Example:
        s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
        Output: ["cats and dog", "cat sand dog"]
        """
        # Build word set for O(1) lookup
        word_set = set(wordDict)
        memo = {}

        def backtrack(start: int) -> List[str]:
            if start in memo:
                return memo[start]

            if start == len(s):
                return [""]

            sentences = []

            for end in range(start + 1, len(s) + 1):
                word = s[start:end]
                if word in word_set:
                    # Get all possible sentences for remaining string
                    sub_sentences = backtrack(end)
                    for sub in sub_sentences:
                        if sub:
                            sentences.append(word + " " + sub)
                        else:
                            sentences.append(word)

            memo[start] = sentences
            return sentences

        return backtrack(0)

    def generate_parentheses_with_constraints(self, n: int, must_include: str = "") -> List[str]:
        """
        LeetCode 22 Extension - Generate Parentheses with Constraints (Hard)

        Generate all valid parentheses combinations.
        Extended: Must include a specific substring pattern.

        Algorithm:
        1. Use backtracking with open/close counts
        2. Check if current path can include required pattern
        3. Prune invalid branches

        Time: O(4^n / sqrt(n)), Space: O(n)

        Example:
        n = 3, must_include = "(())"
        Output: ["((()))", "(()())"] - only those containing "(())"
        """
        result = []

        def is_valid_partial(s: str) -> bool:
            """Check if partial string can lead to valid parentheses."""
            balance = 0
            for char in s:
                if char == '(':
                    balance += 1
                else:
                    balance -= 1
                if balance < 0:
                    return False
            return True

        def can_include_pattern(current: str, remaining_open: int,
                                remaining_close: int) -> bool:
            """Check if pattern can still be included."""
            if must_include in current:
                return True

            # Check if we have enough remaining parentheses
            total_remaining = remaining_open + remaining_close
            if len(must_include) > len(current) + total_remaining:
                return False

            # More sophisticated check could be added here
            return True

        def backtrack(current: str, open_count: int, close_count: int):
            # Base case
            if len(current) == 2 * n:
                if not must_include or must_include in current:
                    result.append(current)
                return

            # Pruning
            if not can_include_pattern(current, n - open_count, n - close_count):
                return

            # Add opening parenthesis
            if open_count < n:
                backtrack(current + "(", open_count + 1, close_count)

            # Add closing parenthesis
            if close_count < open_count:
                backtrack(current + ")", open_count, close_count + 1)

        backtrack("", 0, 0)
        return result

    def permutation_sequence(self, n: int, k: int) -> str:
        """
        LeetCode 60 - Permutation Sequence (Hard)

        Return the kth permutation sequence of numbers 1 to n.
        Use mathematical approach instead of generating all permutations.

        Algorithm:
        1. Use factorial number system
        2. Determine each digit based on k and remaining factorial
        3. Build result incrementally

        Time: O(n^2), Space: O(n)

        Example:
        n = 3, k = 3
        Output: "213" (permutations: "123", "132", "213", ...)
        """
        # Calculate factorials
        factorial = [1]
        for i in range(1, n):
            factorial.append(factorial[-1] * i)

        # Adjust k to 0-indexed
        k -= 1

        # Available digits
        digits = list(range(1, n + 1))
        result = []

        for i in range(n, 0, -1):
            # Determine which digit to use
            index = k // factorial[i - 1]
            result.append(str(digits[index]))
            digits.pop(index)

            # Update k for remaining positions
            k %= factorial[i - 1]

        return ''.join(result)

    def combination_sum_iii_extended(self, k: int, n: int) -> Tuple[List[List[int]], int]:
        """
        LeetCode 216 Extension - Combination Sum III with Analysis (Hard)

        Find all combinations of k numbers that sum to n.
        Use only numbers 1-9, each at most once.
        Extended: Also return total number of recursive calls made.

        Algorithm:
        1. Use backtracking with pruning
        2. Track recursive calls for analysis
        3. Prune based on sum and count constraints

        Time: O(C(9,k)), Space: O(k)

        Example:
        k = 3, n = 7
        Output: ([[1,2,4]], 15) - one combination, 15 recursive calls
        """
        result = []
        call_count = 0

        def backtrack(start: int, path: List[int], remaining_sum: int,
                      remaining_count: int):
            nonlocal call_count
            call_count += 1

            # Base cases
            if remaining_count == 0:
                if remaining_sum == 0:
                    result.append(path[:])
                return

            # Pruning: impossible to reach target
            min_possible = sum(range(start, start + remaining_count))
            max_possible = sum(range(10 - remaining_count, 10))

            if remaining_sum < min_possible or remaining_sum > max_possible:
                return

            # Try each number
            for num in range(start, min(10, remaining_sum + 1)):
                path.append(num)
                backtrack(num + 1, path, remaining_sum - num, remaining_count - 1)
                path.pop()

        backtrack(1, [], n, k)
        return result, call_count

    def beautiful_arrangement_ii(self, n: int, k: int) -> List[int]:
        """
        LeetCode 667 - Beautiful Arrangement II (Hard)

        Construct permutation of 1 to n with exactly k distinct differences
        between adjacent elements.

        Algorithm:
        1. Use constructive approach
        2. Create k distinct differences using alternating pattern
        3. Fill remaining with consecutive numbers

        Time: O(n), Space: O(1)

        Example:
        n = 3, k = 2
        Output: [1,3,2] (differences: |1-3|=2, |3-2|=1)
        """
        result = []

        # Create k distinct differences
        left, right = 1, n

        for i in range(k):
            if i % 2 == 0:
                result.append(left)
                left += 1
            else:
                result.append(right)
                right -= 1

        # Fill remaining positions
        if k % 2 == 0:
            # Last added was from right, continue from right
            while right >= left:
                result.append(right)
                right -= 1
        else:
            # Last added was from left, continue from left
            while left <= right:
                result.append(left)
                left += 1

        return result

    def gray_code_with_explanation(self, n: int) -> Tuple[List[int], List[str]]:
        """
        LeetCode 89 Extension - Gray Code with Binary Representation (Hard)

        Generate n-bit Gray code sequence.
        Extended: Also return binary string representation.

        Algorithm:
        1. Use reflection method
        2. For n-bit code, reflect (n-1)-bit code and add bit
        3. Convert to decimal and binary strings

        Time: O(2^n), Space: O(2^n)

        Example:
        n = 2
        Output: ([0,1,3,2], ["00","01","11","10"])
        """
        if n == 0:
            return [0], ["0"]

        # Generate using reflection
        gray_code = [0]

        for i in range(n):
            # Reflect existing codes and add 2^i
            size = len(gray_code)
            for j in range(size - 1, -1, -1):
                gray_code.append(gray_code[j] | (1 << i))

        # Generate binary representations
        binary_strings = []
        for code in gray_code:
            binary_strings.append(format(code, f'0{n}b'))

        return gray_code, binary_strings


# Example usage and testing
if __name__ == "__main__":
    solver = SubsetsHard()

    # Test 1: Subsets with Constraints
    print("1. Subsets with Multiple Constraints:")
    nums = [1, 2, 2, 3, 3, 4]
    print(f"   Input: nums={nums}, min_size=2, max_size=4, target_sum=8")
    result = solver.subsets_with_constraints(nums, 2, 4, 8)
    print(f"   Output: {result}")
    print()

    # Test 2: Combination Sum IV with Paths
    print("2. Combination Sum IV with Paths:")
    nums = [1, 2, 3]
    target = 4
    print(f"   Input: nums={nums}, target={target}")
    count, paths = solver.combination_sum_iv_with_path(nums, target)
    print(f"   Output: Count={count}, Sample paths={paths[:5]}...")
    print()

    # Test 3: Palindrome Partitioning
    print("3. Palindrome Partitioning:")
    s = "aab"
    print(f"   Input: s='{s}'")
    result = solver.palindrome_partitioning_optimized(s)
    print(f"   Output: {result}")
    print()

    # Test 4: Gray Code
    print("4. Gray Code with Binary:")
    n = 3
    print(f"   Input: n={n}")
    codes, binary = solver.gray_code_with_explanation(n)
    print(f"   Output: Codes={codes}")
    print(f"           Binary={binary}")