"""
Pattern 18: Palindromic Subsequence (Dynamic Programming) - 10 Hard Problems
===========================================================================

The Palindromic Subsequence pattern uses dynamic programming to solve problems
involving palindromes in strings. This includes finding longest palindromic
subsequences, counting palindromes, and transforming strings into palindromes.

Key Concepts:
- State often represents substring boundaries (i, j)
- Consider matching/non-matching characters at boundaries
- Often uses 2D DP table for substring problems
- Can be optimized using expand-around-center for some problems

Time Complexity: Usually O(n²) for subsequence problems
Space Complexity: O(n²), can sometimes be optimized to O(n)
"""

from typing import List, Tuple, Set, Dict
from functools import lru_cache


class PalindromicSubsequenceHard:

    def longest_palindromic_subsequence_with_k_changes(self, s: str, k: int) -> Tuple[int, str]:
        """
        Custom Hard - Longest Palindromic Subsequence with K Character Changes

        Find longest palindromic subsequence after changing at most k characters.
        Return length and one possible palindrome.

        Algorithm:
        1. DP with state (left, right, changes_used)
        2. Try matching characters or changing one to match
        3. Reconstruct palindrome from DP decisions

        Time: O(n² * k), Space: O(n² * k)

        Example:
        s = "abcbea", k = 1
        Output: (5, "abcba") - change 'e' to 'c'
        """
        n = len(s)

        # dp[i][j][changes] = (length, decisions) for s[i:j+1] with 'changes' used
        @lru_cache(maxsize=None)
        def dp(left: int, right: int, changes: int) -> Tuple[int, List[Tuple[int, int, str]]]:
            if left > right:
                return 0, []
            if left == right:
                return 1, [(left, left, s[left])]

            if s[left] == s[right]:
                # Characters match, include both
                length, decisions = dp(left + 1, right - 1, changes)
                return length + 2, [(left, right, s[left])] + decisions
            else:
                # Characters don't match
                results = []

                # Option 1: Skip left character
                length1, decisions1 = dp(left + 1, right, changes)
                results.append((length1, decisions1))

                # Option 2: Skip right character
                length2, decisions2 = dp(left, right - 1, changes)
                results.append((length2, decisions2))

                # Option 3: Change one character to match (if changes available)
                if changes > 0:
                    # Change left to match right
                    length3, decisions3 = dp(left + 1, right - 1, changes - 1)
                    results.append((length3 + 2, [(left, right, s[right])] + decisions3))

                    # Change right to match left
                    length4, decisions4 = dp(left + 1, right - 1, changes - 1)
                    results.append((length4 + 2, [(left, right, s[left])] + decisions4))

                # Return best option
                return max(results, key=lambda x: x[0])

        length, decisions = dp(0, n - 1, k)

        # Reconstruct palindrome
        palindrome_chars = [''] * n
        for left, right, char in decisions:
            palindrome_chars[left] = char
            palindrome_chars[right] = char

        # Build palindrome string
        palindrome = ''.join(c for c in palindrome_chars if c)

        return length, palindrome

    def count_palindromic_subsequences_mod(self, s: str) -> int:
        """
        LeetCode 730 - Count Different Palindromic Subsequences (Hard)

        Count distinct palindromic subsequences modulo 10^9+7.

        Algorithm:
        1. DP where dp[i][j][c] = count of palindromes in s[i:j+1] starting/ending with c
        2. Handle duplicate characters carefully
        3. Use modulo arithmetic throughout

        Time: O(n² * 26), Space: O(n² * 26)

        Example:
        s = "bccb"
        Output: 6 (palindromes: "b", "c", "bb", "cc", "bcb", "bccb")
        """
        MOD = 10 ** 9 + 7
        n = len(s)

        # dp[length][start][char] = count of palindromes of given length starting at position
        dp = [[[0] * 26 for _ in range(n)] for _ in range(n + 1)]

        # Base case: single characters
        for i in range(n):
            dp[1][i][ord(s[i]) - ord('a')] = 1

        # Fill for increasing lengths
        for length in range(2, n + 1):
            for start in range(n - length + 1):
                end = start + length - 1

                for c in range(26):
                    char = chr(c + ord('a'))

                    if s[start] == char and s[end] == char:
                        # Both ends match the character
                        dp[length][start][c] = 2  # Just the character itself repeated

                        # Add all palindromes that can be wrapped
                        for inner_c in range(26):
                            if start + 1 <= end - 1:
                                dp[length][start][c] += dp[length - 2][start + 1][inner_c]
                                dp[length][start][c] %= MOD
                    elif s[start] == char:
                        # Only start matches
                        if start + 1 <= end:
                            dp[length][start][c] = dp[length - 1][start + 1][c]
                    elif s[end] == char:
                        # Only end matches
                        if start <= end - 1:
                            dp[length][start][c] = dp[length - 1][start][c]
                    else:
                        # Neither end matches
                        if start + 1 <= end - 1:
                            # Take from middle, adjust for overcounting
                            dp[length][start][c] = (dp[length - 1][start + 1][c] +
                                                    dp[length - 1][start][c] -
                                                    dp[length - 2][start + 1][c]) % MOD

        # Sum all palindromes
        result = 0
        for length in range(1, n + 1):
            for start in range(n - length + 1):
                for c in range(26):
                    result = (result + dp[length][start][c]) % MOD

        return result

    def minimum_insertion_to_make_palindrome_with_positions(self, s: str) -> Tuple[int, str, List[int]]:
        """
        LeetCode 1312 Extension - Minimum Insertions with Positions (Hard)

        Find minimum insertions to make string palindrome.
        Extended: Return resulting palindrome and insertion positions.

        Algorithm:
        1. DP based on longest common subsequence of s and reverse(s)
        2. Track decisions for reconstruction
        3. Build palindrome by inserting characters

        Time: O(n²), Space: O(n²)

        Example:
        s = "mbadm"
        Output: (2, "mbdadbm", [2, 5]) - insert 'd' at 2, 'b' at 5
        """
        n = len(s)

        # Find longest palindromic subsequence length
        dp = [[0] * n for _ in range(n)]

        # Base case: single characters
        for i in range(n):
            dp[i][i] = 1

        # Fill DP table
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1

                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

        # Minimum insertions = n - longest palindromic subsequence
        lps_length = dp[0][n - 1]
        min_insertions = n - lps_length

        # Reconstruct palindrome
        result = []
        positions = []
        i, j = 0, n - 1
        left_insertions = 0

        while i <= j:
            if i == j:
                result.append(s[i])
                break

            if s[i] == s[j]:
                result.append(s[i])
                i += 1
                j -= 1
            elif dp[i + 1][j] > dp[i][j - 1]:
                # Insert s[j] at the beginning
                result.append(s[j])
                positions.append(len(result) - 1 + left_insertions)
                left_insertions += 1
                j -= 1
            else:
                # Will insert s[i] at the end (symmetric position)
                i += 1

        # Build full palindrome
        if i > j:
            # Even length palindrome
            full_palindrome = result + result[::-1]
        else:
            # Odd length palindrome
            full_palindrome = result[:-1] + result[::-1]

        return min_insertions, ''.join(full_palindrome), positions

    def palindrome_partitioning_min_cuts_with_partitions(self, s: str) -> Tuple[int, List[str]]:
        """
        LeetCode 132 Extension - Palindrome Partitioning II with Partitions (Hard)

        Find minimum cuts for palindrome partitioning.
        Extended: Return one optimal partition.

        Algorithm:
        1. Precompute palindrome table
        2. DP for minimum cuts
        3. Backtrack to find partition

        Time: O(n²), Space: O(n²)

        Example:
        s = "aabbc"
        Output: (2, ["aa", "bb", "c"]) - 2 cuts needed
        """
        n = len(s)

        # Precompute palindrome table
        is_palindrome = [[False] * n for _ in range(n)]

        for i in range(n):
            is_palindrome[i][i] = True

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j]:
                    is_palindrome[i][j] = length == 2 or is_palindrome[i + 1][j - 1]

        # DP for minimum cuts
        dp = [float('inf')] * n
        parent = [-1] * n

        for i in range(n):
            if is_palindrome[0][i]:
                dp[i] = 0
            else:
                for j in range(i):
                    if is_palindrome[j + 1][i] and dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
                        parent[i] = j

        # Backtrack to find partition
        partitions = []
        i = n - 1

        while i >= 0:
            if parent[i] == -1:
                partitions.append(s[0:i + 1])
                break
            else:
                partitions.append(s[parent[i] + 1:i + 1])
                i = parent[i]

        partitions.reverse()

        return dp[n - 1], partitions

    def longest_palindrome_by_concatenating_with_order(self, words: List[str]) -> Tuple[int, List[str]]:
        """
        LeetCode 2131 Extension - Longest Palindrome by Concatenating with Order (Hard)

        Build longest palindrome by concatenating 2-letter words.
        Extended: Return the actual palindrome construction order.

        Algorithm:
        1. Group words by their properties (self-palindrome vs pairs)
        2. Greedily select pairs and at most one center palindrome
        3. Build optimal ordering

        Time: O(n), Space: O(n)

        Example:
        words = ["lc","cl","gg"]
        Output: (6, ["lc","gg","cl"]) - "lcggcl"
        """
        from collections import Counter

        word_count = Counter(words)
        used = set()
        left_part = []
        center = None
        length = 0

        for word in word_count:
            if word in used:
                continue

            reverse_word = word[1] + word[0]

            if word == reverse_word:
                # Self-palindrome
                pairs = word_count[word] // 2
                if pairs > 0:
                    for _ in range(pairs):
                        left_part.append(word)
                    length += pairs * 4

                # Keep one for potential center
                if word_count[word] % 2 == 1 and center is None:
                    center = word
                    length += 2
            else:
                # Non-self-palindrome
                if reverse_word in word_count:
                    pairs = min(word_count[word], word_count[reverse_word])
                    for _ in range(pairs):
                        left_part.append(word)
                    length += pairs * 4
                    used.add(word)
                    used.add(reverse_word)

        # Build final palindrome order
        result_order = []

        # Add left part
        result_order.extend(left_part)

        # Add center if exists
        if center:
            result_order.append(center)

        # Add right part (reverse of left)
        for word in reversed(left_part):
            result_order.append(word[1] + word[0])

        return length, result_order

    def shortest_palindrome_by_adding_in_front(self, s: str) -> str:
        """
        LeetCode 214 - Shortest Palindrome (Hard)

        Find shortest palindrome by adding characters in front.

        Algorithm:
        1. Find longest prefix that is also a palindrome
        2. Use KMP failure function on s + '#' + reverse(s)
        3. Add necessary characters from the end to front

        Time: O(n), Space: O(n)

        Example:
        s = "aacecaaa"
        Output: "aaacecaaa"
        """
        if not s:
            return s

        # Create combined string
        combined = s + '#' + s[::-1]
        n = len(combined)

        # Build KMP failure function
        lps = [0] * n

        for i in range(1, n):
            j = lps[i - 1]

            while j > 0 and combined[i] != combined[j]:
                j = lps[j - 1]

            if combined[i] == combined[j]:
                j += 1

            lps[i] = j

        # Length of longest palindrome prefix
        palindrome_len = lps[-1]

        # Add remaining characters in reverse to front
        return s[palindrome_len:][::-1] + s

    def palindromic_substrings_with_modifications(self, s: str, k: int) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Extension of LeetCode 647 - Count Palindromic Substrings with K Modifications (Hard)

        Count palindromic substrings after at most k character modifications.
        Extended: Return count and sample palindrome positions.

        Algorithm:
        1. For each center, expand and track mismatches
        2. Use sliding window to count valid palindromes
        3. Sample positions for demonstration

        Time: O(n² * k), Space: O(n)

        Example:
        s = "abc", k = 1
        Output: (7, [(0,0), (1,1), (2,2), (0,1), (1,2), (0,2)])
        """
        n = len(s)
        total_count = 0
        sample_positions = []

        # Try each possible center
        for center in range(2 * n - 1):
            left = center // 2
            right = left + center % 2
            mismatches = []

            # Expand around center and track mismatches
            while left >= 0 and right < n:
                if s[left] != s[right]:
                    mismatches.append((left, right))

                # Count palindromes using at most k modifications
                if len(mismatches) <= k:
                    total_count += 1
                    if len(sample_positions) < 20:  # Limit samples
                        sample_positions.append((left, right))
                else:
                    # Can still have palindromes by excluding early mismatches
                    for i in range(len(mismatches) - k):
                        exclude_left, exclude_right = mismatches[i]
                        if exclude_left < left and exclude_right > right:
                            total_count += 1
                            if len(sample_positions) < 20:
                                sample_positions.append((exclude_left + 1, exclude_right - 1))

                left -= 1
                right += 1

                # Stop if too many mismatches
                if len(mismatches) > k + 5:  # Optimization
                    break

        return total_count, sample_positions

    def make_string_palindrome_with_operations(self, s: str) -> Tuple[int, List[Tuple[str, int, int]]]:
        """
        Custom Hard - Make String Palindrome with Insert/Delete/Replace

        Find minimum operations to make string palindrome.
        Operations: insert(char, pos), delete(pos), replace(pos, char)

        Algorithm:
        1. Use edit distance DP variant
        2. Track operations for reconstruction
        3. Favor replacements over insert/delete pairs

        Time: O(n²), Space: O(n²)

        Example:
        s = "abc"
        Output: (1, [("replace", 2, 'a')]) - change to "aba"
        """
        n = len(s)
        rev_s = s[::-1]

        # dp[i][j] = (min_ops, last_operation) to make s[0:i] match rev_s[0:j]
        dp = [[float('inf')] * (n + 1) for _ in range(n + 1)]
        parent = [[None] * (n + 1) for _ in range(n + 1)]

        # Base cases
        for i in range(n + 1):
            dp[i][0] = i  # Delete all
            if i > 0:
                parent[i][0] = ("delete", i - 1, 0)

        for j in range(n + 1):
            dp[0][j] = j  # Insert all
            if j > 0:
                parent[0][j] = ("insert", 0, j - 1)

        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if s[i - 1] == rev_s[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                    parent[i][j] = ("match", i - 1, j - 1)
                else:
                    # Replace
                    if dp[i - 1][j - 1] + 1 < dp[i][j]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                        parent[i][j] = ("replace", i - 1, j - 1)

                    # Delete
                    if dp[i - 1][j] + 1 < dp[i][j]:
                        dp[i][j] = dp[i - 1][j] + 1
                        parent[i][j] = ("delete", i - 1, j)

                    # Insert
                    if dp[i][j - 1] + 1 < dp[i][j]:
                        dp[i][j] = dp[i][j - 1] + 1
                        parent[i][j] = ("insert", i, j - 1)

        # Backtrack to find operations
        operations = []
        i, j = n, n

        while i > 0 or j > 0:
            if parent[i][j] is None:
                break

            op_type, new_i, new_j = parent[i][j]

            if op_type == "replace":
                operations.append(("replace", i - 1, rev_s[j - 1]))
            elif op_type == "delete":
                operations.append(("delete", i - 1, None))
            elif op_type == "insert":
                operations.append(("insert", i, rev_s[j - 1]))

            i, j = new_i, new_j

        operations.reverse()

        # Simplify operations
        simplified = []
        offset = 0

        for op_type, pos, char in operations:
            if op_type == "delete":
                simplified.append((op_type, pos + offset, None))
                offset -= 1
            elif op_type == "insert":
                simplified.append((op_type, pos + offset, char))
                offset += 1
            else:  # replace
                simplified.append((op_type, pos + offset, char))

        return dp[n][n] // 2, simplified[:dp[n][n] // 2]

    def maximum_palindromes_after_change(self, s: str, k: int) -> int:
        """
        HackerRank - Maximum Palindromes After Change (Hard)

        Change at most k characters to maximize number of palindromic substrings.

        Algorithm:
        1. Greedy approach: change characters to create more palindromes
        2. Priority: positions that create most new palindromes
        3. Use frequency analysis

        Time: O(n² * k), Space: O(n)

        Example:
        s = "abcb", k = 1
        Output: 7 (change 'c' to 'b' → "abbb")
        """
        n = len(s)
        chars = list(s)

        # Count current palindromes
        def count_palindromes(string):
            count = 0
            n = len(string)

            # Expand around centers
            for center in range(2 * n - 1):
                left = center // 2
                right = left + center % 2

                while left >= 0 and right < n and string[left] == string[right]:
                    count += 1
                    left -= 1
                    right += 1

            return count

        max_palindromes = count_palindromes(s)

        # Try changing each position to each character
        for _ in range(k):
            best_change = None
            best_count = max_palindromes

            for i in range(n):
                original = chars[i]

                for new_char in 'abcdefghijklmnopqrstuvwxyz':
                    if new_char == original:
                        continue

                    chars[i] = new_char
                    new_count = count_palindromes(chars)

                    if new_count > best_count:
                        best_count = new_count
                        best_change = (i, new_char)

                    chars[i] = original

            if best_change:
                i, new_char = best_change
                chars[i] = new_char
                max_palindromes = best_count
            else:
                break  # No improvement possible

        return max_palindromes


# Example usage and testing
if __name__ == "__main__":
    solver = PalindromicSubsequenceHard()

    # Test 1: Longest Palindromic Subsequence with K Changes
    print("1. Longest Palindromic Subsequence with K Changes:")
    s = "abcbea"
    k = 1
    length, palindrome = solver.longest_palindromic_subsequence_with_k_changes(s, k)
    print(f"   Input: s='{s}', k={k}")
    print(f"   Output: Length={length}, Palindrome='{palindrome}'")
    print()

    # Test 2: Minimum Insertions with Positions
    print("2. Minimum Insertions to Make Palindrome:")
    s = "mbadm"
    insertions, result, positions = solver.minimum_insertion_to_make_palindrome_with_positions(s)
    print(f"   Input: s='{s}'")
    print(f"   Output: Insertions={insertions}, Result='{result}'")
    print(f"   Positions: {positions}")
    print()

    # Test 3: Palindrome Partitioning with Min Cuts
    print("3. Palindrome Partitioning II:")
    s = "aabbc"
    cuts, partitions = solver.palindrome_partitioning_min_cuts_with_partitions(s)
    print(f"   Input: s='{s}'")
    print(f"   Output: Cuts={cuts}, Partitions={partitions}")
    print()

    # Test 4: Shortest Palindrome
    print("4. Shortest Palindrome:")
    s = "aacecaaa"
    result = solver.shortest_palindrome_by_adding_in_front(s)
    print(f"   Input: s='{s}'")
    print(f"   Output: '{result}'")