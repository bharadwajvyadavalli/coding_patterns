"""
Pattern 19: Longest Common Substring/Subsequence (DP) - 10 Hard Problems
========================================================================

The Longest Common Substring/Subsequence pattern finds the longest sequence
that appears in two or more strings. This pattern is fundamental for string
comparison, diff algorithms, and sequence alignment problems.

Key Concepts:
- Subsequence: characters need not be contiguous
- Substring: characters must be contiguous
- 2D DP table where dp[i][j] represents solution for prefixes
- Can be extended to multiple strings and weighted versions

Time Complexity: Usually O(m*n) for two strings
Space Complexity: O(m*n), can be optimized to O(min(m,n))
"""

from typing import List, Tuple, Dict, Set
from collections import defaultdict
import bisect


class LongestCommonSubstringHard:

    def longest_common_subsequence_of_three(self, text1: str, text2: str, text3: str) -> Tuple[int, str]:
        """
        Extension of LeetCode 1143 - LCS of Three Strings (Hard)

        Find longest common subsequence of three strings.
        Return length and the actual subsequence.

        Algorithm:
        1. 3D DP where dp[i][j][k] = LCS of first i,j,k characters
        2. Track parent pointers for reconstruction
        3. Handle three-way character matching

        Time: O(n³), Space: O(n³)

        Example:
        text1 = "abcd", text2 = "acbde", text3 = "aced"
        Output: (3, "acd")
        """
        m, n, p = len(text1), len(text2), len(text3)

        # 3D DP table
        dp = [[[0] * (p + 1) for _ in range(n + 1)] for _ in range(m + 1)]

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                for k in range(1, p + 1):
                    if text1[i - 1] == text2[j - 1] == text3[k - 1]:
                        dp[i][j][k] = dp[i - 1][j - 1][k - 1] + 1
                    else:
                        dp[i][j][k] = max(
                            dp[i - 1][j][k],
                            dp[i][j - 1][k],
                            dp[i][j][k - 1]
                        )

        # Reconstruct LCS
        lcs = []
        i, j, k = m, n, p

        while i > 0 and j > 0 and k > 0:
            if text1[i - 1] == text2[j - 1] == text3[k - 1]:
                lcs.append(text1[i - 1])
                i -= 1
                j -= 1
                k -= 1
            elif dp[i - 1][j][k] >= dp[i][j - 1][k] and dp[i - 1][j][k] >= dp[i][j][k - 1]:
                i -= 1
            elif dp[i][j - 1][k] >= dp[i][j][k - 1]:
                j -= 1
            else:
                k -= 1

        lcs.reverse()
        return dp[m][n][p], ''.join(lcs)

    def shortest_common_supersequence_with_positions(self, str1: str, str2: str) -> Tuple[str, List[Tuple[int, int]]]:
        """
        LeetCode 1092 Extension - Shortest Common Supersequence with Source Tracking (Hard)

        Find shortest string containing both strings as subsequences.
        Extended: Track which characters come from which string.

        Algorithm:
        1. Find LCS first
        2. Build supersequence by merging strings using LCS
        3. Track source of each character

        Time: O(m*n), Space: O(m*n)

        Example:
        str1 = "abac", str2 = "cab"
        Output: ("cabac", [(1,0), (0,0), (0,1), (0,2), (1,2)])
        """
        m, n = len(str1), len(str2)

        # Find LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # Build supersequence
        result = []
        positions = []  # (source: 0=str1, 1=str2, 2=both, index)
        i, j = m, n

        while i > 0 or j > 0:
            if i == 0:
                result.append(str2[j - 1])
                positions.append((1, j - 1))
                j -= 1
            elif j == 0:
                result.append(str1[i - 1])
                positions.append((0, i - 1))
                i -= 1
            elif str1[i - 1] == str2[j - 1]:
                result.append(str1[i - 1])
                positions.append((2, i - 1))  # From both
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                result.append(str1[i - 1])
                positions.append((0, i - 1))
                i -= 1
            else:
                result.append(str2[j - 1])
                positions.append((1, j - 1))
                j -= 1

        result.reverse()
        positions.reverse()

        return ''.join(result), positions

    def longest_palindromic_subsequence_with_k_mismatches(self, s: str, k: int) -> Tuple[int, str]:
        """
        Custom Hard - Longest Palindromic Subsequence with K Mismatches

        Find longest subsequence that becomes palindrome with at most k character changes.

        Algorithm:
        1. DP with state (left, right, mismatches_used)
        2. Allow mismatches when characters don't match
        3. Reconstruct the palindrome

        Time: O(n² * k), Space: O(n² * k)

        Example:
        s = "abcdeca", k = 1
        Output: (7, "abcdcba") - change one character to make palindrome
        """
        n = len(s)

        # dp[i][j][m] = longest palindromic subsequence in s[i:j+1] with m mismatches
        dp = [[[-1] * (k + 1) for _ in range(n)] for _ in range(n)]
        parent = [[[None] * (k + 1) for _ in range(n)] for _ in range(n)]

        # Base case: single characters
        for i in range(n):
            for m in range(k + 1):
                dp[i][i][m] = 1

        # Fill DP table
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1

                for m in range(k + 1):
                    if s[i] == s[j]:
                        if i + 1 <= j - 1:
                            if dp[i + 1][j - 1][m] != -1:
                                dp[i][j][m] = dp[i + 1][j - 1][m] + 2
                                parent[i][j][m] = ('match', i + 1, j - 1, m)
                        else:
                            dp[i][j][m] = 2
                            parent[i][j][m] = ('match', -1, -1, m)
                    else:
                        # Try not including either character
                        if i + 1 <= j and dp[i + 1][j][m] != -1:
                            dp[i][j][m] = dp[i + 1][j][m]
                            parent[i][j][m] = ('skip_left', i + 1, j, m)

                        if i <= j - 1 and dp[i][j - 1][m] != -1 and dp[i][j - 1][m] > dp[i][j][m]:
                            dp[i][j][m] = dp[i][j - 1][m]
                            parent[i][j][m] = ('skip_right', i, j - 1, m)

                        # Try using a mismatch
                        if m > 0 and i + 1 <= j - 1:
                            if dp[i + 1][j - 1][m - 1] != -1:
                                val = dp[i + 1][j - 1][m - 1] + 2
                                if val > dp[i][j][m]:
                                    dp[i][j][m] = val
                                    parent[i][j][m] = ('mismatch', i + 1, j - 1, m - 1)

        # Find best result
        best_length = 0
        best_m = 0

        for m in range(k + 1):
            if dp[0][n - 1][m] > best_length:
                best_length = dp[0][n - 1][m]
                best_m = m

        # Reconstruct palindrome
        palindrome = []

        def reconstruct(i, j, m):
            if i > j or parent[i][j][m] is None:
                return

            action, next_i, next_j, next_m = parent[i][j][m]

            if action == 'match':
                palindrome.append(s[i])
                if next_i != -1:
                    reconstruct(next_i, next_j, next_m)
                palindrome.append(s[i])
            elif action == 'mismatch':
                palindrome.append(s[i])
                if next_i != -1:
                    reconstruct(next_i, next_j, next_m)
                palindrome.append(s[i])
            elif action == 'skip_left':
                reconstruct(next_i, next_j, next_m)
            else:  # skip_right
                reconstruct(next_i, next_j, next_m)

        reconstruct(0, n - 1, best_m)

        return best_length, ''.join(palindrome)

    def edit_distance_with_operations(self, word1: str, word2: str) -> Tuple[int, List[Tuple[str, int, str]]]:
        """
        LeetCode 72 Extension - Edit Distance with Operation Sequence (Hard)

        Find minimum edit distance and the actual operations.
        Operations: insert, delete, replace.

        Algorithm:
        1. Classic edit distance DP
        2. Track operations for reconstruction
        3. Build operation sequence

        Time: O(m*n), Space: O(m*n)

        Example:
        word1 = "horse", word2 = "ros"
        Output: (3, [("replace", 0, "r"), ("delete", 2, ""), ("delete", 3, "")])
        """
        m, n = len(word1), len(word2)

        # DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        parent = [[None] * (n + 1) for _ in range(m + 1)]

        # Base cases
        for i in range(m + 1):
            dp[i][0] = i
            if i > 0:
                parent[i][0] = ('delete', i - 1, 0)

        for j in range(n + 1):
            dp[0][j] = j
            if j > 0:
                parent[0][j] = ('insert', 0, j - 1)

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                    parent[i][j] = ('match', i - 1, j - 1)
                else:
                    # Replace
                    replace_cost = dp[i - 1][j - 1] + 1
                    # Delete
                    delete_cost = dp[i - 1][j] + 1
                    # Insert
                    insert_cost = dp[i][j - 1] + 1

                    if replace_cost <= delete_cost and replace_cost <= insert_cost:
                        dp[i][j] = replace_cost
                        parent[i][j] = ('replace', i - 1, j - 1)
                    elif delete_cost <= insert_cost:
                        dp[i][j] = delete_cost
                        parent[i][j] = ('delete', i - 1, j)
                    else:
                        dp[i][j] = insert_cost
                        parent[i][j] = ('insert', i, j - 1)

        # Reconstruct operations
        operations = []
        i, j = m, n

        while i > 0 or j > 0:
            if parent[i][j] is None:
                break

            op, next_i, next_j = parent[i][j]

            if op == 'replace':
                operations.append(('replace', next_i, word2[next_j]))
            elif op == 'delete':
                operations.append(('delete', next_i, ''))
            elif op == 'insert':
                operations.append(('insert', next_i, word2[next_j]))

            i, j = next_i, next_j

        operations.reverse()

        return dp[m][n], operations

    def longest_common_substring_k_strings(self, strings: List[str]) -> Tuple[int, str, List[int]]:
        """
        Custom Hard - Longest Common Substring of K Strings

        Find longest substring common to all k strings.
        Return length, substring, and starting positions in each string.

        Algorithm:
        1. Use suffix array or rolling hash
        2. Binary search on length
        3. Check if substring of given length exists in all strings

        Time: O(n * k * log n) where n is max string length
        Space: O(n * k)

        Example:
        strings = ["abcdef", "zabcy", "abcdx"]
        Output: (3, "abc", [0, 1, 0])
        """
        if not strings:
            return 0, "", []

        def has_common_substring(length: int) -> Tuple[bool, str, List[int]]:
            """Check if there's a common substring of given length."""
            if length == 0:
                return True, "", [0] * len(strings)

            # Get all substrings of given length from first string
            substrings = defaultdict(list)

            for i in range(len(strings[0]) - length + 1):
                substr = strings[0][i:i + length]
                substrings[substr].append(i)

            # Check each substring against other strings
            for substr, positions in substrings.items():
                found_positions = [positions[0]]
                found_in_all = True

                for j in range(1, len(strings)):
                    pos = strings[j].find(substr)
                    if pos == -1:
                        found_in_all = False
                        break
                    found_positions.append(pos)

                if found_in_all:
                    return True, substr, found_positions

            return False, "", []

        # Binary search on length
        left, right = 0, min(len(s) for s in strings)
        result_length = 0
        result_substring = ""
        result_positions = []

        while left <= right:
            mid = (left + right) // 2
            found, substr, positions = has_common_substring(mid)

            if found:
                result_length = mid
                result_substring = substr
                result_positions = positions
                left = mid + 1
            else:
                right = mid - 1

        return result_length, result_substring, result_positions

    def interleaving_string_with_path(self, s1: str, s2: str, s3: str) -> Tuple[bool, List[int]]:
        """
        LeetCode 97 Extension - Interleaving String with Source Tracking (Hard)

        Check if s3 is interleaving of s1 and s2.
        Extended: Return which characters come from which string.

        Algorithm:
        1. 2D DP where dp[i][j] = can form s3[0:i+j] from s1[0:i] and s2[0:j]
        2. Track source of each character
        3. Reconstruct path

        Time: O(m*n), Space: O(m*n)

        Example:
        s1 = "aab", s2 = "axy", s3 = "aaxaby"
        Output: (True, [1,1,2,1,2,2]) - source string for each char
        """
        m, n, p = len(s1), len(s2), len(s3)

        if m + n != p:
            return False, []

        # DP table
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        parent = [[None] * (n + 1) for _ in range(m + 1)]

        # Base cases
        dp[0][0] = True

        # First row (only using s2)
        for j in range(1, n + 1):
            if s2[j - 1] == s3[j - 1] and dp[0][j - 1]:
                dp[0][j] = True
                parent[0][j] = (0, j - 1, 2)

        # First column (only using s1)
        for i in range(1, m + 1):
            if s1[i - 1] == s3[i - 1] and dp[i - 1][0]:
                dp[i][0] = True
                parent[i][0] = (i - 1, 0, 1)

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                k = i + j - 1  # Position in s3

                # Try using character from s1
                if s1[i - 1] == s3[k] and dp[i - 1][j]:
                    dp[i][j] = True
                    parent[i][j] = (i - 1, j, 1)

                # Try using character from s2
                if s2[j - 1] == s3[k] and dp[i][j - 1]:
                    dp[i][j] = True
                    parent[i][j] = (i, j - 1, 2)

        if not dp[m][n]:
            return False, []

        # Reconstruct path
        path = []
        i, j = m, n

        while i > 0 or j > 0:
            if parent[i][j] is None:
                break

            prev_i, prev_j, source = parent[i][j]
            path.append(source)
            i, j = prev_i, prev_j

        path.reverse()

        return True, path

    def distinct_subsequences_with_all(self, s: str, t: str) -> Tuple[int, List[str]]:
        """
        LeetCode 115 Extension - Distinct Subsequences with All Sequences (Hard)

        Count distinct subsequences of s that equal t.
        Extended: Return all such subsequences (limited).

        Algorithm:
        1. DP for counting
        2. Backtrack to generate actual subsequences
        3. Use indices to track positions

        Time: O(m*n), Space: O(m*n + k) where k is output size

        Example:
        s = "rabbbit", t = "rabbit"
        Output: (3, ["rabbit", "rabbit", "rabbit"]) with different 'b' positions
        """
        m, n = len(s), len(t)

        # DP for counting
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Base case: empty t can be formed in one way
        for i in range(m + 1):
            dp[i][0] = 1

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i - 1][j]  # Don't use s[i-1]

                if s[i - 1] == t[j - 1]:
                    dp[i][j] += dp[i - 1][j - 1]  # Use s[i-1]

        count = dp[m][n]

        # Generate actual subsequences (limit to prevent memory issues)
        subsequences = []

        def backtrack(i: int, j: int, indices: List[int]):
            if len(subsequences) >= 10:  # Limit
                return

            if j == n:
                # Found complete subsequence
                subsequences.append([indices[:], ''.join(s[idx] for idx in indices)])
                return

            if i == m:
                return

            # Try not using s[i]
            backtrack(i + 1, j, indices)

            # Try using s[i] if it matches t[j]
            if s[i] == t[j]:
                indices.append(i)
                backtrack(i + 1, j + 1, indices)
                indices.pop()

        backtrack(0, 0, [])

        # Format output
        formatted = []
        for indices, substr in subsequences:
            formatted.append(f"{substr} (indices: {indices})")

        return count, formatted[:5]  # Limit output

    def maximum_length_of_repeated_subarray_with_k_mismatches(self, nums1: List[int], nums2: List[int], k: int) -> \
    Tuple[int, List[int], List[int]]:
        """
        Extension of LeetCode 718 - Maximum Length of Repeated Subarray with Mismatches (Hard)

        Find longest common subarray allowing k mismatches.
        Return length and the subarrays.

        Algorithm:
        1. Sliding window with mismatch counting
        2. Two pointers for each starting position
        3. Track best subarray positions

        Time: O(m*n), Space: O(1)

        Example:
        nums1 = [1,2,3,2,1], nums2 = [3,2,1,4,7], k = 1
        Output: (3, [2,3,2], [3,2,1])
        """
        m, n = len(nums1), len(nums2)
        max_length = 0
        best_start1 = best_start2 = 0

        # Try each starting position
        for start1 in range(m):
            for start2 in range(n):
                mismatches = 0
                length = 0

                i, j = start1, start2

                while i < m and j < n:
                    if nums1[i] != nums2[j]:
                        mismatches += 1

                    if mismatches <= k:
                        length += 1
                        if length > max_length:
                            max_length = length
                            best_start1 = start1
                            best_start2 = start2
                    else:
                        # Slide window
                        break

                    i += 1
                    j += 1

        # Extract subarrays
        subarray1 = nums1[best_start1:best_start1 + max_length]
        subarray2 = nums2[best_start2:best_start2 + max_length]

        return max_length, subarray1, subarray2

    def wildcard_matching_with_explanation(self, s: str, p: str) -> Tuple[bool, str]:
        """
        LeetCode 44 Extension - Wildcard Matching with Explanation (Hard)

        Check if pattern with ? and * matches string.
        Extended: Explain how pattern matches.

        Algorithm:
        1. DP with optimizations for consecutive *
        2. Track matching decisions
        3. Generate explanation

        Time: O(m*n), Space: O(m*n)

        Example:
        s = "adceb", p = "*a*b"
        Output: (True, "* matches '', a matches 'a', * matches 'dce', b matches 'b'")
        """
        m, n = len(s), len(p)

        # Optimize pattern by removing consecutive *
        optimized_p = []
        for char in p:
            if not optimized_p or char != '*' or optimized_p[-1] != '*':
                optimized_p.append(char)
        p = ''.join(optimized_p)
        n = len(p)

        # DP table
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        parent = [[None] * (n + 1) for _ in range(m + 1)]

        # Base cases
        dp[0][0] = True

        # First row (empty string)
        for j in range(1, n + 1):
            if p[j - 1] == '*' and dp[0][j - 1]:
                dp[0][j] = True
                parent[0][j] = (0, j - 1, '*', '')

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == s[i - 1] or p[j - 1] == '?':
                    if dp[i - 1][j - 1]:
                        dp[i][j] = True
                        parent[i][j] = (i - 1, j - 1, p[j - 1], s[i - 1])
                elif p[j - 1] == '*':
                    # * matches empty
                    if dp[i][j - 1]:
                        dp[i][j] = True
                        parent[i][j] = (i, j - 1, '*', '')
                    # * matches one or more
                    elif dp[i - 1][j]:
                        dp[i][j] = True
                        parent[i][j] = (i - 1, j, '*', s[i - 1])

        if not dp[m][n]:
            return False, "No match possible"

        # Generate explanation
        explanation_parts = []
        i, j = m, n

        while i > 0 or j > 0:
            if parent[i][j] is None:
                break

            prev_i, prev_j, pattern_char, matched_str = parent[i][j]

            if pattern_char == '*':
                # Collect all characters matched by *
                star_match = matched_str
                while i > prev_i:
                    i -= 1
                    if i > 0:
                        star_match = s[i - 1] + star_match

                explanation_parts.append(f"* matches '{star_match}'")
            else:
                explanation_parts.append(f"{pattern_char} matches '{matched_str}'")

            i, j = prev_i, prev_j

        explanation_parts.reverse()
        explanation = ", ".join(explanation_parts)

        return True, explanation

    def minimum_window_subsequence_all_windows(self, s1: str, s2: str) -> List[str]:
        """
        LeetCode 727 Extension - All Minimum Window Subsequences (Hard)

        Find all minimum windows in s1 where s2 is a subsequence.

        Algorithm:
        1. Two pointers to find potential windows
        2. Verify and minimize each window
        3. Collect all minimum windows

        Time: O(n*m), Space: O(k) where k is number of windows

        Example:
        s1 = "abcdebdde", s2 = "bde"
        Output: ["bcde", "bdde"]
        """
        if not s1 or not s2:
            return []

        min_windows = []
        min_length = float('inf')

        # Try each starting position
        start = 0

        while start < len(s1):
            # Find a potential window starting from 'start'
            i = start
            j = 0

            # Forward pass: find window containing s2
            while i < len(s1) and j < len(s2):
                if s1[i] == s2[j]:
                    j += 1
                i += 1

            if j < len(s2):
                # Couldn't find complete s2
                break

            # Backward pass: minimize window
            end = i - 1
            j = len(s2) - 1

            while j >= 0:
                if s1[end] == s2[j]:
                    j -= 1
                end -= 1

            end += 1

            # Found a window
            window_length = i - end

            if window_length < min_length:
                min_length = window_length
                min_windows = [s1[end:i]]
            elif window_length == min_length:
                window = s1[end:i]
                if window not in min_windows:
                    min_windows.append(window)

            # Next search starts after current window start
            start = end + 1

        return min_windows


# Example usage and testing
if __name__ == "__main__":
    solver = LongestCommonSubstringHard()

    # Test 1: LCS of Three Strings
    print("1. Longest Common Subsequence of Three Strings:")
    text1, text2, text3 = "abcd", "acbde", "aced"
    length, lcs = solver.longest_common_subsequence_of_three(text1, text2, text3)
    print(f"   Strings: '{text1}', '{text2}', '{text3}'")
    print(f"   LCS length: {length}, LCS: '{lcs}'")
    print()

    # Test 2: Shortest Common Supersequence
    print("2. Shortest Common Supersequence with Positions:")
    str1, str2 = "abac", "cab"
    scs, positions = solver.shortest_common_supersequence_with_positions(str1, str2)
    print(f"   Strings: '{str1}', '{str2}'")
    print(f"   SCS: '{scs}'")
    print(f"   Source positions: {positions[:5]}...")
    print()

    # Test 3: Edit Distance with Operations
    print("3. Edit Distance with Operations:")
    word1, word2 = "horse", "ros"
    distance, operations = solver.edit_distance_with_operations(word1, word2)
    print(f"   Words: '{word1}' -> '{word2}'")
    print(f"   Distance: {distance}")
    print(f"   Operations: {operations}")
    print()

    # Test 4: Wildcard Matching
    print("4. Wildcard Matching with Explanation:")
    s, p = "adceb", "*a*b"
    matches, explanation = solver.wildcard_matching_with_explanation(s, p)
    print(f"   String: '{s}', Pattern: '{p}'")
    print(f"   Matches: {matches}")
    print(f"   Explanation: {explanation}")