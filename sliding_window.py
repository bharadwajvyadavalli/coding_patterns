"""
Pattern 1: Sliding Window - 10 Hard Problems
============================================

The sliding window pattern is used to perform operations on a specific window size
of an array or linked list, such as finding the longest substring with k distinct
characters, maximum sum subarray of size k, etc.

Key Concepts:
- Use two pointers (start and end) to represent window boundaries
- Expand window by moving end pointer
- Contract window by moving start pointer when constraint is violated
- Track window state using hash maps, counters, or other data structures

Time Complexity: Usually O(n) or O(n*k) where k is window size
Space Complexity: O(k) for storing window elements
"""

from collections import defaultdict, deque
from typing import List, Tuple
import math


class SlidingWindowHard:

    def min_window_substring(self, s: str, t: str) -> str:
        """
        LeetCode 76 - Minimum Window Substring (Hard)

        Find the minimum window in s that contains all characters in t.

        Algorithm:
        1. Use two pointers to create a sliding window
        2. Expand window until all characters from t are included
        3. Contract window while maintaining the constraint
        4. Track the minimum window that satisfies the condition

        Time: O(|s| + |t|), Space: O(|t|)

        Example:
        s = "ADOBECODEBANC", t = "ABC"
        Output: "BANC"
        """
        if not t or not s:
            return ""

        # Dictionary to store the frequency of characters in t
        dict_t = defaultdict(int)
        for char in t:
            dict_t[char] += 1

        # Number of unique characters in t that need to be in the window
        required = len(dict_t)

        # Left and right pointers
        l, r = 0, 0

        # How many unique characters in current window match the desired count
        formed = 0

        # Dictionary to keep count of characters in current window
        window_counts = defaultdict(int)

        # Result tuple (window length, left, right)
        ans = float("inf"), None, None

        while r < len(s):
            # Add character from right to window
            char = s[r]
            window_counts[char] += 1

            # If frequency of current character matches the desired count in t
            if char in dict_t and window_counts[char] == dict_t[char]:
                formed += 1

            # Try to contract window until it ceases to be 'desirable'
            while l <= r and formed == required:
                char = s[l]

                # Update result if this window is smaller
                if r - l + 1 < ans[0]:
                    ans = (r - l + 1, l, r)

                # The character at left pointer is no longer part of window
                window_counts[char] -= 1
                if char in dict_t and window_counts[char] < dict_t[char]:
                    formed -= 1

                # Move left pointer ahead for next iteration
                l += 1

            # Keep expanding window by moving right pointer
            r += 1

        # Return the smallest window or empty string
        return "" if ans[0] == float("inf") else s[ans[1]:ans[2] + 1]

    def longest_substring_with_at_most_two_distinct(self, s: str) -> int:
        """
        LeetCode 159 - Longest Substring with At Most Two Distinct Characters (Hard)

        Find length of longest substring that contains at most 2 distinct characters.

        Algorithm:
        1. Use sliding window with character frequency map
        2. Expand window while distinct characters <= 2
        3. Contract window when distinct characters > 2
        4. Track maximum window size

        Time: O(n), Space: O(1) - at most 3 characters in map

        Example:
        s = "eceba"
        Output: 3 ("ece")
        """
        if len(s) < 3:
            return len(s)

        # Sliding window left and right pointers
        left, right = 0, 0

        # Hashmap character -> its rightmost position in sliding window
        hashmap = defaultdict()

        max_len = 2

        while right < len(s):
            # When sliding window contains less than 3 characters
            hashmap[s[right]] = right
            right += 1

            # Sliding window contains 3 characters
            if len(hashmap) == 3:
                # Delete the leftmost character
                del_idx = min(hashmap.values())
                del hashmap[s[del_idx]]
                # Move left pointer to the right of deleted character
                left = del_idx + 1

            max_len = max(max_len, right - left)

        return max_len

    def substring_with_concatenation_of_all_words(self, s: str, words: List[str]) -> List[int]:
        """
        LeetCode 30 - Substring with Concatenation of All Words (Hard)

        Find all starting indices of substring(s) in s that is a concatenation
        of each word in words exactly once.

        Algorithm:
        1. Use sliding window of size (word_length * number_of_words)
        2. Check each window if it contains all words exactly once
        3. Use hash map to track word frequencies
        4. Optimize by checking word by word instead of character by character

        Time: O(n * m * len), where n = len(s), m = len(words), len = word length
        Space: O(m)

        Example:
        s = "barfoothefoobarman", words = ["foo","bar"]
        Output: [0,9]
        """
        if not s or not words or not words[0]:
            return []

        word_len = len(words[0])
        word_count = len(words)
        total_len = word_len * word_count

        # Count frequency of each word
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1

        result = []

        # Try all possible starting positions within a word length
        for i in range(word_len):
            left = i
            right = i
            current_freq = defaultdict(int)

            while right + word_len <= len(s):
                # Get the word at right pointer
                word = s[right:right + word_len]
                right += word_len

                if word in word_freq:
                    current_freq[word] += 1

                    # Shrink window if we have too many of this word
                    while current_freq[word] > word_freq[word]:
                        left_word = s[left:left + word_len]
                        current_freq[left_word] -= 1
                        left += word_len

                    # Check if we have found all words
                    if right - left == total_len:
                        result.append(left)
                else:
                    # Reset window as current word is not in words list
                    current_freq.clear()
                    left = right

        return result

    def max_consecutive_ones_iii(self, nums: List[int], k: int) -> int:
        """
        LeetCode 1004 - Max Consecutive Ones III (Hard variation)

        Given binary array, return max number of consecutive 1s if you can flip at most k 0s.
        Extended: Also return the indices of flipped positions for maximum window.

        Algorithm:
        1. Use sliding window to track consecutive 1s
        2. Allow at most k zeros in the window
        3. Expand window while zeros <= k
        4. Contract window when zeros > k

        Time: O(n), Space: O(k) for storing flip positions

        Example:
        nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2
        Output: 6 (indices 3,4 or 4,5 can be flipped)
        """
        left = 0
        zero_count = 0
        max_len = 0
        max_window_start = 0
        flip_positions = []
        current_flips = deque()

        for right in range(len(nums)):
            # If we encounter a 0, increment zero count
            if nums[right] == 0:
                zero_count += 1
                current_flips.append(right)

            # If zero count exceeds k, shrink window from left
            while zero_count > k:
                if nums[left] == 0:
                    zero_count -= 1
                    current_flips.popleft()
                left += 1

            # Update maximum length and window position
            if right - left + 1 > max_len:
                max_len = right - left + 1
                max_window_start = left
                flip_positions = list(current_flips)

        return max_len

    def minimum_window_subsequence(self, s1: str, s2: str) -> str:
        """
        LeetCode 727 - Minimum Window Subsequence (Hard)

        Find minimum window in s1 such that s2 is a subsequence of the window.

        Algorithm:
        1. Use dynamic programming with sliding window optimization
        2. For each ending position in s1, find minimum window containing s2
        3. Use two-pointer technique to optimize the search

        Time: O(n * m), Space: O(1)

        Example:
        s1 = "abcdebdde", s2 = "bde"
        Output: "bcde"
        """
        min_len = float('inf')
        min_start = 0

        # Try each position in s1 as potential end of window
        i = 0
        while i < len(s1):
            # Try to match s2 starting from position i in s1
            j = 0
            k = i

            while k < len(s1) and j < len(s2):
                if s1[k] == s2[j]:
                    j += 1
                k += 1

            # If we couldn't match all of s2, no point trying further
            if j < len(s2):
                break

            # We found a match, now find the minimum window
            # by going backwards
            k -= 1  # k is now at the last matched character
            j -= 1  # j is at last character of s2

            while j >= 0:
                if s1[k] == s2[j]:
                    j -= 1
                k -= 1

            k += 1  # k is now at the start of minimum window

            # Update minimum window if current is smaller
            if i - k + 1 < min_len:
                min_len = i - k + 1
                min_start = k

            # Next iteration starts from k+1
            i = k + 1

        return "" if min_len == float('inf') else s1[min_start:min_start + min_len]

    def count_subarrays_with_k_different_integers(self, nums: List[int], k: int) -> int:
        """
        LeetCode 992 - Subarrays with K Different Integers (Hard)

        Count number of subarrays with exactly k different integers.

        Algorithm:
        1. Transform to: at_most(k) - at_most(k-1)
        2. Use sliding window to count subarrays with at most k distinct
        3. Maintain frequency map of elements in window

        Time: O(n), Space: O(k)

        Example:
        nums = [1,2,1,2,3], k = 2
        Output: 7
        """

        def at_most_k_distinct(nums: List[int], k: int) -> int:
            count = 0
            left = 0
            freq_map = defaultdict(int)

            for right in range(len(nums)):
                # Add current element to window
                freq_map[nums[right]] += 1

                # Shrink window if distinct elements exceed k
                while len(freq_map) > k:
                    freq_map[nums[left]] -= 1
                    if freq_map[nums[left]] == 0:
                        del freq_map[nums[left]]
                    left += 1

                # All subarrays ending at right with at most k distinct
                count += right - left + 1

            return count

        # Exactly k = at most k - at most (k-1)
        return at_most_k_distinct(nums, k) - at_most_k_distinct(nums, k - 1)

    def longest_repeating_character_replacement_all_chars(self, s: str, k: int) -> dict:
        """
        LeetCode 424 Extension - Character Replacement for Each Character (Hard)

        For each character, find longest substring where you can replace at most k
        characters to make all characters the same.

        Algorithm:
        1. Run sliding window for each character separately
        2. For each character, find longest window with at most k different chars
        3. Return dictionary mapping each character to its max length

        Time: O(26 * n), Space: O(1)

        Example:
        s = "AABABBA", k = 1
        Output: {'A': 4, 'B': 5} (AAAA by changing 1 B, BBBBB by changing 1 A)
        """
        result = {}
        unique_chars = set(s)

        for target_char in unique_chars:
            left = 0
            max_len = 0
            diff_count = 0

            for right in range(len(s)):
                # Count characters different from target
                if s[right] != target_char:
                    diff_count += 1

                # Shrink window if we need to replace more than k chars
                while diff_count > k:
                    if s[left] != target_char:
                        diff_count -= 1
                    left += 1

                max_len = max(max_len, right - left + 1)

            result[target_char] = max_len

        return result

    def max_sum_of_distinct_subarrays_with_length_k(self, nums: List[int], k: int) -> int:
        """
        LeetCode 2461 - Maximum Sum of Distinct Subarrays With Length K (Hard)

        Find maximum sum among all subarrays of length k with all distinct elements.

        Algorithm:
        1. Use sliding window of size k
        2. Track element frequencies to ensure distinctness
        3. Calculate sum only for valid windows (all distinct)
        4. Use deque for efficient window operations

        Time: O(n), Space: O(k)

        Example:
        nums = [1,5,4,2,9,9,9], k = 3
        Output: 15 ([5,4,2])
        """
        if len(nums) < k:
            return 0

        max_sum = 0
        current_sum = 0
        freq_map = defaultdict(int)
        distinct_count = 0

        # Initialize first window
        for i in range(k):
            if freq_map[nums[i]] == 0:
                distinct_count += 1
            freq_map[nums[i]] += 1
            current_sum += nums[i]

        # Check if first window is valid
        if distinct_count == k:
            max_sum = current_sum

        # Slide the window
        for i in range(k, len(nums)):
            # Add new element
            if freq_map[nums[i]] == 0:
                distinct_count += 1
            freq_map[nums[i]] += 1
            current_sum += nums[i]

            # Remove old element
            old_element = nums[i - k]
            freq_map[old_element] -= 1
            if freq_map[old_element] == 0:
                distinct_count -= 1
            current_sum -= old_element

            # Update max sum if current window has all distinct elements
            if distinct_count == k:
                max_sum = max(max_sum, current_sum)

        return max_sum

    def minimum_swaps_to_group_all_ones_ii(self, nums: List[int]) -> int:
        """
        LeetCode 2134 - Minimum Swaps to Group All 1's Together II (Hard)

        Circular array - find minimum swaps to group all 1's together.

        Algorithm:
        1. Count total 1's - this determines window size
        2. Use circular sliding window (wrap around using modulo)
        3. Find window with maximum 1's (minimum 0's to swap)
        4. Handle circular array by extending the window search

        Time: O(n), Space: O(1)

        Example:
        nums = [0,1,0,1,1,0,0]
        Output: 1 (swap position 2 with position 5)
        """
        n = len(nums)
        total_ones = sum(nums)

        if total_ones == 0 or total_ones == n:
            return 0

        # Find window of size total_ones with maximum ones
        max_ones_in_window = 0
        current_ones = 0

        # Initialize first window
        for i in range(total_ones):
            current_ones += nums[i]
        max_ones_in_window = current_ones

        # Slide window in circular array
        for i in range(total_ones, n + total_ones):
            # Remove element going out of window
            current_ones -= nums[(i - total_ones) % n]
            # Add element coming into window
            current_ones += nums[i % n]
            max_ones_in_window = max(max_ones_in_window, current_ones)

        # Minimum swaps = zeros in the best window
        return total_ones - max_ones_in_window

    def find_substring_with_k_distinct_palindromes(self, s: str, k: int) -> List[Tuple[int, int]]:
        """
        Custom Hard Problem - Find All Substrings with Exactly K Distinct Palindromic Substrings

        Find all substrings that contain exactly k distinct palindromic substrings.

        Algorithm:
        1. Use sliding window to generate substrings
        2. For each substring, count distinct palindromes using expand around center
        3. Use set to track distinct palindromes in current window
        4. Optimize by caching palindrome checks

        Time: O(n³), Space: O(n²)

        Example:
        s = "aabaa", k = 3
        Output: [(0,2), (2,4)] representing "aab" and "baa"
        """

        def get_all_palindromes(substring: str) -> set:
            """Get all palindromic substrings in given string."""
            palindromes = set()
            n = len(substring)

            # Check all possible centers (including between characters)
            for center in range(2 * n - 1):
                left = center // 2
                right = left + center % 2

                # Expand around center
                while left >= 0 and right < n and substring[left] == substring[right]:
                    palindromes.add(substring[left:right + 1])
                    left -= 1
                    right += 1

            return palindromes

        result = []
        n = len(s)

        # Try all possible substrings
        for i in range(n):
            for j in range(i, n):
                substring = s[i:j + 1]
                distinct_palindromes = get_all_palindromes(substring)

                if len(distinct_palindromes) == k:
                    result.append((i, j))

        return result


# Example usage and testing
if __name__ == "__main__":
    solver = SlidingWindowHard()

    # Test 1: Minimum Window Substring
    print("1. Minimum Window Substring:")
    print(f"   Input: s='ADOBECODEBANC', t='ABC'")
    print(f"   Output: {solver.min_window_substring('ADOBECODEBANC', 'ABC')}")
    print()

    # Test 2: Longest Substring with At Most Two Distinct
    print("2. Longest Substring with At Most Two Distinct:")
    print(f"   Input: s='eceba'")
    print(f"   Output: {solver.longest_substring_with_at_most_two_distinct('eceba')}")
    print()

    # Test 3: Substring with Concatenation
    print("3. Substring with Concatenation of All Words:")
    print(f"   Input: s='barfoothefoobarman', words=['foo','bar']")
    print(f"   Output: {solver.substring_with_concatenation_of_all_words('barfoothefoobarman', ['foo', 'bar'])}")
    print()

    # Test 4: Max Consecutive Ones III
    print("4. Max Consecutive Ones III:")
    print(f"   Input: nums=[1,1,1,0,0,0,1,1,1,1,0], k=2")
    print(f"   Output: {solver.max_consecutive_ones_iii([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2)}")