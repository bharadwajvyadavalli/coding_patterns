"""
Pattern 22: Backtracking - 10 Hard Problems
===========================================

The Backtracking pattern systematically explores all possible solutions by building
them incrementally and abandoning solutions that fail to meet constraints. This
pattern is essential for constraint satisfaction problems, puzzles, and optimization.

Key Concepts:
- Build solutions incrementally
- Abandon partial solutions that can't lead to valid complete solutions
- Use state space tree to visualize all possibilities
- Prune branches early to improve efficiency

Time Complexity: Often exponential O(b^d) where b is branching factor, d is depth
Space Complexity: O(d) for recursion stack depth
"""

from typing import List, Set, Tuple, Dict, Optional
from collections import defaultdict, Counter
import copy


class BacktrackingHard:

    def n_queens_with_all_solutions(self, n: int) -> Tuple[int, List[List[str]], Dict[str, int]]:
        """
        LeetCode 51 Extension - N-Queens with Complete Analysis (Hard)

        Place n queens on n×n board so no two queens attack each other.
        Extended: Return count, all solutions, and attack statistics.

        Algorithm:
        1. Place queens row by row
        2. Track attacked columns, diagonals, anti-diagonals
        3. Backtrack when placement is invalid
        4. Collect statistics about placements

        Time: O(n!), Space: O(n)

        Example:
        n = 4
        Output: (2, [[".Q..","...Q","Q...","..Q."], ...], stats)
        """
        solutions = []
        board = [['.' for _ in range(n)] for _ in range(n)]
        cols = set()
        diag1 = set()  # row - col
        diag2 = set()  # row + col
        stats = defaultdict(int)

        def is_safe(row: int, col: int) -> bool:
            return col not in cols and \
                (row - col) not in diag1 and \
                (row + col) not in diag2

        def place_queen(row: int, col: int):
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            stats[f"col_{col}"] += 1

        def remove_queen(row: int, col: int):
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

        def backtrack(row: int):
            if row == n:
                # Found a solution
                solutions.append([''.join(row) for row in board])
                return

            for col in range(n):
                if is_safe(row, col):
                    place_queen(row, col)
                    backtrack(row + 1)
                    remove_queen(row, col)

        backtrack(0)

        return len(solutions), solutions, dict(stats)

    def sudoku_solver_with_techniques(self, board: List[List[str]]) -> Tuple[bool, List[List[str]], Dict[str, int]]:
        """
        LeetCode 37 Extension - Sudoku Solver with Solution Techniques (Hard)

        Solve Sudoku puzzle and track solving techniques used.
        Extended: Return solved board and technique statistics.

        Algorithm:
        1. Use constraint propagation before backtracking
        2. Find cell with minimum remaining values (MRV)
        3. Apply naked singles, hidden singles techniques
        4. Backtrack with intelligent ordering

        Time: O(9^m) where m is empty cells, Space: O(1)
        """
        techniques = defaultdict(int)

        def is_valid(board: List[List[str]], row: int, col: int, num: str) -> bool:
            # Check row
            for j in range(9):
                if board[row][j] == num:
                    return False

            # Check column
            for i in range(9):
                if board[i][col] == num:
                    return False

            # Check 3x3 box
            box_row, box_col = 3 * (row // 3), 3 * (col // 3)
            for i in range(box_row, box_row + 3):
                for j in range(box_col, box_col + 3):
                    if board[i][j] == num:
                        return False

            return True

        def get_candidates(board: List[List[str]], row: int, col: int) -> Set[str]:
            if board[row][col] != '.':
                return set()

            candidates = set('123456789')

            # Remove numbers in same row
            for j in range(9):
                candidates.discard(board[row][j])

            # Remove numbers in same column
            for i in range(9):
                candidates.discard(board[i][col])

            # Remove numbers in same box
            box_row, box_col = 3 * (row // 3), 3 * (col // 3)
            for i in range(box_row, box_row + 3):
                for j in range(box_col, box_col + 3):
                    candidates.discard(board[i][j])

            return candidates

        def find_naked_singles(board: List[List[str]]) -> Optional[Tuple[int, int, str]]:
            """Find cells with only one possible value."""
            for i in range(9):
                for j in range(9):
                    if board[i][j] == '.':
                        candidates = get_candidates(board, i, j)
                        if len(candidates) == 1:
                            techniques['naked_singles'] += 1
                            return i, j, candidates.pop()
            return None

        def find_best_cell(board: List[List[str]]) -> Optional[Tuple[int, int]]:
            """Find empty cell with minimum remaining values (MRV)."""
            min_candidates = 10
            best_cell = None

            for i in range(9):
                for j in range(9):
                    if board[i][j] == '.':
                        candidates = get_candidates(board, i, j)
                        if len(candidates) < min_candidates:
                            min_candidates = len(candidates)
                            best_cell = (i, j)

            return best_cell

        def solve(board: List[List[str]]) -> bool:
            # Try constraint propagation first
            while True:
                naked_single = find_naked_singles(board)
                if naked_single:
                    row, col, num = naked_single
                    board[row][col] = num
                else:
                    break

            # Find best cell for backtracking
            cell = find_best_cell(board)
            if not cell:
                return True  # Solved

            row, col = cell
            candidates = get_candidates(board, row, col)
            techniques['backtrack_calls'] += 1

            for num in sorted(candidates):
                board[row][col] = num
                if solve(board):
                    return True
                board[row][col] = '.'

            return False

        # Make a copy to preserve original
        board_copy = [row[:] for row in board]
        solved = solve(board_copy)

        return solved, board_copy, dict(techniques)

    def word_search_ii_with_pruning(self, board: List[List[str]], words: List[str]) -> Tuple[List[str], Dict[str, int]]:
        """
        LeetCode 212 Extension - Word Search II with Trie Pruning (Hard)

        Find all words from dictionary in 2D board.
        Extended: Use Trie with pruning and return search statistics.

        Algorithm:
        1. Build Trie from word list
        2. DFS with backtracking from each cell
        3. Prune Trie nodes after finding words
        4. Track cells visited and backtracks

        Time: O(m*n*4^L) where L is max word length, Space: O(total_chars)
        """

        class TrieNode:
            def __init__(self):
                self.children = {}
                self.word = None
                self.count = 0  # Number of words below this node

        # Build Trie
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                node.count += 1
            node.word = word

        m, n = len(board), len(board[0])
        found_words = []
        stats = {'cells_visited': 0, 'backtracks': 0, 'trie_prunes': 0}

        def dfs(i: int, j: int, node: TrieNode, visited: Set[Tuple[int, int]]):
            if i < 0 or i >= m or j < 0 or j >= n or (i, j) in visited:
                stats['backtracks'] += 1
                return

            char = board[i][j]
            if char not in node.children or node.children[char].count == 0:
                stats['backtracks'] += 1
                return

            stats['cells_visited'] += 1
            visited.add((i, j))
            node = node.children[char]

            if node.word:
                found_words.append(node.word)
                node.word = None  # Avoid duplicates

                # Prune Trie - decrement counts
                temp = root
                for c in found_words[-1]:
                    temp = temp.children[c]
                    temp.count -= 1
                stats['trie_prunes'] += 1

            # Explore 4 directions
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                dfs(i + di, j + dj, node, visited)

            visited.remove((i, j))

        # Start DFS from each cell
        for i in range(m):
            for j in range(n):
                dfs(i, j, root, set())

        return found_words, stats

    def regular_expression_matching_with_states(self, s: str, p: str) -> Tuple[bool, List[Tuple[int, int]], str]:
        """
        LeetCode 10 Extension - Regular Expression Matching with State Tracking (Hard)

        Match string with pattern containing '.' and '*'.
        Extended: Track matching states and provide explanation.

        Algorithm:
        1. Use memoized backtracking
        2. Handle '*' by trying 0 or more matches
        3. Track state transitions
        4. Build explanation of matching process

        Time: O(m*n), Space: O(m*n)
        """
        memo = {}
        states = []

        def dp(i: int, j: int) -> bool:
            if (i, j) in memo:
                return memo[(i, j)]

            states.append((i, j))

            # Base case: pattern exhausted
            if j == len(p):
                result = i == len(s)
                memo[(i, j)] = result
                return result

            # Check if current characters match
            first_match = i < len(s) and (p[j] == s[i] or p[j] == '.')

            # Handle '*' in pattern
            if j + 1 < len(p) and p[j + 1] == '*':
                # Try 0 matches or 1+ matches
                result = dp(i, j + 2) or (first_match and dp(i + 1, j))
            else:
                # Regular character match
                result = first_match and dp(i + 1, j + 1)

            memo[(i, j)] = result
            return result

        matches = dp(0, 0)

        # Build explanation
        explanation = f"Matching '{s}' with pattern '{p}':\n"
        explanation += f"Total states explored: {len(states)}\n"
        explanation += f"Successful match: {matches}"

        return matches, states, explanation

    def palindrome_partitioning_iii(self, s: str, k: int) -> Tuple[int, List[List[str]]]:
        """
        LeetCode 1278 Extension - Palindrome Partitioning III with Partitions (Hard)

        Partition string into k substrings, minimize total changes to make palindromes.
        Extended: Return actual partitions after changes.

        Algorithm:
        1. Precompute changes needed for each substring
        2. Use DP with backtracking to find optimal partition
        3. Reconstruct partition from DP solution
        4. Apply changes to create palindromes

        Time: O(n²*k), Space: O(n²)
        """
        n = len(s)

        # Precompute changes needed to make s[i:j+1] palindrome
        changes = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                left, right = i, j
                count = 0
                while left < right:
                    if s[left] != s[right]:
                        count += 1
                    left += 1
                    right -= 1
                changes[i][j] = count

        # DP to find minimum changes
        INF = float('inf')
        dp = [[INF] * (k + 1) for _ in range(n + 1)]
        parent = [[(-1, -1)] * (k + 1) for _ in range(n + 1)]

        dp[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                for prev in range(j - 1, i):
                    cost = dp[prev][j - 1] + changes[prev][i - 1]
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        parent[i][j] = (prev, j - 1)

        # Reconstruct partitions
        partitions = []

        def reconstruct(i: int, j: int):
            if i == 0 or j == 0:
                return

            prev_i, prev_j = parent[i][j]
            reconstruct(prev_i, prev_j)

            # Create palindrome from s[prev_i:i]
            substring = list(s[prev_i:i])
            left, right = 0, len(substring) - 1
            while left < right:
                if substring[left] != substring[right]:
                    substring[right] = substring[left]  # Change to make palindrome
                left += 1
                right -= 1

            partitions.append(''.join(substring))

        reconstruct(n, k)

        return dp[n][k], [partitions]

    def combination_sum_with_min_max_constraints(self, candidates: List[int], target: int,
                                                 min_count: int, max_count: int) -> Tuple[List[List[int]], int]:
        """
        Custom Hard - Combination Sum with Count Constraints

        Find combinations summing to target with min/max element count.
        Each number can be used unlimited times.

        Algorithm:
        1. Sort candidates for pruning
        2. Backtrack with count constraints
        3. Prune when sum exceeds or count exceeds
        4. Track total explorations

        Time: O(n^target), Space: O(target)
        """
        candidates.sort()
        result = []
        explorations = 0

        def backtrack(start: int, path: List[int], current_sum: int):
            nonlocal explorations
            explorations += 1

            if current_sum == target and min_count <= len(path) <= max_count:
                result.append(path[:])
                return

            if current_sum > target or len(path) > max_count:
                return

            # Need more elements to reach min_count
            remaining_slots = min_count - len(path)
            if remaining_slots > 0:
                # Check if we can reach target with remaining slots
                min_possible = current_sum + remaining_slots * candidates[0]
                if min_possible > target:
                    return

            for i in range(start, len(candidates)):
                if current_sum + candidates[i] > target:
                    break  # Pruning

                path.append(candidates[i])
                backtrack(i, path, current_sum + candidates[i])
                path.pop()

        backtrack(0, [], 0)
        return result, explorations

    def letter_combinations_with_constraints(self, digits: str, must_contain: Set[str],
                                             forbidden: Set[str]) -> List[str]:
        """
        Custom Hard - Letter Combinations with Constraints

        Generate letter combinations from phone digits with constraints.
        Must contain certain letters, must not contain others.

        Algorithm:
        1. Build digit-to-letters mapping
        2. Backtrack with constraint checking
        3. Prune branches that can't satisfy constraints
        4. Validate complete combinations

        Time: O(4^n * n), Space: O(n)
        """
        if not digits:
            return []

        digit_map = {
            '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
            '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
        }

        result = []
        n = len(digits)

        # Check if constraints are satisfiable
        available_letters = set()
        for digit in digits:
            available_letters.update(digit_map[digit])

        if not must_contain.issubset(available_letters):
            return []  # Impossible to satisfy

        def can_satisfy_constraints(path: List[str], index: int) -> bool:
            # Check if we can still include all must_contain letters
            remaining_positions = n - index
            included = set(path)
            needed = must_contain - included

            return len(needed) <= remaining_positions

        def backtrack(index: int, path: List[str]):
            if index == n:
                path_str = ''.join(path)
                included_set = set(path)

                # Validate constraints
                if must_contain.issubset(included_set) and \
                        forbidden.isdisjoint(included_set):
                    result.append(path_str)
                return

            if not can_satisfy_constraints(path, index):
                return  # Prune

            for letter in digit_map[digits[index]]:
                if letter in forbidden:
                    continue  # Skip forbidden letters

                path.append(letter)
                backtrack(index + 1, path)
                path.pop()

        backtrack(0, [])
        return result

    def word_break_ii_with_optimization(self, s: str, wordDict: List[str]) -> Tuple[List[str], Dict[str, int]]:
        """
        LeetCode 140 Extension - Word Break II with Optimization Stats (Hard)

        Return all possible sentences from string using dictionary.
        Extended: Track optimization statistics.

        Algorithm:
        1. Use memoization to avoid recomputation
        2. Build Trie for efficient prefix matching
        3. Prune impossible substrings early
        4. Track cache hits and explorations

        Time: O(n^3), Space: O(n^3)
        """

        class TrieNode:
            def __init__(self):
                self.children = {}
                self.is_word = False
                self.word = None

        # Build Trie
        root = TrieNode()
        for word in wordDict:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True
            node.word = word

        memo = {}
        stats = {'explorations': 0, 'cache_hits': 0, 'cache_misses': 0}

        def backtrack(start: int) -> List[str]:
            if start in memo:
                stats['cache_hits'] += 1
                return memo[start]

            stats['cache_misses'] += 1
            stats['explorations'] += 1

            if start == len(s):
                return [""]

            result = []
            node = root

            for end in range(start, len(s)):
                char = s[end]
                if char not in node.children:
                    break  # No valid word with this prefix

                node = node.children[char]

                if node.is_word:
                    # Found a valid word, explore remaining
                    word = node.word
                    remaining = backtrack(end + 1)

                    for sentence in remaining:
                        if sentence:
                            result.append(word + " " + sentence)
                        else:
                            result.append(word)

            memo[start] = result
            return result

        sentences = backtrack(0)
        return sentences, dict(stats)

    def optimal_account_balancing(self, transactions: List[List[int]]) -> Tuple[int, List[Tuple[int, int, int]]]:
        """
        LeetCode 465 - Optimal Account Balancing (Hard)

        Minimize transactions to settle all debts.
        Extended: Return actual settlement transactions.

        Algorithm:
        1. Calculate net balance for each person
        2. Separate into debtors and creditors
        3. Use backtracking to find minimum transactions
        4. Prune using balance constraints

        Time: O(n!), Space: O(n)
        """
        # Calculate net balances
        balance = defaultdict(int)
        for giver, receiver, amount in transactions:
            balance[giver] -= amount
            balance[receiver] += amount

        # Get non-zero balances
        balances = [amt for amt in balance.values() if amt != 0]
        n = len(balances)

        if n == 0:
            return 0, []

        min_transactions = float('inf')
        best_solution = []

        def backtrack(index: int, transactions_so_far: List[Tuple[int, int, int]]):
            nonlocal min_transactions, best_solution

            # Skip settled accounts
            while index < n and balances[index] == 0:
                index += 1

            if index == n:
                if len(transactions_so_far) < min_transactions:
                    min_transactions = len(transactions_so_far)
                    best_solution = transactions_so_far[:]
                return

            # Prune if we can't improve
            if len(transactions_so_far) >= min_transactions:
                return

            # Try to settle account[index] with others
            for i in range(index + 1, n):
                if balances[i] * balances[index] < 0:  # Opposite signs
                    # Settle partially or fully
                    amount = min(abs(balances[index]), abs(balances[i]))
                    if balances[index] < 0:
                        amount = -amount

                    balances[i] += amount
                    transactions_so_far.append((index, i, abs(amount)))

                    backtrack(index + 1, transactions_so_far)

                    # Backtrack
                    balances[i] -= amount
                    transactions_so_far.pop()

        backtrack(0, [])

        # Convert indices back to person IDs
        people = list(balance.keys())
        actual_transactions = []
        for i, j, amount in best_solution:
            if balances[i] < 0:
                actual_transactions.append((people[i], people[j], amount))
            else:
                actual_transactions.append((people[j], people[i], amount))

        return min_transactions, actual_transactions

    def android_unlock_patterns(self, m: int, n: int) -> Tuple[int, List[List[int]]]:
        """
        LeetCode 351 - Android Unlock Patterns (Hard)

        Count valid unlock patterns with m to n keys.
        Extended: Return sample patterns.

        Algorithm:
        1. Model skip rules (can't skip unvisited keys)
        2. Use backtracking with visited state
        3. Handle symmetry for optimization
        4. Generate sample patterns

        Time: O(n!), Space: O(n)
        """
        # Skip rules: key -> key -> must visit
        skip = {
            (1, 3): 2, (3, 1): 2,
            (1, 7): 4, (7, 1): 4,
            (1, 9): 5, (9, 1): 5,
            (2, 8): 5, (8, 2): 5,
            (3, 7): 5, (7, 3): 5,
            (3, 9): 6, (9, 3): 6,
            (4, 6): 5, (6, 4): 5,
            (7, 9): 8, (9, 7): 8
        }

        count = 0
        sample_patterns = []

        def is_valid(visited: List[bool], last: int, next_key: int) -> bool:
            if visited[next_key]:
                return False

            if (last, next_key) in skip:
                must_visit = skip[(last, next_key)]
                return visited[must_visit]

            return True

        def backtrack(visited: List[bool], last: int, length: int, path: List[int]):
            nonlocal count

            if length >= m:
                count += 1
                if len(sample_patterns) < 10:  # Collect sample patterns
                    sample_patterns.append(path[:])

            if length == n:
                return

            for next_key in range(1, 10):
                if is_valid(visited, last, next_key):
                    visited[next_key] = True
                    path.append(next_key)

                    backtrack(visited, next_key, length + 1, path)

                    visited[next_key] = False
                    path.pop()

        # Use symmetry: patterns starting with 1,3,7,9 are symmetric
        # Patterns starting with 2,4,6,8 are symmetric
        # Pattern starting with 5 is unique

        visited = [False] * 10

        # Corner keys (1,3,7,9)
        visited[1] = True
        backtrack(visited, 1, 1, [1])
        visited[1] = False
        corner_count = count
        count = corner_count * 4

        # Edge keys (2,4,6,8)
        sample_patterns.clear()
        visited[2] = True
        backtrack(visited, 2, 1, [2])
        visited[2] = False
        edge_count = count - corner_count * 4
        count = corner_count * 4 + edge_count * 4

        # Center key (5)
        sample_patterns.clear()
        visited[5] = True
        backtrack(visited, 5, 1, [5])
        visited[5] = False

        return count, sample_patterns[:5]  # Return first 5 sample patterns