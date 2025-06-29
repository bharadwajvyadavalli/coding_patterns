"""
Pattern 23: Greedy Algorithms - 10 Hard Problems
===============================================

The Greedy Algorithms pattern makes locally optimal choices at each step, hoping
to find a global optimum. This pattern works when the problem has optimal
substructure and the greedy choice property.

Key Concepts:
- Make the best choice at each step
- Never reconsider previous choices
- Prove greedy choice leads to optimal solution
- Often involves sorting or priority queues

Time Complexity: Usually O(n log n) due to sorting
Space Complexity: Usually O(1) to O(n)
"""

from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict, deque
import heapq
import bisect


class GreedyAlgorithmsHard:

    def interval_scheduling_maximization_with_weights(self, intervals: List[List[int]], weights: List[int]) -> Tuple[
        int, List[int], int]:
        """
        Custom Hard - Weighted Interval Scheduling with Maximum Value

        Select non-overlapping intervals to maximize total weight.
        Extended: Return selected intervals and total weight.

        Algorithm:
        1. Sort by end time
        2. Use DP with binary search for optimization
        3. Track selected intervals for reconstruction
        4. Compare with greedy approach

        Time: O(n log n), Space: O(n)

        Example:
        intervals = [[1,3],[2,5],[4,7]], weights = [5,6,4]
        Output: (11, [0,2], 11) - select intervals 0,2 for weight 11
        """
        n = len(intervals)

        # Create jobs with indices
        jobs = [(intervals[i][0], intervals[i][1], weights[i], i)
                for i in range(n)]

        # Sort by end time
        jobs.sort(key=lambda x: x[1])

        # DP approach for weighted interval scheduling
        dp = [0] * n
        parent = [-1] * n

        dp[0] = jobs[0][2]

        for i in range(1, n):
            # Include current job
            incl = jobs[i][2]

            # Find latest non-conflicting job using binary search
            left, right = 0, i - 1
            latest_non_conflict = -1

            while left <= right:
                mid = (left + right) // 2
                if jobs[mid][1] <= jobs[i][0]:
                    latest_non_conflict = mid
                    left = mid + 1
                else:
                    right = mid - 1

            if latest_non_conflict != -1:
                incl += dp[latest_non_conflict]

            # Exclude current job
            excl = dp[i - 1]

            if incl > excl:
                dp[i] = incl
                parent[i] = latest_non_conflict
            else:
                dp[i] = excl
                parent[i] = i - 1

        # Reconstruct selected intervals
        selected = []
        i = n - 1

        while i >= 0:
            if parent[i] == -1 or (parent[i] >= 0 and dp[i] != dp[parent[i]]):
                selected.append(jobs[i][3])  # Original index
                i = parent[i]
            else:
                i -= 1

        selected.reverse()

        return dp[n - 1], selected, dp[n - 1]

    def jump_game_minimum_jumps_with_path(self, nums: List[int]) -> Tuple[int, List[int], bool]:
        """
        LeetCode 45 Extension - Jump Game II with Path (Hard)

        Find minimum jumps to reach end and the path taken.
        Use greedy approach with path reconstruction.

        Algorithm:
        1. Track farthest reachable at each level
        2. Greedy choice: jump when must to maximize range
        3. Record jump positions
        4. Handle unreachable cases

        Time: O(n), Space: O(n)
        """
        n = len(nums)
        if n <= 1:
            return 0, [0], True

        jumps = 0
        current_end = 0
        farthest = 0
        path = [0]

        for i in range(n - 1):
            farthest = max(farthest, i + nums[i])

            # If we can't progress beyond current position
            if farthest <= i:
                return -1, [], False

            # Must jump from current level
            if i == current_end:
                jumps += 1
                current_end = farthest

                # Find best position to jump from
                best_pos = i
                best_reach = i + nums[i]

                for j in range(path[-1] + 1, i + 1):
                    if j + nums[j] >= best_reach:
                        best_reach = j + nums[j]
                        best_pos = j

                if best_pos != path[-1]:
                    path.append(best_pos)

                # Can reach end
                if current_end >= n - 1:
                    break

        path.append(n - 1)
        return jumps, path, True

    def task_scheduler_with_schedule(self, tasks: List[str], n: int) -> Tuple[int, List[str]]:
        """
        LeetCode 621 Extension - Task Scheduler with Actual Schedule (Hard)

        Schedule tasks with cooldown period n between same tasks.
        Extended: Return the actual schedule.

        Algorithm:
        1. Count task frequencies
        2. Use max heap for greedy selection
        3. Track cooldown periods
        4. Build actual schedule

        Time: O(m) where m is total tasks, Space: O(1)
        """
        if not tasks:
            return 0, []

        # Count frequencies
        freq = Counter(tasks)

        # Max heap of frequencies (negative for max heap)
        max_heap = [-f for f in freq.values()]
        heapq.heapify(max_heap)

        schedule = []
        cooldown = deque()  # (available_time, frequency)
        time = 0

        while max_heap or cooldown:
            # Move tasks from cooldown back to heap if ready
            while cooldown and cooldown[0][0] <= time:
                _, freq = cooldown.popleft()
                heapq.heappush(max_heap, freq)

            if max_heap:
                # Execute highest frequency task
                freq = heapq.heappop(max_heap)

                # Find which task has this frequency
                for task, f in freq.items():
                    if -freq == f:
                        schedule.append(task)
                        freq[task] -= 1
                        break

                # Add to cooldown if more instances remain
                if freq < -1:
                    cooldown.append((time + n + 1, freq + 1))
            else:
                # CPU idle
                schedule.append('idle')

            time += 1

        return len(schedule), schedule

    def candy_distribution_with_explanation(self, ratings: List[int]) -> Tuple[int, List[int], str]:
        """
        LeetCode 135 Extension - Candy Distribution with Explanation (Hard)

        Distribute candy with constraints: everyone gets at least 1,
        higher rating gets more than neighbors.
        Extended: Show distribution and explanation.

        Algorithm:
        1. Two passes: left-to-right, right-to-left
        2. Greedy: give minimum satisfying constraints
        3. Track reason for each candy count

        Time: O(n), Space: O(n)
        """
        n = len(ratings)
        candies = [1] * n
        reasons = ['base'] * n

        # Left to right pass
        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                candies[i] = candies[i - 1] + 1
                reasons[i] = f'higher than left ({ratings[i]} > {ratings[i - 1]})'

        # Right to left pass
        for i in range(n - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                if candies[i] <= candies[i + 1]:
                    candies[i] = candies[i + 1] + 1
                    reasons[i] = f'higher than right ({ratings[i]} > {ratings[i + 1]})'

        total = sum(candies)

        # Build explanation
        explanation = f"Total candies: {total}\n"
        explanation += "Distribution:\n"
        for i in range(n):
            explanation += f"  Child {i} (rating {ratings[i]}): "
            explanation += f"{candies[i]} candies - {reasons[i]}\n"

        return total, candies, explanation

    def reorganize_string_with_strategy(self, s: str) -> Tuple[str, bool, Dict[str, int]]:
        """
        LeetCode 767 Extension - Reorganize String with Strategy (Hard)

        Reorganize string so no adjacent characters are same.
        Extended: Show character placement strategy.

        Algorithm:
        1. Count frequencies
        2. Use max heap for greedy placement
        3. Place most frequent chars first
        4. Track placement positions

        Time: O(n log k) where k is unique chars, Space: O(k)
        """
        # Count frequencies
        freq = Counter(s)

        # Check if reorganization is possible
        max_freq = max(freq.values())
        if max_freq > (len(s) + 1) // 2:
            return "", False, {}

        # Max heap of (freq, char)
        heap = [(-f, ch) for ch, f in freq.items()]
        heapq.heapify(heap)

        result = []
        prev_freq = 0
        prev_char = ''

        positions = defaultdict(list)

        while heap:
            # Get most frequent character
            freq1, char1 = heapq.heappop(heap)

            # Add back previous character if any
            if prev_freq < 0:
                heapq.heappush(heap, (prev_freq, prev_char))

            # Place current character
            result.append(char1)
            positions[char1].append(len(result) - 1)

            # Update for next iteration
            prev_freq = freq1 + 1  # Decrease frequency
            prev_char = char1

        return ''.join(result), True, dict(positions)

    def minimum_deletions_to_make_array_beautiful(self, nums: List[int]) -> Tuple[int, List[int], List[int]]:
        """
        LeetCode 2216 Extension - Min Deletions for Beautiful Array (Hard)

        Make array beautiful: every consecutive pair has different values.
        Extended: Return deleted indices and final array.

        Algorithm:
        1. Greedy: delete second of consecutive equal elements
        2. Track deletions and positions
        3. Build final array

        Time: O(n), Space: O(n)
        """
        n = len(nums)
        deletions = 0
        deleted_indices = []
        final_array = []

        i = 0
        while i < n:
            final_array.append(nums[i])

            # Check if we need to skip next element
            if i + 1 < n and nums[i] == nums[i + 1]:
                deleted_indices.append(i + 1)
                deletions += 1
                i += 2  # Skip the duplicate
            else:
                i += 1

        return deletions, deleted_indices, final_array

    def gas_station_with_analysis(self, gas: List[int], cost: List[int]) -> Tuple[int, List[int], int]:
        """
        LeetCode 134 Extension - Gas Station with Journey Analysis (Hard)

        Find starting gas station for circular route.
        Extended: Show gas levels throughout journey.

        Algorithm:
        1. Greedy insight: if total gas >= total cost, solution exists
        2. Start from station after lowest point
        3. Track gas levels at each station

        Time: O(n), Space: O(n)
        """
        n = len(gas)
        total_gas = sum(gas)
        total_cost = sum(cost)

        if total_gas < total_cost:
            return -1, [], 0

        # Find starting point using greedy approach
        tank = 0
        start = 0
        min_tank = float('inf')
        min_station = 0

        for i in range(n):
            tank += gas[i] - cost[i]
            if tank < min_tank:
                min_tank = tank
                min_station = i

        # Start from station after minimum
        start = (min_station + 1) % n

        # Simulate journey
        tank = 0
        gas_levels = []

        for i in range(n):
            station = (start + i) % n
            tank += gas[station]
            gas_levels.append(tank)
            tank -= cost[station]

        return start, gas_levels, tank

    def maximum_performance_team(self, n: int, speed: List[int], efficiency: List[int], k: int) -> Tuple[
        int, List[int]]:
        """
        LeetCode 1383 Extension - Maximum Performance of Team (Hard)

        Select at most k engineers to maximize performance.
        Performance = sum(speed) * min(efficiency)
        Extended: Return selected engineers.

        Algorithm:
        1. Sort by efficiency descending
        2. Greedy: try each engineer as minimum efficiency
        3. Select k-1 fastest engineers with higher efficiency
        4. Use min heap for speed management

        Time: O(n log n), Space: O(k)
        """
        MOD = 10 ** 9 + 7

        # Combine and sort by efficiency descending
        engineers = [(efficiency[i], speed[i], i) for i in range(n)]
        engineers.sort(reverse=True)

        max_performance = 0
        best_team = []
        speed_heap = []
        speed_sum = 0

        for eff, spd, idx in engineers:
            # Add current engineer
            heapq.heappush(speed_heap, spd)
            speed_sum += spd

            # Remove slowest if team too large
            if len(speed_heap) > k:
                speed_sum -= heapq.heappop(speed_heap)

            # Calculate performance with current minimum efficiency
            performance = speed_sum * eff

            if performance > max_performance:
                max_performance = performance
                # Record current team
                best_team = [idx]
                temp_heap = list(speed_heap)

                # Find actual engineers in heap
                for e, s, i in engineers:
                    if s in temp_heap and i != idx:
                        best_team.append(i)
                        temp_heap.remove(s)
                        if len(best_team) == len(speed_heap):
                            break

        return max_performance % MOD, best_team[:k]

    def minimum_cost_to_hire_workers(self, quality: List[int], wage: List[int], k: int) -> Tuple[float, List[int]]:
        """
        LeetCode 857 Extension - Minimum Cost to Hire K Workers (Hard)

        Hire exactly k workers minimizing total cost.
        Pay must be proportional to quality.
        Extended: Return hired workers.

        Algorithm:
        1. Sort by wage/quality ratio
        2. Greedy: try each ratio as base
        3. Select k workers with smallest quality
        4. Use heap for quality management

        Time: O(n log n), Space: O(n)
        """
        n = len(quality)

        # Calculate wage/quality ratio for each worker
        workers = [(wage[i] / quality[i], quality[i], i) for i in range(n)]
        workers.sort()  # Sort by ratio

        min_cost = float('inf')
        hired_workers = []
        quality_heap = []
        quality_sum = 0

        for ratio, q, idx in workers:
            # Add current worker
            heapq.heappush(quality_heap, -q)  # Max heap
            quality_sum += q

            # Remove highest quality if too many workers
            if len(quality_heap) > k:
                quality_sum += heapq.heappop(quality_heap)  # Remove largest

            # Calculate cost with current ratio
            if len(quality_heap) == k:
                cost = quality_sum * ratio

                if cost < min_cost:
                    min_cost = cost
                    # Record current team
                    hired_workers = [idx]
                    temp_qualities = [-h for h in quality_heap]

                    for r, qual, i in workers[:workers.index((ratio, q, idx))]:
                        if qual in temp_qualities and i != idx:
                            hired_workers.append(i)
                            if len(hired_workers) == k:
                                break

        return min_cost, hired_workers

    def find_maximum_number_after_k_swaps(self, num: str, k: int) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Custom Hard - Maximum Number After K Adjacent Swaps

        Make number maximum using at most k adjacent swaps.
        Extended: Return swap operations.

        Algorithm:
        1. Greedy: move largest digits forward
        2. For each position, find best digit within k swaps
        3. Bubble that digit to current position
        4. Track all swaps made

        Time: O(n²), Space: O(k)
        """
        digits = list(num)
        n = len(digits)
        swaps = []
        swaps_remaining = k

        for i in range(n):
            if swaps_remaining == 0:
                break

            # Find the largest digit within reach
            max_digit = digits[i]
            max_pos = i

            for j in range(i + 1, min(i + swaps_remaining + 1, n)):
                if digits[j] > max_digit:
                    max_digit = digits[j]
                    max_pos = j

            # Bubble max digit to position i
            while max_pos > i and swaps_remaining > 0:
                # Swap with previous position
                digits[max_pos], digits[max_pos - 1] = digits[max_pos - 1], digits[max_pos]
                swaps.append((max_pos - 1, max_pos))
                max_pos -= 1
                swaps_remaining -= 1

        return ''.join(digits), swaps