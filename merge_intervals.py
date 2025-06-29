"""
Pattern 4: Merge Intervals - 10 Hard Problems
=============================================

The Merge Intervals pattern deals with problems involving overlapping intervals.
This pattern is essential for solving scheduling problems, time-based queries,
and range merging operations.

Key Concepts:
- Sort intervals by start time (or end time based on problem)
- Iterate through sorted intervals and merge overlapping ones
- Use sweep line algorithm for complex interval problems
- Handle edge cases like touching intervals, nested intervals

Time Complexity: Usually O(n log n) due to sorting
Space Complexity: O(n) for storing results, sometimes O(1) if in-place
"""

from typing import List, Tuple, Optional
import heapq
from collections import defaultdict


class MergeIntervalsHard:

    def employee_free_time(self, schedule: List[List[List[int]]]) -> List[List[int]]:
        """
        LeetCode 759 - Employee Free Time (Hard)

        Find common free time for all employees given their working intervals.
        Each employee may have multiple working intervals.

        Algorithm:
        1. Merge all employee intervals into one list
        2. Sort by start time
        3. Merge overlapping intervals (these are busy times)
        4. Gaps between merged intervals are free times

        Time: O(n log n) where n is total intervals
        Space: O(n)

        Example:
        schedule = [[[1,3],[4,6]],[[1,4]],[[4,5],[6,7]]]
        Output: [[3,4]] - all employees are free from 3 to 4
        """
        # Flatten all intervals
        intervals = []
        for employee in schedule:
            for interval in employee:
                intervals.append(interval)

        # Sort by start time
        intervals.sort(key=lambda x: x[0])

        # Merge overlapping intervals (busy times)
        merged = []
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])

        # Find gaps (free times)
        free_time = []
        for i in range(1, len(merged)):
            free_time.append([merged[i - 1][1], merged[i][0]])

        return free_time

    def interval_list_intersections_extended(self, A: List[List[int]], B: List[List[int]]) -> Tuple[
        List[List[int]], int]:
        """
        LeetCode 986 Extension - Interval List Intersections with Analysis (Hard)

        Find intersection of two interval lists.
        Extended: Also return total intersection length and intersection details.

        Algorithm:
        1. Use two pointers for both lists
        2. Find intersection of current intervals
        3. Move pointer of interval that ends first
        4. Track intersection statistics

        Time: O(m + n), Space: O(min(m, n))

        Example:
        A = [[0,2],[5,10],[13,23],[24,25]]
        B = [[1,5],[8,12],[15,24],[25,26]]
        Output: ([[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]], 18)
        """
        i = j = 0
        intersections = []
        total_length = 0

        while i < len(A) and j < len(B):
            # Find intersection
            start = max(A[i][0], B[j][0])
            end = min(A[i][1], B[j][1])

            if start <= end:
                intersections.append([start, end])
                total_length += end - start + 1

            # Move pointer of interval that ends first
            if A[i][1] < B[j][1]:
                i += 1
            else:
                j += 1

        return intersections, total_length

    def range_module(self):
        """
        LeetCode 715 - Range Module (Hard)

        Implement a data structure to track ranges and handle:
        - addRange(left, right): Add range [left, right)
        - queryRange(left, right): Check if entire range is tracked
        - removeRange(left, right): Remove range [left, right)

        Returns a class implementation with efficient operations.
        """

        class RangeModule:
            def __init__(self):
                # Store intervals as sorted list of [start, end)
                self.intervals = []

            def addRange(self, left: int, right: int) -> None:
                """Add range [left, right) to tracked ranges."""
                new_intervals = []
                i = 0

                # Add all intervals that end before new range starts
                while i < len(self.intervals) and self.intervals[i][1] < left:
                    new_intervals.append(self.intervals[i])
                    i += 1

                # Merge overlapping intervals
                while i < len(self.intervals) and self.intervals[i][0] <= right:
                    left = min(left, self.intervals[i][0])
                    right = max(right, self.intervals[i][1])
                    i += 1

                new_intervals.append([left, right])

                # Add remaining intervals
                while i < len(self.intervals):
                    new_intervals.append(self.intervals[i])
                    i += 1

                self.intervals = new_intervals

            def queryRange(self, left: int, right: int) -> bool:
                """Check if entire range [left, right) is tracked."""
                # Binary search for efficiency
                lo, hi = 0, len(self.intervals) - 1

                while lo <= hi:
                    mid = (lo + hi) // 2
                    if self.intervals[mid][0] <= left < self.intervals[mid][1]:
                        return self.intervals[mid][1] >= right
                    elif self.intervals[mid][0] > left:
                        hi = mid - 1
                    else:
                        lo = mid + 1

                return False

            def removeRange(self, left: int, right: int) -> None:
                """Remove range [left, right) from tracked ranges."""
                new_intervals = []

                for start, end in self.intervals:
                    if end <= left or start >= right:
                        # No overlap
                        new_intervals.append([start, end])
                    elif start < left and end > right:
                        # Remove middle part
                        new_intervals.append([start, left])
                        new_intervals.append([right, end])
                    elif start < left:
                        # Remove right part
                        new_intervals.append([start, left])
                    elif end > right:
                        # Remove left part
                        new_intervals.append([right, end])
                    # else: completely contained, remove entire interval

                self.intervals = new_intervals

        return RangeModule

    def my_calendar_three(self):
        """
        LeetCode 732 - My Calendar III (Hard)

        Implement calendar that tracks maximum K-booking (overlapping events).
        For each new event, return the maximum K-booking after adding it.

        Uses sweep line algorithm for efficient processing.
        """

        class MyCalendarThree:
            def __init__(self):
                # Use sweep line algorithm
                # Track changes at each time point
                self.timeline = defaultdict(int)

            def book(self, start: int, end: int) -> int:
                """
                Book event [start, end) and return max K-booking.

                Time: O(n log n) where n is number of unique time points
                """
                # Mark start and end of event
                self.timeline[start] += 1
                self.timeline[end] -= 1

                # Calculate maximum concurrent events
                max_booking = 0
                current_booking = 0

                # Process timeline in chronological order
                for time in sorted(self.timeline.keys()):
                    current_booking += self.timeline[time]
                    max_booking = max(max_booking, current_booking)

                return max_booking

        return MyCalendarThree

    def remove_covered_intervals(self, intervals: List[List[int]]) -> int:
        """
        LeetCode 1288 - Remove Covered Intervals (Hard variation)

        Remove intervals that are covered by other intervals.
        Extended: Also return which intervals cover which.

        Algorithm:
        1. Sort by start ascending, then by end descending
        2. Track the rightmost point seen
        3. If current interval is covered, mark it
        4. Build coverage relationships

        Time: O(n log n), Space: O(n)

        Example:
        intervals = [[1,4],[3,6],[2,8]]
        Output: 2 (only [2,8] and [3,6] remain, [1,4] is covered by [2,8])
        """
        # Sort by start ascending, end descending
        sorted_intervals = sorted(enumerate(intervals),
                                  key=lambda x: (x[1][0], -x[1][1]))

        remaining = 0
        coverage_map = {}  # interval_idx -> covering_interval_idx
        prev_end = -1

        for idx, (orig_idx, interval) in enumerate(sorted_intervals):
            # If current interval is not covered
            if interval[1] > prev_end:
                remaining += 1
                prev_end = interval[1]

                # Check if this interval covers previous ones
                for prev_idx in range(idx):
                    prev_orig_idx, prev_interval = sorted_intervals[prev_idx]
                    if (interval[0] <= prev_interval[0] and
                            interval[1] >= prev_interval[1]):
                        coverage_map[prev_orig_idx] = orig_idx
            else:
                # Current interval is covered
                # Find which interval covers it
                for prev_idx in range(idx):
                    prev_orig_idx, prev_interval = sorted_intervals[prev_idx]
                    if (prev_interval[0] <= interval[0] and
                            prev_interval[1] >= interval[1]):
                        coverage_map[orig_idx] = prev_orig_idx
                        break

        return remaining

    def meeting_rooms_iii(self, n: int, meetings: List[List[int]]) -> int:
        """
        LeetCode 2402 - Meeting Rooms III (Hard)

        Allocate meetings to n rooms. Each meeting waits if no room available.
        Return the room that held the most meetings.

        Algorithm:
        1. Sort meetings by start time
        2. Use min heap for available rooms
        3. Use min heap for occupied rooms (by end time)
        4. Track meeting count per room

        Time: O(m log m + m log n) where m = meetings
        Space: O(n)

        Example:
        n = 2, meetings = [[0,10],[1,5],[2,7],[3,4]]
        Output: 0 (room 0 hosts meetings at [0,10] and [3,4])
        """
        meetings.sort()  # Sort by start time

        # Min heap of available rooms
        available = list(range(n))
        heapq.heapify(available)

        # Min heap of (end_time, room) for occupied rooms
        occupied = []

        # Count meetings per room
        room_count = [0] * n

        for start, end in meetings:
            # Free up rooms that have ended
            while occupied and occupied[0][0] <= start:
                _, room = heapq.heappop(occupied)
                heapq.heappush(available, room)

            if available:
                # Assign to available room
                room = heapq.heappop(available)
                heapq.heappush(occupied, (end, room))
            else:
                # Wait for earliest room to free up
                prev_end, room = heapq.heappop(occupied)
                # Meeting is delayed
                new_end = prev_end + (end - start)
                heapq.heappush(occupied, (new_end, room))

            room_count[room] += 1

        # Return room with most meetings
        max_meetings = max(room_count)
        for i in range(n):
            if room_count[i] == max_meetings:
                return i

    def rectangle_area_ii(self, rectangles: List[List[int]]) -> int:
        """
        LeetCode 850 - Rectangle Area II (Hard)

        Calculate total area covered by all rectangles, handling overlaps.

        Algorithm:
        1. Use coordinate compression
        2. Create events for rectangle edges
        3. Sweep through x-coordinates
        4. For each x-segment, calculate y-coverage

        Time: O(n² log n), Space: O(n)

        Example:
        rectangles = [[0,0,2,2],[1,0,2,3],[1,0,3,1]]
        Output: 6
        """
        MOD = 10 ** 9 + 7

        # Collect all x-coordinates
        x_coords = set()
        for x1, y1, x2, y2 in rectangles:
            x_coords.add(x1)
            x_coords.add(x2)

        x_coords = sorted(x_coords)

        total_area = 0

        # Process each x-interval
        for i in range(len(x_coords) - 1):
            x1, x2 = x_coords[i], x_coords[i + 1]

            # Collect y-intervals active in [x1, x2]
            y_intervals = []
            for rect_x1, rect_y1, rect_x2, rect_y2 in rectangles:
                if rect_x1 <= x1 and x2 <= rect_x2:
                    y_intervals.append([rect_y1, rect_y2])

            # Merge y-intervals
            y_intervals.sort()
            merged_y = []

            for interval in y_intervals:
                if not merged_y or merged_y[-1][1] < interval[0]:
                    merged_y.append(interval)
                else:
                    merged_y[-1][1] = max(merged_y[-1][1], interval[1])

            # Calculate area for this x-interval
            y_length = sum(y2 - y1 for y1, y2 in merged_y)
            total_area += (x2 - x1) * y_length
            total_area %= MOD

        return total_area

    def falling_squares(self, positions: List[List[int]]) -> List[int]:
        """
        LeetCode 699 - Falling Squares (Hard)

        Squares fall onto the X-axis. Return height after each square falls.
        Each position is [left, side_length].

        Algorithm:
        1. For each square, find maximum height in its range
        2. Place square at that height
        3. Update heights in the range
        4. Use interval tree or coordinate compression

        Time: O(n²), Space: O(n)
        Can be optimized to O(n log n) with segment tree

        Example:
        positions = [[1,2],[2,3],[6,1]]
        Output: [2,5,5]
        """
        heights = []
        intervals = []  # (left, right, height)

        for left, size in positions:
            right = left + size
            max_height = 0

            # Find maximum height in range [left, right)
            for int_left, int_right, int_height in intervals:
                if int_left < right and left < int_right:
                    max_height = max(max_height, int_height)

            # New height for this square
            new_height = max_height + size

            # Update intervals
            new_intervals = []

            # Process existing intervals
            for int_left, int_right, int_height in intervals:
                if int_right <= left or int_left >= right:
                    # No overlap
                    new_intervals.append([int_left, int_right, int_height])
                elif int_left >= left and int_right <= right:
                    # Completely covered, update height
                    new_intervals.append([int_left, int_right, new_height])
                elif int_left < left and int_right > right:
                    # Split into three parts
                    new_intervals.append([int_left, left, int_height])
                    new_intervals.append([left, right, new_height])
                    new_intervals.append([right, int_right, int_height])
                elif int_left < left:
                    # Partial overlap on right
                    new_intervals.append([int_left, left, int_height])
                    new_intervals.append([left, int_right, new_height])
                else:
                    # Partial overlap on left
                    new_intervals.append([int_left, right, new_height])
                    new_intervals.append([right, int_right, int_height])

            # Add new interval if not already covered
            covered = False
            for int_left, int_right, _ in intervals:
                if int_left <= left and right <= int_right:
                    covered = True
                    break

            if not covered:
                new_intervals.append([left, right, new_height])

            # Merge adjacent intervals with same height
            intervals = []
            new_intervals.sort()

            for interval in new_intervals:
                if intervals and intervals[-1][1] == interval[0] and intervals[-1][2] == interval[2]:
                    intervals[-1][1] = interval[1]
                else:
                    intervals.append(interval)

            # Find current maximum height
            current_max = max(h for _, _, h in intervals)
            heights.append(current_max)

        return heights

    def skyline_problem(self, buildings: List[List[int]]) -> List[List[int]]:
        """
        LeetCode 218 - The Skyline Problem (Hard)

        Compute skyline formed by buildings.

        Algorithm:
        1. Create events for building start/end
        2. Process events in order
        3. Track active building heights
        4. Output key points where height changes

        Time: O(n log n), Space: O(n)

        Example:
        buildings = [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]
        Output: [[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]
        """
        # Create events: (x, is_start, height)
        events = []
        for left, right, height in buildings:
            events.append((left, False, height))  # Start event
            events.append((right, True, height))  # End event

        # Sort events: by x, then end before start, then by height
        events.sort(key=lambda x: (x[0], x[1], -x[2] if not x[1] else x[2]))

        result = []
        # Max heap for heights (use negative for max heap)
        active_heights = [0]  # Ground level

        i = 0
        while i < len(events):
            curr_x = events[i][0]

            # Process all events at same x-coordinate
            while i < len(events) and events[i][0] == curr_x:
                x, is_end, height = events[i]

                if not is_end:
                    # Building starts
                    heapq.heappush(active_heights, -height)
                else:
                    # Building ends
                    active_heights.remove(-height)
                    heapq.heapify(active_heights)

                i += 1

            # Current maximum height
            max_height = -active_heights[0]

            # Add key point if height changed
            if not result or result[-1][1] != max_height:
                result.append([curr_x, max_height])

        return result

    def data_stream_as_disjoint_intervals(self):
        """
        LeetCode 352 - Data Stream as Disjoint Intervals (Hard)

        Implement data structure that tracks integers from data stream
        and returns them as disjoint intervals.

        Returns a class implementation.
        """

        class SummaryRanges:
            def __init__(self):
                self.intervals = []

            def addNum(self, val: int) -> None:
                """
                Add number to stream and maintain disjoint intervals.

                Time: O(n) where n is number of intervals
                Can be optimized to O(log n) with balanced BST
                """
                left, right = val, val
                new_intervals = []
                inserted = False

                for start, end in self.intervals:
                    if end < left - 1:
                        # No overlap, interval comes before
                        new_intervals.append([start, end])
                    elif start > right + 1:
                        # No overlap, interval comes after
                        if not inserted:
                            new_intervals.append([left, right])
                            inserted = True
                        new_intervals.append([start, end])
                    else:
                        # Overlap or adjacent, merge
                        left = min(left, start)
                        right = max(right, end)

                if not inserted:
                    new_intervals.append([left, right])

                self.intervals = new_intervals

            def getIntervals(self) -> List[List[int]]:
                """Return current disjoint intervals."""
                return self.intervals

        return SummaryRanges


# Example usage and testing
if __name__ == "__main__":
    solver = MergeIntervalsHard()

    # Test 1: Employee Free Time
    print("1. Employee Free Time:")
    schedule = [[[1, 3], [4, 6]], [[1, 4]], [[4, 5], [6, 7]]]
    print(f"   Input: {schedule}")
    print(f"   Output: {solver.employee_free_time(schedule)}")
    print()

    # Test 2: Interval Intersections
    print("2. Interval List Intersections:")
    A = [[0, 2], [5, 10], [13, 23], [24, 25]]
    B = [[1, 5], [8, 12], [15, 24], [25, 26]]
    intersections, total_length = solver.interval_list_intersections_extended(A, B)
    print(f"   Input: A={A}, B={B}")
    print(f"   Output: Intersections={intersections}, Total Length={total_length}")
    print()

    # Test 3: Remove Covered Intervals
    print("3. Remove Covered Intervals:")
    intervals = [[1, 4], [3, 6], [2, 8]]
    print(f"   Input: {intervals}")
    print(f"   Output: {solver.remove_covered_intervals(intervals)}")
    print()

    # Test 4: Rectangle Area
    print("4. Rectangle Area II:")
    rectangles = [[0, 0, 2, 2], [1, 0, 2, 3], [1, 0, 3, 1]]
    print(f"   Input: {rectangles}")
    print(f"   Output: {solver.rectangle_area_ii(rectangles)}")