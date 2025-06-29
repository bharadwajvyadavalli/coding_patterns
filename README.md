# 23 Essential Coding Patterns - Hard Problems Collection

A comprehensive collection of 23 fundamental coding patterns, each with 10 carefully selected hard problems. This repository serves as an advanced study guide for mastering algorithmic problem-solving techniques.

## 📚 Overview

This collection covers 230 hard problems (10 per pattern) designed to challenge and improve your problem-solving skills. Each pattern includes:
- Detailed explanations of the pattern
- Key concepts and techniques
- Time and space complexity analysis
- 10 hard problems with complete solutions
- Real-world applications

## 🎯 Patterns List

### 1. **Sliding Window**
- **Concept**: Maintain a window of elements while sliding through data
- **Use Cases**: Substring problems, subarray optimization, stream processing
- **Key Problems**: 
  - Minimum Window Substring
  - Longest Substring with At Most K Distinct Characters
  - Substring with Concatenation of All Words
- **Time Complexity**: Usually O(n)
- **File**: `sliding_window.py`

### 2. **Two Pointers**
- **Concept**: Use two pointers to traverse data structure efficiently
- **Use Cases**: Sorted arrays, palindromes, pair finding
- **Key Problems**:
  - 3Sum Closest with All Solutions
  - Trapping Rain Water with Levels
  - Container With Most Water for K Containers
- **Time Complexity**: O(n) or O(n log n)
- **File**: `two_pointers.py`

### 3. **Fast & Slow Pointers**
- **Concept**: Floyd's Tortoise and Hare algorithm for cycle detection
- **Use Cases**: Cycle detection, finding middle element, linked list problems
- **Key Problems**:
  - Find Duplicate with Cycle Analysis
  - Linked List Cycle II Extended
  - Happy Number with Path Analysis
- **Time Complexity**: O(n)
- **File**: `fast_slow_pointers.py`

### 4. **Merge Intervals**
- **Concept**: Combine overlapping or adjacent intervals
- **Use Cases**: Calendar scheduling, resource allocation, time-based events
- **Key Problems**:
  - Insert and Merge Intervals
  - Interval List Intersections
  - Employee Free Time
- **Time Complexity**: O(n log n)
- **File**: `merge_intervals.py`

### 5. **Cyclic Sort**
- **Concept**: Place elements at their correct positions when dealing with ranges
- **Use Cases**: Finding missing/duplicate numbers, array problems with known range
- **Key Problems**:
  - Find All Duplicates and Missing Numbers
  - First Missing Positive with Constraints
  - Minimum Swaps to Sort
- **Time Complexity**: O(n)
- **File**: `cyclic_sort.py`

### 6. **In-place Reversal of Linked List**
- **Concept**: Reverse linked lists or portions without extra space
- **Use Cases**: Linked list manipulation, memory-efficient operations
- **Key Problems**:
  - Reverse Nodes in k-Group Extended
  - Reverse Between Patterns
  - Plus One Linked List
- **Time Complexity**: O(n)
- **File**: `in_place_reverse_linked_list.py`

### 7. **Tree Breadth-First Search (BFS)**
- **Concept**: Level-by-level tree traversal using queue
- **Use Cases**: Level order traversal, shortest path in trees
- **Key Problems**:
  - Vertical Order Traversal with Sorting
  - Maximum Width of Binary Tree
  - All Nodes Distance K
- **Time Complexity**: O(n)
- **File**: `bfs.py`

### 8. **Tree Depth-First Search (DFS)**
- **Concept**: Explore tree paths deeply before backtracking
- **Use Cases**: Path problems, tree validation, parent-child relationships
- **Key Problems**:
  - Binary Tree Maximum Path Sum with Paths
  - Serialize and Deserialize Binary Tree
  - Binary Tree Cameras
- **Time Complexity**: O(n)
- **File**: `dfs.py`

### 9. **Two Heaps**
- **Concept**: Use min-heap and max-heap to track median or balance data
- **Use Cases**: Median finding, data stream processing
- **Key Problems**:
  - Sliding Window Median with Tracking
  - Find Median from Data Stream
  - IPO (Initial Public Offering)
- **Time Complexity**: O(log n) for operations
- **File**: `two_heaps.py`

### 10. **Subsets**
- **Concept**: Generate all possible subsets/combinations/permutations
- **Use Cases**: Combinatorial problems, power set generation
- **Key Problems**:
  - Subsets with Multiple Constraints
  - Combination Sum IV with Paths
  - Permutation Sequence
- **Time Complexity**: O(2^n) or O(n!)
- **File**: `subsets.py`

### 11. **Modified Binary Search**
- **Concept**: Adapt binary search for complex scenarios
- **Use Cases**: Rotated arrays, boundary finding, optimization problems
- **Key Problems**:
  - Median of Two Sorted Arrays
  - Search in Rotated Sorted Array II
  - Find Peak Elements
- **Time Complexity**: O(log n)
- **File**: `modified_binary_search.py`

### 12. **Top K Elements**
- **Concept**: Find K largest/smallest elements efficiently
- **Use Cases**: Priority problems, ranking, statistics
- **Key Problems**:
  - K Closest Points with Distances
  - Top K Frequent Words
  - Kth Largest Element in Stream
- **Time Complexity**: O(n log k)
- **File**: `top_k_elements.py`

### 13. **K-way Merge**
- **Concept**: Merge K sorted arrays/lists efficiently
- **Use Cases**: External sorting, stream merging
- **Key Problems**:
  - Merge k Sorted Lists with Statistics
  - Smallest Range Covering K Lists
  - Kth Smallest in Sorted Matrix
- **Time Complexity**: O(N log K)
- **File**: `k_way_merge.py`

### 14. **Topological Sort**
- **Concept**: Order vertices in directed acyclic graph
- **Use Cases**: Dependency resolution, task scheduling
- **Key Problems**:
  - Alien Dictionary with Analysis
  - Course Schedule III
  - Parallel Courses II
- **Time Complexity**: O(V + E)
- **File**: `topological_sort.py`

### 15. **0/1 Knapsack (Dynamic Programming)**
- **Concept**: Select items with binary choice (take or not take)
- **Use Cases**: Resource allocation, subset selection
- **Key Problems**:
  - Partition to K Equal Sum Subsets
  - Target Sum with Paths
  - Last Stone Weight II
- **Time Complexity**: O(n * capacity)
- **File**: `knapsack.py`

### 16. **Unbounded Knapsack**
- **Concept**: Select items with unlimited quantity
- **Use Cases**: Coin change, cutting problems
- **Key Problems**:
  - Coin Change with Usage Tracking
  - Integer Break with Factors
  - Perfect Squares with Path
- **Time Complexity**: O(n * target)
- **File**: `unbounded_knapsack.py`

### 17. **Fibonacci Numbers**
- **Concept**: Problems with recursive state dependencies
- **Use Cases**: Sequence problems, climbing stairs variations
- **Key Problems**:
  - Climbing Stairs with Variable Steps
  - House Robber III with Strategy
  - Domino and Tromino Tiling
- **Time Complexity**: O(n)
- **File**: `fibonacci_numbers.py`

### 18. **Palindromic Subsequence**
- **Concept**: Find palindromes in strings using DP
- **Use Cases**: String manipulation, subsequence problems
- **Key Problems**:
  - Longest Palindromic Subsequence with K Changes
  - Count Palindromic Subsequences
  - Minimum Deletions to Make Palindrome
- **Time Complexity**: O(n²)
- **File**: `palindromic_subsequence.py`

### 19. **Longest Common Substring/Subsequence**
- **Concept**: Find common sequences in multiple strings
- **Use Cases**: Diff algorithms, DNA sequencing
- **Key Problems**:
  - LCS of Three Strings
  - Shortest Common Supersequence
  - Edit Distance with Operations
- **Time Complexity**: O(m*n)
- **File**: `longest_common_subsequence.py`

### 20. **Longest Increasing Subsequence**
- **Concept**: Find longest subsequence with increasing order
- **Use Cases**: Patience sorting, box stacking
- **Key Problems**:
  - LIS with K Changes
  - Russian Doll Envelopes
  - Maximum Height by Stacking Cuboids
- **Time Complexity**: O(n log n)
- **File**: `longest_increasing_subsequence.py`

### 21. **Bitwise XOR**
- **Concept**: Use XOR properties for efficient solutions
- **Use Cases**: Finding unique elements, bit manipulation
- **Key Problems**:
  - Find Missing and Repeated with Bits
  - Maximum XOR of Two Numbers
  - XOR Queries of Subarray
- **Time Complexity**: O(n)
- **File**: `bitwise_xor.py`

### 22. **Backtracking**
- **Concept**: Build solutions incrementally with ability to backtrack
- **Use Cases**: Puzzles, constraint satisfaction, optimization
- **Key Problems**:
  - N-Queens with Complete Analysis
  - Sudoku Solver with Techniques
  - Word Search II with Trie Pruning
- **Time Complexity**: Often exponential
- **File**: `backtracking.py`

### 23. **Greedy Algorithms**
- **Concept**: Make locally optimal choices for global optimum
- **Use Cases**: Optimization problems, scheduling, resource allocation
- **Key Problems**:
  - Weighted Interval Scheduling
  - Jump Game II with Path
  - Maximum Performance of Team
- **Time Complexity**: Usually O(n log n)
- **File**: `greedy_algorithms.py`

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Basic understanding of data structures and algorithms
- Familiarity with time/space complexity analysis

### Installation
```bash
git clone https://github.com/yourusername/23-coding-patterns.git
cd 23-coding-patterns
```

### Usage
Each pattern is self-contained in its own Python file:

```python
# Example: Using the Sliding Window pattern
from sliding_window import SlidingWindowHard

solver = SlidingWindowHard()
s = "ADOBECODEBANC"
t = "ABC"
result = solver.min_window_substring(s, t)
print(f"Minimum window: {result}")
```

## 📖 How to Study

1. **Start with Pattern Understanding**
   - Read the pattern description and key concepts
   - Understand when and why to use this pattern

2. **Study Example Problems**
   - Start with simpler problems in each pattern
   - Gradually move to harder problems
   - Try solving before looking at solutions

3. **Practice Implementation**
   - Implement solutions from scratch
   - Focus on edge cases and optimization
   - Time yourself to simulate interview conditions

4. **Pattern Recognition**
   - Learn to identify which pattern applies to a problem
   - Some problems may combine multiple patterns
   - Build intuition through practice

## 🎓 Learning Path

### Beginner Level (Weeks 1-4)
- Sliding Window
- Two Pointers
- Fast & Slow Pointers
- Merge Intervals

### Intermediate Level (Weeks 5-8)
- Cyclic Sort
- Tree BFS
- Tree DFS
- Modified Binary Search

### Advanced Level (Weeks 9-12)
- Dynamic Programming patterns (Knapsack, Fibonacci, LCS, LIS)
- Backtracking
- Greedy Algorithms

### Expert Level (Weeks 13-16)
- Two Heaps
- K-way Merge
- Topological Sort
- Bitwise XOR

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Add more problems to existing patterns
- Improve explanations or solutions
- Fix bugs or optimize code
- Add visualizations or diagrams

## 📝 Problem Difficulty Progression

Each pattern includes problems ranging from:
- **Base Hard**: LeetCode Hard level problems
- **Extended Hard**: Original problems extending LeetCode problems
- **Custom Hard**: Completely original hard problems

## 🏆 Challenge Yourself

- [ ] Solve all 230 problems
- [ ] Implement each pattern from memory
- [ ] Create your own hard problems for each pattern
- [ ] Teach someone else these patterns

## 📚 Additional Resources

- [LeetCode](https://leetcode.com)
- [GeeksforGeeks](https://www.geeksforgeeks.org)
- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms)
- [Algorithm Design Manual](http://www.algorist.com)

## 📊 Progress Tracker

Create your own progress tracker:

```markdown
## My Progress

### Pattern Completion
- [ ] Sliding Window (0/10)
- [ ] Two Pointers (0/10)
- [ ] Fast & Slow Pointers (0/10)
...

### Total: 0/230 problems solved
```

## 🌟 Tips for Success

1. **Consistency**: Solve at least one problem daily
2. **Understanding > Speed**: Focus on deep understanding
3. **Write Clean Code**: Practice writing production-quality code
4. **Explain Your Solution**: Practice explaining your approach
5. **Review and Reflect**: Revisit problems after a week

## 📧 Contact

For questions, suggestions, or discussions about these patterns, please open an issue on GitHub.

---

**Happy Coding! 🚀**

*Remember: The journey of a thousand problems begins with a single pattern.*