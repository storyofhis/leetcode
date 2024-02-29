# leetcode

### tips 
- print each value of `vector<vector<int>>`
  ```
  std::vector<std::vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    for (const auto& row : matrix) {
        for (int element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }
  ```
- 
- find the index of element of `vector`
  ```
  void getIndex(vector<int> v, int K) 
  { 
    auto it = find(v.begin(), v.end(), K); 
  
    // If element was found 
    if (it != v.end())  
    { 
      
        // calculating the index 
        // of K 
        int index = it - v.begin(); 
        cout << index << endl; 
    } 
    else { 
        // If the element is not 
        // present in the vector 
        cout << "-1" << endl; 
    } 
  } 
  ```
- copy all elements from maps `mp` to vector `vc`
  ```
  map<int, int> mp;
  vector<pair<int, int>> vc;
  copy(mp.begin(), mp.end(), back_inserter<vector<pair<int, int>>>(vc));
  ```
- sort by lexicographical in vector `vc`:
  ```
  sort(vc.begin(), vc.end(), 
      [](const pair<int, int>& a, 
          const pair<int, int>& b) {
          if (a.second == b.second) {
              return (a.first > b.first);
          } 
          return (a.first < b.first);
  });
  ```
- Sort characters based on frequency : for example you want to sort by frequency, *`note` if two elements has same frequency sort by value
  ```
  map<char, int> m;
  sort(s.begin(), s.end(), [&](char a, char b) {
    if(m[a] == m[b]) {
      return a < b;   // If frequencies are equal, sort by character order
    }
    return m[a] > m[b];
  });
  ```
- count frequency of element : for example you want to count frequency of given string `s`
  ```
  map<char, int> m;
  for (int i = 0; i < s.size(); i++) {
    m[s[i]]++;
  }
  ```
- isAnagram : check if two string or vector are anagram or not ?
  ```
  bool isAnagram(string& s, string& t) {
        sort(s.begin(), s.end());
        sort(t.begin(), t.end());
        return s == t;
  }
  ```
- isPalindrome : checkk if two string or vector are palindromic or not ?
  ```
  bool isPalindrome(string x) {
        string t = x;
        reverse(t.begin(), t.end());
        return x == t;
  }
  ```
  or
  ```
  bool isPalindrome(string &s, int start, int end) {
        while (start < end) {
            if (s[start] != s[end]) {
                return false;
            }
            start++;
            end--;
        }
        return true;
  }
  ```
- hammingWeight : get the total of '1' bits in `binary`
  ```
    int hammingWeight(uint32_t n) {
        int pivot = 1;
        int ans = 0;
        for (int i=0; i < 32; i++) {
            int p = pivot << i;
            if ((int(n) & p) == p) {
                ans += 1;
            }
        }
        return ans;
    }
  ```
- reverseBit : reverse `43261596 (00000010100101000001111010011100)` to `964176192 (00111001011110000010100101000000)`
  ```
    uint32_t reverseBits(uint32_t n) {
        uint32_t ret = 0;
        for (uint32_t i = 0; i < 32; ++i) {
            ret = (ret << 1) + ((n >> i) & 1);
        }
        return ret;
    }
  ```
### some basic theory 
- Rabin Karp :
  ```
  vector<int> rabin_karp(string const& s, string const& t) {
    const int p = 31; 
    const int m = 1e9 + 9;
    int S = s.size(), T = t.size();

    vector<long long> p_pow(max(S, T)); 
    p_pow[0] = 1; 
    for (int i = 1; i < (int)p_pow.size(); i++) 
        p_pow[i] = (p_pow[i-1] * p) % m;

    vector<long long> h(T + 1, 0); 
    for (int i = 0; i < T; i++)
        h[i+1] = (h[i] + (t[i] - 'a' + 1) * p_pow[i]) % m; 
    long long h_s = 0; 
    for (int i = 0; i < S; i++) 
        h_s = (h_s + (s[i] - 'a' + 1) * p_pow[i]) % m; 

    vector<int> occurrences;
    for (int i = 0; i + S - 1 < T; i++) {
        long long cur_h = (h[i+S] + m - h[i]) % m;
        if (cur_h == h_s * p_pow[i] % m)
            occurrences.push_back(i);
    }
    return occurrences;
  }
  ```
- Longest Increasing Subsequence :
  ```
  for (int i = 0; i < nums.size(); i++) {
      for (int j = 0; j < i; j++) {
          if (nums[i] > nums[j]) {
              dp[i] = max(dp[i], dp[j] + 1);
          }
      }
  }
  int ans = dp[0];
  for (int i = 1; i < nums.size(); i++) {
      ans = max(ans, dp[i]);
  }
  ```
- Longest Common Subsequences :
  ```
  int longestCommonSubsequence(string s1, string s2) {
        int m = s1.size(), n = s2.size();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1));
        for(int i = 0; i <= m; i++) {
            for(int j = 0; j <= n; j++) {
                if(i == 0 || j == 0) dp[i][j] = 0; // one or more of the lengths is 0
                else if(s1[i-1] == s2[j-1]) dp[i][j] = 1 + dp[i-1][j-1]; // found a common character
                else dp[i][j] = max(dp[i-1][j], dp[i][j-1]); // take the best of both scenarios
            }
        }
        return dp[m][n];
  }
  ```
- Longest Palindromic Subsequences: 
  ```
  int lps(string& s1, string& s2, int n1, int n2)
  {
      if (n1 == 0 || n2 == 0) {
          return 0;
      }
      if (dp[n1][n2] != -1) {
          return dp[n1][n2];
      }
      if (s1[n1 - 1] == s2[n2 - 1]) {
          return dp[n1][n2] = 1 + lps(s1, s2, n1 - 1, n2 - 1);
      }
      else {
          return dp[n1][n2] = max(lps(s1, s2, n1 - 1, n2),
                                  lps(s1, s2, n1, n2 - 1));
      }
  }
  ```
- edit distance : return the minimum edit of string `str1` to string `str2`: 
  ```
  if (str1[i – 1] == str2[j – 1]) dp[i][j] = dp[i – 1][j – 1];
  if (str1[i – 1] != str2[j – 1]) dp[i][j] = 1 + min(dp[i][j – 1], dp[i – 1][j], dp[i – 1][j – 1]);
  ```
  let see how to implement this :
  ```
  int editDistDP(string str1, string str2, int m, int n)
  {
    // Create a table to store results of subproblems
    int dp[m + 1][n + 1];
 
    // Fill d[][] in bottom up manner
    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            // If first string is empty, only option is to
            // insert all characters of second string
            if (i == 0)
                dp[i][j] = j; // Min. operations = j
 
            // If second string is empty, only option is to
            // remove all characters of second string
            else if (j == 0)
                dp[i][j] = i; // Min. operations = i
 
            // If last characters are same, ignore last char
            // and recur for remaining string
            else if (str1[i - 1] == str2[j - 1])
                dp[i][j] = dp[i - 1][j - 1];
 
            // If the last character is different, consider
            // all possibilities and find the minimum
            else
                dp[i][j]
                    = 1
                      + min(dp[i][j - 1], // Insert
                            dp[i - 1][j], // Remove
                            dp[i - 1][j - 1]); // Replace
        }
    }
 
    return dp[m][n];
  }
  ```
- knapsack:
  ```
  int knapsack(int W, int wt[], int val[], int n)
  {
     int i, w;
     int K[n+1][W+1];
   
     // Build table K[][] in bottom up manner
     for (i = 0; i <= n; i++)
     {
         for (w = 0; w <= W; w++)
         {
             if (i==0 || w==0)
                 K[i][w] = 0;
             else if (wt[i-1] <= w)
                   K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w]);
             else
                   K[i][w] = K[i-1][w];
         }
     }
   
     return K[n][W];
  }
  ```
- flood fill :
  ```
    vector<vector<bool>> visited;
    void floodfill(vector<vector<char>>&grid, int x, int y) {
        if (
            x < 0 || 
            x >= m ||
            y < 0 || 
            y >= n || 
            visited[x][y] || 
            grid[x][y] == '0'
            ) 
            return; 
        visited[x][y] = true;
        floodfill(grid, x-1, y);
        floodfill(grid, x+1, y);
        floodfill(grid, x, y-1);
        floodfill(grid, x, y+1);
    }
  ```

### most function
most common data structure i used to : (vector, set, map, string);
given vector<int> nums as an global variable :
- reverse :
  ```
  reverse(nums.begin(), nums.end())
  ```
  or
  ```
  int i = 0, j = nums.size()-1; 
  while (i < j) {
      swap(nums[i++], nums[j--]);
  }
  ```
  example:
  1. Given an integer array `nums`, rotate the array to the right by `k` steps, where `k` is non-negative.
     `nums` = [1,2,3,4,5,6,7], `k` = 3;
      ```
     //[1,2,3,4,5,6,7], k = 3
      k = k % n;
      reverse(nums.begin(), nums.end() - k);  // [4,3,2,1, | 5,6,7]
      reverse(nums.begin() + n - k, nums.end()); // [4,3,2,1, | 7,6,5]
      reverse(nums.begin(), nums.end());  // [5,6,7,1,2,3,4]
      ```
- sort :
  ```
  sort(nums.begin(), nums.end())
  ```
- permutation :
  ```
  do {
  } while (next_permutation(nums.begin(), nums.end());
  do {
  } while (prev_permutation(nums.begin(), nums.end());
  ```

- substring : concatenation with substring in string builtin function
  ```
  string substr (size_t pos, size_t len) const;
  ```
### 3005. Count Elements With Maximum Frequency
You are given an array nums consisting of positive integers.
Return the total frequencies of elements in nums such that those elements all have the maximum frequency.
The frequency of an element is the number of occurrences of that element in the array.
* input : nums = [1,2,2,3,1,4]
* output : 4
* explanation : The elements 1 and 2 have a frequency of 2 which is the maximum frequency in the array.
So the number of elements in the array with maximum frequency is 4.
```
class Solution {
public:
    int maxFrequencyElements(vector<int>& nums) {
        // count the frequency for each integer
        unordered_map<int, int> freq;
        for (auto& num : nums) freq[num]++;

        // find the maximum frequency
        int maxFreq = 0;
        for (auto& [num, f] : freq) maxFreq = max(maxFreq, f);
        // calculate the sum of the frequencies with `maxFreq`
        int result = 0;
        for (auto& [num, f] : freq) {
            if (f == maxFreq) result += f;
        }
        return result;
    } 
};
```
### 49. Group Anagrams
Given an array of strings strs, group the anagrams together. You can return the answer in any order.
An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
* input : strs = ["eat","tea","tan","ate","nat","bat"]
* output : [["bat"],["nat","tan"],["ate","eat","tea"]]
```
class Solution {
public:
    bool isAnagram(string& s, string& t) {
        sort(s.begin(), s.end());
        sort(t.begin(), t.end());
        return s == t;
    }

    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> anagrams;
        for (const string& str : strs) {
            string sortedStr = str; // eat, tea, tan, ate...
            sort(sortedStr.begin(), sortedStr.end()); // aet, aet, ant, aet...
            anagrams[sortedStr].push_back(str); // aet:[eat, tea, aet], ant:[tan]...
        }
        
        vector<vector<string>> ans;
        for (auto& pair : anagrams) {
            ans.push_back(pair.second);
        }
        return ans;
    }
};

```
### 88. Merge Sorted Array
You are given two integer arrays `nums1` and `nums2`, sorted in non-decreasing order, and two integers `m` and `n`, representing the number of elements in `nums1` and `nums2` respectively.
Merge `nums1` and `nums2` into a single array sorted in non-decreasing order.
The final sorted array should not be returned by the function, but instead be stored inside the array `nums1`. To accommodate this, `nums1` has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. `nums2` has a length of n.
* input : nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
* output : [1,2,2,3,5,6]
```
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int i = m - 1;
        int j = n - 1;
        int k = m + n - 1;
        
        while (j >= 0) {
            if (i >= 0 && nums1[i] > nums2[j]) {
                nums1[k--] = nums1[i--];
            } else {
                nums1[k--] = nums2[j--];
            }
        }
    }
};
```
### 14. Longest Common Prefix
Write a function to find the longest common prefix string amongst an array of strings.
If there is no common prefix, return an empty string "".
* input : strs = ["flower","flow","flight"]
* output : "fl"
```
class Solution {
public:
    // strs = ["flower","flow","flight"]
    string divide_conquer(vector<string>& strs, int l, int r) {
        if (l == r) return strs[l];
        int mid = (l + r) / 2;
        string left = divide_conquer(strs, l, mid); // flower, flow
        string right = divide_conquer(strs, mid + 1, r); // flow, flight
        return commonPrefix(left, right);
    }
    string commonPrefix(string left, string right) { // flower & flow, flow & flight
        for (int i = 0; i < min(left.size(), right.size()); i++) {
            if (left[i] != right[i]) {
                return left.substr(0, i);
            }
        }
        return left.substr(0, min(left.size(), right.size()));
    }
    string longestCommonPrefix(vector<string>& strs) {
        string str = divide_conquer(strs, 0, strs.size() - 1);
        return str;
    }
};
```
### 347. Top K Frequent Elements
Given an integer array `nums` and an integer `k`, return the `k` most frequent elements. You may return the answer in any order.
* Input: nums = [1,1,1,2,2,3], k = 2
* Output: [1,2]
```
class Solution {
public:

    vector<int> topKFrequent(vector<int>& nums, int k) {
        map<int, int> mp;
        for (auto num : nums) {
            mp[num]++;
        }
        // convert map to vector
        vector<pair<int, int>> vc;
        copy(mp.begin(), mp.end(), 
            back_inserter<vector<pair<int, int>>>(vc));
        // sort by lexicographical
        sort(vc.begin(), vc.end(), 
            [](const pair<int, int>& a, 
            const pair<int, int>& b) {
                if (a.second == b.second) {
                    return (a.first > b.first);
                } 
                return (a.first < b.first);
            });
        vector<int> ans;
        for (pair p : vc) {
            ans.push_back(p.first);
            if (--k == 0) break;
        }
        return ans;
    }
};
```
### 56. Merge Intervals
Given an array of `intervals` where `intervals[i] = [starti, endi]`, merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the `intervals` in the input.
* Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
* Output: [[1,6],[8,10],[15,18]]
* Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
```
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end());

        vector<vector<int>> merged;
        for (auto interval : intervals) {
            // if the list of merged intervals is empty or if the current
            // interval does not overlap with the previous, simply append it.
            if (merged.empty() || merged.back()[1] < interval[0]) {
                merged.push_back(interval);
            }
            // otherwise, there is overlap, so we merge the current and previous
            // intervals.
            else {
                merged.back()[1] = max(merged.back()[1], interval[1]);
            }
        }
        return merged;
    }
};
```
or 
```
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        int size = intervals.size();
        if (size <= 1) return intervals;
        sort(intervals.begin(), intervals.end());
        int start = intervals[0][0];
        int end = intervals[0][1];
        vector<vector<int>> res;
        for (int i = 1; i < size; i++) {
            if (intervals[i][0] > end) {
                res.push_back({start, end});
                start = intervals[i][0];
                end = intervals[i][1];
            } else if (intervals[i][1] > end){
                end = intervals[i][1];
            }
            if (i == size - 1) res.push_back({start, end});
        }
        return res;
    }
};
```
### 451. Sort Characters By Frequency
Given a string `s`, sort it in decreasing order based on the frequency of the characters. The frequency of a character is the number of times it appears in the string.
Return the sorted string. If there are multiple answers, return any of them.
* Input: s = "tree"
* Output: "eert"
* Explanation: 'e' appears twice while 'r' and 't' both appear once.
So 'e' must appear before both 'r' and 't'. Therefore "eetr" is also a valid answer.
```
class Solution {
public:
    string frequencySort(string s) {
        map<char, int> m;
        // Step 1: Count character frequencies
        for (int i = 0; i < s.size(); i++) {
            m[s[i]]++;
        }
        // Step 2: Sort characters based on frequency
        sort(s.begin(), s.end(), [&](char a, char b) {
            if(m[a] == m[b]) {
                return a < b;   // If frequencies are equal, sort by character order
            }
            return m[a] > m[b];
        });
        // Step 3: Build the sorted string
        string sortedString;
        for (char c : s) {
            sortedString += c;
        }
        return sortedString;
    }
};
```
### 131. Palindrome Partitioning
Given a string `s`, partition `s` such that every 
substring
 of the partition is a 
palindrome
. Return all possible palindrome partitioning of `s`.
* Input: s = "aab"
* Output: [["a","a","b"],["aa","b"]]
```
class Solution {
public:
    vector<vector<string>> partition(string s) {
        vector<vector<string>> ans;
        vector<string> v = {};
        fnc(s, v, ans);
        return ans;
    }
    void fnc(string s, vector<string>& v, vector<vector<string>>& ans) {
        if (s.size() == 0) {
            ans.push_back(v);
            return;
        }

        for (int len = 1; len <= s.size(); len++) {
            string x = s.substr(0, len);
            if (isPalindrome(x)) {
                v.push_back(x);
                string y = s.substr(len, s.size() - len);
                fnc(y, v, ans);
                v.pop_back();
            }
        }
    }
    bool isPalindrome(string x) {
        string t = x;
        reverse(t.begin(), t.end());
        return x == t;
    }
};
```
### 3. Longest Substring Without Repeating Characters
Given a string `s`, find the length of the longest 
`substring`
 without repeating characters.
* Input: s = "abcabcbb"
* Output: 3
* Explanation: The answer is "abc", with the length of 3.
```
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int n = s.size();
        int maxLength = 0;
        map<char, int> m;
        int left = 0;
        for (int right = 0; right < n; right++) {
            if (
                m.count(s[right]) == 0 || // return 0 if there is no key [right]
                m[s[right]] < left
            ) {
                m[s[right]] = right;
                maxLength = max(maxLength, right - left + 1);
            } else {
                left = m[s[right]] + 1;
                m[s[right]] = right;
            }
        }
        return maxLength;
    }
};
```
### 34. Find First and Last Position of Element in Sorted Array
Given an array of integers `nums` sorted in non-decreasing order, find the starting and ending position of a given `target` value.
If `target` is not found in the array, return `[-1, -1]`.
You must write an algorithm with `O(log n)` runtime complexity.
* Input: nums = [5,7,7,8,8,10], target = 8
* Output: [3,4]
```
class Solution {
public:
    int lower_bound(vector<int>& nums, int low, int high, int target) {
        while(low <= high) {
            int mid = (low + high) >> 1;
            if (nums[mid] < target) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return low;
    }
    vector<int> searchRange(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        int n = nums.size();
        vector<int> ans(2);
        int low = 0, high = n - 1;
        int startPosition = lower_bound(nums, low, high, target);
        int endPosition = lower_bound(nums, low, high, target + 1) - 1;
        cout << startPosition << " " << endPosition << endl;
        if (startPosition < nums.size() && nums[startPosition] == target) {
            ans[0] = startPosition;
            ans[1] = endPosition;
        } else {
            ans[0] = ans[1] = -1;
        }
        return ans;
    }
};
```
### 78. Subsets
Given an integer array nums of unique elements, return all possible 
subsets
 (the power set).
The solution set must not contain duplicate subsets. Return the solution in any order.
* Input: nums = [1,2,3]
* Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```
class Solution {
    int k, n;
public:
    void solve(int first, vector<vector<int>>& ans, vector<int>& nums, vector<int>& curr) {
        if (curr.size() == k) { // k = 1
            ans.push_back(curr);
            return;
        }
        for (int i = first; i < n; ++i) {
            curr.push_back(nums[i]); // nums[0], nums[1], nums[2],
            solve(i + 1, ans, nums, curr); // i = 1, i = 2
            curr.pop_back();
        }
    }
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> ans;
        vector<int> curr = {};
        n = nums.size();
        for (k = 0; k < n + 1; ++k) {
            solve(0, ans, nums, curr);
        }
        return ans;
    }
};
```
### 79. Word Search
Given an `m x n` grid of characters `board` and a string `word`, return true if word exists in the grid.
The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.
```
class Solution {
public:
    vector<vector<bool>> visited;
    bool search(vector<vector<char>>& board, int x, int y, int idx, string word) {
        if (idx == word.size()) return true;
        if (
            x < 0 || 
            y < 0 || 
            x >= board.size() || 
            y >= board[0].size() 
        ) {
            return false;
        }
        bool ans = false;
        if (word[idx] == board[x][y]) {
            board[x][y] = '*';
            ans = search(board, x + 1, y, idx + 1, word) or 
                search(board, x, y + 1, idx + 1, word) or 
                search(board, x - 1, y, idx + 1, word) or 
                search(board, x, y - 1, idx + 1, word);
            board[x][y] = word[idx];
        }
        return ans;
    }
    bool exist(vector<vector<char>>& board, string word) {
        int m = board.size();
        int n = board[0].size();
        visited = vector<vector<bool>> (m, vector<bool>(n, false));
        // search the source
        int x = 0, y = 0;
        int idx = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == word[idx] && 
                    search(board, i, j, idx, word)) {
                    return true;
                }
            }
        }
        return false;
    }
};
```
### 22. Generate Parentheses
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses
* Input: n = 3
* Output: ["((()))","(()())","(())()","()(())","()()()"]
```
class Solution {
public:
    void solve(vector<string>& ans, int i, int j, int n, string s) {
        if (i == j and i == n) {
            ans.push_back(s);
            return;
        }
        if (i < n) solve(ans, i + 1, j, n, s + '(');
        if (j < i) solve(ans, i, j + 1, n, s + ')');
    }
    vector<string> generateParenthesis(int n) {
        vector<string> ans;
        solve(ans, 0, 0, n, "");
        return ans;
    }
};
```
### 32. Longest Valid Parentheses
Given a string containing just the characters '(' and ')', return the length of the longest valid (well-formed) parentheses 
substring.
* Input: s = ")()())"
* Output: 4
* Explanation: The longest valid parentheses substring is "()()".
```
class Solution{
public:
    int longestValidParentheses(string s) {
        int maxans = 0;
        vector<int> dp(s.length(), 0);
        for (int i = 1; i < s.length(); i++) {
            if (s[i] == ')') {
                if (s[i - 1] == '(') {
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                }
                else if (i - dp[i - 1] > 0 && s[i - dp[i - 1] - 1] == '(') {
                    dp[i] = dp[i - 1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
                }
                maxans = max(maxans, dp[i]);
            }
        }
        return maxans;
    }
};
```

