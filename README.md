# leetcode

### tips 
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
