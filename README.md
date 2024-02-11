# leetcode

### most function
most common data structure i used to : (vector, set, map);
given vector<int> nums as an global variable :
- reverse :
  ```
  reverse(nums.begin(), nums.end()) 
  ```
- sort :
  ```
  reverse(nums.begin(), nums.end())
  ```
- next_permutation :
  ```
  do {
  } while (next_permutation(nums.begin(), nums.end())
  ```
### 3005. Count Elements With Maximum Frequency
You are given an array nums consisting of positive integers.
Return the total frequencies of elements in nums such that those elements all have the maximum frequency.
The frequency of an element is the number of occurrences of that element in the array.
> input : nums = [1,2,2,3,1,4]
> output : 4
> explanation :
> The elements 1 and 2 have a frequency of 2 which is the maximum frequency in the array.
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

