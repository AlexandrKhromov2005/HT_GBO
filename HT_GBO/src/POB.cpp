#include "POB.h"
#include <iostream>

inline int comb(int n, int k) {
    if (k < 0 || k > n) return 0;
    int result = 1;
    for (int i = 1; i <= k; ++i) {
        result = result * (n - k + i) / i;
    }
    return result;
}

std::pair<int, int> pob(int val) {
    std::pair<int, int> res = { 0, 0 };
    int ones = 0;
    int temp = 0;
    for (int i = 3; i >= 0; --i) {
        if ((val >> i) & 1) ones++;
    }
    int remaining_ones = ones;
    for (int i = 3; i >= 0; --i) {
        if ((val >> i) & 1) {
            temp += comb(i, remaining_ones);
            remaining_ones--;
        }
    }
    res.first = temp;
    res.second = ones;
    return res;
}

std::vector<int> generate_numbers(int target_ones) {
    std::vector<int> result;
    if (target_ones > 4) return result; 

    for (int num = 0; num < 16; ++num) {
        int ones_count = 0;
        int temp = num;
        while (temp) {
            ones_count += temp & 1;
            temp >>= 1;
        }
        if (ones_count == target_ones) {
            result.push_back(num);
        }
    }
    return result;
}

int inverse_pob(int target_pob, int r) {
    auto trials = generate_numbers(r);
    for (int trial : trials) {
        if (pob(trial).first == target_pob) return trial;
    }
    return 0xFFFF; 
}