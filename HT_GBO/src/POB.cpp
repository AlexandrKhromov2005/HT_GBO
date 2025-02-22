#include "POB.h"

inline unsigned char comb(int n, int k) {
    if (k < 0 || k > n) return 0;
    unsigned char result = 1;
    for (int i = 1; i <= k; ++i) {
        result = result * (n - k + i) / i;
    }
    return result;
}



std::pair<unsigned char, unsigned char> pob(unsigned short val) { // нужно дописать сборку ключа B
    std::pair<unsigned char, unsigned char> res = { 0, 0 };
    int ones = 0;
    for (int i = 9; i >= 0; --i) {
        if ((val >> i) & 1) ones++;
    }
    int remaining_ones = ones;
    for (int i = 9; i >= 0; --i) {
        if ((val >> i) & 1) {
            res.first += comb(i, remaining_ones);
            remaining_ones--;
        }
    }
    res.second = ones;
    return res;
}


std::unordered_map<unsigned char, unsigned short> build_pob_set(int N, int r) {
    std::unordered_map<unsigned char, unsigned short> pob_set;

    std::vector<int> bits(N, 0);
    fill(bits.end() - r, bits.end(), 1);

    do {
        unsigned short num = 0;
        for (int i = 0; i < N; ++i) {
            num |= (bits[i] << (N - 1 - i));
        }

        unsigned char pob_val = 0;
        int p_j = 0;
        for (int j = 0; j < N; ++j) {
            if (bits[j] == 1) {
                pob_val += comb(j, p_j);
                p_j++;
            }
        }

        pob_set[pob_val] = num;
    } while (next_permutation(bits.begin(), bits.end()));

    return pob_set;
}

std::vector<unsigned short> generate_numbers(unsigned char target_ones) {
    std::vector<unsigned short> result;

    if (target_ones < 0 || target_ones > 10) {
        return result;
    }

    for (int num = 0; num < 1024; ++num) {
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

unsigned short inverse_pob(unsigned char target_pob, unsigned char r) {
    std::vector<unsigned short> trials = generate_numbers(r);
    for (unsigned short trial : trials) {
        if (pob(trial).first == target_pob) return trial;
    }
    return 0XFFFF; // error case
}