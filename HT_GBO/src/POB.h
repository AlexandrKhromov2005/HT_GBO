#ifndef POB_H
#define POB_H

#include <unordered_map>
#include <vector>
#include <bitset>
#include <algorithm>
#include <utility>

std::pair<unsigned char, unsigned char> pob(unsigned short val);
unsigned short inverse_pob(unsigned char target_pob, unsigned char r);

#endif // POB_H
