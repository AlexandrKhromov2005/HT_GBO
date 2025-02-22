#ifndef POB_H
#define POB_H

#include <vector>
#include <utility>

// Кодирование 4-битного числа в POB
std::pair<int, int> pob(int val);
// Декодирование POB в 4-битное число
int inverse_pob(int target_pob, int r);

#endif // POB_H