#ifndef RANDOM_UTILS_H
#define RANDOM_UTILS_H

#include <random>
#include <ctime>
#include <cmath>
#include <climits>
#include <cstring>
#include "config.h"
#include <array>

void init_random();

double rand_num();

double randn();

double new_rho(double alpha);

void gen_indexes(std::array<size_t, 4>& indexes, size_t cur_ind, size_t best_ind);

size_t gen_random_index();

double rand_neg_one_to_one();

unsigned char rand_binary();

#endif // RANDOM_UTILS_H