#include "gbo.h"
#include <iostream>


double gsr_func(
    double rho2,
    double best_t,
    double worst_t,
    double current_t,
    double random_t,
    double alpha
) {
    double a = rand_num();
    double b = static_cast<double>(gen_random_index());
    double c = randn();
    double eps = 1e-6;

    // Step 1: Directional component (best vs random)
    double delta = 2.0 * a * fabs(alpha * (best_t - random_t));
    double step = 0.5 * (best_t - random_t + delta);

    // Step 2: Stochastic scaling
    double gsr = c * rho2 * 2.0 * fabs(step) * current_t;
    gsr /= (best_t - worst_t + eps);

    // Step 3: Balance exploration/exploitation
    double p1 = rand_num();
    double p2 = rand_num();
    double yp = p1 * (0.5 * (current_t + best_t) + p2 * step);
    double yq = p1 * (0.5 * (current_t + best_t) - p2 * step);

    // Final adjustment
    return current_t - gsr + (yp - yq) * alpha;
}


void GBO::optimize() {
    Population population;
    population.initOf(hostImage, wm);

    for (size_t m = 0; m < ITERATIONS; ++m) {
        double beta = 0.2 + (1.2 - 0.2) * pow(1.0 - static_cast<double>(m + 1) / ITERATIONS, 2.0);
        double angle = 1.5 * M_PI;
        double alpha = fabs(beta * sin(angle + sin(angle * beta)));

        for (size_t cur_ind = 0; cur_ind < POP_SIZE; ++cur_ind) {
            // Generate random indices
            std::array<size_t, 4> indexes;
            gen_indexes(indexes, cur_ind, population.best_ind);

            // Get reference values
            double best_t = population.vecs[population.best_ind].first;
            double worst_t = population.worst_vec.first;
            double current_t = population.vecs[cur_ind].first;
            double random_t = population.vecs[indexes[0]].first;

            // Apply GSR
            double rho2 = alpha * (2 * rand_num() - 1);
            double t_new = gsr_func(
                rho2, best_t, worst_t,
                current_t, random_t, alpha
            );

            // Clamp and evaluate
            t_new = std::clamp(t_new, 30.0, 60.0);
            double of_new = population.calculateOf(hostImage, wm, t_new);
            population.update({ t_new, of_new }, cur_ind);
        }
    }

    optimal_t = population.vecs[population.best_ind].first;
    std::cout << "Optimized t: " << optimal_t << std::endl;
}

//example of usage
/*cv::Mat host = importImage("host.png");
cv::Mat wm = importImage("watermark.png");
GBO optimizer(host, wm);
optimizer.optimize();
cv::Mat result = embedWatermark(host, wm, optimizer.optimal_t);*/