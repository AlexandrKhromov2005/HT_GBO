#include "affine.h"

inline std::pair<int, int> calc_new_coordinates(int x, int y, int a, int b, int c, int d, int tx, int ty) {
	std::pair <int, int> new_coordinates;

	new_coordinates.first = (a * x + b * y + tx) % WM_SIZE;
	new_coordinates.second = (c * x + d * y + ty) % WM_SIZE;

	if (new_coordinates.first < 0) new_coordinates.first += WM_SIZE;
	if (new_coordinates.second < 0) new_coordinates.second += WM_SIZE;


	return new_coordinates;
}

inline std::pair<int, int> calc_old_coordinates(int x, int y,  int a, int b, int c, int d, int tx, int ty) {
	int det = a * d - b * c;

	int temp_x = d * (x - tx) - b * (y - ty);
	int temp_y = a * (y - ty) - c * (x - tx);

	while (temp_x % det != 0) {
		temp_x += WM_SIZE;
	}

	temp_x /= det;
	temp_x %= WM_SIZE;
	if (temp_x < 0) temp_x += WM_SIZE;

	while (temp_y % det != 0) {
		temp_y += WM_SIZE;
	}

	temp_y /= det;
	temp_y %= WM_SIZE;
	if (temp_y < 0) temp_y += WM_SIZE;


	return {temp_x, temp_y};
}

cv::Mat affineTransform(cv::Mat wm, int a, int b, int c, int d, int tx, int ty) {
	cv::Mat trans_wm = cv::Mat::zeros(wm.size(), wm.type());

	for (int x = 0; x < wm.rows; ++x) {
		for (int y = 0; y < wm.cols; ++y) {
			std::pair<int, int> new_coordinates = calc_new_coordinates(x, y, a, b, c, d, tx, ty);
			trans_wm.at<uchar>(new_coordinates.first, new_coordinates.second) = wm.at<uchar>(x, y);
		}
	}

	return trans_wm;
}

cv::Mat affineTransformInv(cv::Mat wm, int a, int b, int c, int d, int tx, int ty) {
	cv::Mat rest_wm = cv::Mat::zeros(wm.size(), wm.type());

	for (int x = 0; x < wm.rows; ++x) {
		for (int y = 0; y < wm.cols; ++y) {
			std::pair<int, int> old_coordinates = calc_old_coordinates(x, y , a, b, c, d, tx, ty);
			rest_wm.at<uchar>(old_coordinates.first, old_coordinates.second) = wm.at<uchar>(x, y);
		}
	}

	return rest_wm;
}