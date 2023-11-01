#include "mnistools.h"

void MNISTLoader(std::ifstream& data_file, std::vector<Eigen::MatrixXd*>& data_vect,
	std::vector<Eigen::MatrixXd*>& answers_vect, const char* filename) {

	std::cout << "Loading " << filename << " file..." << std::endl << std::endl;

	Eigen::MatrixXd* tempMatrix;
	int get = 0;
	while (data_file.get() != '\n') {}

	while (!data_file.eof()) {
		tempMatrix = new Eigen::MatrixXd(1, 10);
		tempMatrix->setZero();
		answers_vect.push_back(tempMatrix);
		(*answers_vect[answers_vect.size() - 1])(0, data_file.get() - '0') = 1.0;
		data_file.ignore();

		data_vect.push_back(new Eigen::MatrixXd(1, 784));
		for (int i = 0; i < 784; ++i) {
			data_file >> get;
			(*data_vect[data_vect.size() - 1])(0, i) = static_cast<double>(get) / 255.0;
			data_file.ignore();
		}
		while ((isdigit(data_file.peek()) == 0) && data_file.get() != EOF) {}
	}

	std::cout << filename << " file was loaded." << std::endl << std::endl;
}

Eigen::MatrixXd pngToMatrix(const char* file_name) {
	cv::Mat img = cv::imread(file_name, cv::IMREAD_GRAYSCALE);
	cv::Mat gray = 255 - img;

	cv::threshold(gray, gray, 128, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

	cv::Mat col_sum;
	cv::Mat row_sum;

	cv::reduce(gray, col_sum, 0, cv::REDUCE_SUM, CV_32S);
	cv::reduce(gray, row_sum, 1, cv::REDUCE_SUM, CV_32S);

	int index = 0;

	while (col_sum.at<int>(0, index) == 0 && index < col_sum.cols) {
		gray = gray.colRange(1, gray.cols);
		++index;
	}

	index = col_sum.cols - 1;

	while (col_sum.at<int>(0, index) == 0 && index >= 0) {
		gray = gray.colRange(0, gray.cols - 1);
		--index;
	}

	index = 0;

	while (row_sum.at<int>(index, 0) == 0 && index < row_sum.rows) {
		gray = gray.rowRange(1, gray.rows);
		++index;
	}

	index = row_sum.rows - 1;

	while (row_sum.at<int>(index, 0) == 0 && index >= 0) {
		gray = gray.rowRange(0, gray.rows - 1);
		--index;
	}

	if (gray.rows > gray.cols) {
		cv::resize(gray, gray, { (int)(round(gray.cols * 20 / (double)gray.rows)), 20 });
	}
	else {
		cv::resize(gray, gray, { 20, (int)(round(gray.rows * 20 / (double)gray.cols)) });
	}

	cv::Mat colAddMatrix(gray.rows, (28 - gray.cols) / 2, CV_8U, cv::Scalar(0, 0, 0));

	if (gray.cols % 2 == 0) {
		cv::hconcat(gray, colAddMatrix, gray);
		cv::hconcat(colAddMatrix, gray, gray);

	}
	else {
		cv::hconcat(gray, colAddMatrix, gray);
		cv::hconcat(cv::Mat(gray.rows, colAddMatrix.cols + 1, CV_8U, cv::Scalar(0, 0, 0)), gray, gray);
	}

	cv::Mat rowAddMatrix((28 - gray.rows) / 2, gray.cols, CV_8U, cv::Scalar(0, 0, 0));
	if (gray.rows % 2 == 0) {
		cv::vconcat(gray, rowAddMatrix, gray);
		cv::vconcat(rowAddMatrix, gray, gray);
	}
	else {
		cv::vconcat(gray, rowAddMatrix, gray);
		cv::vconcat(cv::Mat(rowAddMatrix.rows + 1, gray.cols, CV_8U, cv::Scalar(0, 0, 0)), gray, gray);
	}

	int m = 0;
	int mrCol = 0;
	int mrRow = 0;
	for (int i = 0; i < gray.cols; ++i) {
		for (int j = 0; j < gray.rows; ++j) {
			mrCol += (i + 1) * gray.at<uchar>(j, i);
			mrRow += (j + 1) * gray.at<uchar>(j, i);
			m += gray.at<uchar>(j, i);
		}
	}
	int cx = round((double)mrCol / (double)m);
	int cy = round((double)mrRow / (double)m);
	int shiftx = round(gray.cols / 2.0 - cx);
	int shifty = round(gray.rows / 2.0 - cy);

	cv::Mat M(2, 3, CV_32F);
	M.at<float>(0, 0) = 1;
	M.at<float>(0, 1) = 0;
	M.at<float>(0, 2) = shiftx;
	M.at<float>(1, 0) = 0;
	M.at<float>(1, 1) = 1;
	M.at<float>(1, 2) = shifty;
	cv::warpAffine(gray, gray, M, { gray.cols, gray.rows });

	Eigen::MatrixXd final(1, 784);
	for (int i = 0; i < gray.rows; ++i) {
		for (int j = 0; j < gray.cols; ++j) {
			final(0, i * 28 + j) = static_cast<double>(gray.at<uchar>(i, j)) / 255.0;
		}
	}

	cv::imwrite("out.png", gray);

	return final;
}