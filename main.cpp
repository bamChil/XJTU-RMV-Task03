#include <opencv2/opencv.hpp>
#include <chrono>
#include <ceres/ceres.h>
#include <vector>
#include "windmill.hpp"

using namespace std;
using namespace cv;

struct XResidual {
    XResidual(double t, double observed_x)
        : t_(t), observed_x_(observed_x) {}

    template <typename T>
    bool operator()(const T* const A0, const T* const A, const T* const omega, const T* const phi, T* residual) const {
        
        residual[0] = T(observed_x_) - cos(A0[0] * T(t_) + A[0] / omega[0] * (sin(omega[0] * T(t_) + phi[0]) - sin(phi[0])));
        return true;
    }

private:
    const double t_;
    const double observed_x_;
};

// 定义一个用于返回内部轮廓的函数
std::vector<std::vector<cv::Point>> getInternalContours(const cv::Mat& src) {
    // 创建二值图像，用于查找轮廓
    cv::Mat binary;

    // 使用阈值将图像转换为二值图像
    cv::threshold(src, binary, 20, 255, cv::THRESH_BINARY);

    // 查找所有轮廓和层次结构
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // 存储内部轮廓
    std::vector<std::vector<cv::Point>> internalContours;

    // 遍历轮廓并找到内部轮廓
    for (size_t i = 0; i < contours.size(); i++) {
        // 如果当前轮廓有父轮廓，则它是内部轮廓
        if (hierarchy[i][3] != -1) {
            internalContours.push_back(contours[i]);
        }
    }

    return internalContours;
}


cv::Point2f findBestMatchRectangleCenter(const cv::Mat& src, const cv::Mat& image) {
    // 1. 提取 src 中的所有轮廓
    std::vector<std::vector<cv::Point>> inter_srcContours = getInternalContours(src);

    std::vector<std::vector<cv::Point>> srcRectangles;

    // 2. 计算 src 中所有轮廓的面积
    std::vector<double> srcContourAreas;
    for (const auto& contour : inter_srcContours) {
        double area = cv::contourArea(contour);
        srcContourAreas.push_back(area);
        srcRectangles.push_back(contour);  // 保留每个轮廓
    }

    // 3. 提取 image 中的轮廓
    std::vector<std::vector<cv::Point>> inter_imageContours = getInternalContours(image);

    std::vector<std::vector<cv::Point>> imageRectangles;

    // 4. 计算 image 的唯一轮廓面积（确保只有一个轮廓）
    if (inter_imageContours.size() != 1) {
        throw std::runtime_error("There must be exactly one rectangle in the known image.");
    }

    std::vector<cv::Point> targetContour = inter_imageContours[0];
    double targetArea = cv::contourArea(targetContour);

    // 5. 找到与 image 轮廓面积最接近的 src 轮廓
    double minAreaDiff = std::numeric_limits<double>::max();
    cv::Point2f bestMatchCenter;

    for (size_t i = 0; i < srcRectangles.size(); ++i) {
        double areaDiff = std::abs(srcContourAreas[i] - targetArea);

        if (areaDiff < minAreaDiff) {
            minAreaDiff = areaDiff;

            // 计算 src 轮廓的中心点
            cv::Rect boundingBox = cv::boundingRect(srcRectangles[i]);
            bestMatchCenter = (boundingBox.tl() + boundingBox.br()) * 0.5;
        }
    }

    return bestMatchCenter;
}


int main()
{
    double t_sum = 0;
    const int N = 10;
    for (int num = 0; num < N; num++)
    {
        std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        double t_start = (double)t.count();
        WINDMILL::WindMill wm(t_start);
        Mat src;

        ceres::Problem problem;
        double A0_init = 0.305, A_init = 1.785, omega_init = 0.884, phi_init = 1.24;
        double A0_gt = 1.305, A_gt = 0.785, omega_gt = 1.884, phi_gt = 0.24;
        int ndata = 200;

        // starttime
        int64 start_time = getTickCount();

        while (1)
        {
            t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            double t_now = (double)t.count();
            src = wm.getMat(t_now); // Get windmill frame

            Mat template_R = imread("../image/R.png", IMREAD_GRAYSCALE);  // R模板
            Mat template_specialBlade = imread("../image/target.png", IMREAD_GRAYSCALE);  // 特殊叶片模板
            Mat gray;
            cvtColor(src, gray, COLOR_BGR2GRAY);

            if (template_R.empty() || template_specialBlade.empty())
            {
                cout << "模板图片加载失败" << endl;
                return -1;
            }

            // 模板匹配检测R中心
            Mat result_R;
            matchTemplate(gray, template_R, result_R, TM_CCOEFF_NORMED);
            double minVal_R, maxVal_R;
            Point minLoc_R, maxLoc_R;
            minMaxLoc(result_R, &minVal_R, &maxVal_R, &minLoc_R, &maxLoc_R);
            Point2f center_R = Point(maxLoc_R.x + template_R.cols / 2, maxLoc_R.y + template_R.rows / 2);


            Point2f center_blade = findBestMatchRectangleCenter(gray, template_specialBlade);

            Point2f center_blade_norm = (center_blade - center_R) / cv::norm(center_R - center_blade);
            // 将两个检测到的点保存到数据列表
            static std::vector<double> time_data;
            static std::vector<double> x_observ;

            double distance = cv::norm(center_R - center_blade);
            // 判断距离是否在(100, 200)之间           
            if (distance >= 100 && distance <= 200)
            {
                // 只有距离满足条件时才保存数据
                time_data.push_back((t_now-t_start)/1000);
                x_observ.push_back(center_blade_norm.x);
            }

            // 当收集到足够的数据时，进行拟合
            if (time_data.size() >= ndata)
            {
                // Initial guess for the parameters
                double A0 = A0_init, A = A_init, omega = omega_init, phi = phi_init;

                // Add residuals for each data point
                for (int i = 0; i < ndata; ++i) {
                    double t = time_data[i];
                    double x_data = x_observ[i];

                    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<XResidual, 1, 1, 1, 1, 1>(
                        new XResidual(t, x_data));
                    problem.AddResidualBlock(cost_function, nullptr, &A0, &A, &omega, &phi);
                }

                // Set up the solver
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                options.minimizer_progress_to_stdout = false;
                ceres::Solver::Summary summary;

                problem.SetParameterLowerBound(&A0, 0, 1);
                problem.SetParameterUpperBound(&A0, 0, 1.4);
                problem.SetParameterLowerBound(&A, 0, 0.75);
                problem.SetParameterUpperBound(&A, 0, 1);
                problem.SetParameterLowerBound(&omega, 0, 1.8);
                problem.SetParameterUpperBound(&omega, 0, 1.9);
                problem.SetParameterLowerBound(&phi, 0, 0.24);
                problem.SetParameterUpperBound(&phi, 0, 0.25);

                ceres::Solve(options, &problem, &summary);

                // Output the results
                // std::cout << summary.BriefReport() << "\n";
                // std::cout << "Fitted parameters:\n";
                // std::cout << "A0: " << A0 << ", A: " << A << ", omega: " << omega << ", phi: " << phi << "\n";

                // Calculate relative errors and check if less than 5%
                double A0_error = std::abs((A0 - A0_gt) / A0_gt) * 100;
                double A_error = std::abs((A - A_gt) / A_gt) * 100;
                double omega_error = std::abs((omega - omega_gt) / omega_gt) * 100;
                double phi_error = std::abs((phi - phi_gt) / phi_gt) * 100;

                if (A0_error < 5 && A_error < 5 && omega_error < 5 && phi_error < 5) {
                    int64 end_time = cv::getTickCount();
                    t_sum += (end_time - start_time) / cv::getTickFrequency();
                    break;
                } else {
                    ndata++;
                    continue;
                }
                
            }

            // 显示图像结果
            // circle(src, center_R, 5, Scalar(0, 255, 0), -1);  // R中心
            // circle(src, center_blade, 5, Scalar(0, 255, 0), -1);  // 特殊叶片中心
            // imshow("windmill", src);
            // waitKey(1);
        }



    }
    std::cout << t_sum / N << std::endl;
}
