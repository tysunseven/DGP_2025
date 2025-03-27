#include <pxr/base/vt/array.h>

#include <iostream>

#include "basic_node_base.h"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(histogram_equalization)
{
    b.add_input<pxr::VtArray<float>>("Input Array");
    b.add_output<pxr::VtArray<float>>("Equalized Array");
}

NODE_EXECUTION_FUNCTION(histogram_equalization)
{
    auto input_array = params.get_input<pxr::VtArray<float>>("Input Array");

    // 处理空数组或只有一个元素的情况
    if (input_array.empty()) {
        std::cerr << "Input array is empty." << std::endl;
        return false;
    }
    if (input_array.size() == 1) {
        pxr::VtArray<float> result = input_array;
        params.set_output("Equalized Array", result);
        return true;
    }

    pxr::VtArray<float> normalized_array(input_array.size());
    pxr::VtArray<float> equalized_array(input_array.size());

    // 找到输入数据范围
    float min_value = *std::min_element(input_array.begin(), input_array.end());
    float max_value = *std::max_element(input_array.begin(), input_array.end());
    float range = max_value - min_value;

    // 归一化数据到 [0,1] 区间
    if (range > std::numeric_limits<float>::epsilon()) {
        for (size_t i = 0; i < input_array.size(); ++i) {
            normalized_array[i] = (input_array[i] - min_value) / range;
        }
    }
    else {
        // 如果所有值相同，则直接返回原数组
        params.set_output("Equalized Array", input_array);
        std::cerr << "Input array has no range (all values are identical)."
                  << std::endl;
        return true;
    }

    // 构建直方图 (256 bins)
    const int NUM_BINS = 256;
    std::vector<int> histogram(NUM_BINS, 0);
    for (size_t i = 0; i < normalized_array.size(); ++i) {
        // 限制在 [0,255] 范围内
        int bin = std::min(static_cast<int>(normalized_array[i] * 255.0f), 255);
        histogram[bin]++;
    }

    // 计算累积分布函数 (CDF)
    std::vector<float> cdf(NUM_BINS, 0.0f);
    cdf[0] = histogram[0];
    for (int i = 1; i < NUM_BINS; ++i) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    // 归一化 CDF
    float total_pixels = static_cast<float>(normalized_array.size());
    float cdf_min = 0.0f;
    // 找到第一个非零的 CDF 值
    for (int i = 0; i < NUM_BINS; ++i) {
        if (cdf[i] > 0) {
            cdf_min = cdf[i];
            break;
        }
    }

    // 应用均衡化，并确保结果映射到原始范围
    for (size_t i = 0; i < normalized_array.size(); ++i) {
        int bin = std::min(static_cast<int>(normalized_array[i] * 255.0f), 255);

        // 应用均衡化公式: (cdf(v) - cdf_min) / (1 - cdf_min)
        float equalized_value = 0.0f;
        if (cdf[NUM_BINS - 1] > cdf_min) {
            equalized_value = (cdf[bin] - cdf_min) / (total_pixels - cdf_min);
        }

        // 映射回原始数据范围
        equalized_array[i] = equalized_value * range + min_value;
    }

    params.set_output("Equalized Array", equalized_array);
    return true;
}

NODE_DECLARATION_UI(histogram_equalization);

NODE_DEF_CLOSE_SCOPE
