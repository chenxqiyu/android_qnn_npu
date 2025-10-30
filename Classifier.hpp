#ifndef __CLASSIFIER_H__
#define __CLASSIFIER_H__

#include "session.hpp"
#include "types.hpp"
#include "DataUtil.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
using namespace qnn;
using namespace tools;
using namespace sample_app;
#include <sstream>
#include <iterator>
#include <opencv2/opencv.hpp>

// #include <nlohmann/json.hpp> // 如果你支持 JSON 标签格式，可选

class Classifier : public session {
private:
    std::vector<const char*> input_node_names = {"input"};
    std::vector<const char*> output_node_names = {"output"};

public:
    Classifier(const std::string &modelPath, const std::string &backendPath, int device_id = 0) {
        int ret = load(modelPath, backendPath, device_id);
        if (ret != 0) {
            std::cerr << "Failed to load model: " << modelPath << std::endl;
        }
    }


// 假设你已有 QNN 相关接口和类定义，这里省略相关 include 和初始化


// 假设 qnn_net 是你已经加载好的模型对象，且它内部存储了 graph info




void printModelMetadata() {
    
    if (!qnn_net) {
        std::cerr << "QNN sample app instance is null!" << std::endl;
        return;
    }

    uint32_t graphCount = qnn_net->m_graphsCount;
    auto** graphsInfo = qnn_net->m_graphsInfo;

    for (uint32_t i = 0; i < graphCount; ++i) {
        auto graph = graphsInfo[i];
        if (!graph) continue;

        std::cout << "Graph Name: " << (graph->graphName ? graph->graphName : "null") << std::endl;



    }
}

// #include <nlohmann/json.hpp> // 如果你支持 JSON 标签格式，可选



std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> exps(logits.size());
    float maxLogit = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = std::exp(logits[i] - maxLogit);
        sum += exps[i];
    }
    for (float& val : exps) {
        val /= sum;
    }
    return exps;
}
std::vector<float> readFloat32Raw(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open raw output file: " << path << std::endl;
        return {};
    }

    std::vector<float> data((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());

    size_t count = data.size() / sizeof(float);
    std::vector<float> result(count);
    file.clear();
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(result.data()), count * sizeof(float));
    return result;
}
std::vector<std::string> loadLabels(const std::string& labelPath) {
    std::vector<std::string> labels;
    std::ifstream in(labelPath);
    if (!in) {
        std::cerr << "Failed to open label file: " << labelPath << std::endl;
        return labels;
    }

    std::stringstream buffer;
    buffer << in.rdbuf();
    std::string content = buffer.str();

  
        // 回退为 txt 文件逐行读取
        std::istringstream iss(content);
        std::string line;
        while (std::getline(iss, line)) {
            labels.push_back(line);
        }
  
    return labels;
}

struct ScoreIndex {
    int index;
    float score;
};

void printTop5(const std::vector<float>& output, const std::vector<std::string>& labels) {
    int topN = 5;
    std::vector<ScoreIndex> scores;
    for (int i = 0; i < output.size(); ++i) {
        scores.push_back({i, output[i]});
    }

    std::partial_sort(scores.begin(), scores.begin() + topN, scores.end(),
                      [](const ScoreIndex& a, const ScoreIndex& b) {
                          return a.score > b.score;
                      });

    std::cout << "Top " << topN << " predictions:" << std::endl;
    for (int i = 0; i < topN; ++i) {
        int idx = scores[i].index;
        float prob = scores[i].score;
        std::cout << i + 1 << ": ID=" << idx
                  << ", Score=" << prob * 100 << "%"
                  << ", Label=" << (idx < labels.size() ? labels[idx] : "Unknown")
                  << std::endl;
    }
}

std::vector<float> preprocessImage(const std::string& imagePath, int height = 224, int width = 224) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);  // BGR
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }

    // Resize
    cv::resize(img, img, cv::Size(width, height));

    // Convert to float and normalize to [0,1]
    img.convertTo(img, CV_32FC3, 1.0 / 255);

    // Convert BGR to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Convert HWC -> CHW
    std::vector<float> chw(height * width * 3);
    int idx = 0;
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                chw[idx++] = img.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    return chw;  // shape: (3, 224, 224)
}
void saveRaw(const std::string& outputPath, const std::vector<float>& data) {
    std::ofstream out(outputPath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + outputPath);
    }
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}
std::vector<float> preprocessImageNCHW(const std::string& imagePath, int height = 224, int width = 224) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);  // BGR
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }

    // Resize
    cv::resize(img, img, cv::Size(width, height));

    // Convert to float and normalize to [0,1]
    img.convertTo(img, CV_32FC3, 1.0 / 255);

    // Convert BGR to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    std::vector<float> nchw(3 * height * width);
    
    // NCHW 排序：先通道，再行，再列
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                cv::Vec3f pixel = img.at<cv::Vec3f>(h, w);
                nchw[c * height * width + h * width + w] = pixel[c];
            }
        }
    }

    return nchw;
}
std::vector<float> preprocessImageNHWC(const std::string& imagePath, int height = 224, int width = 224) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);  // BGR
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }

    // Resize
    cv::resize(img, img, cv::Size(width, height));

    // Convert to float and normalize to [0,1]
    img.convertTo(img, CV_32FC3, 1.0 / 255);

    // Convert BGR to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Convert to NHWC (H, W, C -> float32 array)
    std::vector<float> nhwc(height * width * 3);
    int idx = 0;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            cv::Vec3f pixel = img.at<cv::Vec3f>(h, w);  // [R, G, B]
            nhwc[idx++] = pixel[0]; // R
            nhwc[idx++] = pixel[1]; // G
            nhwc[idx++] = pixel[2]; // B
        }
    }

    return nhwc;  // shape: [H, W, C] (can be reshaped as [1, H, W, C])
}
 //std::vector<uint16_t> preprocessImageNHWC_uint16(const std::string& imagePath) {
 //    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
 //    if (img.empty()) throw std::runtime_error("Failed to load image");

 //    cv::resize(img, img, cv::Size(224, 224));
 //    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

 //    // 转成 float，归一化到 [0, 1]
 //    cv::Mat img_float;
 //    img.convertTo(img_float, CV_32FC3, 1.0 / 255.0);

 //    // 转成 uint16，范围 [0, 65535]
 //    std::vector<uint16_t> data(224 * 224 * 3);

 //    int idx = 0;
 //    for (int h = 0; h < 224; ++h) {
 //        for (int w = 0; w < 224; ++w) {
 //            cv::Vec3f pixel = img_float.at<cv::Vec3f>(h, w);
 //            for (int c = 0; c < 3; ++c) {
 //                float val = pixel[c];
 //                val = std::min(std::max(val, 0.0f), 1.0f);
 //                data[idx++] = static_cast<uint16_t>(val * 65535);
 //            }
 //        }
 //    }

 //    return data;  // NHWC uint16 vector
 //}
//std::vector<uint16_t> preprocessImageNHWC_uint16(const std::string& imagePath) {
//    // 读取图像，彩色模式
//    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
//    if (image.empty()) {
//        throw std::runtime_error("Failed to load image: " + imagePath);
//    }
//
//    // 调整为 224x224（如果模型有要求）
//    cv::resize(image, image, cv::Size(224, 224));
//
//    // OpenCV 默认格式是 BGR，转换为 RGB
//    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
//
//    // 创建输出 buffer（NHWC，uint16）
//    std::vector<uint16_t> output(image.rows * image.cols * image.channels());
//
//    const float scale = 1.5259e-05f;
//    const float inv_scale = 1.0f / scale;  // ≈ 65535
//
//    // 填充 output，像素从 [0,255] → [0,65535]
//    int idx = 0;
//    for (int h = 0; h < image.rows; ++h) {
//        for (int w = 0; w < image.cols; ++w) {
//            for (int c = 0; c < image.channels(); ++c) {
//                uint8_t pixel = image.at<cv::Vec3b>(h, w)[c];
//                output[idx++] = static_cast<uint16_t>(pixel * inv_scale / 255.0f + 0.5f);
//            }
//        }
//    }
//
//    return output;
//}
//
//std::vector<uint16_t> preprocessImageNCHW_uint16(const std::string& imagePath) {
//    // 读取图像，彩色模式
//    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
//    if (image.empty()) {
//        throw std::runtime_error("Failed to load image: " + imagePath);
//    }
//
//    // 调整为 224x224
//    cv::resize(image, image, cv::Size(224, 224));
//
//    // 转换 BGR → RGB
//    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
//
//    const int height = image.rows;
//    const int width = image.cols;
//    const int channels = image.channels();  // 应该是 3
//
//    // NCHW 格式的 buffer: [C][H][W]
//    std::vector<uint16_t> output(channels * height * width);
//
//    const float scale = 1.5259e-05f;
//    const float inv_scale = 1.0f / scale;  // ≈ 65535
//
//    // 填充为 NCHW 格式
//    for (int c = 0; c < channels; ++c) {
//        for (int h = 0; h < height; ++h) {
//            for (int w = 0; w < width; ++w) {
//                uint8_t pixel = image.at<cv::Vec3b>(h, w)[c];
//                size_t idx = c * height * width + h * width + w;
//                output[idx] = static_cast<uint16_t>(pixel * inv_scale / 255.0f + 0.5f);
//            }
//        }
//    }
//
//    return output;
//}
// 
// 
//
// 

std::vector<uint16_t> preprocessImageNHWC_uint16(const std::string& imagePath) {
    //cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    //if (image.empty()) {
    //    throw std::runtime_error("Failed to load image: " + imagePath);
    //}

    //cv::resize(image, image, cv::Size(224, 224));
    //cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    //int height = image.rows;
    //int width = image.cols;
    //int channels = image.channels();  // 通常是 3

    //std::vector<uint16_t> output(height * width * channels);

    //int idx = 0;
    //for (int h = 0; h < height; ++h) {
    //    for (int w = 0; w < width; ++w) {
    //        for (int c = 0; c < channels; ++c) {
    //            uint8_t pixel = image.at<cv::Vec3b>(h, w)[c];
    //            output[idx++] = static_cast<uint16_t>(pixel);
    //        }
    //    }
    //}

    //return output;



    const float scale = 1.5259e-05f;
    const int width = 224, height = 224;

    // 1. 读取并resize图像
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);


    cv::resize(image, image, cv::Size(width, height));
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);  // 归一化为 0~1

    // 2. 转换为量化后的 uint16 数据
    std::vector<uint16_t> inputData(width * height * 3);
    int idx = 0;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            cv::Vec3f pixel = image.at<cv::Vec3f>(h, w);
            for (int c = 0; c < 3; ++c) {
                inputData[idx++] = static_cast<uint16_t>(std::round(pixel[c] / scale));
            }
        }
    }

    return inputData;
}
std::vector<uint16_t> preprocessImageNCHW_uint16(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }

    cv::resize(image, image, cv::Size(224, 224));
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    int height = image.rows;
    int width = image.cols;
    int channels = image.channels();  // 通常是 3

    std::vector<uint16_t> output(channels * height * width);

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                uint8_t pixel = image.at<cv::Vec3b>(h, w)[c];
                size_t idx = c * height * width + h * width + w;
                output[idx] = static_cast<uint16_t>(pixel);
            }
        }
    }

    return output;
}

// std::vector<uint16_t> preprocessImageNHWC_uint16(const std::string& imagePath, int height = 224, int width = 224) {
//     cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);  // BGR
//     if (img.empty()) {
//         throw std::runtime_error("Failed to load image: " + imagePath);
//     }

//     // Resize
//     cv::resize(img, img, cv::Size(width, height));

//     // Convert BGR to RGB
//     cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

//     // Convert to float32, normalize到[0,1]
//     img.convertTo(img, CV_32FC3, 1.0 / 255);

//     // Scale to uint16: [0, 1] -> [0, 65535]
//     img *= 65535.0;

//     // Clamp (有些值可能会超过65535或小于0)
//     cv::Mat img_clamped;
//     cv::min(img, 65535.0, img);
//     cv::max(img, 0.0, img);
//     img.convertTo(img_clamped, CV_16UC3);  // float32 -> uint16

//     // 转为 NHWC 格式的一维数组
//     std::vector<uint16_t> nhwc(height * width * 3);
//     int idx = 0;
//     for (int h = 0; h < height; ++h) {
//         for (int w = 0; w < width; ++w) {
//             cv::Vec3w pixel = img_clamped.at<cv::Vec3w>(h, w);  // R,G,B
//             nhwc[idx++] = pixel[0];  // R
//             nhwc[idx++] = pixel[1];  // G
//             nhwc[idx++] = pixel[2];  // B
//         }
//     }

//     return nhwc;  // NHWC 格式
// }
void saveRawUint16(const std::string& filename, const std::vector<uint16_t>& data) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + filename);
    }
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(uint16_t));
}

// std::vector<float> preprocessImageNHWC_float(const std::string& imagePath) {
//     cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
//     if (img.empty()) throw std::runtime_error("Failed to load image");

//     cv::resize(img, img, cv::Size(224, 224));
//     cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

//     cv::Mat img_float;
//     img.convertTo(img_float, CV_32FC3, 1.0 / 255.0);

//     std::vector<float> data(224 * 224 * 3);
//     int idx = 0;
//     for (int h = 0; h < 224; ++h) {
//         for (int w = 0; w < 224; ++w) {
//             cv::Vec3f pixel = img_float.at<cv::Vec3f>(h, w);
//             for (int c = 0; c < 3; ++c) {
//                 data[idx++] = pixel[c];
//             }
//         }
//     }
//     return data;
// }

// 假设模型输入要求 NHWC uint16，scale 和 zero_point 需要从模型或量化参数确定
// 这里的 scale 和 zero_point 需要你用实际模型量化参数替换
 float QUANT_SCALE = 1.0f / 256.0f;  // 举例，需确认
 int QUANT_ZERO_POINT = 0;

 std::vector<uint16_t> quantizeFloatToUint16(const float* float_data, size_t size, float scale, int32_t zero_point) {
    std::vector<uint16_t> quantized(size);
    for (size_t i = 0; i < size; ++i) {
        int32_t q = static_cast<int32_t>(std::round(float_data[i] / scale)) + zero_point;
        if (q < 0) q = 0;
        if (q > 65535) q = 65535;
        quantized[i] = static_cast<uint16_t>(q);
    }
    return quantized;
}

//0.000015259021893143654
//std::vector<uint16_t> preprocessImageNHWC_uint16(const std::string& imagePath,int width = 224,int height = 224) {
//    // cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR); // BGR
//    // if (img.empty()) {
//    //     throw std::runtime_error("Failed to load image: " + imagePath);
//    // }
//
//    // cv::resize(img, img, cv::Size(width, height));
//    // img.convertTo(img, CV_32FC3, 1.0 / 255);  // 归一化到[0,1]
//
//    // // BGR -> RGB
//    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
//
//    // // NHWC uint16 量化，数据存储顺序为 NHWC
//    // std::vector<uint16_t> quantized(width * height * 3);
//
//    // int idx = 0;
//    // for (int h = 0; h < height; ++h) {
//    //     for (int w = 0; w < width; ++w) {
//    //         cv::Vec3f pixel = img.at<cv::Vec3f>(h, w);
//    //         for (int c = 0; c < 3; ++c) {
//    //             float float_val = pixel[c];
//    //             // 量化公式
//    //             int quant_val = static_cast<int>(std::round(float_val / QUANT_SCALE) + QUANT_ZERO_POINT);
//    //             if (quant_val < 0) quant_val = 0;
//    //             if (quant_val > 65535) quant_val = 65535;
//    //             quantized[idx++] = static_cast<uint16_t>(quant_val);
//    //         }
//    //     }
//    // }
//    // return quantized;
//
//
//
//}
std::vector<uint16_t> convertFloatToUint16_scaled(const std::vector<float>& floatData, float inputScale) {
    std::vector<uint16_t> result(floatData.size());
    for (size_t i = 0; i < floatData.size(); ++i) {
        float value = floatData[i] / inputScale;
        result[i] = static_cast<uint16_t>(std::round(std::clamp(value, 0.0f, 65535.0f)));
    }
    return result;
}

std::vector<float> preprocessImageNCHW_float(const std::string& imagePath, int width = 224, int height = 224) {
    // 读取图像
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Failed to read image: " + imagePath);
    }

    // 调整大小
    cv::resize(img, img, cv::Size(width, height));

    // 转为 float32 并归一化到 [0,1]
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);

    // 转换 BGR 到 RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // 准备输出向量，顺序为 NCHW（这里只处理单张图片，即 N=1）
    std::vector<float> output;
    output.reserve(3 * height * width);

    // 分别按通道遍历
    for (int c = 0; c < 3; ++c) {           // Channel loop: R, G, B
        for (int h = 0; h < height; ++h) {  // Height
            for (int w = 0; w < width; ++w) { // Width
                cv::Vec3f pixel = img.at<cv::Vec3f>(h, w);
                output.push_back(pixel[c]); // c=0:R, c=1:G, c=2:B
            }
        }
    }

    return output;
}

std::vector<float> preprocessImageNHWC_float(const std::string& imagePath, int width = 224, int height = 224) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (img.empty()) throw std::runtime_error("Failed to load image");

    cv::resize(img, img, cv::Size(width, height));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    std::vector<float> result(width * height * 3);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
            for (int c = 0; c < 3; ++c) {
                result[(i * width + j) * 3 + c] = pixel[c] / 255.0f; // float 0~1
            }
        }
    }
    return result;
}
int inferTop2(const std::string &imagePath, const std::string &labelPath, const std::string &way) {

    // Qnn_Tensor_t inputTensor = {};
    // inputTensor.tensorName = const_cast<char*>("input");
    // inputTensor.data = reinterpret_cast<void*>(inputData.data());
    // inputTensor.dataSize = sizeof(uint16_t) * inputElementCount;
    // inputTensor.dataType = QNN_DATATYPE_UINT_16;
    
    // inputTensor.rank = 4;
    // inputTensor.dimensions = new uint32_t[4]{1, 224, 224, 3};
    
    // inputTensor.quantizeParams.encodingType = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    // inputTensor.quantizeParams.scaleOffsetEncoding.scale = inputScale;
    // inputTensor.quantizeParams.scaleOffsetEncoding.offset = inputZeroPoint;
    
    Qnn_Tensor_t inputTensor = {};
    // inputTensor.name = const_cast<char*>("input"); // 确认模型输入名


}

    int inferTop1(const std::string &imagePath, const std::string &labelPath, const std::string &way) {


      // 1. 获取输入维度
      auto inputDims = qnn_net->inputDims[0];
      size_t inputSize = datautil::calculateElementCount(inputDims);
      std::cout << "inputPath: " << imagePath << std::endl;
      std::cout << "labelPath: " << labelPath << std::endl;
      std::cout << "inputSize: " << inputSize << std::endl;
      
            // 1. 获取输入维度
    // auto inputDims = qnn_net->inputDims[0];  // 如：{1, 3, 224, 224}
    // size_t inputSize = datautil::calculateElementCount(inputDims);
    // int channels = inputDims[1];
    // int height = inputDims[2];
    // int width  = inputDims[3];

    // std::cout << "Image path: " << imagePath << std::endl;

    // // 2. 加载图片并 resize
    // cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    // if (img.empty()) {
    //     std::cerr << "Failed to load image: " << imagePath << std::endl;
    //     return -1;
    // }

    // cv::resize(img, img, cv::Size(width, height));

    // // 3. 转换为 float 并归一化到 [0, 1]
    // img.convertTo(img, CV_32FC3, 1.0 / 255);

    // // 4. BGR -> RGB（如果模型需要）
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // // 5. HWC -> CHW
    // std::vector<float> inputData(inputSize);
    // size_t idx = 0;
    // for (int c = 0; c < channels; ++c) {
    //     for (int h = 0; h < height; ++h) {
    //         for (int w = 0; w < width; ++w) {
    //             inputData[idx++] = img.at<cv::Vec3f>(h, w)[c];
    //         }
    //     }
    // }

    std::cout << "Input dims: ";
for (auto d : qnn_net->inputDims[0]) std::cout << d << " ";
std::cout << std::endl;

//定义inputData


std::vector<float> inputData;

if (way == "NHWC") {
    //inputData = preprocessImageNHWC(imagePath, 224, 224);
    //preprocessImageNHWC_uint16(imagePath);

    inputData = preprocessImageNHWC_float(imagePath);

} else if (way == "NCHW") {
    //inputData = preprocessImageNCHW(imagePath, 224, 224);
    inputData = preprocessImageNCHW_float(imagePath);
} 
   
    // saveRaw(outputPath, data);   
    // auto inputData = preprocessImageNHWC_uint16(imagePath);
    // auto inputData = preprocessImageNHWC_float(imagePath);

    // saveRawUint16(outputRawPath, uint16_tensor);

    // 6. 推理
 //std::vector<uint8_t*> input_ptrs = {reinterpret_cast<uint8_t*>(inputData.data())};

std::vector<uint8_t*> input_ptrs = { reinterpret_cast<uint8_t*>(inputData.data()) };
    //std::vector<uint16_t*> input_ptrs= { reinterpret_cast<uint16_t*>(std::get<std::vector<uint16_t>>(inputData).data()) };
    //std::vector<uint16_t*> input_ptrs = { };

    //if (std::holds_alternative<std::vector<float>>(inputData)) {
    //    input_ptrs = { reinterpret_cast<uint8_t*>(std::get<std::vector<float>>(inputData).data()) };
    //} else {
    //     input_ptrs =
    //    
    //}
    //
    std::vector<std::vector<float>> output_vals;
    this->inference(input_ptrs, output_vals);

    std::cout << "Output size: " << output_vals[0].size() << std::endl;

    auto probs = softmax(output_vals[0]);

    // 7. 加载标签并输出
    auto labels = loadLabels(labelPath);
    printTop5(probs, labels);




        // auto output = readFloat32Raw(inputPath);
        // auto probs = softmax(output);
        // auto labels = loadLabels(labelPath);
        // printTop5(probs, labels);





    // printModelMetadata();



// // 2. 读取float32二进制数据（无需归一化）
// std::vector<float> inputData(inputSize);
// std::ifstream in(inputPath, std::ios::binary);
// if (!in) {
//     std::cerr << "Failed to open input file: " << inputPath << std::endl;
//     return -1;
// }
// // in.read(reinterpret_cast<char*>(inputData.data()), inputSize * sizeof(float));
// // if (in.gcount() != static_cast<std::streamsize>(inputSize * sizeof(float))) {
// //     std::cerr << "Input file size mismatch." << std::endl;
// //     return -1;
// // }

// // 3. 推理
// std::vector<uint8_t*> input_ptrs = {reinterpret_cast<uint8_t*>(inputData.data())};
// std::vector<std::vector<float>> output_vals;
// this->inference(input_ptrs, output_vals);

// auto probs = softmax(output_vals[0]);
// auto labels = loadLabels(labelPath);
// printTop5(probs, labels);





    }
};

#endif // __CLASSIFIER_H__
