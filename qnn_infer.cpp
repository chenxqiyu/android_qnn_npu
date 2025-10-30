#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include "Classifier.hpp"

// 简单读取 .raw 文件为 float
std::vector<float> readRawInput(const std::string& filename, size_t size) {
    std::vector<float> buffer(size);
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open input file");
    file.read(reinterpret_cast<char*>(buffer.data()), size * sizeof(float));
    return buffer;
}

// 简单读取 label 文件
std::vector<std::string> readLabels(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) return {};
    std::vector<std::string> labels;
    std::string line;
    while (std::getline(file, line)) labels.push_back(line);
    return labels;
}

// 获取 argmax
int getTopClass(const std::vector<float>& data) {
    int maxIdx = 0;
    float maxVal = data[0];
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i] > maxVal) {
            maxVal = data[i];
            maxIdx = static_cast<int>(i);
        }
    }
    return maxIdx;
}

// 简易命令行参数解析
const char* getArg(int argc, char** argv, const char* key, const char* def = nullptr) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return def;
}

int main(int argc, char** argv) {
    //打印开始了
    std::cout << "Hello, World6666!" << std::endl;

    //htp
    //./qnn_infer.uint16 --model ./efu16.so --input ./dog.jpg --labels ./labels.txt --backend libQnnHtp.so --nc NHWC16
    //cpu版本
    // ./qnn_infer.out --model ./eff.so --input ./dog.jpg --labels ./labels.txt --backend libQnnCpu.so
    //npu版本
    //需要zero scale参数
    //./qnn_infer.out --model ./e-w8a16.so --input ./dog.jpg --labels ./labels.txt --backend libQnnHtp.so

    const char* modelPath = getArg(argc, argv, "--model");
    const char* inputPath = getArg(argc, argv, "--input");
    const char* backEndPath  = getArg(argc, argv, "--backend");
    const char* labelPath = getArg(argc, argv, "--labels");
    const char* nc = getArg(argc, argv, "--nc");

    if (!modelPath || !inputPath ||!labelPath||!backEndPath||!nc) {
        std::cerr << "Usage: " << argv[0]
                  << " --model model.so --input input.raw --backend libQnnHtp.so --labels labels.txt --nc NCHW \n"
                 <<" ./qnn_infer.out --model ./eff.so --input ./dog.jpg --labels ./labels.txt --backend libQnnCpu.so \n"
                 <<" ./qnn_infer.out --model ./e-w8a16.so --input ./dog.jpg --labels ./labels.txt --backend libQnnHtp.so \n";
        return 1;
    }
    // Unet *unet = new Unet(unet_path, htp_backpath, 0);

    // ./qnn_infer.out --model ./e-w8a16.so --input ./output6/Result_0/class_logits.raw --labels ./labels.txt

    Classifier cls(modelPath, backEndPath);
    cls.inferTop1(inputPath, labelPath,nc);



    
    // size_t inputSize = std::stoul(sizeStr);
    // std::vector<float> inputData = readRawInput(inputPath, inputSize);

    // // TODO: 加载模型 .so，创建 QNN context / graph（省略，需集成 QNN SDK）

    // // TODO: 将 inputData 构造为 Qnn_Tensor_t 输入结构并运行模型

    // // 示例模拟输出数据（真实应为从 QNN 推理中获得）
    // std::vector<float> output(1000, 0.0f); // 模拟 1000 类输出
    // output[123] = 0.89f; // 假设类别123的概率最高

    // int topClass = getTopClass(output);
    // std::cout << "Top-1 class index: " << topClass << "\n";
    // std::cout << "Score: " << output[topClass] * 100 << "%\n";

    // if (labelPath) {
    //     std::vector<std::string> labels = readLabels(labelPath);
    //     if (topClass < labels.size())
    //         std::cout << "Label: " << labels[topClass] << "\n";
    // }

    return 0;
}
