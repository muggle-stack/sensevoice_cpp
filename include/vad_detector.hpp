#pragma once

#include <vector>
#include <memory>
#include <deque>
#include <onnxruntime_cxx_api.h>

class VADDetector {
public:
    struct Config {
        std::string model_path;
        int sample_rate = 16000;
        int window_size = 512; // 32ms at 16kHz
        int context_size = 64;
        size_t history_size = 10; // frames for smoothing
    };

    VADDetector(const Config& config);
    ~VADDetector();

    bool initialize();
    void cleanup();
    void reset();
    
    float detectVAD(const std::vector<float>& audio);
    float detectVAD(const float* audio, size_t length);

private:
    Config config_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Env> env_;
    Ort::MemoryInfo memory_info_;
    
    // Model state
    std::vector<float> state_h_;
    std::vector<float> state_c_;
    std::vector<float> context_;
    
    // History for smoothing
    std::deque<float> prob_history_;
    
    // Input/output tensor info
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    bool initializeSession();
    void resampleIfNeeded(const float* input, size_t input_length, 
                         std::vector<float>& output);
};