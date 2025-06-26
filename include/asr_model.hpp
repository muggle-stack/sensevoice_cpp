#pragma once

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <onnxruntime_cxx_api.h>

class AudioProcessor;
class Tokenizer;

class ASRModel {
public:
    struct Config {
        std::string model_path;
        std::string config_path;
        std::string vocab_path;
        std::string decoder_path;
        int batch_size = 1;
        int sample_rate = 16000;
        std::string language = "zh";
        bool use_itn = true;
        bool quantized = true;
    };

    ASRModel(const Config& config);
    ~ASRModel();

    bool initialize();
    void cleanup();
    
    std::string recognize(const std::vector<float>& audio);
    std::string recognize(const float* audio, size_t length);
    
    // Batch processing
    std::vector<std::string> recognizeBatch(const std::vector<std::vector<float>>& audio_batch);

private:
    Config config_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Env> env_;
    Ort::MemoryInfo memory_info_;
    
    std::unique_ptr<AudioProcessor> audio_processor_;
    std::unique_ptr<Tokenizer> tokenizer_;
    
    // Model parameters
    int blank_id_ = 0;
    std::map<std::string, int> language_dict_;
    std::map<std::string, int> textnorm_dict_;
    
    // Input/output tensor info
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    bool initializeSession();
    bool loadConfig();
    void initializeLanguageMaps();
    
    std::vector<float> extractFeatures(const std::vector<float>& audio);
    std::vector<int> decodeCTC(const std::vector<float>& logits, int sequence_length);
    std::string postProcess(const std::vector<int>& token_ids);
    
    int getLanguageId(const std::string& language);
    int getTextnormId(bool use_itn);
};