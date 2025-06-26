#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <onnxruntime_cxx_api.h>

class Tokenizer {
public:
    struct Config {
        std::string vocab_file;
        std::string decoder_model_path;
        std::string ort_extensions_path = "";
    };

    Tokenizer(const Config& config);
    ~Tokenizer();

    bool initialize();
    void cleanup();
    
    std::string decode(const std::vector<int>& token_ids);
    std::vector<int> encode(const std::string& text);
    
    // Vocabulary operations
    std::string idToToken(int id) const;
    int tokenToId(const std::string& token) const;
    size_t getVocabSize() const { return vocab_size_; }

private:
    Config config_;
    
    // ONNX decoder session
    std::unique_ptr<Ort::Session> decoder_session_;
    std::unique_ptr<Ort::Env> env_;
    Ort::MemoryInfo memory_info_;
    
    // Vocabulary
    std::unordered_map<int, std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    size_t vocab_size_ = 0;
    
    // Special tokens
    int pad_token_id_ = 0;
    int unk_token_id_ = 1;
    int bos_token_id_ = 2;
    int eos_token_id_ = 3;
    
    // Model tensor info
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    
    bool loadVocabulary();
    bool initializeDecoder();
    
    // Text processing utilities
    std::string postProcessText(const std::string& text);
    std::vector<std::string> splitByDelimiters(const std::string& text);
    std::string joinTokens(const std::vector<std::string>& tokens);
};