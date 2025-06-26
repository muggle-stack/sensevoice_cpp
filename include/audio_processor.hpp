#pragma once

#include <vector>
#include <string>
#include <complex>
#include <cmath>

class AudioProcessor {
public:
    struct Config {
        int sample_rate = 16000;
        int frame_length = 400;  // 25ms
        int frame_shift = 160;   // 10ms
        int n_mels = 80;
        int n_fft = 512;  // Changed to ensure power-of-2 for efficient FFT
        float preemphasis = 0.97f;
        bool apply_cmvn = true;
        std::string cmvn_file;
    };

    AudioProcessor(const Config& config);
    ~AudioProcessor();

    bool initialize();
    void loadCMVN(const std::string& cmvn_file);
    
    // Main feature extraction
    std::vector<std::vector<float>> extractFeatures(const std::vector<float>& audio);
    std::vector<std::vector<float>> extractFeatures(const float* audio, size_t length);
    
    // Individual processing steps
    std::vector<float> preprocess(const std::vector<float>& audio);
    std::vector<std::vector<float>> computeFbank(const std::vector<float>& audio);
    std::vector<std::vector<float>> applyLFR(const std::vector<std::vector<float>>& features);
    void applyCMVN(std::vector<std::vector<float>>& features);

private:
    Config config_;
    
    // CMVN parameters
    bool cmvn_loaded_ = false;
    std::vector<float> cmvn_mean_;
    std::vector<float> cmvn_var_;
    
    // Mel filterbank
    std::vector<std::vector<float>> mel_filterbank_;
    
    // Window function
    std::vector<float> window_;
    
    void initializeMelFilterbank();
    void initializeWindow();
    
    // FFT and spectral processing
    std::vector<std::complex<float>> fft(const std::vector<float>& signal);
    std::vector<float> computePowerSpectrum(const std::vector<std::complex<float>>& fft_result);
    std::vector<float> applyMelFilterbank(const std::vector<float>& power_spectrum);
    
    // Utility functions
    float melScale(float freq);
    float invMelScale(float mel);
    std::vector<float> createHammingWindow(int size);
    
    // Frame processing
    std::vector<std::vector<float>> frameSignal(const std::vector<float>& signal);
    void padFeatures(std::vector<std::vector<float>>& features, int target_length);
};