#include "audio_recorder.hpp"
#include "vad_detector.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>

AudioRecorder::AudioRecorder()
    : config_(Config{}), stream_(nullptr), is_recording_(false), 
      speech_detected_(false), should_stop_(false), vad_detector_(nullptr) {
}

AudioRecorder::AudioRecorder(const Config& config)
    : config_(config), stream_(nullptr), is_recording_(false), 
      speech_detected_(false), should_stop_(false), vad_detector_(nullptr) {
}

AudioRecorder::~AudioRecorder() {
    cleanup();
}

bool AudioRecorder::initialize() {
    // Suppress ALSA error messages
    FILE* null_file = std::freopen("/dev/null", "w", stderr);
    (void)null_file; // Suppress unused warning
    
    PaError err = Pa_Initialize();
    
    // Restore stderr
    FILE* tty_file = std::freopen("/dev/tty", "w", stderr);
    (void)tty_file; // Suppress unused warning
    
    if (err != paNoError) {
        std::cerr << "Failed to initialize PortAudio: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    // Print available devices for debugging
    int numDevices = Pa_GetDeviceCount();
    std::cout << "Available audio devices:" << std::endl;
    for (int i = 0; i < numDevices; i++) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
        std::cout << "Device " << i << ": " << deviceInfo->name 
                  << " (inputs: " << deviceInfo->maxInputChannels << ")" << std::endl;
    }

    // Set up stream parameters
    PaStreamParameters inputParameters;
    inputParameters.device = (config_.device_index >= 0) ? 
        config_.device_index : Pa_GetDefaultInputDevice();
    
    if (inputParameters.device == paNoDevice) {
        std::cerr << "No default input device available." << std::endl;
        return false;
    }

    inputParameters.channelCount = config_.channels;
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;

    // Open stream
    err = Pa_OpenStream(&stream_,
                        &inputParameters,
                        nullptr, // no output
                        config_.sample_rate,
                        config_.frames_per_buffer,
                        paClipOff,
                        audioCallback,
                        this);

    if (err != paNoError) {
        std::cerr << "Failed to open stream: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    return true;
}

void AudioRecorder::cleanup() {
    if (is_recording_.load()) {
        stopRecording();
    }

    if (stream_) {
        Pa_CloseStream(stream_);
        stream_ = nullptr;
    }
    
    Pa_Terminate();
}

int AudioRecorder::audioCallback(const void* input_buffer, void* output_buffer,
                                unsigned long frames_per_buffer,
                                const PaStreamCallbackTimeInfo* time_info,
                                PaStreamCallbackFlags status_flags,
                                void* user_data) {
    AudioRecorder* recorder = static_cast<AudioRecorder*>(user_data);
    const float* input = static_cast<const float*>(input_buffer);
    
    if (input) {
        recorder->processAudioFrame(input, frames_per_buffer);
    }
    
    return recorder->should_stop_.load() ? paComplete : paContinue;
}

void AudioRecorder::processAudioFrame(const float* input, unsigned long frame_count) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    // Convert to vector for easier processing
    std::vector<float> frame(input, input + frame_count * config_.channels);
    
    // Add to pre-speech buffer (circular buffer)
    if (pre_speech_buffer_.size() > config_.frames_per_buffer * 10) {
        pre_speech_buffer_.erase(pre_speech_buffer_.begin(), 
                                pre_speech_buffer_.begin() + config_.frames_per_buffer);
    }
    pre_speech_buffer_.insert(pre_speech_buffer_.end(), frame.begin(), frame.end());
    
    // Choose VAD method based on configuration
    bool is_speech = false;
    
    if (useEnergyVAD()) {
        // Energy-based VAD
        float energy_prob = computeEnergyVAD(frame.data(), frame.size());
        is_speech = energy_prob > config_.trigger_threshold;
    } else if (useSileroVAD() && vad_detector_) {
        // Silero VAD
        float silero_prob = computeSileroVAD(frame);
        is_speech = silero_prob > config_.trigger_threshold;
        
    } else {
        // Fallback to energy VAD if Silero not available
        float energy_prob = computeEnergyVAD(frame.data(), frame.size());
        is_speech = energy_prob > config_.trigger_threshold;
    }
    
    // Call external VAD callback if available
    if (vad_callback_) {
        vad_callback_(frame);
    }
    
    auto now = std::chrono::steady_clock::now();
    
    if (is_speech) {
        last_speech_time_ = now;
        if (!speech_detected_.load()) {
            speech_detected_.store(true);
            std::cout << "▶ Speech detected, starting recording..." << std::endl;
            // Add pre-speech buffer to main buffer
            audio_buffer_.insert(audio_buffer_.end(), 
                               pre_speech_buffer_.begin(), pre_speech_buffer_.end());
        }
    }
    
    if (speech_detected_.load()) {
        audio_buffer_.insert(audio_buffer_.end(), frame.begin(), frame.end());
        
        // Check stopping conditions
        auto silence_duration = std::chrono::duration<double>(now - last_speech_time_).count();
        auto total_duration = std::chrono::duration<double>(now - recording_start_time_).count();
        
        if (silence_duration > config_.silence_duration) {
            std::cout << "⏹ Silence detected, stopping recording" << std::endl;
            should_stop_.store(true);
        } else if (total_duration > config_.max_record_time) {
            std::cout << "⏹ Max recording time reached, stopping recording" << std::endl;
            should_stop_.store(true);
        }
    }
}

std::vector<float> AudioRecorder::recordAudio() {
    if (!stream_) {
        std::cerr << "Stream not initialized" << std::endl;
        return {};
    }

    // Reset state
    audio_buffer_.clear();
    pre_speech_buffer_.clear();
    vad_buffer_.clear(); // Clear VAD buffer for new recording
    speech_detected_.store(false);
    should_stop_.store(false);
    recording_start_time_ = std::chrono::steady_clock::now();
    last_speech_time_ = recording_start_time_;
    
    // Reset VAD detector state if using Silero VAD
    if (vad_detector_) {
        vad_detector_->reset();
    }

    // Start stream
    PaError err = Pa_StartStream(stream_);
    if (err != paNoError) {
        std::cerr << "Failed to start stream: " << Pa_GetErrorText(err) << std::endl;
        return {};
    }

    // Wait for recording to complete
    while (!should_stop_.load()) {
        Pa_Sleep(100); // Sleep 100ms
    }

    // Stop stream
    err = Pa_StopStream(stream_);
    if (err != paNoError) {
        std::cerr << "Failed to stop stream: " << Pa_GetErrorText(err) << std::endl;
    }

    std::lock_guard<std::mutex> lock(buffer_mutex_);
    return audio_buffer_;
}

void AudioRecorder::startRecording() {
    if (is_recording_.load()) {
        return;
    }

    is_recording_.store(true);
    recording_thread_ = std::thread(&AudioRecorder::recordingThread, this);
}

void AudioRecorder::stopRecording() {
    if (!is_recording_.load()) {
        return;
    }

    should_stop_.store(true);
    is_recording_.store(false);
    
    if (recording_thread_.joinable()) {
        recording_thread_.join();
    }
}

void AudioRecorder::recordingThread() {
    auto result = recordAudio();
    
    std::unique_lock<std::mutex> lock(recording_mutex_);
    last_recording_ = std::move(result);
    lock.unlock();
    
    recording_cv_.notify_all();
}

std::vector<float> AudioRecorder::getLastRecording() {
    std::unique_lock<std::mutex> lock(recording_mutex_);
    recording_cv_.wait(lock, [this] { return !is_recording_.load(); });
    return last_recording_;
}

float AudioRecorder::computeEnergyVAD(const float* input, unsigned long frame_count) {
    // Calculate RMS energy
    float energy = 0.0f;
    float max_sample = 0.0f;
    
    for (unsigned long i = 0; i < frame_count; ++i) {
        energy += input[i] * input[i];
        max_sample = std::max(max_sample, std::abs(input[i]));
    }
    energy = std::sqrt(energy / frame_count);
    
    // Convert energy to probability-like value (0-1)
    // Adjust these thresholds based on your audio environment
    const float min_energy = 0.0001f;
    const float max_energy = 0.1f;
    
    if (energy < min_energy) return 0.0f;
    if (energy > max_energy) return 1.0f;
    
    // Linear mapping to 0-1 range
    return (energy - min_energy) / (max_energy - min_energy);
}

float AudioRecorder::computeSileroVAD(const std::vector<float>& audio_chunk) {
    if (!vad_detector_) {
        return 0.0f;
    }
    
    // Accumulate audio chunks for Silero VAD (using member variable)
    static const size_t VAD_WINDOW_SIZE = 512; // 32ms at 16kHz
    
    // Resample audio_chunk to 16kHz if needed
    std::vector<float> resampled_chunk;
    if (config_.sample_rate != 16000) {
        // Simple decimation for 48kHz -> 16kHz (3:1 ratio)
        if (config_.sample_rate == 48000) {
            resampled_chunk.reserve(audio_chunk.size() / 3);
            for (size_t i = 0; i < audio_chunk.size(); i += 3) {
                resampled_chunk.push_back(audio_chunk[i]);
            }
        } else {
            // For other sample rates, use simple linear interpolation
            double ratio = static_cast<double>(config_.sample_rate) / 16000.0;
            size_t new_size = static_cast<size_t>(audio_chunk.size() / ratio);
            resampled_chunk.reserve(new_size);
            for (size_t i = 0; i < new_size; ++i) {
                size_t src_idx = static_cast<size_t>(i * ratio);
                if (src_idx < audio_chunk.size()) {
                    resampled_chunk.push_back(audio_chunk[src_idx]);
                }
            }
        }
    } else {
        resampled_chunk = audio_chunk;
    }
    
    // Add resampled chunk to buffer
    vad_buffer_.insert(vad_buffer_.end(), resampled_chunk.begin(), resampled_chunk.end());
    
    // Process when we have enough data
    if (vad_buffer_.size() >= VAD_WINDOW_SIZE) {
        // Use the latest VAD_WINDOW_SIZE samples
        std::vector<float> vad_input(vad_buffer_.end() - VAD_WINDOW_SIZE, vad_buffer_.end());
        
        // Keep only recent data in buffer (sliding window)
        if (vad_buffer_.size() > VAD_WINDOW_SIZE * 2) {
            vad_buffer_.erase(vad_buffer_.begin(), vad_buffer_.end() - VAD_WINDOW_SIZE);
        }
        
        // Use Silero VAD detector
        float prob = vad_detector_->detectVAD(vad_input);
        return prob;
    }
    
    // Not enough data yet, return low probability
    return 0.0f;
}
