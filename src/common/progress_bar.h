#pragma once

#include <chrono>
#include <cstddef>
#include <iostream>

class ProgressBar {
public:
    ProgressBar(std::string name, std::size_t total, std::size_t width = 100)
        : name_{name}, total_{total}, width_{width}, progress_{0},
          start_{std::chrono::steady_clock::now()} {
        std::cout << name_ << " [" << std::string(width_, ' ') << "] 0%";
        std::cout.flush();
    }

    void update(std::size_t progress = 1) {
        progress_ += progress;
        std::size_t percent{100 * progress_ / total_};
        if (percent > 0) {
            std::cout << '\r' << name_ << " [" << std::string(percent * width_ / 100 - 1, '=')
                      << ">" << std::string(width_ - percent * width_ / 100, ' ')
                      << "] "<< percent << '%';
        }
        std::cout.flush();
    }

    ~ProgressBar() {
        std::cout << '\r' << name_ << " [" << std::string(width_, '=') << "] 100%" << std::endl;
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start_);
        std::cout << "Elapsed time: " << duration.count() << " s" << std::endl;
    }

private:
    std::string name_;
    std::size_t total_;
    std::size_t width_;
    std::size_t progress_;
    std::chrono::steady_clock::time_point start_;
};
