#pragma once

#include <chrono>
#include <cstddef>
#include <iostream>

class ProgressBar {
public:
    ProgressBar(std::size_t total, std::size_t width = 100)
        : total_{total}, width_{width}, progress_{0},
          start_{std::chrono::steady_clock::now()} {
        std::cout << '[' << std::string(width_, ' ') << "] 0%";
        std::cout.flush();
    }

    void update(std::size_t progress = 1) {
        progress_ += progress;
        std::size_t percent{100 * progress_ / total_};
        std::cout << '\r' << '[' << std::string(percent * width_ / 100, '=')
                  << std::string(width_ - percent * width_ / 100, ' ') << "] "
                  << percent << '%';
        std::cout.flush();
    }

    ~ProgressBar() {
        std::cout << '\r' << '[' << std::string(width_, '=') << "] 100%" << std::endl;
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
        std::cout << "Elapsed time: " << duration.count() << " ms" << std::endl;
    }

private:
    std::size_t total_;
    std::size_t width_;
    std::size_t progress_;
    std::chrono::steady_clock::time_point start_;
};
