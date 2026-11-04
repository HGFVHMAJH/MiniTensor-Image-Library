
#include "../Image.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

volatile double benchmarkChecksum = 0.0;

// Measures the execution time of one function call.
template <typename Function>
double measureTime(Function function)
{
    auto start = Clock::now();

    function();

    auto end = Clock::now();

    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

// Returns the median execution time.
double calculateMedian(std::vector<double> times)
{
    std::sort(times.begin(), times.end());

    return times[times.size() / 2];
}

// Reads every pixel so the complete output is observable.
double calculateChecksum(const Image& image)
{
    double sum = 0.0;

    for (std::uint8_t value : image.data())
    {
        sum += value;
    }

    return sum;
}

// Benchmarks one image filter.
void benchmarkFilter(
    std::ofstream& csv,
    const std::string& operationName,
    const Image& original,
    const std::function<void(Image&)>& operation)
{
    constexpr int repetitions = 15;

    std::vector<double> times;
    times.reserve(repetitions);

    // Warm-up run.
    {
        Image working = original;
        operation(working);
        benchmarkChecksum += calculateChecksum(working);
    }

    for (int i = 0; i < repetitions; ++i)
    {
        // Copying is deliberately outside the timed section.
        Image working = original;

        double elapsed = measureTime([&]()
        {
            operation(working);
        });

        times.push_back(elapsed);

        // Consume the complete result after timing.
        benchmarkChecksum += calculateChecksum(working);
    }

    double medianTime = calculateMedian(times);

    csv << operationName << ','
        << original.width() << ','
        << original.height() << ','
        << repetitions << ','
        << medianTime << '\n';

    std::cout << operationName
              << " | Median: " << medianTime
              << " ms\n";
}

int main()
{
    const std::string inputFile = "yard.bmp";
    const std::string outputFile =
        "benchmarks/results/temp_output.bmp";

    constexpr int repetitions = 15;

    std::ofstream csv(
        "benchmarks/results/image_cpp.csv"
    );

    if (!csv)
    {
        std::cerr << "Could not create benchmark CSV file.\n";
        return 1;
    }

    csv << "operation,width,height,repetitions,median_ms\n";

    try
    {
        // Load the original image once for filter benchmarks.
        Image original(inputFile);

        std::cout << "Image loaded successfully.\n";
        std::cout << "Dimensions: "
                  << original.width() << " x "
                  << original.height() << '\n';

        // ------------------------------------
        // 1. BMP LOADING
        // ------------------------------------

        std::vector<double> loadingTimes;
        std::unique_ptr<Image> loaded;

        // Warm-up.
        {
            Image warmup(inputFile);
            benchmarkChecksum += calculateChecksum(warmup);
        }

        for (int i = 0; i < repetitions; ++i)
        {
            loaded.reset();

            double elapsed = measureTime([&]()
            {
                loaded = std::make_unique<Image>(inputFile);
            });

            loadingTimes.push_back(elapsed);

            // Checksum outside the timed section.
            benchmarkChecksum += calculateChecksum(*loaded);
        }

        double loadingMedian = calculateMedian(loadingTimes);

        csv << "loading,"
            << original.width() << ','
            << original.height() << ','
            << repetitions << ','
            << loadingMedian << '\n';

        std::cout << "loading"
                  << " | Median: " << loadingMedian
                  << " ms\n";

        // ------------------------------------
        // 2. GRAYSCALE
        // ------------------------------------

        benchmarkFilter(
            csv,
            "grayscale",
            original,
            [](Image& image)
            {
                image.grayscale();
            }
        );

        // ------------------------------------
        // 3. HORIZONTAL REFLECTION
        // ------------------------------------

        benchmarkFilter(
            csv,
            "horizontal_reflection",
            original,
            [](Image& image)
            {
                image.reflect();
            }
        );

        // ------------------------------------
        // 4. VERTICAL FLIP
        // ------------------------------------

        benchmarkFilter(
            csv,
            "vertical_flip",
            original,
            [](Image& image)
            {
                image.flipVertical();
            }
        );

        // ------------------------------------
        // 5. BLUR
        // ------------------------------------

        benchmarkFilter(
            csv,
            "blur",
            original,
            [](Image& image)
            {
                image.blur();
            }
        );

        // ------------------------------------
        // 6. EDGE DETECTION
        // ------------------------------------

        benchmarkFilter(
            csv,
            "edges",
            original,
            [](Image& image)
            {
                image.edges();
            }
        );

        // ------------------------------------
        // 7. SEPIA
        // ------------------------------------

        benchmarkFilter(
            csv,
            "sepia",
            original,
            [](Image& image)
            {
                image.sepia();
            }
        );

        // ------------------------------------
        // 8. BRIGHTNESS ADJUSTMENT
        // ------------------------------------

        benchmarkFilter(
            csv,
            "brightness",
            original,
            [](Image& image)
            {
                image.adjustBrightness(1.2);
            }
        );

        // ------------------------------------
        // 9. BMP SAVING
        // ------------------------------------

        std::vector<double> savingTimes;

        // Warm-up.
        original.save(outputFile);

        for (int i = 0; i < repetitions; ++i)
        {
            double elapsed = measureTime([&]()
            {
                original.save(outputFile);
            });

            savingTimes.push_back(elapsed);
        }

        double savingMedian = calculateMedian(savingTimes);

        csv << "saving,"
            << original.width() << ','
            << original.height() << ','
            << repetitions << ','
            << savingMedian << '\n';

        std::cout << "saving"
                  << " | Median: " << savingMedian
                  << " ms\n";

        std::cout << "\nC++ image benchmark completed.\n";
        std::cout << "Checksum: " << benchmarkChecksum << '\n';
        std::cout << "Results saved to: "
                  << "benchmarks/results/image_cpp.csv\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << "Benchmark error: "
                  << error.what() << '\n';

        return 1;
    }

    return 0;
}
