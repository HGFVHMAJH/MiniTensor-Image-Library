#ifndef MINI_IMAGE_HPP
#define MINI_IMAGE_HPP

#include "Tensor.hpp"
#include <cstdint>
#include <string>
#include <vector>

// Image is stored in memory as an RGB tensor with shape {height, width, 3}.
// The BMP reader/writer currently supports uncompressed 24-bit BMP files.
class Image : public Tensor<std::uint8_t>
{
private:
#pragma pack(push, 1)
    struct BitmapFileHeader
    {
        std::uint16_t type;
        std::uint32_t size;
        std::uint16_t reserved1;
        std::uint16_t reserved2;
        std::uint32_t pixelOffset;
    };

    struct BitmapInfoHeader
    {
        std::uint32_t headerSize;
        std::int32_t width;
        std::int32_t height;
        std::uint16_t planes;
        std::uint16_t bitsPerPixel;
        std::uint32_t compression;
        std::uint32_t imageSize;
        std::int32_t xPixelsPerMeter;
        std::int32_t yPixelsPerMeter;
        std::uint32_t colorsUsed;
        std::uint32_t importantColors;
    };
#pragma pack(pop)

    struct LoadedBMP
    {
        BitmapFileHeader fileHeader{};
        BitmapInfoHeader infoHeader{};
        std::size_t width = 0;
        std::size_t height = 0;
        std::vector<std::uint8_t> rgbPixels;
    };

    BitmapFileHeader fileHeader_{};
    BitmapInfoHeader infoHeader_{};
    std::string sourceFilename_;

    explicit Image(LoadedBMP loaded, const std::string& filename);
    static LoadedBMP loadBMP(const std::string& filename);

    std::size_t pixelOffset(std::size_t y, std::size_t x, std::size_t channel) const;
    static std::uint8_t clampToByte(double value);

public:
    explicit Image(const std::string& bmpFilename);

    std::size_t width() const noexcept { return shape_[1]; }
    std::size_t height() const noexcept { return shape_[0]; }
    std::size_t channels() const noexcept { return shape_[2]; }
    const std::string& sourceFilename() const noexcept { return sourceFilename_; }

    // Filters modify this Image in place, matching the original CS50 helpers.
    void grayscale();
    void reflect();                 // Horizontal reflection.
    void flipVertical();            // Top-to-bottom reflection.
    void blur();                    // 3x3 box blur, matching the CS50 implementation.
    void edges();                    // Sobel edge detection, black outside image bounds.
    void sepia();
    void adjustBrightness(double factor);

    // Writes the current pixels to a 24-bit BMP file.
    void save(const std::string& outputFilename) const;
};

#endif
