#include "Image.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

Image::LoadedBMP Image::loadBMP(const std::string& filename)
{
    static_assert(sizeof(BitmapFileHeader) == 14, "BMP file header must be 14 bytes");
    static_assert(sizeof(BitmapInfoHeader) == 40, "BMP info header must be 40 bytes");
    std::ifstream input(filename, std::ios::binary);
    if (!input)
        throw std::runtime_error("Could not open BMP file: " + filename);

    LoadedBMP loaded;
    input.read(reinterpret_cast<char*>(&loaded.fileHeader), sizeof(loaded.fileHeader));
    input.read(reinterpret_cast<char*>(&loaded.infoHeader), sizeof(loaded.infoHeader));
    if (!input)
        throw std::runtime_error("BMP file is too small or has an incomplete header: " + filename);

    const auto& fh = loaded.fileHeader;
    const auto& ih = loaded.infoHeader;

    // Deliberately support the same simple format as the supplied CS50 program.
    if (fh.type != 0x4D42 || fh.pixelOffset != 54 || ih.headerSize != 40 ||
        ih.planes != 1 || ih.bitsPerPixel != 24 || ih.compression != 0)
    {
        throw std::runtime_error("Unsupported BMP: expected a standard uncompressed 24-bit BMP with a 54-byte header");
    }
    if (ih.width <= 0 || ih.height == 0 || ih.height == std::numeric_limits<std::int32_t>::min())
        throw std::runtime_error("Invalid BMP dimensions");

    loaded.width = static_cast<std::size_t>(ih.width);
    loaded.height = static_cast<std::size_t>(ih.height < 0 ? -ih.height : ih.height);

    if (loaded.width > std::numeric_limits<std::size_t>::max() / loaded.height / 3)
        throw std::runtime_error("BMP dimensions are too large");

    const std::size_t rowBytes = loaded.width * 3;
    const std::size_t rowStride = (rowBytes + 3) & ~std::size_t(3);
    std::vector<std::uint8_t> row(rowStride);
    loaded.rgbPixels.resize(loaded.height * loaded.width * 3);

    input.seekg(static_cast<std::streamoff>(fh.pixelOffset), std::ios::beg);
    if (!input)
        throw std::runtime_error("Could not seek to BMP pixel data");

    for (std::size_t fileY = 0; fileY < loaded.height; ++fileY)
    {
        input.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(rowStride));
        if (!input)
            throw std::runtime_error("Unexpected end of BMP pixel data");

        // Positive BMP height means bottom-up storage; negative means top-down.
        const std::size_t y = ih.height > 0 ? loaded.height - 1 - fileY : fileY;
        for (std::size_t x = 0; x < loaded.width; ++x)
        {
            const std::size_t src = x * 3; // BMP channel order: B, G, R.
            const std::size_t dst = (y * loaded.width + x) * 3;
            loaded.rgbPixels[dst] = row[src + 2];     // R
            loaded.rgbPixels[dst + 1] = row[src + 1]; // G
            loaded.rgbPixels[dst + 2] = row[src];     // B
        }
    }

    return loaded;
}

Image::Image(LoadedBMP loaded, const std::string& filename)
    : Tensor<std::uint8_t>(
          std::vector<std::size_t>{loaded.height, loaded.width, 3},
          loaded.rgbPixels),
      fileHeader_(loaded.fileHeader),
      infoHeader_(loaded.infoHeader),
      sourceFilename_(filename)
{
}

Image::Image(const std::string& bmpFilename)
    : Image(loadBMP(bmpFilename), bmpFilename)
{
}

std::size_t Image::pixelOffset(std::size_t y, std::size_t x, std::size_t channel) const
{
    if (y >= height() || x >= width() || channel >= 3)
        throw std::out_of_range("Image pixel coordinate or channel is out of range");
    return (y * width() + x) * 3 + channel;
}

std::uint8_t Image::clampToByte(double value)
{
    if (value < 0.0) return 0;
    if (value > 255.0) return 255;
    return static_cast<std::uint8_t>(std::lround(value));
}

void Image::grayscale()
{
    for (std::size_t y = 0; y < height(); ++y)
    {
        for (std::size_t x = 0; x < width(); ++x)
        {
            const std::size_t p = pixelOffset(y, x, 0);
            const auto average = static_cast<std::uint8_t>(std::lround(
                (static_cast<double>(data_[p]) + data_[p + 1] + data_[p + 2]) / 3.0));
            data_[p] = average;
            data_[p + 1] = average;
            data_[p + 2] = average;
        }
    }
}

void Image::reflect()
{
    for (std::size_t y = 0; y < height(); ++y)
    {
        for (std::size_t x = 0; x < width() / 2; ++x)
        {
            const std::size_t left = pixelOffset(y, x, 0);
            const std::size_t right = pixelOffset(y, width() - 1 - x, 0);
            for (std::size_t channel = 0; channel < 3; ++channel)
                std::swap(data_[left + channel], data_[right + channel]);
        }
    }
}

void Image::flipVertical()
{
    for (std::size_t y = 0; y < height() / 2; ++y)
    {
        const std::size_t oppositeY = height() - 1 - y;
        for (std::size_t x = 0; x < width(); ++x)
        {
            const std::size_t top = pixelOffset(y, x, 0);
            const std::size_t bottom = pixelOffset(oppositeY, x, 0);
            for (std::size_t channel = 0; channel < 3; ++channel)
                std::swap(data_[top + channel], data_[bottom + channel]);
        }
    }
}

void Image::blur()
{
    const std::vector<std::uint8_t> original = data_;

    for (std::size_t y = 0; y < height(); ++y)
    {
        for (std::size_t x = 0; x < width(); ++x)
        {
            unsigned int sums[3] = {0, 0, 0};
            unsigned int count = 0;

            const std::size_t yStart = y == 0 ? 0 : y - 1;
            const std::size_t yEnd = std::min(height() - 1, y + 1);
            const std::size_t xStart = x == 0 ? 0 : x - 1;
            const std::size_t xEnd = std::min(width() - 1, x + 1);

            for (std::size_t ny = yStart; ny <= yEnd; ++ny)
            {
                for (std::size_t nx = xStart; nx <= xEnd; ++nx)
                {
                    const std::size_t p = (ny * width() + nx) * 3;
                    for (std::size_t c = 0; c < 3; ++c)
                        sums[c] += original[p + c];
                    ++count;
                }
            }

            const std::size_t out = pixelOffset(y, x, 0);
            for (std::size_t c = 0; c < 3; ++c)
                data_[out + c] = static_cast<std::uint8_t>(std::lround(
                    static_cast<double>(sums[c]) / count));
        }
    }
}

void Image::edges()
{
    const std::vector<std::uint8_t> original = data_;
    const int gxKernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int gyKernel[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (std::size_t y = 0; y < height(); ++y)
    {
        for (std::size_t x = 0; x < width(); ++x)
        {
            int gx[3] = {0, 0, 0};
            int gy[3] = {0, 0, 0};

            for (int ky = 0; ky < 3; ++ky)
            {
                for (int kx = 0; kx < 3; ++kx)
                {
                    const long long ny = static_cast<long long>(y) + ky - 1;
                    const long long nx = static_cast<long long>(x) + kx - 1;

                    // The original CS50 implementation treats out-of-image pixels as black.
                    if (ny < 0 || nx < 0 || ny >= static_cast<long long>(height()) ||
                        nx >= static_cast<long long>(width()))
                        continue;

                    const std::size_t p = (static_cast<std::size_t>(ny) * width() +
                                           static_cast<std::size_t>(nx)) * 3;
                    for (int c = 0; c < 3; ++c)
                    {
                        gx[c] += static_cast<int>(original[p + c]) * gxKernel[ky][kx];
                        gy[c] += static_cast<int>(original[p + c]) * gyKernel[ky][kx];
                    }
                }
            }

            const std::size_t out = pixelOffset(y, x, 0);
            for (int c = 0; c < 3; ++c)
                data_[out + c] = clampToByte(std::sqrt(
                    static_cast<double>(gx[c]) * gx[c] + static_cast<double>(gy[c]) * gy[c]));
        }
    }
}

void Image::sepia()
{
    for (std::size_t y = 0; y < height(); ++y)
    {
        for (std::size_t x = 0; x < width(); ++x)
        {
            const std::size_t p = pixelOffset(y, x, 0);
            const double red = data_[p];
            const double green = data_[p + 1];
            const double blue = data_[p + 2];

            const double sepiaRed = 0.393 * red + 0.769 * green + 0.189 * blue;
            const double sepiaGreen = 0.349 * red + 0.686 * green + 0.168 * blue;
            const double sepiaBlue = 0.272 * red + 0.534 * green + 0.131 * blue;

            data_[p] = clampToByte(sepiaRed);
            data_[p + 1] = clampToByte(sepiaGreen);
            data_[p + 2] = clampToByte(sepiaBlue);
        }
    }
}

void Image::adjustBrightness(double factor)
{
    if (!std::isfinite(factor) || factor < 0.0)
        throw std::invalid_argument("Brightness factor must be finite and non-negative");

    for (std::uint8_t& component : data_)
        component = clampToByte(static_cast<double>(component) * factor);
}

void Image::save(const std::string& outputFilename) const
{
    std::ofstream output(outputFilename, std::ios::binary);
    if (!output)
        throw std::runtime_error("Could not create BMP file: " + outputFilename);

    BitmapFileHeader fileHeader = fileHeader_;
    BitmapInfoHeader infoHeader = infoHeader_;

    const std::size_t rowBytes = width() * 3;
    const std::size_t rowStride = (rowBytes + 3) & ~std::size_t(3);
    const std::size_t imageBytes = rowStride * height();
    if (imageBytes > std::numeric_limits<std::uint32_t>::max() - fileHeader.pixelOffset)
        throw std::runtime_error("Image is too large to save as this BMP format");

    infoHeader.imageSize = static_cast<std::uint32_t>(imageBytes);
    fileHeader.size = static_cast<std::uint32_t>(fileHeader.pixelOffset + imageBytes);

    output.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
    output.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));
    if (!output)
        throw std::runtime_error("Failed while writing BMP headers");

    std::vector<std::uint8_t> row(rowStride, 0);
    for (std::size_t fileY = 0; fileY < height(); ++fileY)
    {
        const std::size_t y = infoHeader.height > 0 ? height() - 1 - fileY : fileY;
        for (std::size_t x = 0; x < width(); ++x)
        {
            const std::size_t src = pixelOffset(y, x, 0);
            const std::size_t dst = x * 3;
            row[dst] = data_[src + 2];     // BMP B
            row[dst + 1] = data_[src + 1]; // BMP G
            row[dst + 2] = data_[src];     // BMP R
        }
        // Padding bytes remain zero.
        output.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size()));
        if (!output)
            throw std::runtime_error("Failed while writing BMP pixel data");
    }
}
