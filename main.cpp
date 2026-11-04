#include "Image.hpp"
#include <exception>
#include <iostream>

int main()
{
    try
    {
        // Each Image loads its own copy of the original BMP.
        Image grayscaleImage("yard.bmp");
        grayscaleImage.grayscale();
        grayscaleImage.save("yard_grayscale.bmp");

        Image reflectedImage("yard.bmp");
        reflectedImage.reflect();
        reflectedImage.save("yard_reflected.bmp");

        Image blurredImage("yard.bmp");
        blurredImage.blur();
        blurredImage.save("yard_blur.bmp");

        Image edgesImage("yard.bmp");
        edgesImage.edges();
        edgesImage.save("yard_edges.bmp");

        Image sepiaImage("yard.bmp");
        sepiaImage.sepia();
        sepiaImage.save("yard_sepia.bmp");

        std::cout << "Image filters completed successfully.\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
