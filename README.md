# sobel-edge-detection

# Project Title
Sobel Edge Detection using C++ with OpenMP.

# Program Description
The program takes a PPM image as input and applies Sobel Edge Detection technique to it and generates a binary image with the edges of the image displayed. Also, it displays a grayscale image of the input image. Convolution operation is applied to it along with the Sobel masks as the edge detection technique. The image is divided into different parts and each part is assigned to the processors where the edge detection operation takes place.

# Method Used
OpenMP

# Input
The input of this program is entered by the users through the keyboard. The program asks users for input file name, output file name and number of processors to be used during runtime.

# Output
The program creates two new PPM image files that consists of grayscale image and binarized image that displays the edges of the image. It also displays the number of processors used and time taken for the process.
