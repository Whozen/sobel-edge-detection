#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <omp.h> //for openmp

using namespace std;

 
/*
Class Name: Pixel
Class Variables Datatype: unsigned integer
Class Description: This class represents the three colour values (R,G,B) in each pixel. It has four methods to set and get each color pixel in the image matrix
*/
class Pixel {
    private:
    unsigned int P1, P2, P3;

    public:
    Pixel () {};   
    void setPixels (unsigned int Pixel1, unsigned int Pixel2, unsigned int Pixel3);        
    unsigned int getPixel1 ();
    unsigned int getPixel2 ();
    unsigned int getPixel3 ();
};

//Set each color pixel value
void Pixel::setPixels (unsigned int Pixel1, unsigned int Pixel2, unsigned int Pixel3) {
    P1 = Pixel1;
    P2 = Pixel2;
    P3 = Pixel3;
}

//Get first color pixel value i.e, Red color pixel value
unsigned int Pixel::getPixel1 () {
    return P1;
}

//Get second color pixel value i.e, Green color pixel value
unsigned int Pixel::getPixel2 () {
    return P2;
}

//Get third color pixel value i.e, Blue color pixel value
unsigned int Pixel::getPixel3 () {
    return P3;
}



/*
Function Name: sobel_convolution
Function Parameter: Integers Total Rows, Total Columns, Grayscale Pixels and Output Pixels in the image
Function Return Type: void
Function Description: This function uses Sobel operation on the grayscale image using the masks Gx and Gy. The grayscale image is divided into different parts and each part is assigned to each processor. Then convolution operation is applied in each processor for Sobel operation.
*/
void sobel_convolution(int TotRows, int TotCol, Pixel **PixelVal, Pixel **OutputVal) {
    unsigned int valX, valY = 0;
    unsigned int GX [3][3];
    unsigned int GY [3][3];
    unsigned int val1, val2, val3;
    int i,j,x,y;

    //Get processor id and number of processors
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    
    //Calculate local row and columns
    int local_row = TotRows/thread_count;
    int local_col = TotCol/thread_count;

    //Calculate the first and last pixel value for the matrices in each processor
    int my_first = (my_rank*local_row);
    int my_last = my_first + local_row;

    //Sobel Horizontal Mask     
    GX[0][0] = 1; GX[0][1] = 0; GX[0][2] = -1; 
    GX[1][0] = 2; GX[1][1] = 0; GX[1][2] = -2;  
    GX[2][0] = 1; GX[2][1] = 0; GX[2][2] = -1;

    //Sobel Vertical Mask   
    GY[0][0] =  1; GY[0][1] = 2; GY[0][2] =  1;    
    GY[1][0] =  0; GY[1][1] = 0; GY[1][2] =  0;    
    GY[2][0] = -1; GY[2][1] =-2; GY[2][2] = -1;


    //SOBEL edge detector implementation. 
    //In each Pixel, the values of the 3 colours is the same.Therefore the calculation is performed on the first one only. The other 2 colours are then set to be = to the first one.
    #pragma omp parallel for private(j,x,y)
        for(i=my_first; i < my_last; i++) {
            for(j=0; j < TotCol; j++) {

                // setting the pixels around the border to 0, because the Sobel kernel cannot be allied to the border pixels
                if ((i == my_first)||(i == my_last-1)||(j == 0)||(j == TotCol-1)) {
                    valX = 0;
                    valY = 0;
                } else {
                    //calculating the X and Y convolutions
                    for (int x = -1; x <= 1; x++) {
                        valX = 0;
                        valY = 0;
                        for (int y = -1; y <= 1; y++) {
                            valX = valX + (PixelVal[i+x][j+y].getPixel1() * GX[1+x][1+y]);
                            valY = valY + (PixelVal[i+x][j+y].getPixel1() * GY[1+x][1+y]);
                        }
                    }
                }

                //Calculate the Gradient Magnitude
                val1 = sqrt((valX*valX) + (valY*valY));

                //The threshold value is set to 200. The image is binarized using the threshold value to obtain a binary image
                if( val1 > 200 ) {
                    val1 = 255;
                } else {
                    val1 = 0;
                }

                //Setting the new pixel value in Output matrix
                OutputVal[i][j].setPixels(val1, val1, val1);
            }
        }
}


/*
Function Name: convert_grayscale
Function Parameter: Integers Total Rows, Total Columns and Pixels in the image
Function Return Type: void
Function Description: This function converts colored PPM image into grayscale using parallel for. Grayscale value is average of the 3's color in a pixel.
*/
void convert_grayscale(int TotRows, int TotCol, Pixel **PixelVal) {
    unsigned int val1, val2, val3;
    int i,j;

    #pragma omp parallel for private(j)
        for(i=0; i < TotRows; i++) {
            for(j=0; j < TotCol; j++) {
                val1 = (PixelVal[i][j].getPixel1()+PixelVal[i][j].getPixel1()+PixelVal[i][j].getPixel1())/3;
                val2 = val1;
                val3 = val1;
                PixelVal[i][j].setPixels(val1, val2, val3);
            }
        }
}



//Declaring Global variables
Pixel **PixelVal;
Pixel **OutputVal;


int main () {
    //information contained in the header file is represented by the following variables

    unsigned char Magic [2];
    unsigned int TotRows = 5000;
    unsigned int TotCol = 5000;
    unsigned int MaxVal = 255;
    double start_time; 
    double end_time;
    int thread_count, i, j, T = 0;
    string img, output_img;

    int size = (3 * TotRows * TotCol);

    char *charImage = new char [size];
    char *greyImage = new char [size];

    //Get the name of the input image file from the users
    cout<<"Enter the name of the input PPM image (with extension)\n";
    cin>>img;

    //Get the name of the image file where output image will be created
    cout<<"Enter the name to be given to the output file (without extension)\n";
    cin>>output_img;

    //Get the total number of processors
    cout<<"Number of processors\n";
    cin>>thread_count;

    //Opening input image
    ifstream OldImage;
    OldImage.open (img, ios::in | ios::binary);

    if (!OldImage) {
       cout << "\nError: Cannot open image file! " << endl;
    }

    //Reading the header of the original image file
    OldImage >> Magic [0] >> Magic [1] >>  TotRows >>  TotCol >> MaxVal;
    //printf("Magic [0] = %c, Magic [1] =  %c, TotRows =  %d, TotCol =  %d, MaxVal = %d\n", Magic [0], Magic [1], TotRows, TotCol, MaxVal);

    OldImage.read(charImage, size);

    unsigned int val1, val2, val3;

    //Initializing array of pixels, which is used to represent the image
    PixelVal = new Pixel* [TotRows];
    OutputVal = new Pixel* [TotRows];


    //Reading the image data and setting the pixels values as unsigned integers
    for(i=0; i < TotRows; i++) {
        PixelVal[i] = new Pixel [TotCol];
        OutputVal[i] = new Pixel [TotCol];

        for(j=0; j < TotCol; j++) {
            val1 = (unsigned int)charImage[T];
            val2 = (unsigned int)charImage[T+1];
            val3 = (unsigned int)charImage[T+2];
            PixelVal[i][j].setPixels (val1, val2, val3);
            T=T+3;
        }
    }


    if (OldImage.fail()) {
        //cout << "Can't read image " << endl;
    }

    OldImage.close();

    //Get the start time of the process
    start_time = omp_get_wtime(); 

    //convert_grayscale function is called to convert input image into grayscale image using multiple processors
    #pragma omp parallel num_threads(thread_count)
    {
        convert_grayscale(TotRows,TotCol,PixelVal);
    }
    
    //sobel_convolution function is called to convert grayscale image into binary image using multiple processors.
    #pragma omp parallel num_threads(thread_count)
    {
        sobel_convolution(TotRows, TotCol, PixelVal, OutputVal);
    }


    //The output matrix is assigned to the output Pixel object matrix
    T=0;
    for(i=0; i < TotRows; i++) {
        for(j=0; j < TotCol; j++) {
            val1 = OutputVal[i][j].getPixel1();
            val2 = OutputVal[i][j].getPixel2();
            val3 = OutputVal[i][j].getPixel3();
            charImage[T]=(unsigned char)val1;
            charImage[T+1]=(unsigned char)val2;
            charImage[T+2]=(unsigned char)val3;
            T=T+3;
        }
    }

    //The grayscale matrix is assigned to the grayscale Pixel object matrix
    T=0;
    for(i=0; i < TotRows; i++) {
        for(j=0; j < TotCol; j++) {
            val1 = PixelVal[i][j].getPixel1();
            val2 = PixelVal[i][j].getPixel2();
            val3 = PixelVal[i][j].getPixel3();
            greyImage[T]=(unsigned char)val1;
            greyImage[T+1]=(unsigned char)val2;
            greyImage[T+2]=(unsigned char)val3;
            T=T+3;
        }
    }

    //Get the end time of the process
    end_time = omp_get_wtime(); 
    printf("\n\nUsing P = %d, Time taken = %lf sec", thread_count, end_time - start_time);


    //Creating two new files to host the grayscale image and the output image
    ofstream NewImage;
    ofstream GreyImage;

    NewImage.open (output_img + ".ppm", ios::out | ios::binary);
    GreyImage.open (output_img + "_grayscale.ppm", ios::out | ios::binary);

    if ((!NewImage) || (!GreyImage)) {
        cout << "\nError: Cannot open image file! " <<endl;
    }

    //Writing the header in the new image files
    NewImage << "P6" << endl << TotRows << " " << TotCol << " " << MaxVal << endl;
    GreyImage << "P6" << endl << TotRows << " " << TotCol << " " << MaxVal << endl;

    NewImage.write(charImage, size);
    GreyImage.write(greyImage, size);

    if (NewImage.fail() || GreyImage.fail()) {
        cout << "Can't write image " << endl;
    }


    //Closing the output files and deleting the variables
    NewImage.close();
    delete [] charImage;

    GreyImage.close(); 
    delete [] greyImage;

    for (int i = 0; i < TotRows; i++) {
        delete [] PixelVal[i];
        delete [] OutputVal[i];
    }

    delete [] PixelVal;
    delete [] OutputVal;

    return 0;
}