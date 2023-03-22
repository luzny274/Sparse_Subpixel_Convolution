/*
 * Author: Vitezslav Luzny
 * Contact: luzny274@gmail.com
 * Description: This file imports the convolution function from "conv_func.cpp" to be used with specific datatypes: fl64, fl32, ui32 and ui16
*/


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <malloc.h>
#include <omp.h>
#include <math.h>

#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>

void initialize_conv(){

    printf("Sparse Subpixel Convolution initialized\n");
}
#include <chrono>
using namespace std::chrono;

#define indint int64_t

// include the convolution for a datatype fl64
#define my_decimal double
namespace conv_fl64{
    #include "convolution_function.cpp"
}

// include the convolution for a datatype fl32
#undef my_decimal
#define my_decimal float
namespace conv_fl32{
    #include "convolution_function.cpp"
}



// include the convolution for a datatype ui32
#undef my_decimal
#define my_decimal uint32_t
namespace conv_ui32{
    #include "convolution_function.cpp"
}


// include the convolution for a datatype ui16
#undef my_decimal
#define my_decimal uint16_t
namespace conv_ui16{
    #include "convolution_function.cpp"
}