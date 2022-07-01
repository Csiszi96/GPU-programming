#include <vector>
#include <iostream>

#include "gpu.h"
#include "field.h"
#include "python_binding.h"

void test() {
    GPU_Field f;
    std::cout << "01" << std::endl;

    int width = 10;
    int length = 10;

    f.init(width, length);

    f.simulate_frame();

    f.print_field();
    // f.print_shadows();
}


// int main() {
//     test();
// 	return 0;
// }
