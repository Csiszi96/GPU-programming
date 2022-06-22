#include "Field.h"
#include <iostream>
// #include "python_binding.h"


// const float SHADOW_ANGLE = 15;

void test() {
    int WIDTH = 10;
    int LENGTH = 10;
    float HOPPING_PRECENT = 0.1f;

    Field field(WIDTH, LENGTH, HOPPING_PRECENT);

    field.simulate_frame();

    field.print_heights();
    std::cout << std::endl << std::endl;
    field.print_shadows();
}

int main () {
    test();
    return 0;
}