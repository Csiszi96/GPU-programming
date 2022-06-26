#include "Field.h"
#include <iostream>
#include "python_binding.h"


// const float SHADOW_ANGLE = 15;

void test() {
    int WIDTH = 10;
    int LENGTH = 10;
    float HOPPING_PRECENT = 0.1f;

    Field field(WIDTH, LENGTH, HOPPING_PRECENT);

    field.simulate_frame();

    field.print_heights();
    std::cout << std::endl;
    field.print_shadows();
}

std::shared_ptr<Field> make_ptr(int x, int y) {
    // return new Field(x, y);
    return std::make_shared<Field>(x, y);
}

void destroy_ptr(std::shared_ptr<Field> &f) {
    // delete f;
    // f = NULL;
}

std::vector<int> get_h(std::shared_ptr<Field> &f) {
    return f->get_heights_arr();
}

// int main () {
//     test();
//     return 0;
// }