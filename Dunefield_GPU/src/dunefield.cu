#include <vector>
#include <iostream>

#include "gpu.h"
#include "field.h"


int main() {
    
    Field f(64,64);

    std::vector<int> tmp = f.get_heights();

    for (auto x : tmp)
        std::cout << x << " ";

	return 0;
}
