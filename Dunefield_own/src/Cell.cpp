#include "Cell.h"
#include "random_gen.h"

Cell::Cell(int h) : height(h)
    {}

void Cell::erode() 
    { height--; }

void Cell::stack() 
    { height++; }

bool Cell::deposited() {
    if (shadow || random.real() <= DEPOSITE_CHANCE_IN_LIGHT) {
        stack();
        return true;
    }
    else {
        return false;
    }
}

void Cell::jump(int counter) {
    counter--;
    if (counter == 0) {
        if (!deposited()) {
            forward->jump(JUMP_LENGTH);
        }
    }
    else {
        forward->jump(counter);
    }
}

void Cell::fix_cell() 
    { height = temp_height; }