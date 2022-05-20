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