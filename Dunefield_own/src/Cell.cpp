#include "Cell.h"
#include "mersenne_twister.h"

Cell::Cell(int h) : height(h)
    {}

void Cell::erode() 
    { temp_height--; }

void Cell::stack() 
    { temp_height++; }

bool Cell::deposited() {
    if (shadow || random.real() <= DEPOSITE_CHANCE_IN_LIGHT) {
        stack();
        return true;
    }
    else {
        return false;
    }
}

void Cell::jump()
    { jump(JUMP_LENGTH); }

void Cell::jump(int counter) {
    counter--;
    if (counter == 0) {
        if (!deposited()) {
            forward->jump();
        }
    }
    else {
        forward->jump(counter);
    }
}

void Cell::fix_cell() 
    { height = temp_height; }

int Cell::get_height()
    { return height; }

void Cell::set_forward(Cell* f) 
    { forward = f; }

void Cell::set_backward(Cell* b) 
    { backward = b; }

void Cell::set_sides(Cell* n)
    { sides.push_back(n); }

void Cell::set_back_diags(Cell* d)
    { back_diags.push_back(d); }

void Cell::set_front_diags(Cell* d)
    { front_diags.push_back(d); }

void Cell::set_neigbours() {
    neigbours.push_back(forward);
    neigbours.push_back(backward);
    neigbours.insert(neigbours.end(), sides.begin(), sides.end());
    neigbours.insert(neigbours.end(), back_diags.begin(), back_diags.end());
    neigbours.insert(neigbours.end(), front_diags.begin(), front_diags.end());
}

void Cell::landslide() {
    for (auto c : neigbours) {
        if ( c->get_height() - temp_height >= LANDSLIDE_DELTA) {
            c->erode();
            stack();
        }
    }
}

void Cell::calculate_shadow(float init_shadow, int counter) {
    if (init_shadow > 0) shadow = true;
    float next_shadow = init_shadow - 1 + (height - forward->get_height()) * SHADOW_LENGTH;
    if (counter > 0) 
        forward->calculate_shadow(next_shadow, counter -1);
}