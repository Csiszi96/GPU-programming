#include "Field.h"


Field::Field(int w, int l, double p) : width(w), length(l), hopping_prec(p) {
    for(const auto &c: field) {
        field_vector.insert(field_vector.end(), c.begin(), c.end()); 
    }
    load_neigbours();
}

Field::Field(int w, int l) 
    { Field(w, l, 0.1); }

void Field::simulate_step() {
    calculate_sadows();
    hopping();
    fix_cells();

    calculate_gradients();
    landslide();
    fix_cells();
}

void Field::fix_cells() {
    for (Cell* c : field_vector) {
        c->fix_cell();
    }
}