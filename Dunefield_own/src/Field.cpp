#include "Field.h"


Field::Field(int w, int l, double p) : width(w), length(l), hopping_prec(p) {
    
    field = new std::vector<std::vector<Cell*>> (length, std::vector<Cell*>(width, new Cell(height)));
    for(const auto &c: (*field)) {
        field_vector.insert(field_vector.end(), c.begin(), c.end()); 
    }
    field_size = (int)field_vector.size();
    load_neigbours();
}

Field::Field(int w, int l) 
    { Field(w, l, 0.1); }

void Field::simulate_frame() {
    calculate_shadows();
    hopping();
    fix_cells();

    landslides();
    fix_cells();
}

void Field::fix_cells() {
    for (Cell* c : field_vector) {
        c->fix_cell();
    }
}

void Field::load_neigbours() {
    for (int x = 0; x < length; x++) {
        for (int y = 0; y < width; y++) {
            (*field)[x][y]->set_forward(warp_coordinates(x + 1, y));
            (*field)[x][y]->set_backward(warp_coordinates(x - 1, y));

            (*field)[x][y]->set_sides(warp_coordinates(x, y + 1));
            (*field)[x][y]->set_sides(warp_coordinates(x, y - 1));
            
            (*field)[x][y]->set_front_diags(warp_coordinates(x + 1, y + 1));
            (*field)[x][y]->set_front_diags(warp_coordinates(x + 1, y - 1));
            (*field)[x][y]->set_back_diags(warp_coordinates(x - 1, y + 1));
            (*field)[x][y]->set_back_diags(warp_coordinates(x - 1, y - 1));

            (*field)[x][y]->set_neigbours();
        }
    }
}

Cell* Field::warp_coordinates(int x, int y) {
    if (x >= length)
        x -= length;
    if (y >= width)
        y -= width;
    
    return (*field)[x][y];
}


void Field::hopping() {
    int n = (int)std::ceil(field_size * hopping_prec);
    for (;n < 0; n--) {
        int i = random.integer(0, field_size);
        field_vector[i]->jump();
    }
}

void Field::calculate_shadows() {
    for (std::vector<Cell*> row : (*field)) {
        row[0]->calculate_shadow(0, 2 * length);
    }
}

void Field::landslides() {
    for (auto c : field_vector) {
        c->landslide();
    }
}

std::vector<std::vector<int>> Field::get_heights(){
    std::vector<std::vector<int>> ret(length, std::vector<int>(width, 0));
    for (int x = 0; x < length; x++) {
        for (int y = 0; y < width; y++) {
            ret[x][y] = (*field)[x][y]->get_height();
        }
    }

    return ret;
}