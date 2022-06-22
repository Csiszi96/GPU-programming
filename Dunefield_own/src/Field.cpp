#include "Field.h"
#include <algorithm>
#include <iostream>

void print(std::string msg) {
    std::cout << msg << std::endl;
}

Field::Field(int w, int l, double p) : width(w), length(l), hopping_prec(p) {
    
    field = new std::vector<std::vector<Cell*>> (length, std::vector<Cell*>(width, new Cell(height)));
    for(const auto &c: (*field)) {
        field_vector.insert(field_vector.end(), c.begin(), c.end()); 
    }
    field_size = (int)field_vector.size();
    load_neigbours();
}

Field::Field(int w, int l) {
    Field(w, l, 0.1);
}

void Field::simulate_frame() {
    print("1");
    // calculate_shadows();
    print("2");
    // hopping();
    print("3");
    // landslides();
}

void Field::fix_cells() {
    for (Cell* c : field_vector)
        c->fix_cell();
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
            (*field)[x][y]->set_diags();
        }
    }
}

Cell* Field::warp_coordinates(int x, int y) {
    if (x >= length)
        x -= length;
    if (y >= width)
        y -= width;
    if (x < 0)
        x += length;
    if (y < 0)
        y += width;
    
    return (*field)[x][y];
}

void Field::hopping() {
    int n_hops = (int)std::ceil(field_size * hopping_prec);

    if (DISTINCT_HOP) {
        std::random_shuffle(rnd_field_vector.begin(), rnd_field_vector.end());
        for (int i = 0; i < n_hops; i++){
            // rnd_field_vector[i]->hop();
        }
    }

    else {
        for (;n_hops > 0; n_hops--) {
            int i = random.integer(0, field_size);
            field_vector[i]->hop();
        }
    }
}

void Field::calculate_shadows() {
    // NOTE: for_each
    for (std::vector<Cell*> row : (*field)) {
        row[0]->calculate_shadow(0, 2 * length);
    }
}

void Field::landslides() {
    // NOTE: for_each
    // std::for_each(field_vector.begin(), field_vector.end(), [](auto c){c->landslide(MOORE);} );

    for (auto c : field_vector) {
        c->landslide(MOORE);
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

std::vector<std::vector<bool>> Field::get_shadows(){
    std::vector<std::vector<bool>> ret(length, std::vector<bool>(width));
    for (int x = 0; x < length; x++) {
        for (int y = 0; y < width; y++) {
            ret[x][y] = (*field)[x][y]->get_shadow();
        }
    }

    return ret;
}

void Field::set_shadow_length(float x){
    // NOTE: for_each
    for (auto c : field_vector) {
        c->set_shadow_length(x);
    }
}

void Field::set_jump_length(int x) {
    // NOTE: for_each
    for (auto c : field_vector) {
        c->set_jump_length(x);
    }
}

void Field::set_landslide_delta(int x){
    // NOTE: for_each
    for (auto c : field_vector) {
        c->set_landslide_delta(x);
    }
}

void Field::print_heights() {
    auto heigths = get_heights();
    for (auto row : heigths) {
        for (auto c : row) {
            std::cout << c << " ";
        }
        std::cout << std::endl;
    }
}

void Field::print_shadows() {
    auto heigths = get_shadows();
    for (auto row : heigths) {
        for (auto c : row) {
            std::cout << c << " ";
        }
        std::cout << std::endl;
    }
}
