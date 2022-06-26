#include "Field.h"
#include <algorithm>
#include <iostream>

void print(std::string msg) {
    std::cout << msg << std::endl;
}

Field::Field(int w, int l, double p) : width(w), length(l), hopping_prec(p) {
    init(w, l, p);
}

Field::Field(int w, int l) {
    Field(w, l, 0.1);
}

Field::Field() {
    Field(10, 10);
}

void Field::init(int w, int l, double p) {
    if (bool_init) return;
    else std::cout << "Object is already initialized!" << std::endl;

    width = w;
    length = l;
    hopping_prec = p;
    
    field_vector = std::vector<Cell*> (length * width);
    for (int i = 0; i < field_vector.size(); i++) {
        if (i == 0 || i == 21) field_vector[i] = new Cell(20);
        else field_vector[i] = new Cell(height);
    }


    for (int y = 0; y < width; y ++) {
        std::vector<Cell*>::const_iterator first = field_vector.begin() + y * length;
        std::vector<Cell*>::const_iterator last = field_vector.begin() + (y + 1) * length;
        std::vector<Cell*> row(first, last);

        field.push_back(row);
        first_column.push_back(row[0]);
    }

    rnd_field_vector = field_vector;
    field_size = (int)field_vector.size();
    load_neigbours();

    // field_vector[0]->erode(); // NOTE: testing line
    std::cout << "pointer: " << this << std::endl;

    bool_init = true;
}

bool Field::initialized() {
    if (bool_init) 
        return true;
    else {
        std::cout << "Object not jet initialized! Run this.init(int width, int height, double hopping_precision!)" << std::endl;
        return false;
    }
}

void Field::simulate_frame() {
    hopping();
    landslides();
    calculate_shadows();
}

void Field::fix_cells() {
    // NOTE: for_each
    for (Cell* c : field_vector)
        c->fix_cell();
}

void Field::load_neigbours() {
    for (int y = 0; y < length; y++) {
        for (int x = 0; x < width; x++) {
            field[y][x]->set_forward(warp_coordinates(y, x + 1));
            field[y][x]->set_backward(warp_coordinates(y, x - 1));

            field[y][x]->set_sides(warp_coordinates(y + 1, x));
            field[y][x]->set_sides(warp_coordinates(y - 1, x));
            
            field[y][x]->set_front_diags(warp_coordinates(y + 1, x + 1));
            field[y][x]->set_front_diags(warp_coordinates(y - 1, x + 1));
            field[y][x]->set_back_diags(warp_coordinates(y + 1, x - 1));
            field[y][x]->set_back_diags(warp_coordinates(y - 1, x - 1));

            field[y][x]->set_neigbours();
            field[y][x]->set_diags();
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
    
    return field[x][y];
}

void Field::hopping() {
    int n_hops = (int)std::ceil(field_size * hopping_prec);

    if (DISTINCT_HOP) {
        std::random_shuffle(rnd_field_vector.begin(), rnd_field_vector.end());
        for (int i = 0; i < n_hops; i++){
            rnd_field_vector[i]->hop();
        }
    }

    else {
        for (;n_hops > 0; n_hops--) {
            int i = random.integer(0, field_size - 1);
            field_vector[i]->hop();
        }
    }
}

void Field::calculate_shadows() {
    // NOTE: for_each
    for (auto c : first_column) {
        c->calculate_shadow(0, 2 * length);
    }
}

void Field::landslides() {
    // NOTE: for_each
    // std::for_each(field_vector.begin(), field_vector.end(), [](auto c){c->landslide(MOORE);} );

    for (auto c : field_vector) {
        c->landslide(MOORE);
    }

    fix_cells();
}

std::vector<std::vector<int>> Field::get_heights(){
    std::vector<std::vector<int>> ret(length, std::vector<int>(width, 0));
    for (int x = 0; x < length; x++) {
        for (int y = 0; y < width; y++) {
            ret[x][y] = field[x][y]->get_height();
        }
    }

    return ret;
}

std::vector<std::vector<bool>> Field::get_shadows(){
    std::vector<std::vector<bool>> ret(length, std::vector<bool>(width));
    for (int x = 0; x < length; x++) {
        for (int y = 0; y < width; y++) {
            ret[x][y] = field[x][y]->get_shadow();
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

std::vector<int> Field::get_heights_arr() {
    std::cout << "pointer: " << this << std::endl;
    
    std::vector<int> ret;
    for (auto c : field_vector)
        ret.push_back(c->get_height());

    return ret;
}

std::vector<bool> Field::get_shadows_arr() {
    std::cout << "pointer: " << this << std::endl;
    std::vector<bool> ret;
    for (auto c : field_vector)
        ret.push_back(c->get_shadow());
    return ret;
}

int Field::get_width() {
    std::cout << "pointer: " << this << std::endl;
    return width;
}