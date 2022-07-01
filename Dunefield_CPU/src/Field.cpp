#include "Field.h"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <numeric>
#include <execution>

void print(std::string msg) {
    std::cout << msg << std::endl;
}

CPU_Field::CPU_Field(int w, int l, double p) : width(w), length(l), hopping_prec(p) {
    init(w, l, p);
}

CPU_Field::CPU_Field(int w, int l) {
    CPU_Field(w, l, 0.1);
}

CPU_Field::CPU_Field() {
    CPU_Field(10, 10);
}

void CPU_Field::init(int w, int l) {
    init(w,l,0.1);
}

void CPU_Field::init(int w, int l, double p) {
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

    std::cout << "pointer: " << this << std::endl;

    bool_init = true;

    auto tmp = get_heights_arr();
    no_blocks = std::accumulate(tmp.begin(), tmp.end(), decltype(tmp)::value_type(0));
}

bool CPU_Field::initialized() {
    if (bool_init) 
        return true;
    else {
        std::cout << "Object not jet initialized! Run this.init(int width, int height, double hopping_precision!)" << std::endl;
        return false;
    }
}

int CPU_Field::simulate_frame() {
    auto start = std::chrono::high_resolution_clock::now();

    hopping();
    landslides();
    calculate_shadows();

    auto stop = std::chrono::high_resolution_clock::now();
    return (int)std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
}

void CPU_Field::fix_cells() {
    std::for_each(
        // std::execution::par,
        std::begin(field_vector),
        std::end(field_vector),
        [](Cell* &c) {  c->fix_cell(); }
    );
}

void CPU_Field::load_neigbours() {
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

Cell* CPU_Field::warp_coordinates(int x, int y) {
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

void CPU_Field::hopping() {
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

void CPU_Field::calculate_shadows() {
    std::for_each(
        // std::execution::par,
        std::begin(first_column),
        std::end(first_column),
        [&](Cell* &c) {  c->calculate_shadow(0, 2 * length); }
    );
}

void CPU_Field::landslides() {
    std::for_each(
        // std::execution::par,
        std::begin(field_vector),
        std::end(field_vector),
        [&](Cell* &c) {  c->landslide(MOORE); }
    );

    fix_cells();
}

std::vector<std::vector<int>> CPU_Field::get_heights(){
    std::vector<std::vector<int>> ret(length, std::vector<int>(width, 0));
    for (int x = 0; x < length; x++) {
        for (int y = 0; y < width; y++) {
            ret[x][y] = field[x][y]->get_height();
        }
    }

    return ret;
}

std::vector<std::vector<bool>> CPU_Field::get_shadows(){
    std::vector<std::vector<bool>> ret(length, std::vector<bool>(width));
    for (int x = 0; x < length; x++) {
        for (int y = 0; y < width; y++) {
            ret[x][y] = field[x][y]->get_shadow();
        }
    }

    return ret;
}

void CPU_Field::set_shadow_length(float x){
    std::for_each(
        // std::execution::par,
        std::begin(field_vector),
        std::end(field_vector),
        [&x](Cell* &c) {  c->set_shadow_length(x); }
    );
}

void CPU_Field::set_jump_length(int x) {
    std::for_each(
        // std::execution::par,
        std::begin(field_vector),
        std::end(field_vector),
        [&x](Cell* &c) {  c->set_jump_length(x); }
    );
}

void CPU_Field::set_landslide_delta(int x){
    std::for_each(
        // std::execution::par,
        std::begin(field_vector),
        std::end(field_vector),
        [&x](Cell* &c) {  c->set_landslide_delta(x); }
    );
}

void CPU_Field::print_heights() {
    auto heigths = get_heights();
    for (auto row : heigths) {
        for (auto c : row) {
            std::cout << c << " ";
        }
        std::cout << std::endl;
    }
}

void CPU_Field::print_shadows() {
    auto heigths = get_shadows();
    for (auto row : heigths) {
        for (auto c : row) {
            std::cout << c << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<int> CPU_Field::get_heights_arr() {    
    std::vector<int> ret;
    for (auto c : field_vector)
        ret.push_back(c->get_height());

    return ret;
}

std::vector<bool> CPU_Field::get_shadows_arr() {
    std::cout << "pointer: " << this << std::endl;
    std::vector<bool> ret;
    for (auto c : field_vector)
        ret.push_back(c->get_shadow());
    return ret;
}

int CPU_Field::get_width() {
    std::cout << "pointer: " << this << std::endl;
    return width;
}

int CPU_Field::check_block_level() {
    auto tmp = get_heights_arr();
    int sum = std::accumulate(tmp.begin(), tmp.end(), decltype(tmp)::value_type(0));
    return sum - no_blocks;
}

