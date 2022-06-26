#include "Cell.h"
#include "mersenne_twister.h"
#include <algorithm>
#include <iostream>
#include <limits>

template<typename T>
T max_val(T x) {
    return std::numeric_limits<T>::max();
}

Cell::Cell(int h) : height(h), temp_height(h)
    {}

bool Cell::erode() {
    if (height > 0){
        height--; 
        return true;
    }
    else{
        std::cout << std::endl << "ERROR: tried to erode cell with height: "<< height << std::endl;
        return false;
    }
}

void Cell::stack() { 
    if (height < max_val(height))
        height++; 
    else
        std::cout << std::endl << "ERROR: tried to stack cell with height: "<< height << std::endl;    
}

void Cell::stack_tmp() { 
    if (temp_height < max_val(temp_height))
        temp_height++; 
    else
        std::cout << std::endl << "ERROR: tried to stack cell with height: "<< temp_height << std::endl;    
}

bool Cell::deposited() {
    if (shadow || random.real() <= DEPOSITE_CHANCE_IN_LIGHT) {
        stack();
        return true;
    }
    else {
        return false;
    }
}

void Cell::jump() {
    Cell* tmp = forward;
    for (int i = 0; i < JUMP_LENGTH - 1; i++)
        tmp = tmp->forward;

    if (!tmp->deposited())
        tmp->jump();
}

void Cell::hop() {
    if (erode())
        jump();
}

void Cell::fix_cell() 
    { height = temp_height; }

int Cell::get_height() { 
    return height; 
}

bool Cell::get_shadow() {
    return shadow;
}

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
}

void Cell::set_diags() {
    diags.insert(diags.end(), back_diags.begin(), back_diags.end());
    diags.insert(diags.end(), front_diags.begin(), front_diags.end());
}

void Cell::landslide(bool moore) {
    // Newton style landslide (only neighbours)
    std::random_shuffle(neigbours.begin(), neigbours.end());
    for (auto c : neigbours) {
        if ( temp_height - c->get_height() >= LANDSLIDE_DELTA) {
            c->stack_tmp();
            temp_height--; // not erode() !!
        }
    }

    // If moore style landslide is used include diags
    if (moore) {
        std::random_shuffle(diags.begin(), diags.end());
        for (auto c : diags) {
            if ( temp_height - c->get_height() >= LANDSLIDE_DELTA) {
                c->stack_tmp();
                temp_height --; // not erode() !!
            }
        }
    }
}

void Cell::calculate_shadow(float init_shadow, int counter) {
    if (init_shadow > 0) shadow = true;
    else shadow = false;

    if (counter > 0) {
        // How much does the previous shadow over reach?
        float remainder = init_shadow - 1;
        if (remainder < 0) remainder = 0;

        // Get next shadow by calculating height difference and adding remainder
        float next_shadow = remainder + (height - forward->get_height()) * SHADOW_LENGTH;
        if (next_shadow < 0) next_shadow = 0;

        forward->calculate_shadow(next_shadow, counter -1);
    }
}

void Cell::set_shadow_length(float x){
    SHADOW_LENGTH = x;
}

void Cell::set_jump_length(int x) {
    JUMP_LENGTH = x;
}

void Cell::set_landslide_delta(int x){
    LANDSLIDE_DELTA = x;
}
