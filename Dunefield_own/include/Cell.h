#include <vector>

class Cell {
    private:
        int height;
        bool shadow;

        Cell* forward;
        Cell* backward;
        std::vector<Cell*> neigbours;
        std::vector<Cell*> diags;

    public:
        Cell(int height);
        void erode();
        void stack();
        bool deposited();

        // Probably needs to be rewritten for GPU
        void jump(int counter);
};