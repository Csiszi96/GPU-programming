#ifndef python_bynding
#define python_bynding

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(cpp_arit, mod) {
    py::class_<Field>(mod, "Field")
        .def(py::init<>())
        .def("simulate_frame", &Field::simulate_frame)
        .def("get_heights", &Field::get_heights)
        .def("get_shadows", &Field::get_shadows);
}

#endif /* python_bynding */