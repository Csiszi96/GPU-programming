#ifndef python_binding
#define python_binding

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

namespace py = pybind11;

std::shared_ptr<Field> make_ptr(int x, int y);
void destroy_ptr(std::shared_ptr<Field> &f);
std::vector<int> get_h(std::shared_ptr<Field> &f);

PYBIND11_MODULE(dunefield, mod) {
    py::add_ostream_redirect(mod, "ostream_redirect");
    py::class_<Field, std::shared_ptr<Field>>(mod, "Field")
    // py::class_<Field>(mod, "Field")
        .def(py::init<int&, int&>(), py::arg("width"), py::arg("length"))
        .def(py::init<>(), py::arg("width"), py::arg("length"))
        .def("simulate_frame", &Field::simulate_frame)
        .def("get_heights", &Field::get_heights_arr)
        .def("get_shadows", &Field::get_shadows_arr)
        .def("width", &Field::get_width)
        .def("init", &Field::init);
    mod.def("make_ptr", &make_ptr);
    mod.def("delete_ptr", &destroy_ptr);
    mod.def("heights", &get_h);
}

#endif /* python_binding */