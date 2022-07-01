#ifndef python_binding
#define python_binding

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

namespace py = pybind11;

std::shared_ptr<CPU_Field> make_ptr(int x, int y);
void destroy_ptr(std::shared_ptr<CPU_Field> &f);
std::vector<int> get_h(std::shared_ptr<CPU_Field> &f);

PYBIND11_MODULE(cpu_dunefield, mod) {
    py::add_ostream_redirect(mod, "ostream_redirect");
    py::class_<CPU_Field, std::shared_ptr<CPU_Field>>(mod, "CPU_Field")
    // py::class_<CPU_Field>(mod, "CPU_Field")
        .def(py::init<int&, int&>(), py::arg("width"), py::arg("length"))
        .def(py::init<>())
        .def("simulate_frame", &CPU_Field::simulate_frame)
        .def("get_heights", &CPU_Field::get_heights_arr)
        .def("get_shadows", &CPU_Field::get_shadows_arr)
        .def("width", &CPU_Field::get_width)
        .def( "initialize", py::overload_cast<int, int>(&CPU_Field::init))
        .def("check_block_level", &CPU_Field::check_block_level);
    mod.def("make_ptr", &make_ptr);
    mod.def("delete_ptr", &destroy_ptr);
    mod.def("heights", &get_h);
}

#endif /* python_binding */