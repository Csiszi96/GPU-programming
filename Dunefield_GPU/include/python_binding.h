#ifndef python_bynding
#define python_bynding

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

namespace py = pybind11;

PYBIND11_MODULE(gpu_dunefield, mod) {
    mod.doc() = "A simulation based on Werner's sandune model, written in CUDA.";
    
    py::add_ostream_redirect(mod, "ostream_redirect");
    py::class_<GPU_Field, std::shared_ptr<GPU_Field>>(mod, "GPU_Field")
        .def(py::init<>())
        .def("initialize", py::overload_cast<int, int, int, float, float, int, float, int>(&GPU_Field::init))
        .def("initialize", py::overload_cast<int, int>(&GPU_Field::init))
        .def("simulate_frame", &GPU_Field::simulate_frame)
        .def("get_heights", &GPU_Field::get_heights)
        .def("get_shadows", &GPU_Field::get_shadows)
        .def("check_block_level", &GPU_Field::check_block_level);
}

#endif /* python_bynding */