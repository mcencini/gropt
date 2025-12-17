#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <vector>
#include <memory>
#include <cstdlib>

extern "C" {
#include "_gropt_binding.h"
}

namespace py = pybind11;

constexpr int N_DDEBUG = 100;

// Helper to build py::array_t with shape
template <typename T>
py::array_t<T> make_array(const std::vector<T>& data, std::vector<py::ssize_t> shape) {
    return py::array_t<T>(py::array::ShapeContainer(shape.begin(), shape.end()), data.data());
}

// run_kernel_diff_fixeddt wrapper
std::pair<py::array_t<double>, py::array_t<double>> py_run_kernel_diff_fixeddt(
    double dt0, double gmax, double smax, double TE, const py::array_t<double>& moments_params,
    double PNS_thresh, double T_readout, double T_90, double T_180, int diffmode, double dt_out,
    const py::array_t<double>& eddy_params, double search_bval, double slew_reg, int Naxis, int verbose = 0)
{
    int N_moments = static_cast<int>(moments_params.size());
    int N_eddy = static_cast<int>(eddy_params.size());
    std::vector<double> ddebug(N_DDEBUG, 0.0);

    double* G_out_ptr = nullptr;
    int N_out = 0;
    double* ddebug_ptr = ddebug.data();

    run_kernel_diff_fixeddt(&G_out_ptr, &N_out, &ddebug_ptr, verbose, dt0, gmax, smax, TE,
                            N_moments, moments_params.data(), PNS_thresh, T_readout, T_90,
                            T_180, diffmode, dt_out, N_eddy, eddy_params.data(), search_bval, slew_reg, Naxis);

    std::unique_ptr<double, decltype(&free)> guard(G_out_ptr, &free);
    std::vector<double> G_out(G_out_ptr, G_out_ptr + N_out);

    py::array_t<double> G_return(
        py::array::ShapeContainer{
            static_cast<py::ssize_t>(Naxis),
            static_cast<py::ssize_t>(G_out.size() / Naxis)},
        G_out.data());

    py::array_t<double> debug_out(
        py::array::ShapeContainer{static_cast<py::ssize_t>(ddebug.size())},
        ddebug.data());

    return {G_return, debug_out};
}

// run_kernel_diff_fixedN wrapper
std::pair<py::array_t<double>, py::array_t<double>> py_run_kernel_diff_fixedN(
    int N0, double gmax, double smax, double TE, const py::array_t<double>& moments_params,
    double PNS_thresh, double T_readout, double T_90, double T_180, int diffmode, double dt_out,
    const py::array_t<double>& eddy_params, double search_bval, double slew_reg, int verbose = 0)
{
    int N_moments = static_cast<int>(moments_params.size());
    int N_eddy = static_cast<int>(eddy_params.size());
    std::vector<double> ddebug(N_DDEBUG, 0.0);

    double* G_out_ptr = nullptr;
    int N_out = 0;
    double* ddebug_ptr = ddebug.data();

    run_kernel_diff_fixedN(&G_out_ptr, &N_out, &ddebug_ptr, verbose, N0, gmax, smax, TE,
                           N_moments, moments_params.data(), PNS_thresh, T_readout, T_90,
                           T_180, diffmode, dt_out, N_eddy, eddy_params.data(), search_bval, slew_reg);

    std::unique_ptr<double, decltype(&free)> guard(G_out_ptr, &free);
    std::vector<double> G_out(G_out_ptr, G_out_ptr + N_out);

    py::array_t<double> G_return(
        py::array::ShapeContainer{static_cast<py::ssize_t>(N_out)},
        G_out.data());

    py::array_t<double> debug_out(
        py::array::ShapeContainer{static_cast<py::ssize_t>(ddebug.size())},
        ddebug.data());

    return {G_return, debug_out};
}

// run_kernel_diff_fixedN_Gin wrapper
std::pair<py::array_t<double>, py::array_t<double>> py_run_kernel_diff_fixedN_Gin(
    const py::array_t<double>& G_in, int N0, double gmax, double smax, double TE,
    const py::array_t<double>& moments_params, double PNS_thresh, double T_readout, double T_90,
    double T_180, int diffmode, double dt_out, const py::array_t<double>& eddy_params,
    double search_bval, double slew_reg, int verbose = 0)
{
    int N_moments = static_cast<int>(moments_params.size());
    int N_eddy = static_cast<int>(eddy_params.size());
    std::vector<double> ddebug(N_DDEBUG, 0.0);

    double* G_out_ptr = nullptr;
    int N_out = 0;
    double* ddebug_ptr = ddebug.data();

    run_kernel_diff_fixedN_Gin(&G_out_ptr, &N_out, &ddebug_ptr, verbose, G_in.data(), N0, gmax, smax, TE,
                               N_moments, moments_params.data(), PNS_thresh, T_readout, T_90,
                               T_180, diffmode, dt_out, N_eddy, eddy_params.data(), search_bval, slew_reg);

    std::unique_ptr<double, decltype(&free)> guard(G_out_ptr, &free);
    std::vector<double> G_out(G_out_ptr, G_out_ptr + N_out);

    py::array_t<double> G_return(
        py::array::ShapeContainer{static_cast<py::ssize_t>(N_out)},
        G_out.data());

    py::array_t<double> debug_out(
        py::array::ShapeContainer{static_cast<py::ssize_t>(ddebug.size())},
        ddebug.data());

    return {G_return, debug_out};
}

// run_kernel_diff_fixeddt_fixG wrapper
std::pair<py::array_t<double>, py::array_t<double>> py_run_kernel_diff_fixeddt_fixG(
    double dt0, double gmax, double smax, double TE, const py::array_t<double>& moments_params,
    double PNS_thresh, double T_readout, double T_90, double T_180, int diffmode, double dt_out,
    const py::array_t<double>& eddy_params, double search_bval, int N_gfix,
    const py::array_t<double>& gfix, double slew_reg, int verbose = 0)
{
    int N_moments = static_cast<int>(moments_params.size());
    int N_eddy = static_cast<int>(eddy_params.size());
    std::vector<double> ddebug(N_DDEBUG, 0.0);

    double* G_out_ptr = nullptr;
    int N_out = 0;
    double* ddebug_ptr = ddebug.data();

    run_kernel_diff_fixeddt_fixG(&G_out_ptr, &N_out, &ddebug_ptr, verbose, dt0, gmax, smax, TE,
                                 N_moments, moments_params.data(), PNS_thresh, T_readout, T_90,
                                 T_180, diffmode, dt_out, N_eddy, eddy_params.data(), search_bval,
                                 N_gfix, gfix.data(), slew_reg);

    std::unique_ptr<double, decltype(&free)> guard(G_out_ptr, &free);
    std::vector<double> G_out(G_out_ptr, G_out_ptr + N_out);

    py::array_t<double> G_return(
        py::array::ShapeContainer{static_cast<py::ssize_t>(N_out)},
        G_out.data());

    py::array_t<double> debug_out(
        py::array::ShapeContainer{static_cast<py::ssize_t>(ddebug.size())},
        ddebug.data());

    return {G_return, debug_out};
}

PYBIND11_MODULE(_gropt_binding, m) {
    m.doc() = "Pybind11 wrapper for GrOpt kernel (bindings)";

    m.def("run_kernel_diff_fixeddt", &py_run_kernel_diff_fixeddt, "Run kernel with fixed dt");
    m.def("run_kernel_diff_fixedN", &py_run_kernel_diff_fixedN, "Run kernel with fixed N");
    m.def("run_kernel_diff_fixedN_Gin", &py_run_kernel_diff_fixedN_Gin, "Run kernel with fixed N and G_in");
    m.def("run_kernel_diff_fixeddt_fixG", &py_run_kernel_diff_fixeddt_fixG, "Run kernel with fixed dt and fixG");
}