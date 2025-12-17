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

struct BufInfo {
    std::size_t rows;
    const double* ptr;
};

// Compute rows similar to original Cython:
// - If ndim >= 2: use shape[0].
// - If ndim == 1:
//     * if size % stride_fallback == 0, use size/stride_fallback
//     * else, if size > 0, use 1
// - If size == 0, rows = 0
inline BufInfo get_buf_info(const py::array_t<double>& arr, std::size_t stride_fallback) {
    auto info = arr.request();
    std::size_t rows = 0;
    if (info.ndim >= 2) {
        rows = static_cast<std::size_t>(info.shape[0]);
    } else if (info.ndim == 1) {
        if (info.size == 0) {
            rows = 0;
        } else if (stride_fallback > 0 && info.size % stride_fallback == 0) {
            rows = static_cast<std::size_t>(info.size / stride_fallback);
        } else {
            rows = 1; // flattened single row (e.g., length 7 for moments)
        }
    } else { // ndim == 0
        rows = (info.size > 0) ? 1 : 0;
    }
    return {rows, static_cast<const double*>(info.ptr)};
}

// run_kernel_diff_fixeddt wrapper
std::pair<py::array_t<double>, py::array_t<double>> py_run_kernel_diff_fixeddt(
    double dt0, double gmax, double smax, double TE, const py::array_t<double>& moments_params,
    double PNS_thresh, double T_readout, double T_90, double T_180, int diffmode, double dt_out,
    const py::array_t<double>& eddy_params, double search_bval, double slew_reg, int Naxis, int verbose = 0)
{
    auto mp = get_buf_info(moments_params, 6);
    auto ep = get_buf_info(eddy_params, 4);

    double* G_out_ptr = nullptr;
    int N_out = 0;
    double* ddebug_ptr = nullptr;

    run_kernel_diff_fixeddt(&G_out_ptr, &N_out, &ddebug_ptr, verbose, dt0, gmax, smax, TE,
                            static_cast<int>(mp.rows), mp.ptr, PNS_thresh, T_readout, T_90,
                            T_180, diffmode, dt_out, static_cast<int>(ep.rows), ep.ptr,
                            search_bval, slew_reg, Naxis);

    std::unique_ptr<double, decltype(&free)> guard_G(G_out_ptr, &free);
    std::unique_ptr<double, decltype(&free)> guard_D(ddebug_ptr, &free);

    std::vector<double> G_out(G_out_ptr, G_out_ptr + N_out);
    std::vector<double> ddebug;
    if (ddebug_ptr) ddebug.assign(ddebug_ptr, ddebug_ptr + N_DDEBUG);
    else ddebug.assign(N_DDEBUG, 0.0);

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
    auto mp = get_buf_info(moments_params, 6);
    auto ep = get_buf_info(eddy_params, 4);

    double* G_out_ptr = nullptr;
    int N_out = 0;
    double* ddebug_ptr = nullptr;

    run_kernel_diff_fixedN(&G_out_ptr, &N_out, &ddebug_ptr, verbose, N0, gmax, smax, TE,
                           static_cast<int>(mp.rows), mp.ptr, PNS_thresh, T_readout, T_90,
                           T_180, diffmode, dt_out, static_cast<int>(ep.rows), ep.ptr,
                           search_bval, slew_reg);

    std::unique_ptr<double, decltype(&free)> guard_G(G_out_ptr, &free);
    std::unique_ptr<double, decltype(&free)> guard_D(ddebug_ptr, &free);

    std::vector<double> G_out(G_out_ptr, G_out_ptr + N_out);
    std::vector<double> ddebug;
    if (ddebug_ptr) ddebug.assign(ddebug_ptr, ddebug_ptr + N_DDEBUG);
    else ddebug.assign(N_DDEBUG, 0.0);

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
    auto mp = get_buf_info(moments_params, 6);
    auto ep = get_buf_info(eddy_params, 4);
    auto gi = G_in.request();

    double* G_out_ptr = nullptr;
    int N_out = 0;
    double* ddebug_ptr = nullptr;

    run_kernel_diff_fixedN_Gin(&G_out_ptr, &N_out, &ddebug_ptr, verbose,
                               static_cast<const double*>(gi.ptr), N0, gmax, smax, TE,
                               static_cast<int>(mp.rows), mp.ptr, PNS_thresh, T_readout, T_90,
                               T_180, diffmode, dt_out, static_cast<int>(ep.rows), ep.ptr,
                               search_bval, slew_reg);

    std::unique_ptr<double, decltype(&free)> guard_G(G_out_ptr, &free);
    std::unique_ptr<double, decltype(&free)> guard_D(ddebug_ptr, &free);

    std::vector<double> G_out(G_out_ptr, G_out_ptr + N_out);
    std::vector<double> ddebug;
    if (ddebug_ptr) ddebug.assign(ddebug_ptr, ddebug_ptr + N_DDEBUG);
    else ddebug.assign(N_DDEBUG, 0.0);

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
    auto mp = get_buf_info(moments_params, 6);
    auto ep = get_buf_info(eddy_params, 4);
    auto gf = gfix.request();

    double* G_out_ptr = nullptr;
    int N_out = 0;
    double* ddebug_ptr = nullptr;

    run_kernel_diff_fixeddt_fixG(&G_out_ptr, &N_out, &ddebug_ptr, verbose, dt0, gmax, smax, TE,
                                 static_cast<int>(mp.rows), mp.ptr, PNS_thresh, T_readout, T_90,
                                 T_180, diffmode, dt_out, static_cast<int>(ep.rows), ep.ptr,
                                 search_bval, N_gfix, static_cast<const double*>(gf.ptr), slew_reg);

    std::unique_ptr<double, decltype(&free)> guard_G(G_out_ptr, &free);
    std::unique_ptr<double, decltype(&free)> guard_D(ddebug_ptr, &free);

    std::vector<double> G_out(G_out_ptr, G_out_ptr + N_out);
    std::vector<double> ddebug;
    if (ddebug_ptr) ddebug.assign(ddebug_ptr, ddebug_ptr + N_DDEBUG);
    else ddebug.assign(N_DDEBUG, 0.0);

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