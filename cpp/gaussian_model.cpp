#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

constexpr double kSqrtTwoPi = 2.5066282746310002;
constexpr double kSqrtTwo = 1.4142135623730950488;

double normal_pdf(double value) {
    return std::exp(-0.5 * value * value) / kSqrtTwoPi;
}

struct IntegratedValues {
    double probability;
    double derivative_center;
    double derivative_sigma;
};

IntegratedValues integrated_probability_and_derivatives(
    double coordinate,
    double center,
    double sigma
) {
    const double edge_low = coordinate - 0.5;
    const double edge_high = coordinate + 0.5;
    const double z_low = (edge_low - center) / sigma;
    const double z_high = (edge_high - center) / sigma;
    const double probability = 0.5 * (
        std::erf(z_high / kSqrtTwo) - std::erf(z_low / kSqrtTwo)
    );
    const double derivative_center = (
        normal_pdf(z_low) - normal_pdf(z_high)
    ) / sigma;
    const double derivative_sigma = (
        z_low * normal_pdf(z_low) - z_high * normal_pdf(z_high)
    ) / sigma;
    return {probability, derivative_center, derivative_sigma};
}

}  // namespace

std::tuple<py::array_t<double>, py::array_t<double>> model_and_jacobian(
    py::array_t<double, py::array::c_style | py::array::forcecast> parameters,
    py::array_t<double, py::array::c_style | py::array::forcecast> x_coords,
    py::array_t<double, py::array::c_style | py::array::forcecast> y_coords,
    double background_center_x,
    double background_center_y
) {
    const auto parameters_info = parameters.request();
    const auto x_info = x_coords.request();
    const auto y_info = y_coords.request();
    if (parameters_info.ndim != 1 || parameters_info.shape[0] != 7) {
        throw std::invalid_argument("parameters must have shape (7,)");
    }
    if (x_info.ndim != 1 || y_info.ndim != 1) {
        throw std::invalid_argument("x_coords and y_coords must be one-dimensional");
    }

    const auto* parameter_values = static_cast<const double*>(parameters_info.ptr);
    const auto* x_values = static_cast<const double*>(x_info.ptr);
    const auto* y_values = static_cast<const double*>(y_info.ptr);
    const std::size_t width = static_cast<std::size_t>(x_info.shape[0]);
    const std::size_t height = static_cast<std::size_t>(y_info.shape[0]);
    const double x0 = parameter_values[0];
    const double y0 = parameter_values[1];
    const double total_flux = parameter_values[2];
    const double sigma = parameter_values[3];
    const double b0 = parameter_values[4];
    const double bx = parameter_values[5];
    const double by = parameter_values[6];
    if (!(sigma > 0.0)) {
        throw std::invalid_argument("sigma must be positive");
    }

    py::array_t<double> model({height, width});
    py::array_t<double> jacobian({height * width, static_cast<std::size_t>(7)});
    auto model_view = model.mutable_unchecked<2>();
    auto jacobian_view = jacobian.mutable_unchecked<2>();

    std::vector<IntegratedValues> x_integrals(width);
    std::vector<IntegratedValues> y_integrals(height);
    for (std::size_t column = 0; column < width; ++column) {
        x_integrals[column] = integrated_probability_and_derivatives(
            x_values[column], x0, sigma
        );
    }
    for (std::size_t row = 0; row < height; ++row) {
        y_integrals[row] = integrated_probability_and_derivatives(
            y_values[row], y0, sigma
        );
    }

    for (std::size_t row = 0; row < height; ++row) {
        for (std::size_t column = 0; column < width; ++column) {
            const auto& x = x_integrals[column];
            const auto& y = y_integrals[row];
            const double x_delta = x_values[column] - background_center_x;
            const double y_delta = y_values[row] - background_center_y;
            const double row_value = total_flux * y.probability * x.probability;
            const std::size_t flat_index = row * width + column;
            model_view(row, column) = row_value + b0 + bx * x_delta + by * y_delta;

            jacobian_view(flat_index, 0) = total_flux * y.probability * x.derivative_center;
            jacobian_view(flat_index, 1) = total_flux * y.derivative_center * x.probability;
            jacobian_view(flat_index, 2) = y.probability * x.probability;
            jacobian_view(flat_index, 3) = total_flux * (
                y.derivative_sigma * x.probability
                + y.probability * x.derivative_sigma
            );
            jacobian_view(flat_index, 4) = 1.0;
            jacobian_view(flat_index, 5) = x_delta;
            jacobian_view(flat_index, 6) = y_delta;
        }
    }
    return {model, jacobian};
}

PYBIND11_MODULE(_gaussian_model, module) {
    module.doc() = "Pixel-integrated Gaussian model and analytical Jacobian.";
    module.def(
        "model_and_jacobian",
        &model_and_jacobian,
        py::arg("parameters"),
        py::arg("x_coords"),
        py::arg("y_coords"),
        py::arg("background_center_x"),
        py::arg("background_center_y")
    );
}
