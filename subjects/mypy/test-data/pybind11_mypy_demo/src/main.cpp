/**
 * This file contains the pybind11 reference implementation for the stugen tests,
 * and was originally inspired by:
 *
 * https://github.com/sizmailov/pybind11-mypy-demo
 *
 * Copyright (c) 2016 The Pybind Development Team, All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * You are under no obligation whatsoever to provide any bug fixes, patches, or
 * upgrades to the features, functionality or performance of the source code
 * ("Enhancements") to anyone; however, if you choose to make your Enhancements
 * available either publicly, or directly to the author of this software, without
 * imposing a separate written license agreement for such Enhancements, then you
 * hereby grant the following license: a non-exclusive, royalty-free perpetual
 * license to install, use, modify, prepare derivative works, incorporate into
 * other computer software, distribute, and sublicense such enhancements or
 * derivative works thereof, in binary and source code form.
 */

#include <cmath>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace basics {

int answer() {
  return 42;
}

int sum(int a, int b) {
  return a + b;
}

double midpoint(double left, double right){
  return left + (right - left)/2;
}

double weighted_midpoint(double left, double right, double alpha=0.5) {
  return left + (right - left) * alpha;
}

struct Point {

  enum class LengthUnit {
    mm=0,
    pixel,
    inch
  };

  enum class AngleUnit {
    radian=0,
    degree
  };

  Point() : Point(0, 0) {}
  Point(double x, double y) : x(x), y(y) {}

  static const Point origin;
  static const Point x_axis;
  static const Point y_axis;

  static LengthUnit length_unit;
  static AngleUnit angle_unit;

  double length() const {
    return std::sqrt(x * x + y * y);
  }

  double distance_to(double other_x, double other_y) const {
    double dx = x - other_x;
    double dy = y - other_y;
    return std::sqrt(dx*dx + dy*dy);
  }

  double distance_to(const Point& other) const {
    return distance_to(other.x, other.y);
  }

  double x, y;
};

const Point Point::origin = Point(0, 0);
const Point Point::x_axis = Point(1, 0);
const Point Point::y_axis = Point(0, 1);

Point::LengthUnit Point::length_unit = Point::LengthUnit::mm;
Point::AngleUnit Point::angle_unit = Point::AngleUnit::radian;

} // namespace: basics

void bind_basics(py::module& basics) {

  using namespace basics;

  // Functions
  basics.def("answer", &answer);
  basics.def("sum", &sum);
  basics.def("midpoint", &midpoint, py::arg("left"), py::arg("right"));
  basics.def("weighted_midpoint", weighted_midpoint, py::arg("left"), py::arg("right"), py::arg("alpha")=0.5);

  // Classes
  py::class_<Point> pyPoint(basics, "Point");
  py::enum_<Point::LengthUnit> pyLengthUnit(pyPoint, "LengthUnit");
  py::enum_<Point::AngleUnit> pyAngleUnit(pyPoint, "AngleUnit");

  pyPoint
    .def(py::init<>())
    .def(py::init<double, double>(), py::arg("x"), py::arg("y"))
    .def("distance_to", py::overload_cast<double, double>(&Point::distance_to, py::const_), py::arg("x"), py::arg("y"))
    .def("distance_to", py::overload_cast<const Point&>(&Point::distance_to, py::const_), py::arg("other"))
    .def_readwrite("x", &Point::x)
    .def_property("y",
        [](Point& self){ return self.y; },
        [](Point& self, double value){ self.y = value; }
    )
    .def_property_readonly("length", &Point::length)
    .def_property_readonly_static("x_axis", [](py::object cls){return Point::x_axis;})
    .def_property_readonly_static("y_axis", [](py::object cls){return Point::y_axis;})
    .def_readwrite_static("length_unit", &Point::length_unit)
    .def_property_static("angle_unit",
        [](py::object& /*cls*/){ return Point::angle_unit; },
        [](py::object& /*cls*/, Point::AngleUnit value){ Point::angle_unit = value; }
     );

  pyPoint.attr("origin") = Point::origin;

  pyLengthUnit
    .value("mm", Point::LengthUnit::mm)
    .value("pixel", Point::LengthUnit::pixel)
    .value("inch", Point::LengthUnit::inch);

  pyAngleUnit
    .value("radian", Point::AngleUnit::radian)
    .value("degree", Point::AngleUnit::degree);

  // Module-level attributes
  basics.attr("PI") = std::acos(-1);
  basics.attr("__version__") = "0.0.1";
}

PYBIND11_MODULE(pybind11_mypy_demo, m) {
  auto basics = m.def_submodule("basics");
  bind_basics(basics);
}