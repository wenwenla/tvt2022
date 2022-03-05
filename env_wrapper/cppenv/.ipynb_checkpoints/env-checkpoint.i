%module env

%{
#include "env.h"
%}

%include "std_string.i"
%include "std_vector.i"
%include "std_array.i"
%include "env.h"

%template(Observation) std::array<double, 50>;
%template(Observations) std::vector<std::array<double, 50>>;
%template(ControlInfo) std::vector<std::array<double, 2>>;
%template(DoubleVector) std::vector<double>;
%template(IntVector) std::vector<int>;
%template(BoolVector) std::vector<bool>;
%template(Point2DVector) std::vector<Point2D>;

