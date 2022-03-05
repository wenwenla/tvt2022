# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.12
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info >= (2, 7, 0):
    def swig_import_helper():
        import importlib
        pkg = __name__.rpartition('.')[0]
        mname = '.'.join((pkg, '_env')).lstrip('.')
        try:
            return importlib.import_module(mname)
        except ImportError:
            return importlib.import_module('_env')
    _env = swig_import_helper()
    del swig_import_helper
elif _swig_python_version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_env', [dirname(__file__)])
        except ImportError:
            import _env
            return _env
        try:
            _mod = imp.load_module('_env', fp, pathname, description)
        finally:
            if fp is not None:
                fp.close()
        return _mod
    _env = swig_import_helper()
    del swig_import_helper
else:
    import _env
del _swig_python_version_info

try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr(self, class_type, name):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    raise AttributeError("'%s' object has no attribute '%s'" % (class_type.__name__, name))


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except __builtin__.Exception:
    class _object:
        pass
    _newclass = 0

class SwigPyIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SwigPyIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SwigPyIterator, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _env.delete_SwigPyIterator
    __del__ = lambda self: None

    def value(self) -> "PyObject *":
        return _env.SwigPyIterator_value(self)

    def incr(self, n: 'size_t'=1) -> "swig::SwigPyIterator *":
        return _env.SwigPyIterator_incr(self, n)

    def decr(self, n: 'size_t'=1) -> "swig::SwigPyIterator *":
        return _env.SwigPyIterator_decr(self, n)

    def distance(self, x: 'SwigPyIterator') -> "ptrdiff_t":
        return _env.SwigPyIterator_distance(self, x)

    def equal(self, x: 'SwigPyIterator') -> "bool":
        return _env.SwigPyIterator_equal(self, x)

    def copy(self) -> "swig::SwigPyIterator *":
        return _env.SwigPyIterator_copy(self)

    def next(self) -> "PyObject *":
        return _env.SwigPyIterator_next(self)

    def __next__(self) -> "PyObject *":
        return _env.SwigPyIterator___next__(self)

    def previous(self) -> "PyObject *":
        return _env.SwigPyIterator_previous(self)

    def advance(self, n: 'ptrdiff_t') -> "swig::SwigPyIterator *":
        return _env.SwigPyIterator_advance(self, n)

    def __eq__(self, x: 'SwigPyIterator') -> "bool":
        return _env.SwigPyIterator___eq__(self, x)

    def __ne__(self, x: 'SwigPyIterator') -> "bool":
        return _env.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n: 'ptrdiff_t') -> "swig::SwigPyIterator &":
        return _env.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n: 'ptrdiff_t') -> "swig::SwigPyIterator &":
        return _env.SwigPyIterator___isub__(self, n)

    def __add__(self, n: 'ptrdiff_t') -> "swig::SwigPyIterator *":
        return _env.SwigPyIterator___add__(self, n)

    def __sub__(self, *args) -> "ptrdiff_t":
        return _env.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self
SwigPyIterator_swigregister = _env.SwigPyIterator_swigregister
SwigPyIterator_swigregister(SwigPyIterator)

class Point2D(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Point2D, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Point2D, name)
    __repr__ = _swig_repr
    __swig_setmethods__["x"] = _env.Point2D_x_set
    __swig_getmethods__["x"] = _env.Point2D_x_get
    if _newclass:
        x = _swig_property(_env.Point2D_x_get, _env.Point2D_x_set)
    __swig_setmethods__["y"] = _env.Point2D_y_set
    __swig_getmethods__["y"] = _env.Point2D_y_get
    if _newclass:
        y = _swig_property(_env.Point2D_y_get, _env.Point2D_y_set)

    def __init__(self):
        this = _env.new_Point2D()
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this
    __swig_destroy__ = _env.delete_Point2D
    __del__ = lambda self: None
Point2D_swigregister = _env.Point2D_swigregister
Point2D_swigregister(Point2D)

class UAVModel2D(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, UAVModel2D, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, UAVModel2D, name)
    __repr__ = _swig_repr

    def __init__(self, x: 'double', y: 'double', v: 'double', w: 'double'):
        this = _env.new_UAVModel2D(x, y, v, w)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def step(self, ang: 'double', acc: 'double') -> "void":
        return _env.UAVModel2D_step(self, ang, acc)

    def x(self) -> "double":
        return _env.UAVModel2D_x(self)

    def y(self) -> "double":
        return _env.UAVModel2D_y(self)

    def v(self) -> "double":
        return _env.UAVModel2D_v(self)

    def w(self) -> "double":
        return _env.UAVModel2D_w(self)
    __swig_destroy__ = _env.delete_UAVModel2D
    __del__ = lambda self: None
UAVModel2D_swigregister = _env.UAVModel2D_swigregister
UAVModel2D_swigregister(UAVModel2D)

class ManyUavEnv(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ManyUavEnv, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ManyUavEnv, name)
    __repr__ = _swig_repr

    def __init__(self, uav_cnt: 'int', random_seed: 'int', uav_die: 'bool'=True):
        this = _env.new_ManyUavEnv(uav_cnt, random_seed, uav_die)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def reset(self) -> "void":
        return _env.ManyUavEnv_reset(self)

    def step(self, control: 'ControlInfo') -> "void":
        return _env.ManyUavEnv_step(self, control)

    def getObservations(self) -> "std::vector< Observation,std::allocator< Observation > >":
        return _env.ManyUavEnv_getObservations(self)

    def getRewards(self) -> "std::vector< double,std::allocator< double > >":
        return _env.ManyUavEnv_getRewards(self)

    def getObstacles(self) -> "std::vector< Point2D,std::allocator< Point2D > >":
        return _env.ManyUavEnv_getObstacles(self)

    def getUavs(self) -> "std::vector< Point2D,std::allocator< Point2D > >":
        return _env.ManyUavEnv_getUavs(self)

    def getCollision(self) -> "std::vector< bool,std::allocator< bool > >":
        return _env.ManyUavEnv_getCollision(self)

    def getTarget(self) -> "Point2D":
        return _env.ManyUavEnv_getTarget(self)

    def isDone(self) -> "bool":
        return _env.ManyUavEnv_isDone(self)

    def getCollisionWithObs(self) -> "int":
        return _env.ManyUavEnv_getCollisionWithObs(self)

    def getCollisionWithUav(self) -> "int":
        return _env.ManyUavEnv_getCollisionWithUav(self)

    def getInTargetArea(self) -> "int":
        return _env.ManyUavEnv_getInTargetArea(self)

    def getSuccCnt(self) -> "int":
        return _env.ManyUavEnv_getSuccCnt(self)

    def getRadius(self) -> "double":
        return _env.ManyUavEnv_getRadius(self)

    def getVel(self) -> "double":
        return _env.ManyUavEnv_getVel(self)

    def getStatus(self) -> "std::vector< bool,std::allocator< bool > >":
        return _env.ManyUavEnv_getStatus(self)

    def getShapedReward(self) -> "std::vector< double,std::allocator< double > >":
        return _env.ManyUavEnv_getShapedReward(self)
    __swig_destroy__ = _env.delete_ManyUavEnv
    __del__ = lambda self: None
ManyUavEnv_swigregister = _env.ManyUavEnv_swigregister
ManyUavEnv_swigregister(ManyUavEnv)

class Observation(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Observation, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Observation, name)
    __repr__ = _swig_repr

    def iterator(self) -> "swig::SwigPyIterator *":
        return _env.Observation_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self) -> "bool":
        return _env.Observation___nonzero__(self)

    def __bool__(self) -> "bool":
        return _env.Observation___bool__(self)

    def __len__(self) -> "std::array< double,50 >::size_type":
        return _env.Observation___len__(self)

    def __getslice__(self, i: 'std::array< double,50 >::difference_type', j: 'std::array< double,50 >::difference_type') -> "std::array< double,50 > *":
        return _env.Observation___getslice__(self, i, j)

    def __setslice__(self, *args) -> "void":
        return _env.Observation___setslice__(self, *args)

    def __delslice__(self, i: 'std::array< double,50 >::difference_type', j: 'std::array< double,50 >::difference_type') -> "void":
        return _env.Observation___delslice__(self, i, j)

    def __delitem__(self, *args) -> "void":
        return _env.Observation___delitem__(self, *args)

    def __getitem__(self, *args) -> "std::array< double,50 >::value_type const &":
        return _env.Observation___getitem__(self, *args)

    def __setitem__(self, *args) -> "void":
        return _env.Observation___setitem__(self, *args)

    def __init__(self, *args):
        this = _env.new_Observation(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def empty(self) -> "bool":
        return _env.Observation_empty(self)

    def size(self) -> "std::array< double,50 >::size_type":
        return _env.Observation_size(self)

    def swap(self, v: 'Observation') -> "void":
        return _env.Observation_swap(self, v)

    def begin(self) -> "std::array< double,50 >::iterator":
        return _env.Observation_begin(self)

    def end(self) -> "std::array< double,50 >::iterator":
        return _env.Observation_end(self)

    def rbegin(self) -> "std::array< double,50 >::reverse_iterator":
        return _env.Observation_rbegin(self)

    def rend(self) -> "std::array< double,50 >::reverse_iterator":
        return _env.Observation_rend(self)

    def front(self) -> "std::array< double,50 >::value_type const &":
        return _env.Observation_front(self)

    def back(self) -> "std::array< double,50 >::value_type const &":
        return _env.Observation_back(self)

    def fill(self, u: 'std::array< double,50 >::value_type const &') -> "void":
        return _env.Observation_fill(self, u)
    __swig_destroy__ = _env.delete_Observation
    __del__ = lambda self: None
Observation_swigregister = _env.Observation_swigregister
Observation_swigregister(Observation)

class Observations(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Observations, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Observations, name)
    __repr__ = _swig_repr

    def iterator(self) -> "swig::SwigPyIterator *":
        return _env.Observations_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self) -> "bool":
        return _env.Observations___nonzero__(self)

    def __bool__(self) -> "bool":
        return _env.Observations___bool__(self)

    def __len__(self) -> "std::vector< std::array< double,50 > >::size_type":
        return _env.Observations___len__(self)

    def __getslice__(self, i: 'std::vector< std::array< double,50 > >::difference_type', j: 'std::vector< std::array< double,50 > >::difference_type') -> "std::vector< std::array< double,50 >,std::allocator< std::array< double,50 > > > *":
        return _env.Observations___getslice__(self, i, j)

    def __setslice__(self, *args) -> "void":
        return _env.Observations___setslice__(self, *args)

    def __delslice__(self, i: 'std::vector< std::array< double,50 > >::difference_type', j: 'std::vector< std::array< double,50 > >::difference_type') -> "void":
        return _env.Observations___delslice__(self, i, j)

    def __delitem__(self, *args) -> "void":
        return _env.Observations___delitem__(self, *args)

    def __getitem__(self, *args) -> "std::vector< std::array< double,50 > >::value_type const &":
        return _env.Observations___getitem__(self, *args)

    def __setitem__(self, *args) -> "void":
        return _env.Observations___setitem__(self, *args)

    def pop(self) -> "std::vector< std::array< double,50 > >::value_type":
        return _env.Observations_pop(self)

    def append(self, x: 'Observation') -> "void":
        return _env.Observations_append(self, x)

    def empty(self) -> "bool":
        return _env.Observations_empty(self)

    def size(self) -> "std::vector< std::array< double,50 > >::size_type":
        return _env.Observations_size(self)

    def swap(self, v: 'Observations') -> "void":
        return _env.Observations_swap(self, v)

    def begin(self) -> "std::vector< std::array< double,50 > >::iterator":
        return _env.Observations_begin(self)

    def end(self) -> "std::vector< std::array< double,50 > >::iterator":
        return _env.Observations_end(self)

    def rbegin(self) -> "std::vector< std::array< double,50 > >::reverse_iterator":
        return _env.Observations_rbegin(self)

    def rend(self) -> "std::vector< std::array< double,50 > >::reverse_iterator":
        return _env.Observations_rend(self)

    def clear(self) -> "void":
        return _env.Observations_clear(self)

    def get_allocator(self) -> "std::vector< std::array< double,50 > >::allocator_type":
        return _env.Observations_get_allocator(self)

    def pop_back(self) -> "void":
        return _env.Observations_pop_back(self)

    def erase(self, *args) -> "std::vector< std::array< double,50 > >::iterator":
        return _env.Observations_erase(self, *args)

    def __init__(self, *args):
        this = _env.new_Observations(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x: 'Observation') -> "void":
        return _env.Observations_push_back(self, x)

    def front(self) -> "std::vector< std::array< double,50 > >::value_type const &":
        return _env.Observations_front(self)

    def back(self) -> "std::vector< std::array< double,50 > >::value_type const &":
        return _env.Observations_back(self)

    def assign(self, n: 'std::vector< std::array< double,50 > >::size_type', x: 'Observation') -> "void":
        return _env.Observations_assign(self, n, x)

    def resize(self, *args) -> "void":
        return _env.Observations_resize(self, *args)

    def insert(self, *args) -> "void":
        return _env.Observations_insert(self, *args)

    def reserve(self, n: 'std::vector< std::array< double,50 > >::size_type') -> "void":
        return _env.Observations_reserve(self, n)

    def capacity(self) -> "std::vector< std::array< double,50 > >::size_type":
        return _env.Observations_capacity(self)
    __swig_destroy__ = _env.delete_Observations
    __del__ = lambda self: None
Observations_swigregister = _env.Observations_swigregister
Observations_swigregister(Observations)

class ControlInfo(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ControlInfo, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ControlInfo, name)
    __repr__ = _swig_repr

    def iterator(self) -> "swig::SwigPyIterator *":
        return _env.ControlInfo_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self) -> "bool":
        return _env.ControlInfo___nonzero__(self)

    def __bool__(self) -> "bool":
        return _env.ControlInfo___bool__(self)

    def __len__(self) -> "std::vector< std::array< double,2 > >::size_type":
        return _env.ControlInfo___len__(self)

    def __getslice__(self, i: 'std::vector< std::array< double,2 > >::difference_type', j: 'std::vector< std::array< double,2 > >::difference_type') -> "std::vector< std::array< double,2 >,std::allocator< std::array< double,2 > > > *":
        return _env.ControlInfo___getslice__(self, i, j)

    def __setslice__(self, *args) -> "void":
        return _env.ControlInfo___setslice__(self, *args)

    def __delslice__(self, i: 'std::vector< std::array< double,2 > >::difference_type', j: 'std::vector< std::array< double,2 > >::difference_type') -> "void":
        return _env.ControlInfo___delslice__(self, i, j)

    def __delitem__(self, *args) -> "void":
        return _env.ControlInfo___delitem__(self, *args)

    def __getitem__(self, *args) -> "std::vector< std::array< double,2 > >::value_type const &":
        return _env.ControlInfo___getitem__(self, *args)

    def __setitem__(self, *args) -> "void":
        return _env.ControlInfo___setitem__(self, *args)

    def pop(self) -> "std::vector< std::array< double,2 > >::value_type":
        return _env.ControlInfo_pop(self)

    def append(self, x: 'std::vector< std::array< double,2 > >::value_type const &') -> "void":
        return _env.ControlInfo_append(self, x)

    def empty(self) -> "bool":
        return _env.ControlInfo_empty(self)

    def size(self) -> "std::vector< std::array< double,2 > >::size_type":
        return _env.ControlInfo_size(self)

    def swap(self, v: 'ControlInfo') -> "void":
        return _env.ControlInfo_swap(self, v)

    def begin(self) -> "std::vector< std::array< double,2 > >::iterator":
        return _env.ControlInfo_begin(self)

    def end(self) -> "std::vector< std::array< double,2 > >::iterator":
        return _env.ControlInfo_end(self)

    def rbegin(self) -> "std::vector< std::array< double,2 > >::reverse_iterator":
        return _env.ControlInfo_rbegin(self)

    def rend(self) -> "std::vector< std::array< double,2 > >::reverse_iterator":
        return _env.ControlInfo_rend(self)

    def clear(self) -> "void":
        return _env.ControlInfo_clear(self)

    def get_allocator(self) -> "std::vector< std::array< double,2 > >::allocator_type":
        return _env.ControlInfo_get_allocator(self)

    def pop_back(self) -> "void":
        return _env.ControlInfo_pop_back(self)

    def erase(self, *args) -> "std::vector< std::array< double,2 > >::iterator":
        return _env.ControlInfo_erase(self, *args)

    def __init__(self, *args):
        this = _env.new_ControlInfo(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x: 'std::vector< std::array< double,2 > >::value_type const &') -> "void":
        return _env.ControlInfo_push_back(self, x)

    def front(self) -> "std::vector< std::array< double,2 > >::value_type const &":
        return _env.ControlInfo_front(self)

    def back(self) -> "std::vector< std::array< double,2 > >::value_type const &":
        return _env.ControlInfo_back(self)

    def assign(self, n: 'std::vector< std::array< double,2 > >::size_type', x: 'std::vector< std::array< double,2 > >::value_type const &') -> "void":
        return _env.ControlInfo_assign(self, n, x)

    def resize(self, *args) -> "void":
        return _env.ControlInfo_resize(self, *args)

    def insert(self, *args) -> "void":
        return _env.ControlInfo_insert(self, *args)

    def reserve(self, n: 'std::vector< std::array< double,2 > >::size_type') -> "void":
        return _env.ControlInfo_reserve(self, n)

    def capacity(self) -> "std::vector< std::array< double,2 > >::size_type":
        return _env.ControlInfo_capacity(self)
    __swig_destroy__ = _env.delete_ControlInfo
    __del__ = lambda self: None
ControlInfo_swigregister = _env.ControlInfo_swigregister
ControlInfo_swigregister(ControlInfo)

class DoubleVector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, DoubleVector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, DoubleVector, name)
    __repr__ = _swig_repr

    def iterator(self) -> "swig::SwigPyIterator *":
        return _env.DoubleVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self) -> "bool":
        return _env.DoubleVector___nonzero__(self)

    def __bool__(self) -> "bool":
        return _env.DoubleVector___bool__(self)

    def __len__(self) -> "std::vector< double >::size_type":
        return _env.DoubleVector___len__(self)

    def __getslice__(self, i: 'std::vector< double >::difference_type', j: 'std::vector< double >::difference_type') -> "std::vector< double,std::allocator< double > > *":
        return _env.DoubleVector___getslice__(self, i, j)

    def __setslice__(self, *args) -> "void":
        return _env.DoubleVector___setslice__(self, *args)

    def __delslice__(self, i: 'std::vector< double >::difference_type', j: 'std::vector< double >::difference_type') -> "void":
        return _env.DoubleVector___delslice__(self, i, j)

    def __delitem__(self, *args) -> "void":
        return _env.DoubleVector___delitem__(self, *args)

    def __getitem__(self, *args) -> "std::vector< double >::value_type const &":
        return _env.DoubleVector___getitem__(self, *args)

    def __setitem__(self, *args) -> "void":
        return _env.DoubleVector___setitem__(self, *args)

    def pop(self) -> "std::vector< double >::value_type":
        return _env.DoubleVector_pop(self)

    def append(self, x: 'std::vector< double >::value_type const &') -> "void":
        return _env.DoubleVector_append(self, x)

    def empty(self) -> "bool":
        return _env.DoubleVector_empty(self)

    def size(self) -> "std::vector< double >::size_type":
        return _env.DoubleVector_size(self)

    def swap(self, v: 'DoubleVector') -> "void":
        return _env.DoubleVector_swap(self, v)

    def begin(self) -> "std::vector< double >::iterator":
        return _env.DoubleVector_begin(self)

    def end(self) -> "std::vector< double >::iterator":
        return _env.DoubleVector_end(self)

    def rbegin(self) -> "std::vector< double >::reverse_iterator":
        return _env.DoubleVector_rbegin(self)

    def rend(self) -> "std::vector< double >::reverse_iterator":
        return _env.DoubleVector_rend(self)

    def clear(self) -> "void":
        return _env.DoubleVector_clear(self)

    def get_allocator(self) -> "std::vector< double >::allocator_type":
        return _env.DoubleVector_get_allocator(self)

    def pop_back(self) -> "void":
        return _env.DoubleVector_pop_back(self)

    def erase(self, *args) -> "std::vector< double >::iterator":
        return _env.DoubleVector_erase(self, *args)

    def __init__(self, *args):
        this = _env.new_DoubleVector(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x: 'std::vector< double >::value_type const &') -> "void":
        return _env.DoubleVector_push_back(self, x)

    def front(self) -> "std::vector< double >::value_type const &":
        return _env.DoubleVector_front(self)

    def back(self) -> "std::vector< double >::value_type const &":
        return _env.DoubleVector_back(self)

    def assign(self, n: 'std::vector< double >::size_type', x: 'std::vector< double >::value_type const &') -> "void":
        return _env.DoubleVector_assign(self, n, x)

    def resize(self, *args) -> "void":
        return _env.DoubleVector_resize(self, *args)

    def insert(self, *args) -> "void":
        return _env.DoubleVector_insert(self, *args)

    def reserve(self, n: 'std::vector< double >::size_type') -> "void":
        return _env.DoubleVector_reserve(self, n)

    def capacity(self) -> "std::vector< double >::size_type":
        return _env.DoubleVector_capacity(self)
    __swig_destroy__ = _env.delete_DoubleVector
    __del__ = lambda self: None
DoubleVector_swigregister = _env.DoubleVector_swigregister
DoubleVector_swigregister(DoubleVector)

class IntVector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, IntVector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, IntVector, name)
    __repr__ = _swig_repr

    def iterator(self) -> "swig::SwigPyIterator *":
        return _env.IntVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self) -> "bool":
        return _env.IntVector___nonzero__(self)

    def __bool__(self) -> "bool":
        return _env.IntVector___bool__(self)

    def __len__(self) -> "std::vector< int >::size_type":
        return _env.IntVector___len__(self)

    def __getslice__(self, i: 'std::vector< int >::difference_type', j: 'std::vector< int >::difference_type') -> "std::vector< int,std::allocator< int > > *":
        return _env.IntVector___getslice__(self, i, j)

    def __setslice__(self, *args) -> "void":
        return _env.IntVector___setslice__(self, *args)

    def __delslice__(self, i: 'std::vector< int >::difference_type', j: 'std::vector< int >::difference_type') -> "void":
        return _env.IntVector___delslice__(self, i, j)

    def __delitem__(self, *args) -> "void":
        return _env.IntVector___delitem__(self, *args)

    def __getitem__(self, *args) -> "std::vector< int >::value_type const &":
        return _env.IntVector___getitem__(self, *args)

    def __setitem__(self, *args) -> "void":
        return _env.IntVector___setitem__(self, *args)

    def pop(self) -> "std::vector< int >::value_type":
        return _env.IntVector_pop(self)

    def append(self, x: 'std::vector< int >::value_type const &') -> "void":
        return _env.IntVector_append(self, x)

    def empty(self) -> "bool":
        return _env.IntVector_empty(self)

    def size(self) -> "std::vector< int >::size_type":
        return _env.IntVector_size(self)

    def swap(self, v: 'IntVector') -> "void":
        return _env.IntVector_swap(self, v)

    def begin(self) -> "std::vector< int >::iterator":
        return _env.IntVector_begin(self)

    def end(self) -> "std::vector< int >::iterator":
        return _env.IntVector_end(self)

    def rbegin(self) -> "std::vector< int >::reverse_iterator":
        return _env.IntVector_rbegin(self)

    def rend(self) -> "std::vector< int >::reverse_iterator":
        return _env.IntVector_rend(self)

    def clear(self) -> "void":
        return _env.IntVector_clear(self)

    def get_allocator(self) -> "std::vector< int >::allocator_type":
        return _env.IntVector_get_allocator(self)

    def pop_back(self) -> "void":
        return _env.IntVector_pop_back(self)

    def erase(self, *args) -> "std::vector< int >::iterator":
        return _env.IntVector_erase(self, *args)

    def __init__(self, *args):
        this = _env.new_IntVector(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x: 'std::vector< int >::value_type const &') -> "void":
        return _env.IntVector_push_back(self, x)

    def front(self) -> "std::vector< int >::value_type const &":
        return _env.IntVector_front(self)

    def back(self) -> "std::vector< int >::value_type const &":
        return _env.IntVector_back(self)

    def assign(self, n: 'std::vector< int >::size_type', x: 'std::vector< int >::value_type const &') -> "void":
        return _env.IntVector_assign(self, n, x)

    def resize(self, *args) -> "void":
        return _env.IntVector_resize(self, *args)

    def insert(self, *args) -> "void":
        return _env.IntVector_insert(self, *args)

    def reserve(self, n: 'std::vector< int >::size_type') -> "void":
        return _env.IntVector_reserve(self, n)

    def capacity(self) -> "std::vector< int >::size_type":
        return _env.IntVector_capacity(self)
    __swig_destroy__ = _env.delete_IntVector
    __del__ = lambda self: None
IntVector_swigregister = _env.IntVector_swigregister
IntVector_swigregister(IntVector)

class BoolVector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, BoolVector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, BoolVector, name)
    __repr__ = _swig_repr

    def iterator(self) -> "swig::SwigPyIterator *":
        return _env.BoolVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self) -> "bool":
        return _env.BoolVector___nonzero__(self)

    def __bool__(self) -> "bool":
        return _env.BoolVector___bool__(self)

    def __len__(self) -> "std::vector< bool >::size_type":
        return _env.BoolVector___len__(self)

    def __getslice__(self, i: 'std::vector< bool >::difference_type', j: 'std::vector< bool >::difference_type') -> "std::vector< bool,std::allocator< bool > > *":
        return _env.BoolVector___getslice__(self, i, j)

    def __setslice__(self, *args) -> "void":
        return _env.BoolVector___setslice__(self, *args)

    def __delslice__(self, i: 'std::vector< bool >::difference_type', j: 'std::vector< bool >::difference_type') -> "void":
        return _env.BoolVector___delslice__(self, i, j)

    def __delitem__(self, *args) -> "void":
        return _env.BoolVector___delitem__(self, *args)

    def __getitem__(self, *args) -> "std::vector< bool >::value_type":
        return _env.BoolVector___getitem__(self, *args)

    def __setitem__(self, *args) -> "void":
        return _env.BoolVector___setitem__(self, *args)

    def pop(self) -> "std::vector< bool >::value_type":
        return _env.BoolVector_pop(self)

    def append(self, x: 'std::vector< bool >::value_type') -> "void":
        return _env.BoolVector_append(self, x)

    def empty(self) -> "bool":
        return _env.BoolVector_empty(self)

    def size(self) -> "std::vector< bool >::size_type":
        return _env.BoolVector_size(self)

    def swap(self, v: 'BoolVector') -> "void":
        return _env.BoolVector_swap(self, v)

    def begin(self) -> "std::vector< bool >::iterator":
        return _env.BoolVector_begin(self)

    def end(self) -> "std::vector< bool >::iterator":
        return _env.BoolVector_end(self)

    def rbegin(self) -> "std::vector< bool >::reverse_iterator":
        return _env.BoolVector_rbegin(self)

    def rend(self) -> "std::vector< bool >::reverse_iterator":
        return _env.BoolVector_rend(self)

    def clear(self) -> "void":
        return _env.BoolVector_clear(self)

    def get_allocator(self) -> "std::vector< bool >::allocator_type":
        return _env.BoolVector_get_allocator(self)

    def pop_back(self) -> "void":
        return _env.BoolVector_pop_back(self)

    def erase(self, *args) -> "std::vector< bool >::iterator":
        return _env.BoolVector_erase(self, *args)

    def __init__(self, *args):
        this = _env.new_BoolVector(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x: 'std::vector< bool >::value_type') -> "void":
        return _env.BoolVector_push_back(self, x)

    def front(self) -> "std::vector< bool >::value_type":
        return _env.BoolVector_front(self)

    def back(self) -> "std::vector< bool >::value_type":
        return _env.BoolVector_back(self)

    def assign(self, n: 'std::vector< bool >::size_type', x: 'std::vector< bool >::value_type') -> "void":
        return _env.BoolVector_assign(self, n, x)

    def resize(self, *args) -> "void":
        return _env.BoolVector_resize(self, *args)

    def insert(self, *args) -> "void":
        return _env.BoolVector_insert(self, *args)

    def reserve(self, n: 'std::vector< bool >::size_type') -> "void":
        return _env.BoolVector_reserve(self, n)

    def capacity(self) -> "std::vector< bool >::size_type":
        return _env.BoolVector_capacity(self)
    __swig_destroy__ = _env.delete_BoolVector
    __del__ = lambda self: None
BoolVector_swigregister = _env.BoolVector_swigregister
BoolVector_swigregister(BoolVector)

class Point2DVector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Point2DVector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Point2DVector, name)
    __repr__ = _swig_repr

    def iterator(self) -> "swig::SwigPyIterator *":
        return _env.Point2DVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self) -> "bool":
        return _env.Point2DVector___nonzero__(self)

    def __bool__(self) -> "bool":
        return _env.Point2DVector___bool__(self)

    def __len__(self) -> "std::vector< Point2D >::size_type":
        return _env.Point2DVector___len__(self)

    def __getslice__(self, i: 'std::vector< Point2D >::difference_type', j: 'std::vector< Point2D >::difference_type') -> "std::vector< Point2D,std::allocator< Point2D > > *":
        return _env.Point2DVector___getslice__(self, i, j)

    def __setslice__(self, *args) -> "void":
        return _env.Point2DVector___setslice__(self, *args)

    def __delslice__(self, i: 'std::vector< Point2D >::difference_type', j: 'std::vector< Point2D >::difference_type') -> "void":
        return _env.Point2DVector___delslice__(self, i, j)

    def __delitem__(self, *args) -> "void":
        return _env.Point2DVector___delitem__(self, *args)

    def __getitem__(self, *args) -> "std::vector< Point2D >::value_type const &":
        return _env.Point2DVector___getitem__(self, *args)

    def __setitem__(self, *args) -> "void":
        return _env.Point2DVector___setitem__(self, *args)

    def pop(self) -> "std::vector< Point2D >::value_type":
        return _env.Point2DVector_pop(self)

    def append(self, x: 'Point2D') -> "void":
        return _env.Point2DVector_append(self, x)

    def empty(self) -> "bool":
        return _env.Point2DVector_empty(self)

    def size(self) -> "std::vector< Point2D >::size_type":
        return _env.Point2DVector_size(self)

    def swap(self, v: 'Point2DVector') -> "void":
        return _env.Point2DVector_swap(self, v)

    def begin(self) -> "std::vector< Point2D >::iterator":
        return _env.Point2DVector_begin(self)

    def end(self) -> "std::vector< Point2D >::iterator":
        return _env.Point2DVector_end(self)

    def rbegin(self) -> "std::vector< Point2D >::reverse_iterator":
        return _env.Point2DVector_rbegin(self)

    def rend(self) -> "std::vector< Point2D >::reverse_iterator":
        return _env.Point2DVector_rend(self)

    def clear(self) -> "void":
        return _env.Point2DVector_clear(self)

    def get_allocator(self) -> "std::vector< Point2D >::allocator_type":
        return _env.Point2DVector_get_allocator(self)

    def pop_back(self) -> "void":
        return _env.Point2DVector_pop_back(self)

    def erase(self, *args) -> "std::vector< Point2D >::iterator":
        return _env.Point2DVector_erase(self, *args)

    def __init__(self, *args):
        this = _env.new_Point2DVector(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x: 'Point2D') -> "void":
        return _env.Point2DVector_push_back(self, x)

    def front(self) -> "std::vector< Point2D >::value_type const &":
        return _env.Point2DVector_front(self)

    def back(self) -> "std::vector< Point2D >::value_type const &":
        return _env.Point2DVector_back(self)

    def assign(self, n: 'std::vector< Point2D >::size_type', x: 'Point2D') -> "void":
        return _env.Point2DVector_assign(self, n, x)

    def resize(self, *args) -> "void":
        return _env.Point2DVector_resize(self, *args)

    def insert(self, *args) -> "void":
        return _env.Point2DVector_insert(self, *args)

    def reserve(self, n: 'std::vector< Point2D >::size_type') -> "void":
        return _env.Point2DVector_reserve(self, n)

    def capacity(self) -> "std::vector< Point2D >::size_type":
        return _env.Point2DVector_capacity(self)
    __swig_destroy__ = _env.delete_Point2DVector
    __del__ = lambda self: None
Point2DVector_swigregister = _env.Point2DVector_swigregister
Point2DVector_swigregister(Point2DVector)

# This file is compatible with both classic and new-style classes.


