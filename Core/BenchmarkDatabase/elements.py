from enum import Enum
import re

class Value_Comparable(object):
    def __eq__(self, other):
        if hasattr(other, '__dict__'):
            for key in other.__dict__.keys():
                v1 = self.__getattribute__(key)
                v2 = other.__getattribute__(key)
                if isinstance(v1, type(v2)):
                    try:
                        if not v1 == v2:
                            return False
                    except:
                        return False
                else:
                    return False
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.__dict__.values())


class Operator(object):
    """
    Benchmarkable Operator
    Operators are compared by their name and length of the parameter descriptor.
    The values of the individual parameter descriptors are ignored since they are only for debugging.
    """
    def __init__(self, name, param_desc):
        self.name = name  # type: str
        self.parameter_descriptors = param_desc  # type: tuple

    def __eq__(self, other):
        return self.name == other.name and len(self.parameter_descriptors) == len(other.parameter_descriptors)

    def __hash__(self):
        return hash(self.__dict__.values())

    def __str__(self):
        return "{} [{}]".format(self.name, str(self.parameter_descriptors))


class Operation(Value_Comparable):
    """An operator with associated parameters."""
    def __init__(self, operator, params):
        self.operator = operator
        self.parameters = params

    def __str__(self):
        return "Operator: {} \nParameters {}".format(str(self.operator), str(self.parameters))


class Time(object):
    """Simple container for a time duration."""
    def __init__(self, t, u):
        self.time = t
        self.unit = u

    @classmethod
    def fromString(cls, s):
        r = re.compile(r"^\d*[.]?\d*")
        num = r.match(s)
        if num:
            return cls(float(num[0]), s.replace(num[0], ""))

    def __add__(self, other):
        if other is None:
            return self
        if self.unit == other.unit:
            return Time(self.time + other.time, self.unit)
        else:
            diffe = Time.sfactor[other.unit] / Time.sfactor[self.unit]
            return Time(self.time + other.time * diffe, self.unit)

    def __sub__(self, other):
        return self.__add__(Time(-other.time, other.unit))

    def __str__(self):
        return str(self.time) + self.unit

    def __gt__(self, other):
        return self.time > other.time * (Time.sfactor[other.unit] / Time.sfactor[self.unit])

    sfactor = {
        "d" : 60 * 60 * 24,
        "h" : 60 * 60,
        "m" : 60,
        "s" : 1,
        "ms": 1.0e-3,
        "us": 1.0e-6,
        "ns": 1.0e-9
    }


class Measurement(Value_Comparable):
    """Performance of an operation that ran on a specific system."""
    def __init__(self, operation : Operation, system : str, time : Time):
        self.operation = operation
        self.system = system
        self.time = time

    def __str__(self):
        return "Measurement: {} \nTime: {}".format(str(self.operation), self.time)


class Target(Enum):
    """This directly refers to the different measurement lists in the db."""
    OP = "operator_results"
    TRAINING = "training_results"
    PREDICTION = "perf_predictions"
