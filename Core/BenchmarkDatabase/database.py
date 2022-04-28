import jsonpickle
import os
import time
from BenchmarkDatabase import default_opset, runtime_filepath, root_path
from BenchmarkDatabase.elements import Operator, Operation, Measurement, Target, Time
from typing import List, Dict


class Database(object):
    """
    Main lightweight database class

    Attributes:
        _operators (Operator): All kinds of Operators, this includes low-level and training level operators.
        low_level_ops (Operation): Low-level operations of network operators.
        training_ops (Operation): Network training operations.
        operator_results (Measurement): Low-level operator results.
        training_results (Measurement): Results of full network training runs.
        perf_prediction (Measurement): Performance predictions of operations on specific systems.
    """

    db_filepath = runtime_filepath + "db.json" # type: str

    def __init__(self):
        """Loads the db.json file from the Runtimefiles directory if possible. Otherwise empty."""
        if os.path.isfile(self.db_filepath):
            fp = open(self.db_filepath, "r")
            json = fp.read()
            db = jsonpickle.decode(json)                # type: Database
            self._operators = db._operators             # type: List[Operator]
            self.low_level_ops = db.low_level_ops       # type: List[Operation]
            self.training_ops = db.training_ops         # type: List[Operation]
            self.operator_results = db.operator_results # type: List[Measurement]
            self.training_results = db.training_results # type: List[Measurement]
            self.perf_predictions = db.perf_predictions   # type: List[Measurement]
        else:
            self._operators = default_opset.default_operators
            self.low_level_ops = []
            self.training_ops = []
            self.operator_results = []
            self.training_results = []
            self.perf_predictions = []

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type is not None:
            self.store(runtime_filepath + "crashed_db_backup_" + time.strftime("%Y%m%d-%H%M%S") + ".json")
            return
        else :
            self.store()

    #TODO: Implement storage as separate json files
    def store(self, path=""):
        """
        The Database is stored as pickled json files.
        """
        if path == "":
            path = self.db_filepath
        json = jsonpickle.encode(self)
        outfile = open(path, "w")
        outfile.write(json)
        outfile.close()

    def add_operator(self, operator: Operator) -> Operator:
        """
        Add an operator to the database.
        If the operator is already in the db the reference to the db object is returned.
        :param operator: Operator that shall be added.
        :return: operator that is in the db.
        """
        operators = set(self._operators)
        for op in operators:
            if op == operator:
                return op
        self._operators.append(operator)
        return operator

    def add_operators(self, operators: List[Operator]):
        """
        Add a list of operators to the database.
        :param operators: Operators that shall be added.
        """
        for op in operators:
            self.add_operator(op)

    def remove_operator(self, operator: Operator):
        """Remove operator from the db. This also deletes all objects that reference the Operator!"""
        for collection in self.low_level_ops, self.training_ops:
            for operation in collection:
                if operation.operator == operator:
                    self.remove_operation(operation)
        self._operators.remove(operator)

    def get_measurement(self, operation: Operation) -> Measurement:
        measurements = self.operator_results
        for measurement in measurements:
            if measurement.operation == operation:
                return operation
        print("No Measurements in the database for" + str(operation))

    def add_operation(self, operation: Operation, target) -> Operation:
        """
        Add an operation to the database.
        This also adds the operator of the operation if it is not present in the database.
        :param operation: Operation to be added.
        :param target: Is the operation part of the operator benchmarks or is it a full network training operation. (Target.OP or Target.TRAINING)
        :return: operation that is in the db.
        """
        operation.operator = self.add_operator(operation.operator)

        if target == Target.OP:
            for op in self.low_level_ops:
                if op == operation:
                    return op
            self.low_level_ops.append(operation)
            return operation
        elif target == Target.TRAINING:
            operations = set(self.training_ops)
            for op in operations:
                if op == operation:
                    return op
            self.training_ops.append(operation)
            return operation

        return NotImplemented

    def add_operations(self, operations: List[Operation], target):
        """
        Add operations to the database.
        This also adds the operator of the operations if it is not present in the database.
        :param operations: List of operations to be added.
        :param target: Is the operation part of the operator benchmarks or is it a full network training operation. (Target.OP or Target.TRAINING)
        """
        for operation in operations:
            self.add_operation(operation, target)

    def remove_operation(self, operation: Operation):
        """This removes an operation from the database.
        This also removes all measurements that reference this operation!"""
        if operation in self.low_level_ops:
            self.low_level_ops.remove(operation)
        elif operation in self.training_ops:
            self.training_ops.remove(operation)
        else:
            print("Operation not in the database.")
            return
        for collection in self.operator_results, self.training_results, self.perf_prediction:
            for measurement in collection:
                if measurement.operation == operation:
                    collection.remove(measurement)

    def add_measurement(self, measurement: Measurement, target: Target) -> Measurement:
        """
        Add a measurement to the database. This does not add duplicates.
        :param measurement: The measurement to be added.
        :param target: The target list of measurement which the new measurement shall be added to.
        Valid options are: (Target.OP, Target.TRAINING, Target.PREDICTION)
        :return: measurement that is in the db.
        """
        measurement.operation = self.add_operation(measurement.operation, target)

        if not hasattr(self, str(target.value)):
            print("Cannot add measurement. Valid targets are: (Target.OP, Target.TRAINING, Target.PREDICTION)")
        else:
            result_set = getattr(self, str(target.value))
            for m in result_set:
                if measurement == m:
                    print("Skip adding duplicate. {}".format(measurement))
                    return m
            result_set.append(measurement)
            return measurement
        return NotImplemented

    def add_measurements(self, measurements: List[Measurement], target: Target):
        """
        Add a list of measurements to the database. This does not add duplicates. All measurements need to be for the same target.
        :param measurements: The list of measurements to be added.
        :param target: The target list of measurement which the new measurements shall be added to.
        Valid options are: (Target.OP, Target.TRAINING, Target.PREDICTION)
        """
        if not hasattr(self, str(target.value)):
            print("Cannot add measurements. Valid targets are: (Target.OP, Target.TRAINING, Target.PREDICTION)")
        else:
            for measurement in measurements:
                self.add_measurement(measurement, target)

    def remove_measurement(self, measurement: Measurement, target: str):
        """
        Remove a measurement from the database.
        :param measurement: The measurement to remove.
        :param target: The target list of measurement which the new measurement shall be removed from.
        Valid options are: (Target.OP, Target.TRAINING, Target.PREDICTION)
        """
        if not hasattr(self, str(target.value)):
            print("Cannot remove measurement. Valid targets are: (Target.OP, Target.TRAINING, Target.PREDICTION)")
        else:
            result_set = getattr(self, str(target.value)) # type: list
            result_set.remove(measurement)

    def get_measurements(self, target) -> list:
        print("Get Measurements from: {}".format(str(target.value)))
        return getattr(self, str(target.value))

    def get_operations_by_type(self, target, filter=None) -> dict:
        """
        Sort all operations by their operator from a target group in a dict.
        :param target: Is the operation part of the operator benchmarks or is it a full network training operation. (Target.OP or Target.TRAINING)
        :param filter: If the filter is set, only operators present (by name) in the filter are returned.
        :return: Returns a dict with an entry per operator each with a list of the associated operations.
        """
        if target == Target.OP:
            operations = self.low_level_ops
        else:
            operations = self.training_ops

        op_dict = {}
        for op in operations:
            if filter is None or op.operator.name in filter:
                if op.operator.name not in op_dict:
                    op_dict[op.operator.name] = []
                op_dict[op.operator.name].append(op)

        return op_dict

    def log_op_and_training_differences(self):
        """
        For all operations that are both present in the Operator results and the training results print the measured differences.
        """
        '''print("Operation Results:")
        for m in self.operator_results:
            print("{} {} {}".format(m.operation.operator.name, m.operation.parameters, m.time))
        print("All training results:")
        for m in self.training_results:
            print("{} {} {}".format(m.operation.operator.name, m.operation.parameters, m.time))
        '''
        print("Operator Name Parameters -> OP Benchmark Time | Training Time")
        for m1 in self.operator_results:
            train_time = Time(0, "ms")
            for m2 in self.training_results:
                if m1.operation == m2.operation:
                    train_time = train_time + m2.time
            if train_time.time == 0:
                print("No pair found for:" + str(m1))

            print("{}{} -> {} | {}".format(m1.operation.operator.name, m1.operation.parameters, m1.time, train_time))

    def log_op_results(self, ops : Dict[str, List[Operation]]):
        print("{} operations.".format(len(ops)))
        print("Operation -> Predicted -> Measured")
        for optype, oplist in ops.items():
            for op in oplist:
                #measurement = next((x for x in self.operator_results if x.operation == op), None)
                measurement_op = None
                measurement_tn = None
                for m_op in self.operator_results:
                    if m_op.operation == op:
                        measurement_op = m_op
                        break
                for m_tn in self.training_results:
                    if m_tn.operation == op:
                        measurement_tn = m_tn

                m1 = (measurement_op.time + Time(0, "ms")) if measurement_op else "No Result"
                m2 = (measurement_tn.time + Time(0, "ms")) if measurement_tn else "No Result"
                print("{} -> {} -> {}".format(op, m1, m2))



    @staticmethod
    def op_list_to_dict(ops: List[Operation]):
        op_dict = {}
        for op in ops:
            if op.operator.name in op_dict:
                op_dict[op.operator.name].append(op)
            else:
                op_dict[op.operator.name] = [op]
        return op_dict




