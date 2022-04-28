import onnx
from onnx import shape_inference, helper
from onnx.tools import update_model_dims
import argparse
import glob
from BenchmarkDatabase.elements import Operator, Operation
from OnnxUtils.symbolic_shape_infer import SymbolicShapeInference
import Runtimetools

def test_extraction():
    models =  []
    #for model in glob.glob("../../models/*.onnx"):
    for model in glob.glob("../../Runtimefiles/Ex_lm_LSTM.onnx"):
        models.append(model)

    for model in models:
        print("Testing extraction on: " + model + " with batchsize 64")
        extract_operations(model, 64)

    #https://pytorch.org/docs/stable/onnx.html#torch.onnx.export



def extract_operations(path: str, batch_size: int):
    """
    Extract all operations from an onnx specified network.
    :param path: Path to the .onnx file of the network.
    :param input_shape: The input shape for the model. This includes the batch size etc.
    :param output_shape: The output shape for the model. This includes the batch size etc.
    :return: List of operations that are in the network.
    This list can contain duplicates if operations are done multiple times!
    """
    model = onnx.load(path)
    print("Loading model from {} for extraction".format(path))
    onnx.checker.check_model(model)

    # TODO: Check, feels incorrect, not flexible for different input formats...
    input_node = model.graph.input[0]
    output_node = model.graph.output[0]

    # input_node.type.tensor_type.shape.dim[0].dim_value = batch_size
    # output_node.type.tensor_type.shape.dim[0].dim_value = batch_size

    inferred_model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred_model)

    int_max = 2 ** 31 - 1
    auto_merge = False
    guess_output_rank = False
    verbose = 0
    symbolic_shape_inference = SymbolicShapeInference(int_max, auto_merge, guess_output_rank, verbose)
    all_shapes_inferred = False
    symbolic_shape_inference._preprocess(inferred_model)
    while symbolic_shape_inference.run_:
        all_shapes_inferred = symbolic_shape_inference._infer_impl(inferred_model)
    symbolic_shape_inference._update_output_from_vi()
    inferred_model = symbolic_shape_inference.out_mp_

    onnx.save(inferred_model, path)

    # Code for variable sizes in definition:
    # model_input_dict = {}
    # for elem in model.graph.input:
    #     model_input_dict[elem.name] = [x.dim_value for x in elem.type.tensor_type.shape.dim]
    # model_output_dict = {}
    # for elem in model.graph.output:
    #     model_output_dict[elem.name] = [x.dim_value for x in elem.type.tensor_type.shape.dim]
    # variable_model = update_model_dims.update_inputs_outputs_dims(model, model_input_dict,
    #                                                               model_output_dict)



    # for node in model.graph.node:
    #    print(node)
    # print("Input:")
    # print(inferred_model.graph.input[0])
    # print("Output:")
    # print(inferred_model.graph.output[0])

    # Generate a dict of nodename:output shape
    output_shapes = {}
    for node in inferred_model.graph.value_info:
        try:
            output_shapes[node.name] = node.type.tensor_type.shape
            # print("Node {} output: {}".format(node.name, node.type.tensor_type.shape))
        except:
            print("Node doesn't follow the style: node.type.tensor_type.shape. Node content: {}".format(node))

    # for node in inferred_model.graph.node:
    #     print("Node " + node.name)
    #     print(node)
    #
    nodes_by_name = { x.name : x for x in inferred_model.graph.node}
    shapeinfo_by_name = { x.name : x for x in inferred_model.graph.value_info}

    # for node in set(nodes_by_name) - set(shapeinfo_by_name):
    #     print(node)
        # if("Mul_2" in node):
        #     print(nodes_by_name[node])
        #     print(shapeinfo_by_name[nodes_by_name[node].input[0]])
        #     return

    # print("Missing node shapes: {}".format(len(inferred_model.graph.node) - len(inferred_model.graph.value_info)))

    #return
        #print(node)
        #for output in node.output:
        #    print("{}".format(output))
            #print("{} Output: {} Input: {}".format(node.name, node.input, node.output))
    #
    #for node in inferred_model.graph.value_info:
    #    if node.name not in output_shapes:
    #        print(inferred_model.graph.output)
    #        output_overlap = next((x for x in inferred_model.graph.output if x.name == node.name), None)
    #        try:
    #            output_shapes[node.name] = output_overlap.type.tensor_type.shape
    #        except:
    #            print("Output not shaped as expected.")


    # Extract the constant value shapes
    constant_shapes = {}
    for initializer in inferred_model.graph.initializer:
        if hasattr(initializer, 'name'):
            constant_shapes[initializer.name] = initializer.dims

    operator_set = []
    operations_list = []

    for node in inferred_model.graph.node:
        # Generate a dict of nodeoutputname:[shape, shape, ...] for the inputs of each node.
        inlist = {}
        for input in node.input:
            if input in output_shapes:
                # intermediate input, connect to output of previous
                inlist[input] = output_shapes[input]
            else:
                if input not in constant_shapes:
                    # Non intermediate inputs, non constant inputs
                    # print("Searching for: " + input)
                    # print(inferred_model.graph.input)
                    input_overlap = next((x for x in inferred_model.graph.input if x.name == input), None)
                    try:
                        inlist[input_overlap.name] = input_overlap.type.tensor_type.shape
                    except:
                        print("Input not shaped as expected.")

        # inlist contains all non-constant inputs of the current node.
        if len(inlist) == 1:
            dims = list(inlist.values())[0].dim
            op = Operator(node.op_type, ())
            attribute_descriptors = []
            attribute_values = []
            for attrib in node.attribute:
                if attrib.type == onnx.AttributeProto.INTS:
                    attribute_descriptors += [attrib.name + "_" + str(i) for i, x in enumerate(attrib.ints)]
                    attribute_values += [x for x in attrib.ints]
                elif attrib.type == onnx.AttributeProto.FLOATS:
                    attribute_descriptors += [attrib.name + "_" + str(i) for i, x in enumerate(attrib.floats)]
                    attribute_values += [x for x in attrib.floats]
                elif attrib.type == onnx.AttributeProto.FLOAT:
                    attribute_descriptors += [attrib.name]
                    attribute_values += [attrib.f]
                elif attrib.type == onnx.AttributeProto.INT:
                    attribute_descriptors += [attrib.name]
                    attribute_values += [attrib.i]
                else:
                    print("Todo implement. Attribute value type not accepted: {}".format(attrib))
            op.parameter_descriptors = ["?" for _ in dims] + attribute_descriptors
            operation = Operation(op, [x.dim_value for x in dims] + attribute_values)
            operations_list.append(operation)
            # print("Added {}".format(operation))
            # Operator specific parameters. TODO: Improve?
            if(node.op_type == "Conv"):
                #Add Number of Filters to the parameters
                op.parameter_descriptors.insert(4, "K")
                operation.parameters.insert(4, output_shapes[node.output[0]].dim[1].dim_value)
                print(op)
        elif len(inlist) > 1:
            if node.op_type == "Add" and len(inlist) == 2:
                op = Operator("Add", ())
                op.parameter_descriptors = ["A", "B"]
                dim1 = [x.dim_value for x in list(inlist.values())[0].dim]
                dim2 = [x.dim_value for x in list(inlist.values())[1].dim]
                operations_list.append(Operation(op, [dim1, dim2]))
            else:
                op = Operator(node.op_type, ())
                op.parameter_descriptors = ["?"] * len(list(inlist.values()))
                dims = []
                for inp in list(inlist.values()):
                    dims.append([x.dim_value for x in inp.dim])
                operations_list.append(Operation(op, dims))
                #print("Todo implement. Multiple Inputs for node {}: {}".format(node.op_type, inlist))

    new_oplist = []
    for op in operations_list:
        if op.operator.name in Runtimetools.pytorch_op_unfolding:
            print("OP Extract: {}".format(op))
            added_ops = parse_strategies(op, Runtimetools.pytorch_op_unfolding)
        else:
            added_ops = [op]
        new_oplist += added_ops

#        print(op.operator)

#        print("Conversions: ")
#        print(node)
#        print("-------->")
        print("Operator:")
        print(operations_list[len(operations_list) - 1].operator)
        print("Parameters:")
        print(operations_list[len(operations_list) - 1].parameters)
    #for op in operations_list:
    #    print(op)
    return new_oplist

def parse_strategies(op, strategies):
    # print("parsing: {},{}".format(op.parameters, strategies))
    return_params = []
    for key in strategies:
        if key == op.operator.name:
            for strat in strategies[key]:
                # TODO: Extract only fitting strats!
                # print(strat)
                # print(key)
                [ls, rs] = strat.split(" -> ")
                # print(ls)
                # print(op.parameters)
                params = Runtimetools.TrainingBenchmark.flatten(op.parameters)
                # print(params)
                if len(ls.replace(',','')) != len(params):
                    # return_params += [op]
                    # print("Incompatible: {} <-> {}".format(op.parameters, strat))
                    continue
                else:
                    ls = ls.replace(',','')
                    # print("ls:{}".format(ls))
                    symbols = {}
                    idx = 0
                    for lit in ls:
                        if lit not in symbols:
                            symbols[lit] = idx
                            idx = idx + 1
                    values = [0] * idx
                    for i, v in enumerate(params):
                        values[symbols[ls[i]]] = v
                        #print("{} = {}".format(ls[i], v))
                    all_changed = False
                    added_symbol = 0
                    while not all_changed:
                        all_changed = True
                        for i, lit in enumerate(rs):
                            if lit == '*':
                                a = rs[i - 1]
                                b = rs[i + 1]
                                symbols[str(added_symbol)] = idx + added_symbol
                                values.append(values[symbols[a]] * values[symbols[b]])
                                all_changed = False
                                new_rs = rs[:i - 1] + str(added_symbol) + rs[i + 2:]
                                new_rs = new_rs.replace(a + '*' + b, str(added_symbol))
                                rs = new_rs
                                added_symbol = added_symbol + 1
                                break
                            if lit == "/":
                                a = rs[i - 1]
                                b = rs[i + 1]
                                symbols[str(added_symbol)] = idx + added_symbol
                                values.append(values[symbols[a]] / values[symbols[b]])
                                all_changed = False
                                new_rs = rs[:i - 1] + str(added_symbol) + rs[i + 2:]
                                new_rs = new_rs.replace(a + '/' + b, str(added_symbol))
                                rs = new_rs
                                added_symbol = added_symbol + 1
                                break
                    # print(rs)
                    sublists = rs.split(',')
                    #print(sublists)
                    params = []
                    for ls in sublists:
                        params.append([values[symbols[l]] for l in ls])

                        return_params.append(Operation(op.operator, params))

    #print("Reordered params:")
    #for el in return_params:
    #    print(el)
    # print(return_params)
    return return_params


if __name__ == '__main__':
    test_extraction()
