import tensorflow as tf
model_name = "muzzle_detector"
path_to_model = f"built_graph/{model_name}.pb" 

def get_input_output_tensor_names(pb_file_path):
    """
    Get the input and output tensor names from a TensorFlow .pb file.

    Parameters:
        pb_file_path (str): Path to the input .pb file.

    Returns:
        input_tensor_name (str): Name of the input tensor.
        output_tensor_name (str): Name of the output tensor.
    """
    # Load the TensorFlow GraphDef from the .pb file
    with tf.io.gfile.GFile(pb_file_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Get the input and output tensor names from the graph
    input_tensor_name = graph_def.node[0].name
    output_tensor_name = graph_def.node[-1].name

    return input_tensor_name, output_tensor_name

# Example usage:
input_name, output_name = get_input_output_tensor_names(path_to_model)
print("Input Tensor Name:", input_name)
print("Output Tensor Name:", output_name)
