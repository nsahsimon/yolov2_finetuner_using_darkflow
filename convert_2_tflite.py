import tensorflow as tf
model_name = "muzzle_detector"
path_to_model = f"built_graph/{model_name}.pb" 


# gf = tf.compat.v1.GraphDef()
# m_file = open(path_to_model,'rb')
# for n in gf.node:
#     print( n.name )

    
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file = path_to_model, 
    input_arrays = ['input'],
    output_arrays = ['output'] 
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with tf.io.gfile.GFile(f'{model_name}.tflite', 'wb') as f:
  f.write(tflite_model)