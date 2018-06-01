# OLD_CHECKPOINT_FILE = "models/cnn_deep_dnn/model282.ckpt"
# NEW_CHECKPOINT_FILE = "models/fix/model282.ckpt"
OLD_CHECKPOINT_FILE = "models/cnn_lstm/model183.ckpt"
NEW_CHECKPOINT_FILE = "models/fix/model183.ckpt"

import tensorflow as tf
vars_to_rename = {
    "lstm_2/lstm_cell/weights" : "cnn_lstm/lstm_2/lstm_cell/kernel",
    "filter_5" : "cnn_lstm/filter_5",
    "lstm_2/lstm_cell/biases" : "cnn_lstm/lstm_2/lstm_cell/bias",
    "filter_3/Adam" : "cnn_lstm/cnn_lstm/filter_3/Adam",
    "filter_7" : "cnn_lstm/filter_7",
    "cnn_embed/weight_input_layer/Adam_1" : "cnn_lstm/cnn_lstm/cnn_embed/weight_input_layer/Adam_1",
    "cnn_embed/weight_input_layer/Adam" : "cnn_lstm/cnn_lstm/cnn_embed/weight_input_layer/Adam",
    "cnn_embed/bias_input_layer/Adam_1" : "cnn_lstm/cnn_lstm/cnn_embed/bias_input_layer/Adam_1",
    "cnn_embed/bias_input_layer" : "cnn_lstm/cnn_embed/bias_input_layer",
    "filter_3/Adam_1" : "cnn_lstm/cnn_lstm/filter_3/Adam_1",
    "filter_3" : "cnn_lstm/filter_3",
    "output_layer/weight_output_layer/Adam" : "cnn_lstm/cnn_lstm/output_layer/weight_output_layer/Adam",
    "output_layer/bias_output_layer" : "cnn_lstm/output_layer/bias_output_layer",
    "filter_7/Adam" : "cnn_lstm/cnn_lstm/filter_7/Adam",
    "beta2_power" : "cnn_lstm/beta2_power",
    "filter_5/Adam_1" : "cnn_lstm/cnn_lstm/filter_5/Adam_1",
    "output_layer/weight_output_layer/Adam_1" : "cnn_lstm/cnn_lstm/output_layer/weight_output_layer/Adam_1",
    "output_layer/bias_output_layer/Adam_1" : "cnn_lstm/cnn_lstm/output_layer/bias_output_layer/Adam_1",
    "lstm_1/lstm_cell/weights" : "cnn_lstm/lstm_1/lstm_cell/kernel",
    "filter_5/Adam" : "cnn_lstm/cnn_lstm/filter_5/Adam",
    "cnn_embed/bias_input_layer/Adam" : "cnn_lstm/cnn_lstm/cnn_embed/bias_input_layer/Adam",
    "lstm_0/lstm_cell/biases" : "cnn_lstm/lstm_0/lstm_cell/bias",
    "beta1_power" : "cnn_lstm/beta1_power",
    "lstm_0/lstm_cell/weights" : "cnn_lstm/lstm_0/lstm_cell/kernel",
    "cnn_embed/weight_input_layer" : "cnn_lstm/cnn_embed/weight_input_layer",
    "filter_7/Adam_1" : "cnn_lstm/cnn_lstm/filter_7/Adam_1",
    "output_layer/weight_output_layer" : "cnn_lstm/output_layer/weight_output_layer",
    "lstm_1/lstm_cell/biases" : "cnn_lstm/lstm_1/lstm_cell/bias",
    "output_layer/bias_output_layer/Adam" : "cnn_lstm/cnn_lstm/output_layer/bias_output_layer/Adam"
}
# vars_to_rename = {}
new_checkpoint_vars = {}
reader = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE)
for old_name in reader.get_variable_to_shape_map():
    print(old_name)
    if old_name in vars_to_rename:
        new_name = vars_to_rename[old_name]
    else:
        new_name = old_name
    new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))

init = tf.global_variables_initializer()
saver = tf.train.Saver(new_checkpoint_vars)

with tf.Session() as sess:
    sess.run(init)
    saver.save(sess, NEW_CHECKPOINT_FILE)