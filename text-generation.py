import numpy as np
import tensorflow as tf

print("Setup")
filename = './wonderland.txt'

data = open(filename, 'r').read()
unique_chars = list(set(data))
no_unique_chars = len(unique_chars)

num_to_char = {i:char for i, char in enumerate(unique_chars)}
char_to_num = {char:i for i, char in enumerate(unique_chars)}

no_of_features = no_unique_chars
length_of_sequence = 100
no_of_hidden = 700
no_of_layers = 10
generate_text_length = 100
print_per = 1
no_of_epochs = 1
no_of_sequences = int(len(data)/length_of_sequence)
print("no of sequences: {}".format( no_of_sequences))

X = np.zeros((int(len(data)/length_of_sequence), length_of_sequence, no_of_features))
Y = np.zeros((int(len(data)/length_of_sequence), length_of_sequence, no_of_features))

# data generation
print("Data Generation")
for i in range(0, int(len(data)/length_of_sequence)):
    X_sequence = data[i*length_of_sequence:(i+1)*length_of_sequence]
    X_sequence_in_num = [char_to_num[char] for char in X_sequence]

    input_sequence = np.zeros((length_of_sequence, no_of_features))

    for j in range(length_of_sequence):
        input_sequence[j][X_sequence_in_num[j]] = 1
    X[i] = input_sequence

    Y_sequence = data[i*length_of_sequence+1:(i+1)*length_of_sequence+1]
    Y_sequence_in_num = [char_to_num[char] for char in Y_sequence]
    output_sequence = np.zeros((length_of_sequence, no_of_features))
    for j in range(length_of_sequence):
        output_sequence[j][Y_sequence_in_num[j]] = 1
    Y[i] = output_sequence

# create model
print("Creating Model")
batchX_placeholder = tf.placeholder(tf.float32, [length_of_sequence, no_of_features])
batchY_placeholder = tf.placeholder(tf.int32, [length_of_sequence, no_of_features])

cell_state = tf.placeholder(tf.float32, [length_of_sequence, no_of_hidden])
hidden_state = tf.placeholder(tf.float32, [length_of_sequence, no_of_hidden])
init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

W2 = tf.Variable(np.random.rand(no_of_hidden, no_of_features), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, no_of_features)), dtype=tf.float32)

inputs_series = tf.split(batchX_placeholder, no_of_features, 1)
labels_series = tf.unstack(batchY_placeholder, axis=1)
cell = tf.nn.rnn_cell.BasicLSTMCell(no_of_hidden, state_is_tuple=True)
states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series, init_state)

logits_series = [tf.matmul(state, W2) + b2 for state in states_series]
prediction_series = [tf.nn.softmax(logits) for logits in logits_series]
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series, labels_series)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


loss_list = []
print("Run")
def generate_text(sess, generate_text_length):
    generated_num = [np.random.randint(no_unique_chars)]
    generated_char = [num_to_char[generated_num[-1]]]
    X = np.zeros((1, generate_text_length, no_unique_chars))
    for i in range(generate_text_length):
        X[0][i][generated_num[-1]] = 1
        print(num_to_char[generated_num[-1]], end="")
        generated_num = np.argmax(sess.run(prediction_series, feed_dict ={batchX_placeholder: X[:, :i+1, :]})[0], 1)
        generated_char.append(num_to_char[generated_num[-1]])
    return "".join(generated_char)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(no_of_epochs):
    _current_cell_state = np.zeros((length_of_sequence, no_of_hidden))
    _current_hidden_state = np.zeros((length_of_sequence, no_of_hidden))
    for sequence_idx in range(no_of_sequences):
        _total_loss, _train_step, _current_state, _prediction_series = sess.run(
                [total_loss, train_step, current_state, prediction_series],
                feed_dict = {
                    batchX_placeholder: X[sequence_idx],
                    batchY_placeholder: Y[sequence_idx],
                    cell_state: _current_cell_state,
                    hidden_state: _current_hidden_state
                    }
                )
        loss_list.append(_total_loss)
        if sequence_idx%print_per == 0:
            print("Epoch {}, Sequence {}, Total Loss {}".format(epoch, sequence_idx, _total_loss))

saver = tf.train.Saver()
save_path = saver.save(sess, "./model.ckpt")
print("Model saved in %s" % save_path)
#sess.close()

