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
no_of_epochs = 5
pkeep=0.8
batch_size = 1
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

batchX_placeholder = tf.placeholder(tf.int32, [batch_size, no_of_sequences, no_of_features], name='X')
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, no_of_sequences, no_of_features], name ='Y')

Hin = tf.placeholder(tf.float32, [None, no_of_hidden*no_of_layers], name='Hin')

cells = [tf.nn.rnn_cell.BasicLSTMCell(int(no_of_hidden/2), state_is_tuple=False) for _ in range(no_of_layers)]
dropcells = [tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in cells]
multicell = tf.nn.rnn_cell.MultiRNNCell(dropcells, state_is_tuple=False)
multicell = tf.nn.rnn_cell.DropoutWrapper(multicell, output_keep_prob=pkeep)  
Yr, H = tf.nn.dynamic_rnn(multicell, tf.to_float(batchX_placeholder), dtype=tf.float32, initial_state=Hin)

Yflat = tf.reshape(Yr, [-1, int(no_of_hidden/2)])
Ylogits = tf.contrib.layers.linear(Yflat, no_of_features)
# https://github.com/martin-gorner/tensorflow-rnn-shakespeare/blob/master/rnn_train.py
break
W = tf.get_variable("W", [no_of_hidden, no_of_features])
b = tf.get_variable("b", [no_of_features])
embedding = tf.get_variable("embedding", [no_of_features, no_of_hidden])
inputs = tf.nn.embedding_lookup(embedding, batchX_placeholder)

init_state_placeholder = tf.nn.rnn_cell.BasicLSTMCell.zero_state(batch_size, tf.float32)
state_per_layer_list = tf.unstack(init_state_placeholder, axis=0)
rnn_tuple_state = tuple(
    [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
     for idx in range(no_of_layers)]
)

W2 = tf.Variable(np.random.rand(no_of_hidden, no_of_features), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, no_of_features)), dtype=tf.float32)

inputs_series = tf.split(batchX_placeholder, 1)
labels_series = tf.unstack(batchY_placeholder, axis=1) #tf.split(batchY_placeholder, 1)
cell = tf.nn.rnn_cell.BasicLSTMCell(no_of_hidden, state_is_tuple=True)
stacked_lstm = tf.contrib.rnn.MultiRNNCell(
    [ tf.contrib.rnn.BasicLSTMCell(no_of_hidden) for _ in range(no_of_layers) ]
, state_is_tuple=True)
states_series, current_state = tf.contrib.rnn.static_rnn(stacked_lstm, inputs_series, rnn_tuple_state)

logits_series = [tf.matmul(state, W2) + b2 for state in states_series]
prediction_series = [tf.nn.softmax(logits) for logits in logits_series]
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series, labels_series)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


def vector_to_letter(vector):
    return num_to_char[np.argmax(vector)]

def letter_to_vector(letter):
    result = np.zeros(no_of_features)
    result[char_to_num[letter]] = 1
    return result

def generate_text(sess, generate_text_length):
    generated_num = np.random.randint(no_unique_chars)
    generated_char = num_to_char[generated_num]
    _current_state = np.zeros((no_of_layers, 2, 1, no_of_hidden))
    input_vector = letter_to_vector(generated_char)
    for i in range(generate_text_length):
        print(vector_to_letter(input_vector), end="")
        _current_state, _prediction_series = sess.run(
            [current_state, prediction_series], 
            feed_dict = {
                batchX_placeholder: [input_vector],
                init_state_placeholder: _current_state
            }
        )
        input_vector = letter_to_vector(vector_to_letter(_prediction_series[0]))

print("Run")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch_idx in range(no_of_epochs):
    for sequence_idx in range(no_of_sequences):
        _current_state = np.zeros((no_of_layers, 2, 1, no_of_hidden))
        input = X[sequence_idx]
        output = Y[sequence_idx]
        loss_list = []
        for char_idx in range(len(input)):
            _total_loss, _train_step, _current_state, _prediction_series = sess.run(
                [total_loss, train_step, current_state, prediction_series],
                feed_dict = {
                    batchX_placeholder: [input[char_idx]],
                    batchY_placeholder: [output[char_idx]],
                    init_state_placeholder: _current_state
                }
            )
            #print(vector_to_letter(_prediction_series[0]), end="")
            #print("["+vector_to_letter(output[char_idx])+"]",end="")
            
            loss_list.append(_total_loss)

        if sequence_idx%print_per == 0:
            print("Epoch {}, Sequence {}, Total Loss {}".format(epoch_idx, sequence_idx, np.mean(loss_list)))

saver = tf.train.Saver()
save_path = saver.save(sess, "./model.ckpt")
print("Model saved in %s" % save_path)
#sess.close()

