import numpy as np
import tensorflow as tf

print("Setup")
filename = './wonderland.txt'

data = open(filename, 'r').read()
unique_chars = sorted(list(set(data)))
no_unique_chars = len(unique_chars)

num_to_char = {i:char for i, char in enumerate(unique_chars)}
char_to_num = {char:i for i, char in enumerate(unique_chars)}

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

no_of_features = no_unique_chars
length_of_sequence = 100
no_of_hidden = 700
no_of_layers = 3
generate_text_length = 100
print_per = 1
no_of_epochs = 5
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

batchX_placeholder = tf.placeholder(tf.float32, [None, length_of_sequence, no_unique_chars], name='X')
batchY_placeholder = tf.placeholder(tf.float32, [None, length_of_sequence, no_unique_chars], name='Y')
pkeep = tf.placeholder(tf.float32, name='pkeep')

Hin = tf.placeholder(tf.float32, [None, no_of_hidden*no_of_layers], name='Hin')

cells = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(no_of_hidden), input_keep_prob=pkeep) for _ in range(no_of_layers)]
multicell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=False), output_keep_prob=pkeep)
Yr, Hout = tf.nn.dynamic_rnn(multicell, batchX_placeholder, dtype=tf.float32, initial_state=Hin)

Yfloat = tf.reshape(Yr, [-1, no_of_hidden])
Ylogits = tf.contrib.layers.linear(Yfloat, no_unique_chars)
Ysoftmax = tf.nn.softmax(Ylogits)

Yinput_flat = tf.reshape(batchY_placeholder, [-1, no_unique_chars])
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels = Yinput_flat)


train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for epoch in range(20):
    inH = np.zeros([batch_size, no_of_hidden*no_of_layers])
    for i in range(no_of_sequences-1):
        print("{}/{} in epoch {}".format(i, no_of_sequences-1, epoch), end="\r")
        batchX = [X[i]] 
        batchY = [Y[i]]
        _, softmaxY, outH = sess.run([train_step, Ysoftmax, Hout], feed_dict = {batchX_placeholder: batchX, batchY_placeholder: batchY, Hin: inH, pkeep: 0.7})    
        inH = outH


