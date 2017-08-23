# First check the Python version
import sys
try:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    from skimage import data
    from scipy.misc import imresize
    import IPython.display as ipyd
except ImportError:
    print('You are missing some packages! ')

# Import Tensorflow
try:
    import tensorflow as tf
except ImportError:
    print("You do not have tensorflow installed!")

# This cell includes the provided libraries from the zip file
# and a library for displaying images from ipython, which
# we will use to display the gif
try:
    from libs import utils, gif, datasets, dataset_utils, vae, dft
except ImportError:
    print("Make sure you have started notebook in the same directory" +
          " as the provided zip file which includes the 'libs' folder" +
          " and the file 'utils.py' inside of it.  You will NOT be able"
          " to complete this assignment unless you restart jupyter"
          " notebook inside the directory created by extracting"
          " the zip file or cloning the github repo.")

# We'll tell matplotlib to inline any drawn figures like so:
plt.style.use('ggplot')






dst = 'gtzan_music_speech'
if not os.path.exists(dst):
    dataset_utils.gtzan_music_speech_download(dst)


# Get the full path to the directory
music_dir = os.path.join(os.path.join(dst, 'music_speech'), 'music_wav')

# Now use list comprehension to combine the path of the directory with any wave files
music = [os.path.join(music_dir, file_i)
         for file_i in os.listdir(music_dir)
         if file_i.endswith('.wav')]

# Similarly, for the speech folder:
speech_dir = os.path.join(os.path.join(dst, 'music_speech'), 'speech_wav')
speech = [os.path.join(speech_dir, file_i)
          for file_i in os.listdir(speech_dir)
          if file_i.endswith('.wav')]

# Let's see all the file names
print(music, speech)


file_i = speech[0]
print(speech[0])
s = utils.load_audio(file_i,b_normalize=True)


fft_size = 512
hop_size = 256
re, im = dft.dft_np(s, hop_size=256, fft_size=512)
mag, phs = dft.ztoc(re, im)
print(mag.shape)
plt.imshow(mag.T)

# The sample rate from our audio is 22050 Hz.
sr = 22050

# We can calculate how many hops there are in a second
# which will tell us how many frames of magnitudes
# we have per second

n_frames_per_second = sr // hop_size
print(n_frames_per_second)

# We want 500 milliseconds of audio in our window
n_frames = n_frames_per_second // 2

# And we'll move our window by 250 ms at a time
frame_hops = n_frames_per_second // 4

# We'll therefore have this many sliding windows:
n_hops = (len(mag) - n_frames) // frame_hops

# Store every magnitude frame and its label of being music: 0 or speech: 1
Xs, ys = [], []

# Let's start with the music files
for i in music:
    # Load the ith file:
    s = utils.load_audio(i)

    # Now take the dft of it (take a DSP course!):
    re, im = dft.dft_np(s, fft_size=fft_size, hop_size=hop_size)

    # And convert the complex representation to magnitudes/phases (take a DSP course!):
    mag, phs = dft.ztoc(re, im)

    # This is how many sliding windows we have:
    n_hops = (len(mag) - n_frames) // frame_hops

    # Let's extract them all:
    for hop_i in range(n_hops):

        # Get the current sliding window
        frames = mag[(hop_i * frame_hops):(hop_i * frame_hops + n_frames)]

        # We'll take the log magnitudes, as this is a nicer representation:
        this_X = np.log(np.abs(frames[..., np.newaxis]) + 1e-10)

        # And store it:
        Xs.append(this_X)

        # And be sure that we store the correct label of this observation:
        ys.append(0)

# Now do the same thing with speech (TODO)!
for i in speech:

    # Load the ith file:
    s = utils.load_audio(i)

    # Now take the dft of it (take a DSP course!):
    re, im =  dft.dft_np(s, fft_size=fft_size, hop_size=hop_size)

    # And convert the complex representation to magnitudes/phases (take a DSP course!):
    mag, phs = dft.ztoc(re, im)

    # This is how many sliding windows we have:
    n_hops = (len(mag) - n_frames) // frame_hops

    # Let's extract them all:
    for hop_i in range(n_hops):

        # Get the current sliding window
        frames = mag[(hop_i * frame_hops):(hop_i * frame_hops + n_frames)]

        # We'll take the log magnitudes, as this is a nicer representation:
        this_X = np.log(np.abs(frames[..., np.newaxis]) + 1e-10)

        # And store it:
        Xs.append(this_X)

        # Make sure we use the right label (TODO!)!
        ys.append(1)

# Convert them to an array:
Xs = np.array(Xs)
ys = np.array(ys)

print(Xs.shape, ys.shape)

# Just to make sure you've done it right.  If you've changed any of the
# parameters of the dft/hop size, then this will fail.  If that's what you
# wanted to do, then don't worry about this assertion.
assert(Xs.shape == (15360, 43, 256, 1) and ys.shape == (15360,))



n_observations, n_height, n_width, n_channels = Xs.shape

ds = datasets.Dataset(Xs=Xs, ys=ys, split=[0.8, 0.1, 0.1], one_hot=True)


Xs_i, ys_i = next(ds.train.next_batch())

# Notice the shape this returns.  This will become the shape of our input and output of the network:
print(Xs_i.shape, ys_i.shape)

assert(ys_i.shape == (100, 2))


tf.reset_default_graph()

# Create the input to the network.  This is a 4-dimensional tensor!
# Don't forget that we should use None as a shape for the first dimension
# Recall that we are using sliding windows of our magnitudes (TODO):
X = tf.placeholder(name='X', shape=[None, 43, 256, 1], dtype=tf.float32)

# Create the output to the network.  This is our one hot encoding of 2 possible values (TODO)!
Y = tf.placeholder(name='Y', shape=[None,2], dtype=tf.float32)


# TODO:  Explore different numbers of layers, and sizes of the network
n_filters = [9, 9, 9, 9]

# Now let's loop over our n_filters and create the deep convolutional neural network
H = X
for layer_i, n_filters_i in enumerate(n_filters):

    # Let's use the helper function to create our connection to the next layer:
    # TODO: explore changing the parameters here:
    H, W = utils.conv2d(
        H, n_filters_i, k_h=3, k_w=3, d_h=2, d_w=2,
        name=str(layer_i))

    # And use a nonlinearity
    # TODO: explore changing the activation here:
    H = tf.nn.relu(H)

    # Just to check what's happening:
    print(H.get_shape().as_list())


# Connect the last convolutional layer to a fully connected network (TODO)!
fc, W = utils.linear(H, n_output=100,name='W',activation=tf.nn.softmax)

# And another fully connected layer, now with just 2 outputs, the number of outputs that our
# one hot encoding has (TODO)!
Y_pred, W = utils.linear(fc, n_output=2,name='W2',activation=tf.nn.softmax)


loss = utils.binary_cross_entropy(Y_pred, Y)
cost = tf.reduce_mean(tf.reduce_sum(loss, 1))
predicted_y = tf.argmax(Y_pred, 1)
actual_y = tf.argmax(Y, 1)
correct_prediction = tf.equal(predicted_y, actual_y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)



# Explore these parameters: (TODO)
n_epochs = 10
batch_size = 100

# Create a session and init!
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Now iterate over our dataset n_epoch times
for epoch_i in range(n_epochs):
    print('Epoch: ', epoch_i)

    # Train
    this_accuracy = 0
    its = 0

    # Do our mini batches:
    for Xs_i, ys_i in ds.train.next_batch(batch_size):
        # Note here: we are running the optimizer so
        # that the network parameters train!
        this_accuracy += sess.run([accuracy, optimizer], feed_dict={
                X:Xs_i, Y:ys_i})[0]
        its += 1
        print(this_accuracy / its)
    print('Training accuracy: ', this_accuracy / its)

    # Validation (see how the network does on unseen data).
    this_accuracy = 0
    its = 0

    # Do our mini batches:
    for Xs_i, ys_i in ds.valid.next_batch(batch_size):
        # Note here: we are NOT running the optimizer!
        # we only measure the accuracy!
        this_accuracy += sess.run(accuracy, feed_dict={
                X:Xs_i, Y:ys_i})
        its += 1
    print('Validation accuracy: ', this_accuracy / its)



'''
g = tf.get_default_graph()
for layer_i in range(len(n_filters)):
    W = sess.run(g.get_tensor_by_name('{}/W:0'.format(layer_i)))
    plt.figure(figsize=(5, 5))
    plt.imshow(utils.montage_filters(W))
    plt.title('Layer {}\'s Learned Convolution Kernels'.format(layer_i))

'''
