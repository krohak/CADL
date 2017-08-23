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

#plt.plot(s)


'''
# Parameters for our dft transform.  Sorry we can't go into the
# details of this in this course.  Please look into DSP texts or the
# course by Perry Cook linked above if you are unfamiliar with this.
fft_size = 512
hop_size = 256

re, im = dft.dft_np(s, hop_size=256, fft_size=512)
mag, phs = dft.ztoc(re, im)
print(mag.shape)
plt.imshow(mag.T)
'''


'''
plt.figure(figsize=(10, 4))
plt.imshow(np.log(mag.T))
plt.xlabel('Time')
plt.ylabel('Frequency Bin')
'''


'''
We could just take just a single row (or column in the second plot of the magnitudes just above, as we transposed it in that plot) as an input 
to a neural network. However, that just represents about an 80th of a second of audio data, and is not nearly enough data to say whether 
something is music or speech. We'll need to use more than a single row to get a decent length of time. One way to do this is to use a sliding 
2D window from the top of the image down to the bottom of the image (or left to right). Let's start by specifying how large our sliding window 
is.
'''

'''
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
'''

'''
Xs = []
ys = []
for hop_i in range(n_hops):
    # Creating our sliding window
    frames = mag[(hop_i * frame_hops):(hop_i * frame_hops + n_frames)]
    
    # Store them with a new 3rd axis and as a logarithmic scale
    # We'll ensure that we aren't taking a log of 0 just by adding
    # a small value, also known as epsilon.
    Xs.append(np.log(np.abs(frames[..., np.newaxis]) + 1e-10))
    
    # And then store the label 
    ys.append(0)
print(Xs, ys)
'''

'''
The code below will perform this for us, as well as create the inputs and outputs to our classification network by specifying 0s for the music 
dataset and 1s for the speech dataset. Let's just take a look at the first sliding window, and see it's label:
In [45]:
'''

'''
plt.imshow(Xs[0][..., 0])
plt.title('label:{}'.format(ys[9]))
'''


'''
Since this was the first audio file of the music dataset, we've set it to a label of 0. And now the second one, which should have 50% overlap 
with the previous one, and still a label of 0:
In [44]:
'''
'''
plt.imshow(Xs[1][..., 0])
plt.title('label:{}'.format(ys[1]))
'''


