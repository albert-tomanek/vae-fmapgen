from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations
from keras.models import load_model
from matplotlib import pyplot as plt

from main import GANModel, AutoEncoder

plt.rcParams['figure.figsize'] = (12, 12)
encoder = load_model('weights_enc.h5')

for filter_no in range(64):
    print("Visualizing filter", filter_no)
    img = visualize_activation(encoder, 2, filter_indices=filter_no, max_iter=500, verbose=False)
    plt.subplot(8, 64//8, filter_no+1)
    plt.imshow(img.reshape(img.shape[:-1]), cmap='Greys')

print('\nSaving image...')
plt.savefig('activ.png')
