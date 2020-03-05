from vis.utils import utils
from keras import activations
from vis.visualization import visualize_saliency,visualize_activation


def prep_model_output(model):
	# https://github.com/raghakot/keras-vis/issues/119
	# Utility to search for layer index by name. 
	# Alternatively we can specify this as -1 since it corresponds to the last layer.
	layer_idx = -1

	# Swap softmax with linear
	model.layers[layer_idx].activation = activations.linear
	model = utils.apply_modifications(model)

	# This is the output node we want to maximize.
	return(model)

def visualize_activation_wrapper(model,filter_indices=0):
	model = prep_model_output(model)
	img = visualize_activation(model=model, layer_idx=-1, filter_indices=filter_indices)
	return(img)

def visualize_saliency_wrapper(model,seed_input,filter_indices=0):
	model = prep_model_output(model)
	img = visualize_saliency(model=model, layer_idx=-1, filter_indices=filter_indices, seed_input=seed_input)
	return(img)
