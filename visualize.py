from vis.utils import utils
from keras import activations
from vis.visualization import visualize_saliency,visualize_activation


def prep_model_output(model):
	# Utility to search for layer index by name. 
	# Alternatively we can specify this as -1 since it corresponds to the last layer.
	layer_idx = -1

	# Swap softmax with linear
	model.layers[layer_idx].activation = activations.linear
	model = utils.apply_modifications(model)

	# This is the output node we want to maximize.
	filter_idx = 0
	return(model)

def visualize_activation_wrapper(model):
	model = prep_model_output(model)
	img = visualize_activation(model=model, layer_idx=layer_idx, filter_indices=filter_idx)
	return(img)

def visualize_activation_wrapper(model,seed_input):
	model = prep_model_output(model)
	img = visualize_saliency(model, layer_idx, filter_indices=filter_idx, seed_input=seed_input)
	return(img)
