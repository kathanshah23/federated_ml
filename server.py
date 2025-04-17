import numpy as np
import tensorflow as tf

def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model

def average_weights(weight_list):
    avg_weights = list()
    for weights in zip(*weight_list):
        avg_weights.append(np.mean(weights, axis=0))
    return avg_weights

client_weights = []
input_shape = 10

global_model = create_model(input_shape)
global_model.build(input_shape=(None, input_shape))

for i in range(3):
    client_model = create_model(input_shape)
    client_model.build(input_shape=(None, input_shape))
    client_model.set_weights(global_model.get_weights())
    updated_weights = [w + np.random.normal(0, 0.01, w.shape) for w in client_model.get_weights()]
    client_weights.append(updated_weights)

aggregated_weights = average_weights(client_weights)
global_model.set_weights(aggregated_weights)

global_model.save("global_model.h5")
print("Global model updated and saved.")
