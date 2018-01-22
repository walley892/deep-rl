from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, ELU,LeakyReLU
from keras.optimizers import SGD, Adam

class CNN:
    def __init__(self, input_shape = (250,250, 3), n_out = 2, conv_layer_filters = None, conv_layer_sizes = None, conv_layer_strides = None, conv_activations = None, dense_layer_sizes=None, dense_activations = None, loss = 'mse', optimizer=Adam(lr = 0.01, decay = 0.1)):
        self.model = Sequential()
        

        if conv_layer_filters == None:
            #conv_layer_filters = [16 for _ in range(4)]
            conv_layer_filters = [64,32]
        if conv_layer_sizes == None:
            conv_layer_sizes = [1, 2, 1, 2, 1]
            
        if conv_layer_strides == None:
            conv_layer_strides = [2, 1, 2, 2, 2]
        
        if conv_activations == None:
            conv_activations = [[Activation('relu'), Activation('relu')] for _ in range(5)]
        
        if dense_layer_sizes == None:
            dense_layer_sizes = [800 for _ in range(2)]
        if dense_activations == None:
            dense_activations = ['relu' for _ in range(1)]
        dense_activations.append('linear')



        for i in range(len(conv_layer_filters)):

            if i == 0:
                self.model.add(Conv2D(input_shape = input_shape, filters = conv_layer_filters[0], kernel_size = conv_layer_sizes[0], strides = conv_layer_strides[0]))

            else:
                 self.model.add(Conv2D( padding = 'same', filters = conv_layer_filters[i], kernel_size = conv_layer_sizes[i], strides = conv_layer_strides[i]))
#            self.model.add(conv_activations[i][0])
            self.model.add(Conv2D( padding = 'same', filters = conv_layer_filters[i], kernel_size = conv_layer_sizes[i], strides = conv_layer_strides[i]))
 
#            self.model.add(conv_activations[i][1])
            self.model.add(MaxPooling2D())
        self.model.add(Flatten())


        for i in range(len(dense_layer_sizes)-1):
            self.model.add(Dense(dense_layer_sizes[i],activation =  dense_activations[i]))


        self.model.add(Dense(n_out, activation = dense_activations[-1]))

        self.model.compile(loss = loss, optimizer = optimizer)
    

    def save(self, path):
        self.model.save(path)
    def load(self, path):
        self.model = load_model(path)
    def train(self, data, labels, epochs = 10, batch_size = 32):
        self.model.fit(data, labels, epochs = epochs, batch_size = batch_size)
    def predict(self, data):
        return self.model.predict(data)


class FCNN:
    def __init__(self, n_in, n_out, dense_layers = None, dense_activations = None, loss = 'mse', optimizer = 'adagrad'):
        self.model = Sequential()
        

        if dense_layers == None:
            dense_layers = [200 for i in range(5)]
        if dense_activations == None:
            dense_activations = ['relu' for _ in range(4)]
        dense_activations.append('linear')

        self.model.add(Dense(dense_layers[0], activation = dense_activations[0], input_dim = n_in ) )

        for i in range(1, len(dense_layers)-1):
            self.model.add(Dense(dense_layers[i], activation = dense_activations[i]))
        self.model.add(Dense(n_out, activation = dense_activations[-1]))
        self.model.compile(loss = loss, optimizer = optimizer)

    def save(self, path):
        self.model.save(path)
    def load(self, path):
        self.model = load_model(path)
    def train(self, data, labels, epochs = 10, batch_size = 32):
        self.model.fit(data, labels, epochs = epochs, batch_size = batch_size)
    def predict(self, data):
        return self.model.predict( data)


