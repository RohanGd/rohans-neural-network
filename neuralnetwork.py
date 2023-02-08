import numpy as np
class neuralnetwork:
    def __init__(self,_topology: list,) -> None:
        self.topology = _topology
        sizes_of_wtmat = []
        # geting size of wt matrix. If topolgy is [2,3,1] wt matrix should have weights of dimensions: [(3x2),(1x3)]
        for i in range(len(_topology)-1):
            sizes_of_wtmat.append((_topology[i+1], _topology[i]))

        # weight matrix will contain all weights of all layers
        self.weight_matrix = []
        for _size in sizes_of_wtmat:
            self.weight_matrix.append(np.random.uniform(size=_size))

        # bias matrix of size of the next layer, for topology[2,3,1] bias matrix is of size[(3x1), (1x1)]
        self.bias_matrix = []
        for i in range(1,len(_topology)):
            self.bias_matrix.append(np.random.uniform(size=(_topology[i],1)))

        # genrating valuyes for nodes, call it value_matrix
        self.value_matrix = []
        for i in range(1,len(_topology)):
            self.value_matrix.append(np.random.uniform(size=(_topology[i],1)))

    def forward(self, weights, values, biases, output):
        assert(weights.shape[1] == values.shape[0])
        assert(weights.shape[0] == biases.shape[0])
        assert(biases.shape == output.shape)
        output = np.add(np.matmul(weights, values), biases)
        output = self.softmax(output)

    def feedForward(self, inputs):
        if type(inputs) == list:
            n = len(inputs)
            inputs = np.array(inputs)
            inputs = inputs.reshape((n,-1))
        for i in range(len(self.topology)-1):
            print(i)
            if i == 0:
                self.forward(self.weight_matrix[0], inputs, self.bias_matrix[0], self.value_matrix[0])
            else:
                self.forward(self.weight_matrix[i], self.value_matrix[i-1], self.bias_matrix[i], self.value_matrix[i])
        
        return self.value_matrix[-1]

    # defining our activation function
    def softmax(self,layer):
        exps = np.exp(layer)
        return exps / exps.sum()

    # defining our cost function
    def mae(self, prediction, output):
        res = output - prediction 
        res = res ** 2
        res = res / 2
        return res 
    




