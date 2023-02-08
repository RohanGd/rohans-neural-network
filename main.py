from neuralnetwork import *
import numpy as np
# def main():
#     print("Hello World!")
#     weights = np.random.uniform(size=(3,2))
#     inputs = np.random.uniform(size=(2,1))
#     biases = np.random.uniform(size=(3,1))
#     print('\nWEIGHTS: ')
#     print(weights)
#     print('\nINPUTS: ')
#     print(inputs)
#     print('\nBIASES: ')
#     print(biases)


#     c = np.add(np.matmul(weights, inputs), biases)
#     print('\nFIRST FORWARD PASS:')
#     myfunc(c)

#     print(c.shape[1])

#     k = [1,2,3]
#     print(str(type(k)))

# def myfunc(t : np.array):
#     print(t)
    
def main():

    # t = np.array([2,3,1])
    # print(t)
    # print()
    # t = t.reshape((len(t),-1))
    # print(t)
    # print(t.shape)
    nn = neuralnetwork([2,3,1])
    t = nn.feedForward([1,2])
    print(t)

if __name__ == '__main__':
    main()


'''
input = [[1] 
         [5]]
output = [6]

'''