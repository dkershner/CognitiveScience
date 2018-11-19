# Some potentially useful modules
# Whether or not you use these (or others) depends on your implementation!
import random
import numpy
import math
import matplotlib.pyplot as plt

class NeuralMMAgent(object):
    '''
    Class to for Neural Net Agents
    '''

    def __init__(self, num_in_nodes, num_hid_nodes, num_hid_layers, num_out_nodes, \
                learning_rate = 0.2, max_epoch=10000, max_sse=.01, momentum=0.5, \
                creation_function=None, activation_function=None, random_seed=1):
        '''
        Arguments:
            num_in_nodes -- total # of input layers for Neural Net
            num_hid_nodes -- total # of hidden nodes for each hidden layer
                in the Neural Net
            num_hid_layers -- total # of hidden layers for Neural Net
            num_out_nodes -- total # of output layers for Neural Net
            learning_rate -- learning rate to be used when propagating error
            creation_function -- function that will be used to create the
                neural network given the input
            activation_function -- list of two functions:
                1st function will be used by network to determine activation given a weighted summed input
                2nd function will be the derivative of the 1st function
            random_seed -- used to seed object random attribute.
                This ensures that we can reproduce results if wanted
        '''
        assert num_in_nodes > 0 and num_hid_layers > 0 and num_hid_nodes and\
            num_out_nodes > 0, "Illegal number of input, hidden, or output layers!"

        rand_obj = random.Random()
        rand_obj.seed(random_seed)
        self.num_in_nodes = num_in_nodes
        self.num_hid_nodes = num_hid_nodes
        self.num_hid_layers = num_hid_layers
        self.num_out_nodes = num_out_nodes
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.max_sse = max_sse
        self.momentum = momentum
        self.creation_function = creation_function
        self.activation_function = activation_function
        self.random_seed = random_seed

        self.weights, self.weightDeltas, self.activations, self.errors, \
            self.thetas, self.thetaDeltas = \
            self.create_neural_structure(num_in_nodes, num_hid_nodes, num_hid_layers, num_out_nodes, rand_obj)

        #array to track little deltas for each node
        self.little_deltas = [[0 for x in range(num_in_nodes)], \
            [0 for x in range(num_hid_nodes)] * num_hid_layers, [0 for x in range(num_out_nodes)]]

    def train_net(self, input_list, output_list, max_num_epoch=100000, \
                    max_sse=0.1):
        ''' Trains neural net using incremental learning
            (update once per input-output pair)
            Arguments:
                input_list -- 2D list of inputs
                output_list -- 2D list of outputs matching inputs
        '''
        
        all_err = []
        
        for x in range(max_num_epoch):
            for y in range(len(input_list)):
                #setup activations by feeding forward
                self._feedForward(input_list[y])
                self._calculate_deltas(output_list[y])
                #print("activations: ", self.activations)
                #print("weight deltas: ", self.weightDeltas)
                #print("error grad: ", self.little_deltas)
                self._adjust_weights_thetas()
                #print("weights: ", self.weights)

            total_err = 0
            for err in self.errors:
                total_err += err**2

            #print(total_err)
            self.errors = []

            all_err.append(total_err)


            if (total_err < max_sse):
                break

    			#Show us how our error has changed
        plt.plot(all_err)
        plt.show()

        #print(self.all_err)


    def _calculate_deltas(self, output_list):
        '''Used to calculate all weight deltas for our neural net
            Arguments:
                out_error -- output error (typically SSE), obtained using target
                    output and actual output
        '''
        #Calculate error gradient for each output node & propgate error
        #   (calculate weight deltas going backward from output_nodes)
        for i in range(len(self.activations) - 1, 0, -1):
            # if its the weights to the outputs
            if i == len(self.activations) - 1:
                #loop through each output node
                for j in range(len(self.activations[i])):
                    error = output_list[j] - self.activations[i][j]
                    self.errors.append(error)
                    little_delta = self.sigmoid_af_deriv(self.activations[i][j]) * error
                    self.little_deltas[i][j] = little_delta
                    #loop through previous layer to update weight deltas coming from it to output nodes
                    for k in range(len(self.activations[i-1])):
                        big_delta = self.learning_rate * self.activations[i-1][k] * little_delta + self.momentum * self.weightDeltas[i-1][k*self.num_out_nodes + j]
                        self.weightDeltas[i-1][k*self.num_out_nodes + j] = big_delta
            else:
                #loop through each node
                for j in range(len(self.activations[i])):
                    little_deltas_sum = 0
                    for x in range(len(self.little_deltas[i + 1])):
                        little_deltas_sum += self.little_deltas[i+1][x] * self.weights[i][x*len(self.little_deltas[i+1]) + j]
                    little_delta = self.sigmoid_af_deriv(self.activations[i][j]) * little_deltas_sum
                    self.little_deltas[i][j] = little_delta
                    #loop through previous layer to update weight deltas coming from it to output nodes
                    for k in range(len(self.activations[i-1])):
                        big_delta = self.learning_rate * self.activations[i-1][k] * little_delta + self.momentum * self.weightDeltas[i-1][k*self.num_hid_nodes + j]
                        self.weightDeltas[i-1][k*self.num_hid_nodes + j] = big_delta


    def _adjust_weights_thetas(self):
        '''Used to apply deltas
        '''
        #apply all weight deltas to current weights
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] += self.weightDeltas[i][j]
                #zero out the weight deltas once its been applied just in case
                #self.weightDeltas[i][j] = 0

        #apply all theta deltas to current thetas
        for i in range(len(self.thetas)):
            for j in range(len(self.thetas[i])):
                self.thetas[i][j] += self.thetaDeltas[i][j]
                #zero out the theta deltas once its been applied just in case
                self.thetaDeltas[i][j] = 0

    def _feedForward(self, input_list):
        #set inputs
        self.activations[0] = input_list
        #get each layer (list of nodes)
        for i in range(1, len(self.activations)):
            #get each node in the layer
            for node in range(len(self.activations[i])):
                sum = 0
                #loop through nodes in previous layer
                for index in range(len(self.activations[i-1])):
                    sum += self.activations[i-1][index] * self.weights[i-1][index*len(self.activations[i])+node]
                self.activations[i][node] = self.sigmoid_af(sum)


    @staticmethod
    def create_neural_structure(num_in, num_hid, num_hid_layers, num_out, rand_obj):
        ''' Creates the structures needed for a simple backprop neural net
        This method creates random weights [-0.5, 0.5]
        Arguments:
            num_in -- total # of input nodes for Neural Net
            num_hid -- total # of hidden nodes for each hidden layer
                in the Neural Net
            num_hid_layers -- total # of hidden layers for Neural Net
            num_out -- total # of output nodes for Neural Net
            rand_obj -- the random object that will be used to selecting
                random weights
        Outputs:
            Tuple w/ the following items
                1st - 2D list of initial weights
                2nd - 2D list for weight deltas
                3rd - 2D list for activations
                4th - 2D list for errors
                5th - 2D list of thetas for threshold
                6th - 2D list for thetas deltas
        '''
        # setup the initial weights array
        weights = []
         #first row of weights from inputs to first hidden layers
        weights.append([rand_obj.uniform(-.5, .5) for x in range(num_in * num_hid)])
        #weights between hidden layers 1 to num_hid_layers
        weights += [[rand_obj.uniform(-.5, .5) for x in range(num_hid * num_hid)] for layer in range(num_hid_layers - 1)]
        #weights between final hidden layer and Outputs
        weights.append([rand_obj.uniform(-.5, .5) for x in range(num_out * num_hid)])

        #setup the weight deltas array to initially be all 0's
        weightDeltas = [[0 for x in range(num_in * num_hid)]]
        for layer in range(num_hid_layers - 1):
            weightDeltas.append([0 for x in range(num_hid * num_hid)])
        weightDeltas.append([0 for x in range(num_out * num_hid)])

        #setup the activations array
        activations = [[0 for x in range(num_in)], [0 for x in range(num_hid)] * num_hid_layers, [0 for x in range(num_out)]]

        #setup the errors array
        errors = []

        #setup the thetas array
        thetas = [[0 for x in range(num_in)], [0 for x in range(num_hid)] * num_hid_layers, [0 for x in range(num_out)]]

        #setup the theta deltas
        thetaDeltas = [[0 for x in range(num_in)], [0 for x in range(num_hid)] * num_hid_layers, [0 for x in range(num_out)]]

        return (weights, weightDeltas, activations, errors, thetas, thetaDeltas)


    #-----Begin ACCESSORS-----#
    def set_weights(self, aList):
        self.weights = aList

    def set_thetas(self, aList):
        self.thetas = aList

		#-----End ACCESSORS-----#


    @staticmethod
    def sigmoid_af(summed_input):
        #Sigmoid function
        return 1 / (1 +  math.exp(-summed_input))

    @staticmethod
    def sigmoid_af_deriv(sig_output):
        #the derivative of the sigmoid function
        return sig_output * (1 - sig_output)


test_agent = NeuralMMAgent(2, 2, 1, 1,random_seed=5, max_epoch=1000000, \
                            learning_rate=0.2, momentum=0)
test_in = [[1,0],[0,0],[1,1],[0,1]]
test_out = [[1],[0],[0],[1]]
#test_agent.set_weights([[-.37,.26,.1,-.24],[-.01,-.05]])
test_agent.set_thetas([[0,0],[0,0],[0]])
test_agent.train_net(test_in, test_out, max_sse = test_agent.max_sse, \
                     max_num_epoch = test_agent.max_epoch)
