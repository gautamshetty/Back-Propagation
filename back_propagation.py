# Shetty, Gautam
# 100-144-6742
# 2016-11-13
# Assignment_05

import numpy as np
import scipy.misc
import os
import theano
import theano.tensor as T
from theano import function, printing, pp

T.config.exception_verbosity='high'
T.config.on_unused_input='ignore'

"""
    Hidden Layer output
"""
l1_weight_matrix = T.dmatrix("l1_weight_matrix")
l1_input_matrix = T.dmatrix("l1_input_matrix")
l1_net_input = T.dmatrix("l1_input_matrix")
l1_a = T.dmatrix("l1_input_matrix")

l1_net_input = T.dot(l1_weight_matrix, T.transpose(l1_input_matrix))

l1_a = T.nnet.relu(l1_net_input)

"""
    Output Layer output.
"""
l2_weight_matrix = T.dmatrix("l2_weight_matrix") #100 x 10
l2_net_input = T.dmatrix('l2_net_input')
l2_a = T.dmatrix('l2_net_input')
output_scores = T.lvector("output_scores")

l2_net_input = T.dot(T.transpose(l2_weight_matrix), l1_a) # 10 x 100

l2_a = T.nnet.softmax(l2_net_input)
l2_loss = T.nnet.categorical_crossentropy(l2_a, output_scores)

avg_loss = T.sum(l2_loss)/l1_input_matrix.shape[0]

l2_lossa2_grad = T.grad(avg_loss, l2_a)

l2_a2n2_grad = T.dot(T.transpose(l2_a), (1 - l2_a))

l2_n2w2_grad = l1_a

total_l2_grad = T.dot(l2_lossa2_grad, T.dot(l2_a2n2_grad, l2_n2w2_grad))

l2_weight = l2_weight_matrix - 0.001 * T.transpose(total_l2_grad * l2_a)


l1_lossa1_grad = T.dot(T.transpose(l2_lossa2_grad), T.transpose(l2_weight_matrix))

#l1_a1n1_grad = T.nnet.relu(l1_input_matrix)

l1_n1w1_grad = l1_input_matrix

total_l1_grad = T.dot(l1_lossa1_grad, l1_n1w1_grad)
#total_l1_grad = T.dot(T.transpose(T.dot(l1_lossa1_grad, l1_a1n1_grad)), l1_n1w1_grad)
#total_l1_grad = T.nnet.relu(total_l1_grad)

l1_weight = l1_weight_matrix - 0.001 * total_l1_grad

back_propagation = function([l1_input_matrix, l1_weight_matrix, l2_weight_matrix, output_scores], [avg_loss, l1_weight, l2_weight, l2_a, l2_net_input, l2_loss])


l1_weight_matrix = T.dmatrix("l1_weight_matrix")
l1_input_matrix = T.dmatrix("l1_input_matrix")
l1_net_input = T.dmatrix("l1_input_matrix")
l1_a = T.dmatrix("l1_input_matrix")
output_scores = T.lvector("output_scores")

l1_net_input = T.dot(l1_weight_matrix, T.transpose(l1_input_matrix))

#l1_a = T.nnet.relu(l1_net_input)
l1_a = T.nnet.sigmoid(l1_net_input)

"""
    Output Layer output.
"""
l2_weight_matrix = T.dmatrix("l2_weight_matrix") #100 x 10
l2_net_input = T.dmatrix('l2_net_input')
l2_a = T.dmatrix('l2_net_input')

l2_net_input = T.dot(T.transpose(l2_weight_matrix), l1_a) # 10 x 100

l2_a = T.nnet.softmax(l2_net_input)
l2_loss = T.nnet.categorical_crossentropy(l2_a, output_scores)

classify_image = function([l1_input_matrix, l1_weight_matrix, l2_weight_matrix, output_scores], [l2_loss, l2_a])

"""
    Reads image file and converts it to vector.
    Normalizes the vector before returning.
"""
def read_one_image_and_convert_to_vector(file_name):

    img = scipy.misc.imread(file_name).astype(np.float32)  # read image and convert to float
    img = img.ravel()

    """
        Normalize - subtract mean from all values.
    """
    mean_value = np.mean(img)
    img = img - mean_value

    std_dev = np.std(img)

    img /= std_dev
    #print img

    return img

"""
    Creates input samples by reading image files and converting them to vectors.
"""
def create_samples(fileDirectory, j, num_of_samples=100):

    file_list = []

    for k in range(10):
        for i in range(num_of_samples / 10):
            imgVector = read_one_image_and_convert_to_vector(fileDirectory + "/" + str(k) + "_" + str(j) + ".png")
            file_list.append(imgVector)
            j += 1
        j -= num_of_samples / 10

    return np.array(file_list)

def create_weights(x, y):

    weight_matrix = 0.01 * np.random.randn(x, y)
    #weight_matrix = 0.01 * np.random.uniform(-1, 1, (x, y))
    #weight_matrix = np.where(weight_matrix < 0, -1 * weight_matrix, weight_matrix)

    #if y == 3072:
    #    weight_matrix *= np.sqrt(2/(x * y))
    #else:
    weight_matrix /= np.sqrt(x * y)

    #print weight_matrix

    return np.array(weight_matrix)

def main():

    num_pixels = 3072
    num_class = 10
    num_neurons = 100

    output_scores = np.array([0,0,0,0,0,0,0,0,0,0])

    l1_weight_matrix = create_weights(num_neurons, num_pixels)
    #print "L1 Weights : "
    #print l1_weight_matrix
    l2_weight_matrix = create_weights(num_neurons, num_class)
    #print "L2 Weights : "
    #print l2_weight_matrix
    #target_vector = create_target(10)
    lowest_loss = 1
    lowest_index = 0
    l1_input_samples = create_samples("train", 0 * (num_neurons / 10), num_neurons)
    for i in range(200):

        if (i < 10):
            l1_input_samples = create_samples("train", i * (num_neurons / 10), num_neurons)

        updated_values = back_propagation(l1_input_samples, l1_weight_matrix, l2_weight_matrix, output_scores)

        if i % 10 == 0:
            print "Iteration : ", i
            print "Loss : ", updated_values[0]

            if updated_values[0] < lowest_loss:
                lowest_loss = updated_values[0]
                lowest_index = i
        #print "L1 Output : ", updated_values[3]
        #print "L2 Output : ", updated_values[4]

        #print updated_values[0].shape
        #print updated_values[1].shape
        #print updated_values[2].shape
        #print updated_values[3].shape
        #print updated_values[4].shape
        #print updated_values[5].shape

        #if i == 2:
        #    print "L1 Weights : ", updated_values[1]
        #    print "L2 Weights : ", updated_values[2]

        l1_weight_matrix = updated_values[1]
        l2_weight_matrix = updated_values[2]

        #print updated_values[4]
        #print "L2 Weights : ", l2_weight_matrix
    print "Lowest Loss : ", lowest_loss
    print "Lowest Loss Index : ", lowest_index

    predicted_class = np.argmax(updated_values[3], axis=1)
    # print predicted_class
    # print predicted_class.shape
    print 'training accuracy: %.2f' % (np.mean(predicted_class == output_scores))

    test_samples = create_samples("test", 0)
    result = classify_image(test_samples, l1_weight_matrix, l2_weight_matrix, output_scores)
    # print result[0]
    # print result[0].shape
    # predicted_class = np.argmax(result[1], axis=0)
    # print result[1]
    # print result[1].shape
    # print predicted_class.shape
    print 'training accuracy: %.2f' % (np.mean(result[1] == output_scores))

    img_trans = np.transpose(result[1])

    con_index = -1
    index_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(100):
        img_arr = np.array(img_trans[i])
        # print img_trans[i]
        # print "Max : ", np.max(img_trans[i])
        # print np.where(img_trans[i] == np.max(img_trans[i]))
        index_tuple = np.where(img_trans[i] == np.max(img_trans[i]))
        index_array[index_tuple[0]] += 1
    print index_array

if __name__ == '__main__':
	main()
