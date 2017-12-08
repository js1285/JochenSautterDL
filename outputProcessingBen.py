import matplotlib.pyplot as plt
import numpy as np



def skipLinesUntilStartsWith(data, i, phrase):
    line = data[i]
    while(not line.startswith(phrase)):
        i += 1
        line = data[i]
    return i



def readExampleImmages(data, i, numberImmages, labels, filename):
	for label in labels:
		i = skipLinesUntilStartsWith(data, i, label)
		print(data[i])

	for picnr in range(numberImmages):
		i = skipLinesUntilStartsWith(data, i, "IM")
		print(data[i])
		i += 1

		immage = []
		for _ in range(28*28):
			immage.append(float(data[i].replace(']', '').replace('[', '')))
			i += 1
		immage = np.array(immage).reshape((28, 28))
		plt.imshow(immage, cmap='gray', interpolation='none')
		plt.axis('off')
		plt.savefig('picsNoise/' + filename + '{}'.format(picnr))
	return i




with open('autoencoderNoisetestOut.txt', 'r') as file:
    data = file.readlines()
i = 0


#
# reading for learning rate 0.1
#
i = readExampleImmages(data, i, numberImmages=10, 
						labels=["EXAMPLE RESULTS", "EXAMPLE TRAIN IMMAGES"],
						filename='LR01/picA_train_im')

i = readExampleImmages(data, i, numberImmages=10, 
						labels=["EXAMPLE TRAIN PROCESSED"],
						filename='LR01/picB_train_proc')

i = readExampleImmages(data, i, numberImmages=10, 
						labels=["EXAMPLE TEST IMMAGES"],
						filename='LR01/picC_test_im')

i = readExampleImmages(data, i, numberImmages=10, 
						labels=["EXAMPLE TEST PROCESSED"],
						filename='LR01/picD_test_proc')


#
# reading for learning rate 0.01
#

i = readExampleImmages(data, i, numberImmages=10, 
						labels=["EXAMPLE RESULTS", "EXAMPLE TRAIN IMMAGES"],
						filename='LR001/picA_train_im')

i = readExampleImmages(data, i, numberImmages=10, 
						labels=["EXAMPLE TRAIN PROCESSED"],
						filename='LR001/picB_train_proc')

i = readExampleImmages(data, i, numberImmages=10, 
						labels=["EXAMPLE TEST IMMAGES"],
						filename='LR001/picC_test_im')

i = readExampleImmages(data, i, numberImmages=10, 
						labels=["EXAMPLE TEST PROCESSED"],
						filename='LR001/picD_test_proc')


#
# reading for learning rate 0.001
#


i = readExampleImmages(data, i, numberImmages=10, 
						labels=["EXAMPLE RESULTS", "EXAMPLE TRAIN IMMAGES"],
						filename='LR0001/picA_train_im')

i = readExampleImmages(data, i, numberImmages=10, 
						labels=["EXAMPLE TRAIN PROCESSED"],
						filename='LR0001/picB_train_proc')

i = readExampleImmages(data, i, numberImmages=10, 
						labels=["EXAMPLE TEST IMMAGES"],
						filename='LR0001/picC_test_im')

i = readExampleImmages(data, i, numberImmages=10, 
						labels=["EXAMPLE TEST PROCESSED"],
						filename='LR0001/picD_test_proc')


