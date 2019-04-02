import math

sample_data = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

# defines the mean squared error
def mse(predictions, actual):
	loss_sum_sqared = 0
	for i in range(0, len(predictions)):
		loss_sum_sqared += (predictions[i] - actual[i])**2
	return loss_sum_sqared / len(predictions)

# y = w*x + b
# find values of w and b through gradient descent and update w,b after each iteration
w = 0.0 # weight
b = 10.0 # bias
n = 0.005 # learning coefficient

# keep iterating till convergence
t = 0
prevloss = 0
length = len(sample_data)
while True:
	print("Iteration " + str(t))
	# calculate prediction using equation y = wx + b
	predictions = [(x, w*x + b) for (x,y) in sample_data]

	# calculate the loss using mean squared error
	loss = mse([x[1] for x in predictions], [x[1] for x in sample_data])
	print("Loss in " + str(t) + " iteration: " + str(loss))

	# check if previous loss and current loss have stopped changing - difference is less than some threshold
	if abs(loss - prevloss) < 0.01:
		print("Breaking! Convergence achieved")
		break

	# update values
	b_gradient = -(2.0/length) * sum([(y - (w*x + b)) for (x,y) in sample_data])
	w_gradient = -(2.0/length) * sum([x*(y - (w*x + b)) for (x,y) in sample_data])

	w = w - n*w_gradient
	b = b - n*b_gradient

	print("Now w: " + str(w) + " and b: " + str(b))
	# raw_input("Enter")
	prevloss = loss
	t += 1

print("Final w: " + str(w) + " and b: " + str(b))
