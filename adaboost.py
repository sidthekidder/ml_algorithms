import numpy as np
from decision_tree import id3, predict_example, print_tree
from sklearn.model_selection import train_test_split

def adaboost_classify(Xtrn, ytrn, Xtst, ytst, iterations):
	# init weights to 1/N each
	wts = np.ones(len(Xtrn)) / len(Xtrn)
	pred_train = np.zeros(len(Xtrn))
	pred_test = np.zeros(len(Xtst))

	for i in range(iterations):
		print("Training stump in iteration " + str(i))
		dtree = id3(x=Xtrn, y=ytrn, attribute_value_pairs=None, depth=0, max_depth=1, weights=wts)
		ytrn_pred = [predict_example(X, dtree) for X in Xtrn]
		ytst_pred = [predict_example(X, dtree) for X in Xtst]

		# number of misclassified examples for training set
		misclassified = [int(x) for x in (ytrn_pred != ytrn)]

		# multiply misclassified examples by weights
		err = np.dot(wts, misclassified) / sum(wts)

		# calculate alpha
		alpha = np.log((1 - err) / err)

		# convert misclassified from 0/1 to -1/1
		misclassified = [x if x == 1 else -1 for x in misclassified]

		# recalculate weights 
		wts = [x * np.exp(alpha*(x != ytrn_pred[i])) for i,x in enumerate(misclassified)]

		# make predictions with current test observations
		pred_train = [sum(x) for x in zip(pred_train, [x * alpha for x in ytrn_pred])] 
		pred_test = [sum(x) for x in zip(pred_test, [x * alpha for x in ytst_pred])]

	pred_test = np.sign(pred_test)
	return (sum(pred_test != ytst) / len(pred_test))

if __name__ == "__main__":
	dataset = np.genfromtxt("./Skin_NonSkin.txt", delimiter='\t', dtype=int)
	ytrn = dataset[:, 3] # column to predict
	Xtrn = dataset[:, :3] # feature columns

	Xtrn, Xtst, ytrn, ytst = train_test_split(Xtrn, ytrn, test_size=0.99, random_state=42)

	print("Starting to run adaboost")

	# fit adaboost ensemble of trees using baseline decision tree
	err = adaboost_classify(Xtrn, ytrn, Xtst, ytst, iterations=10)

	print("Error rate is " + str(err))

