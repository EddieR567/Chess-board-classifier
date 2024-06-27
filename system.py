"""Classification system.

Solution for the COM2004/3004 assignment by Edward Ranyard (December 2023).

version: v1.4
"""
from typing import List

import numpy as np
from scipy.stats import norm

N_DIMENSIONS = 10


def classify(model:dict, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    I have had to add the model to the classify arguments but I checked on the discussion
    board that this is allowed. 

    I have adopted to use a naive Bayes classifier, calculating log likelihoods of a particular 
    image with respect to each class and coupling that with prior probabilities to compute the
    posterior probability. I have then normalised the posterior data and calculated the probability 
    of the most likely label (this is for use in classify_boards, alongside the second-most likely label)

    It then returns the highest-scoring label.

    Args:
        model (dictionary): a dictionary containing the class means, standard deviations, priors, labels 
            and reduced feature vectors as well as later storing the probabilities and second predicitons.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    classes = np.unique(model["labels_train"]).tolist()

    means = model["class_means"]
    stds = model["class_stds"]
    priors = model["class_priors"]

    predictions = []
    probabilities = []
    secondPredictions = []
    epsilon = 1e-100 #miniscule value added to prevent a dividing by zero error

    for sample in test:
        posteriors = []
        for label in means:
            #using scipy.norm for the probability density function
            likelihoods = np.sum(np.log(norm.pdf(sample, means[label], stds[label]) + epsilon))
            posterior = likelihoods + np.log(priors[label])
            posteriors.append(posterior)

        maxLogPosterior = np.max(posteriors)
        normalisedPosteriors = np.exp(posteriors - maxLogPosterior)
        normalisedPosteriors /= np.sum(normalisedPosteriors) 
        probability = np.max(normalisedPosteriors)  

        indicesSorted = np.argsort(normalisedPosteriors)[::-1] 
        secondIndex = indicesSorted[1]


        predictedLabel = classes[np.argmax(posteriors)]

        secondPredictions.append(classes[secondIndex])
        probabilities.append(probability)
        predictions.append(predictedLabel)

    model["probabilities"] = probabilities
    model["seconds"] = secondPredictions


    return predictions


# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    Note: My reduce_dimensions function evidently is not very long, as the majority of my 
    dimensionality reduction is done in process_training_data.

    This code performs a dot product transformation on the training and test feature vector datasets,
    with the N selected eigenvectors computed in process_training_data using LDA.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """

    selected_eigenvectors = np.array(model["selected_eigenvectors"])
    reduced_data = data.dot(selected_eigenvectors)

    return reduced_data

def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Within this function, I've executed a procedure for Linear Discriminant Analysis. By calculating the 
    within-class and between-class scatter matrices, it identifies the most discriminative 
    dimensions in the dataset. The eigenvectors derived from these matrices are then transformed
    into a lower-dimensional space. 

    I also have used this function to set up the class-specific statistics in the model data,
    more specifically the class means, standard deviations and priors, which are all made relevant 
    in the classify function.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels.
    model = {}
    model["labels_train"] = labels_train.tolist()


    classes = np.unique(labels_train)
    totalMean = np.mean(fvectors_train, axis=0)
    featureSize = fvectors_train.shape[1]
    withinClassScatter = np.zeros((featureSize,featureSize))
    betweenClassScatter = np.zeros((len(totalMean), len(totalMean)))
    
    for label in classes:
        #fills in the scatter matrices
        indices = np.where(labels_train == label)
        sampleSize = np.sum(labels_train == label)
        classVectors = fvectors_train[indices]
        meanVector = np.mean(classVectors, axis=0)
        deviation = meanVector - totalMean
        deviationMatrix = classVectors - meanVector
        scatterMatrix = np.dot(deviationMatrix.T, deviationMatrix)
        withinClassScatter += scatterMatrix
        classScatter = sampleSize * np.outer(deviation, deviation)
        betweenClassScatter += classScatter


    combined = np.linalg.inv(withinClassScatter).dot(betweenClassScatter)
    eigenvalues, eigenvectors = np.linalg.eig(combined)

    #sorts and filters N best eigenvectors
    sortedIndices = np.argsort(eigenvalues)[::-1] 
    eigenvectors = eigenvectors[:, sortedIndices] 
    selectedEigenvectors = eigenvectors[:, :N_DIMENSIONS]
    selectedEigenvectors = np.real(selectedEigenvectors) #for serialization

    model["selected_eigenvectors"] = selectedEigenvectors.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()


    means = {}
    stds = {}
    priors = {}

    #calculates class data
    for label in classes:
        trainingData = fvectors_train_reduced[labels_train == label]
        means[label] = np.mean(trainingData, axis=0).tolist()
        stds[label] = np.std(trainingData, axis=0).tolist()
        priors[label] = len(trainingData) / len(labels_train)

    
    model["class_means"] = means
    model["class_stds"] = stds
    model["class_priors"] = priors

    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Calls the classify function.
    labels = classify(model, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    I have added 2 nested subroutines, countPieces to count how many of each piece 
    appears on a particular board, and validatePieces to change the least likely piece
    to its second highest label in the event there are too many (eg. 3 kings).

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
            
    def countPieces(piece,board): 
    #takes piece letter and which board to iterate through
        counter = 0
        boardRange = board * 64
        probList = []
        indexList = []
        for square in classifiedSquares[boardRange:boardRange + 64]:
            index = boardRange + counter
            if square.lower() == piece:
                indexList.append(index)
                probList.append(probabilities[index])
            counter+=1
        return indexList,probList
    #returns list of indices where the piece is found, and each probability

    def validatePieces(maxOnBoard,actualOnBoard,indexList:List,probList:List):
    #takes max possible number of certain piece, actual number on the board & the two lists
        while actualOnBoard > maxOnBoard:
            minIndex = indexList[probList.index(min(probList))]
            classifiedSquares[minIndex] = seconds[minIndex]
            indexList.pop(probList.index(min(probList)))
            probList.remove(min(probList))
            actualOnBoard -= 1

    classifiedSquares = classify_squares(fvectors_test, model)
    boardNum = len(classifiedSquares) // 64
    probabilities = model["probabilities"]
    seconds = model["seconds"]

    for board in range(0,boardNum-1): 
        kingList,kingProbs = countPieces('k',board)
        validatePieces(2,len(kingList),kingList,kingProbs)
        bishList,bishProbs = countPieces('b',board)
        validatePieces(4,len(bishList),bishList,bishProbs)
        knightList,knightProbs = countPieces('n',board)
        validatePieces(4,len(knightList),knightList,knightProbs)
        rookList,rookProbs = countPieces('r',board)
        validatePieces(4,len(rookList),rookList,rookProbs)
        pawnList,pawnProbs = countPieces('p',board)
        validatePieces(16,len(pawnList),pawnList,pawnProbs)
        emptiesList,emptiesProbs = countPieces('.',board)
        validatePieces(62,len(emptiesList),emptiesList,emptiesProbs)

    return classifiedSquares 
