#!/usr/bin/octave

close; clear all;
LEARNINGRATE=0.2;

MAXITERATIONS=input("Select number of MAXITERATIONS: ");
file = input("Select the dataset to use: ", "s");
dataset = load(file);

INPUTNODES = input("Select the number of input nodes: ");
HIDDENNODES = input("Select the number of hidden nodes: ");
OUTPUTNODES = input("Select the number of output nodes: ");

features = dataset(:,1:INPUTNODES);
class = dataset(:,end-OUTPUTNODES+1:end);

# Initialize weights as random
#added bias neuron

HintonInput2HiddenLayerWeights = randn(INPUTNODES+1,HIDDENNODES) ./ 100;
HintonInput2HiddenLayerWeights(1,:) = 0;

HintonHidden2OutputLayerWeights = randn(HIDDENNODES+1,OUTPUTNODES) ./ 100;
HintonHidden2OutputLayerWeights(1,:) = 0;

# start training
for i=1:MAXITERATIONS
    
    # 1 get random training example
    index = randi(4);
    inputValues = features(index,:);
    expectedOutputValues = class(index,:);

    # 2 Run the input through the network
    hiddenNodesValues = sigmoid([ones(size(inputValues,1),1) inputValues] * HintonInput2HiddenLayerWeights);
    outputNodesValues = sigmoid([ones(size(hiddenNodesValues,1),1) hiddenNodesValues] * HintonHidden2OutputLayerWeights);

    # 3 Compare the actual output to the desired output (delta)
    outputDelta = expectedOutputValues - outputNodesValues;

    # 4 Backpropagate the error and adjust the weights in the network

    for outputNode=1:OUTPUTNODES
	
      # 4.1 calculate error gradient on outputNodes
      outputGradient = outputNodesValues(outputNode) * (1.0 - outputNodesValues(outputNode)) * outputDelta(outputNode);

      for hiddenNode=1:HIDDENNODES

	# 4.2 Update weights of hidden -> output nodes according to learning rate and gradient
	hiddenDelta = LEARNINGRATE * hiddenNodesValues(hiddenNode) * outputGradient;
	HintonHidden2OutputLayerWeights(hiddenNode+1,outputNode) += hiddenDelta; # do not update bias in this way, it will update at the end
	
        # 4.3 Calculate error gradient of hidden nodes as part of the error of the output nodes
	hiddenGradient = hiddenNodesValues(hiddenNode) * (1 - hiddenNodesValues(hiddenNode))  * outputGradient * HintonHidden2OutputLayerWeights(hiddenNode+1, outputNode);
	
	for inputNode=1:INPUTNODES
	    
	  # 4.4 Update weights of input -> hidden according to learning rate
	  inputDelta = LEARNINGRATE * inputValues(inputNode) * hiddenGradient;
	  HintonInput2HiddenLayerWeights(inputNode+1,hiddenNode) += inputDelta;  # do not update bias in this way, it will update at the end
	 
	endfor
	# endfor inputNode

	BiasHiddenDelta = LEARNINGRATE * hiddenGradient;
	HintonInput2HiddenLayerWeights(1, hiddenNode) += BiasHiddenDelta;

      endfor
      # endfor hiddenNode
						       
      BiasOutputDelta = LEARNINGRATE * outputGradient;
      HintonHidden2OutputLayerWeights(1, outputNode) += BiasOutputDelta;

    endfor
    # endfor outputNode

						       
endfor
#nedfor MAXITERATIONS


hiddenNodesValues = sigmoid([ones(size(features,1),1) features] * HintonInput2HiddenLayerWeights);
outputNodesValues = sigmoid([ones(size(hiddenNodesValues,1),1) hiddenNodesValues] * HintonHidden2OutputLayerWeights);

[features class outputNodesValues]
