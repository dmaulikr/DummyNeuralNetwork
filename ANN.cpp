#include </usr/include/octave-3.8.1/octave/oct.h>
#include <octave-3.8.1/octave/parse.h>

#define HELP_TEXT "Usage [...] = ANN(Matrix features, Matrix classValue)"

#define MAXITERATIONS 10000
#define LEARNINGRATE 0.1

#define INPUTNODES 3
#define HIDDENNODES 3
#define OUTPUTNODES 1

#define E 2.71828182845905


void printMatrix(Matrix m)
{
    int nRows = m.rows();
    int nColumns = m.columns();
    
    const double * mPointer = m.fortran_vec();
    
    for ( size_t i=0; i<nRows*nColumns; i++)    
        printf("%lf \t", mPointer[i]);
    printf("\n");    
}


Matrix createRandomMatrix(int nRows, int nColumns)
{
  octave_value_list args;
  args.append(octave_value(nRows));
  args.append(octave_value(nColumns));
  octave_value_list out = feval("randn", args, 1);
  Matrix randMatrix( out(0).matrix_value() );
  return randMatrix;
}


Matrix addBiasValues(Matrix m)
{
    int nRows = m.rows();

    // create bias matrix
    octave_value_list args;
    args.append(octave_value(nRows));
    args.append(octave_value(1));
    octave_value_list out = feval("ones", args, 1);    
    Matrix onesMatrix( out(0).matrix_value() );
    
    // combine matrix m with bias matrix
    octave_value_list args2;
    args2.append(onesMatrix);
    args2.append(m);
    octave_value_list out2 = feval("horzcat", args2, 1);
    Matrix outMatrix( out2(0).matrix_value() );
    return outMatrix;
}


Matrix tanHActivation(Matrix m)
{
    octave_value_list args;
    args.append(m);
    octave_value_list out = feval("tanh", args, 1);
    Matrix outMatrix( out(0).matrix_value() );
    return outMatrix;
}


Matrix sigmoidActivation(Matrix m)
{
    int nRows = m.rows();
    int nColumns = m.columns();
    
    Matrix outMatrix(nRows, nColumns, 0);
    
    const double * mPointer = m.fortran_vec();
    double * outMatrixPointer = outMatrix.fortran_vec();
    
    for ( size_t i=0; i<nRows*nColumns; i++)
    {
        outMatrixPointer[i] = ( 1.0 / ( 1.0 + pow(E, - (mPointer[i]) ) )  );
    }
    
    return outMatrix;
}



Matrix calculateOutputErrorGradient(Matrix outputNodesValues, Matrix absoluteError)
{
    int nRows = outputNodesValues.rows();
    int nColumns = outputNodesValues.columns();
    
    Matrix gradientMatrix(nRows, nColumns, 0);
    double * gradientMatrixPointer = gradientMatrix.fortran_vec();
    const double * outputNodesValuesPointer = outputNodesValues.fortran_vec();
    const double * absoluteErrorPointer = absoluteError.fortran_vec();
    
    for ( size_t i=0; i<nRows*nColumns; i++)
    {
        gradientMatrixPointer[i] = outputNodesValuesPointer[i] * (1 - outputNodesValuesPointer[i]) * absoluteErrorPointer[i];
    }    
    
    return gradientMatrix;
}


Matrix calculateHiddenErrorGradient(Matrix hiddenNodesValues, Matrix outputErrorGradient, Matrix HintonOutputLayerWeights)
{
    int nRows = hiddenNodesValues.rows();
    int nColumns = hiddenNodesValues.columns();
    
    Matrix gradientMatrix(nRows, nColumns, 0);
    double * gradientMatrixPointer = gradientMatrix.fortran_vec();
    const double * hiddenNodesValuesPointer = hiddenNodesValues.fortran_vec();
    const double * outputErrorGradientPointer = outputErrorGradient.fortran_vec();
    const double * HintonOutputLayerWeightsPointer = HintonOutputLayerWeights.fortran_vec();

    
    for ( size_t i=0; i<nRows*nColumns; i++)
    {
        gradientMatrixPointer[i] = hiddenNodesValuesPointer[i] * (1 - hiddenNodesValuesPointer[i]) * outputErrorGradientPointer[(int)round(i/nRows)] * HintonOutputLayerWeightsPointer[(int)round((i/nRows)+1)];
    }    
    
    return gradientMatrix;
}




Matrix calculateDeltaOutputNodes(Matrix HintonOutputLayerWeights, Matrix outputNodesValues, Matrix errorGradient)
{
    int nRows = HintonOutputLayerWeights.rows();
    int nColumns = HintonOutputLayerWeights.columns();
    
    Matrix deltaMatrix(nRows, nColumns, 0);
    double * deltaMatrixPointer = deltaMatrix.fortran_vec();
    const double * errorGradientPointer = errorGradient.fortran_vec();
    const double * outputNodesValuesPointer = outputNodesValues.fortran_vec();
    
    for ( size_t i=0; i<nRows*nColumns; i++)
    {
        deltaMatrixPointer[i] = LEARNINGRATE * errorGradientPointer[(int)round(i/nRows)] * outputNodesValuesPointer[(int)round(i/nRows)];
    }    
    
    return deltaMatrix;
}



Matrix calculateDeltaHiddenNodes(Matrix HintonHiddenLayerWeights, Matrix hiddenNodesValues, Matrix errorGradient)
{    
    int nRows = HintonHiddenLayerWeights.rows();
    int nColumns = HintonHiddenLayerWeights.columns();
    
    Matrix deltaMatrix(nRows, nColumns, 0);
    double * deltaMatrixPointer = deltaMatrix.fortran_vec();
    const double * errorGradientPointer = errorGradient.fortran_vec();
    const double * hiddenNodesValuesPointer = hiddenNodesValues.fortran_vec();
    
    for ( size_t i=0; i<nRows*nColumns; i++)
    {
        deltaMatrixPointer[i] = LEARNINGRATE * errorGradientPointer[(int)round(i/nRows)] * hiddenNodesValuesPointer[(int)round(i/nRows)];
    }    
    
    return deltaMatrix;    
}




DEFUN_DLD(ANN, argv, nargout, HELP_TEXT)
{
  octave_value_list retval;

  int nargs = argv.length();
  if (nargs != 2 )
    {
      error(HELP_TEXT);
      return retval;
    }
  
  Matrix features(argv(0).matrix_value());
  Matrix featuresBias = addBiasValues(features);
  Matrix classValue(argv(1).matrix_value());
  
  // initialize neural network weights - HIDDEN LAYER
  Matrix HintonHiddenLayerWeights = createRandomMatrix(INPUTNODES,HIDDENNODES-1);
  Matrix HintonOutputLayerWeights = createRandomMatrix(HIDDENNODES,OUTPUTNODES);

  // start training
    
  Matrix sumSquareError;
  
  for (int iteration = 1; iteration <= MAXITERATIONS; iteration++) 
  {
      printf("\n\nIteration %d...\n", iteration);
      
      // 1.- Get example inputs and outputs
      Matrix inputNodeValues = featuresBias;
      Matrix expectedValues = classValue;
      
      // 2.- Run the inputs through the network
      Matrix hiddenNodesValues = sigmoidActivation(inputNodeValues * HintonHiddenLayerWeights);
      Matrix outputNodesValues = sigmoidActivation(addBiasValues(hiddenNodesValues) * HintonOutputLayerWeights);

      // 3.- Compare the actual outputs to the desired outputs
      Matrix absoluteError = expectedValues - outputNodesValues;
      printf("Training instance\n");
      printMatrix(inputNodeValues);
      printf("output of network -> ");
      printMatrix(outputNodesValues);
      printf("expected value -> ");
      printMatrix(expectedValues);
      printf("absolute error -> ");
      printMatrix(absoluteError);
                            
      // 4.- Adjust the weights in the network
      
      // calculate error gradient of output nodes
      Matrix outputErrorGradient = calculateOutputErrorGradient(outputNodesValues, absoluteError);
      
      // update weights of output nodes according to learning rate
      HintonOutputLayerWeights += calculateDeltaOutputNodes(HintonOutputLayerWeights, outputNodesValues, outputErrorGradient);
      
      // calculate error gradient of hidden nodes as part of the error of the output nodes
      Matrix hiddenErrorGradient = calculateHiddenErrorGradient(hiddenNodesValues, outputErrorGradient, HintonOutputLayerWeights);
      
      // update weights according to learning rate
      HintonHiddenLayerWeights += calculateDeltaHiddenNodes(HintonHiddenLayerWeights, hiddenNodesValues, hiddenErrorGradient);
  }
  
  
  printMatrix(HintonHiddenLayerWeights);
  retval.append(featuresBias);
  retval.append(classValue);
  retval.append(HintonHiddenLayerWeights);
  retval.append(HintonOutputLayerWeights);
  return retval;
}
