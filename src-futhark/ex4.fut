def sigmoid (x: f32) : f32 = 1.0 / (1.0 + f32.exp (-x))

def sig_derivative (x: f32) : f32 =
  let fx = sigmoid x in fx * (1.0 - fx)

def label_to_output (i: i64) (label: i8) : [i]f32 =
  let label64 = i64.i8 label
  in map (\index -> if index == label64 then 1.0 else 0.0) (iota i)

def potential [n] (inputs: [n]f32) (weights: [n]f32) : f32 =
  f32.sum (map2 (*) inputs weights)

def learn [i] [j] (lr: f32) (weights: [i][j]f32) (deltas: [i]f32) (outputs: [j]f32) : [i][j]f32 =
  map2 (\wI deltaI ->
          map2 (\wIJ xJ -> wIJ + lr * deltaI * xJ)
               wI
               outputs)
       weights
       deltas

-- Train the model on one image
def train_iteration (j: i64) (h: i64) (i: i64) (lr: f32) (inputs: [j]f32) (hiddenWeights: [h][j]f32) (outputWeights: [i][h]f32) (label: i8) : ([h][j]f32, [i][h]f32, f32) =
  let Y = label_to_output i label
  let hiddenPotentials = map (potential inputs) (hiddenWeights)
  let hiddenOutputs = map sigmoid hiddenPotentials
  let outputPotentials = map (potential hiddenOutputs) (outputWeights)
  let outputOutputs = map sigmoid outputPotentials
  let outputDeltas = map3 (\potI Yi Xi -> sig_derivative potI * (Yi - Xi)) outputPotentials Y outputOutputs
  let hiddenDeltas =
    map2 (\potH weights -> sig_derivative potH * f32.sum (map2 (*) outputDeltas weights))
         hiddenPotentials
         (transpose outputWeights)
  let updatedHiddenWeights = learn lr hiddenWeights hiddenDeltas inputs
  let updatedOutputWeights = learn lr outputWeights outputDeltas hiddenOutputs
  let sumDelta = f32.sum (map f32.abs outputDeltas)
  in (updatedHiddenWeights, updatedOutputWeights, sumDelta)

-- Train the model on a batch of N images
entry train_batch (n: i64) (j: i64) (h: i64) (i: i64) (lr: f32) (inputss: [n][j]f32) (hiddenWeights: [h][j]f32) (outputWeights: [i][h]f32) (labels: [n]i8) : ([h][j]f32, [i][h]f32, f32) =
  let train_with_params = train_iteration j h i lr
  in foldl (\(hW, oW, sumDeltas) (inputs, label) ->
              let (updatedHW, updatedOW, d) = train_with_params inputs hW oW label
              in (updatedHW, updatedOW, sumDeltas + d))
           (hiddenWeights, outputWeights, 0)
           (zip inputss labels)

entry test: i32 = 5
