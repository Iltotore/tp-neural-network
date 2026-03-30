#define _REENTRANT
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "load_mnist.h"

// bias => + 1
#define INPUT_N (__SIZE_IMAGE * __SIZE_IMAGE + 1)
#define HIDDEN_N (100 + 1)
#define OUTPUT_N (10)
#define HIDDEN_W_N (INPUT_N * HIDDEN_N)
#define OUTPUT_W_N (HIDDEN_N * OUTPUT_N)
#define LEARNING_RATE_BASE (0.5)
#define LEARNING_RATE_DECAY (0.1)
#define TRAINING_BATCH_SIZE (2000)
#define TRAIN_ERROR_THRESHOLD (0.02)
#define TRAINING_SAMPLE_SIZE (100)
#define TESTING_SAMPLE_SIZE (10000)


#define DEBUG (true)

// Based on Makefile's FUTHARK variable.
// You might need to run the `clean` task when changing the value whether here or via `make`.
#ifndef FUTHARK
#define FUTHARK (false)
#endif

#if FUTHARK
#include "../src-generated/ex4.h"
#endif

char gl[10] = {' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'};
char bigl[96] =
    "@MBHENR#KWXDFPQASUZbdehx*8Gm&04LOVYkpq5Tagns69owz$CIu23Jcfry%1v7l+it[]{}?j|()=~!-/"
    "<>^^^_';,:`. ";

void print_image(image* img) {
    int i;
    char grey_level;

    /* print the image "pixels" in grey_level using ascii*/
    for (i = 0; i < __SIZE_IMAGE * __SIZE_IMAGE; i++) {
        /* grey scale ascii art*/

        grey_level = (img->imgbuf[i]) / 2.7;
        fprintf(stderr, "%c ", bigl[(int)grey_level]);

        if (i % __SIZE_IMAGE == 0) fprintf(stderr, "\n");
    }

    /* print the label */

    fprintf(stderr, "\nlabel : %d \n", img->label);
}

float randomWeight(int weight_count) { return ((float)rand() / RAND_MAX - 0.5) * 2 / sqrt(weight_count); }

void initializeWeights(float* weights, int size) {
    for (int i = 0; i < size; i++) weights[i] = randomWeight(size);
}

void initializeOutputs(float* outputs, int size) {
    for (int i = 0; i < size; i++) outputs[i] = 0;
}

float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

float sig_derivative(float x) {
    // TODO check
    float fx = sigmoid(x);
    return fx * (1.0 - fx);
}

float getPotential(float* inputs, float* weights, int count) {
    float potential = 0;
    for (int i = 0; i < count; i++) {
        potential += inputs[i] * weights[i];
    }
    return potential;
}

void toInputs(image ret, float* inputs) {
    for (int i = 0; i < INPUT_N - 1; i++) {
        inputs[i] = ret.imgbuf[i] / 255.0;
    }
}

void toExpectedOutputs(image ret, float* Y) {
    for (int i = 0; i < OUTPUT_N; i++) {
        Y[i] = i == ret.label ? 1 : 0;
    }
}

void testPredict(float* inputs, float* hiddenWeights, float* hiddenPotential, float* hiddenOutputs,
                 float* outputWeights, float* outputPotential, float* outputOutputs) {
    for (int i = 0; i < HIDDEN_N; i++) {
        hiddenPotential[i] = getPotential(inputs, hiddenWeights + i * INPUT_N, INPUT_N);
        hiddenOutputs[i] = sigmoid(hiddenPotential[i]);
    }

    for (int i = 0; i < OUTPUT_N; i++) {
        outputPotential[i] = getPotential(hiddenOutputs, outputWeights + (i * HIDDEN_N), HIDDEN_N);
        outputOutputs[i] = sigmoid(outputPotential[i]);
    }
}

char predictLabel(float* outputOutputs) {
    int predicted = 0;
    float predictedValue = outputOutputs[0];

    for (int i = 1; i < OUTPUT_N; i++) {
        if (outputOutputs[i] > predictedValue) {
            predicted = i;
            predictedValue = outputOutputs[i];
        }
    }

    return predicted;
}

float calculateError(float* Y, float* outputOutputs) {
    float error = 0;
    for (int i = 0; i < OUTPUT_N; i++) error += fabs(Y[i] - outputOutputs[i]);
    return error;
}

int main() {
    time_t seed = time(0);

#if DEBUG
    fprintf(stderr, "Seed: %ld\n", seed);
#endif

    srand(seed);

    float hiddenWeights[HIDDEN_W_N];
    float hiddenPotential[HIDDEN_N];
    float hiddenOutputs[HIDDEN_N];

    float outputWeights[OUTPUT_W_N];
    float outputPotential[OUTPUT_N];
    float outputOutputs[OUTPUT_N];

    float inputs[INPUT_N];
    float Y[OUTPUT_N];

    initializeWeights(hiddenWeights, HIDDEN_W_N);
    initializeWeights(outputWeights, OUTPUT_W_N);

    initializeOutputs(hiddenOutputs, HIDDEN_N);
    initializeOutputs(outputOutputs, OUTPUT_N);

    hiddenOutputs[HIDDEN_N - 1] = 1;

#if FUTHARK
    struct futhark_context_config* cfg = futhark_context_config_new();
    futhark_context_config_set_device(cfg, "UHD");
    struct futhark_context* ctx = futhark_context_new(cfg);

    float inputss[INPUT_N * TRAINING_BATCH_SIZE];
    for (int i = 0; i < TRAINING_BATCH_SIZE; i++) inputss[i * INPUT_N + INPUT_N - 1] = 1.0;

    signed char labels[TRAINING_BATCH_SIZE];
#else
    float hiddenDeltas[HIDDEN_N];
    float outputDeltas[OUTPUT_N];


    inputs[INPUT_N - 1] = 1;
#endif

    // Training part

    bool running = true;
    int epoch = 0;
    image ret;

    open_training_files();
    open_test_files();

    while (running) {
        float learningRate = LEARNING_RATE_BASE / (1 + LEARNING_RATE_DECAY * epoch);
        float sumDelta = 0;

#if FUTHARK
        for (int k = 0; k < TRAINING_BATCH_SIZE; k++) {
            int class = rand() % 59000;
            read_training_image(class, &ret);

            toInputs(ret, inputss + k * INPUT_N);

            labels[k] = ret.label;
        }

        struct futhark_f32_2d* inputss_arr = futhark_new_f32_2d(ctx, inputss, TRAINING_BATCH_SIZE, INPUT_N);
        struct futhark_i8_1d* labels_arr = futhark_new_i8_1d(ctx, labels, TRAINING_BATCH_SIZE);
        struct futhark_f32_2d* hiddenWeights_arr = futhark_new_f32_2d(ctx, hiddenWeights, HIDDEN_N, INPUT_N);
        struct futhark_f32_2d* outputWeights_arr = futhark_new_f32_2d(ctx, outputWeights, OUTPUT_N, HIDDEN_N);

        int res = futhark_entry_train_batch(ctx, &hiddenWeights_arr, &outputWeights_arr, &sumDelta,
                                            TRAINING_BATCH_SIZE, INPUT_N, HIDDEN_N, OUTPUT_N, learningRate,
                                            inputss_arr, hiddenWeights_arr, outputWeights_arr, labels_arr);

        int sync = futhark_context_sync(ctx);

        if (res != 0 || sync != 0) {
            fprintf(stderr, "Something wrong happened in Futhark code. Result: %d, Sync: %d\n", res, sync);
            exit(1);
        }

        futhark_values_f32_2d(ctx, hiddenWeights_arr, hiddenWeights);
        futhark_values_f32_2d(ctx, outputWeights_arr, outputWeights);
        futhark_free_f32_2d(ctx, hiddenWeights_arr);
        futhark_free_f32_2d(ctx, outputWeights_arr);
        futhark_free_f32_2d(ctx, inputss_arr);
        futhark_free_i8_1d(ctx, labels_arr);
#else
        for (int k = 0; k < TRAINING_BATCH_SIZE; k++) {
            int class = rand() % 59000;
            read_training_image(class, &ret);

            toInputs(ret, inputs);
            toExpectedOutputs(ret, Y);

            testPredict(inputs, hiddenWeights, hiddenPotential, hiddenOutputs, outputWeights, outputPotential,
                        outputOutputs);

            for (int i = 0; i < OUTPUT_N; i++) {
                outputDeltas[i] = sig_derivative(outputPotential[i]) * (Y[i] - outputOutputs[i]);
                sumDelta += fabs(outputDeltas[i]);
            }

            for (int i = 0; i < HIDDEN_N; i++) {
                float sum = 0;

                for (int j = 0; j < OUTPUT_N; j++) {
                    sum += outputDeltas[j] * outputWeights[(j * HIDDEN_N) + i];
                }

                hiddenDeltas[i] = sig_derivative(hiddenPotential[i]) * sum;
            }

            for (int i = 0; i < OUTPUT_N; i++) {
                for (int j = 0; j < HIDDEN_N; j++) {
                    outputWeights[(i * HIDDEN_N) + j] += learningRate * outputDeltas[i] * hiddenOutputs[j];
                }
            }

            for (int i = 0; i < HIDDEN_N; i++) {
                for (int j = 0; j < INPUT_N; j++) {
                    hiddenWeights[(i * INPUT_N) + j] += learningRate * hiddenDeltas[i] * inputs[j];
                }
            }
        }
#endif
        float errorSum = 0;

        for (int k = 0; k < TRAINING_SAMPLE_SIZE; k++) {
            int class = rand() % 59000;
            read_test_image(class, &ret);

            toInputs(ret, inputs);
            toExpectedOutputs(ret, Y);
            testPredict(inputs, hiddenWeights, hiddenPotential, hiddenOutputs, outputWeights, outputPotential,
                        outputOutputs);

            errorSum += calculateError(Y, outputOutputs);
        }

        float error = errorSum;

#if DEBUG
        fprintf(stderr, "Epoch %d, LR: %f, Delta avg: %f, Error (%d): %.3f\n", epoch, learningRate,
                sumDelta / TRAINING_BATCH_SIZE, TRAINING_SAMPLE_SIZE, error);
#endif

        epoch++;

        running = sumDelta / TRAINING_BATCH_SIZE > TRAIN_ERROR_THRESHOLD;
    }

    close_training_files();

    // Testing part

#if DEBUG
    fprintf(stderr, "Testing NN...\n");
#endif

    int success = 0;
    for (int k = 0; k < TESTING_SAMPLE_SIZE; k++) {
        int class = rand() % 59000;
        read_test_image(class, &ret);

        toInputs(ret, inputs);
        testPredict(inputs, hiddenWeights, hiddenPotential, hiddenOutputs, outputWeights, outputPotential,
                    outputOutputs);
        if (predictLabel(outputOutputs) == ret.label) success++;
    }

#if DEBUG
    fprintf(stderr, "Accuracy: %.3f%%, Seed: %ld\n", success * 100.0 / TESTING_SAMPLE_SIZE, seed);
#endif

    return 0;
}
