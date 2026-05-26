#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "forward.h"

// Runs the full forward pass of the Transformer encoder-decoder example step by step.
int main() {
    /*
        Vocabulary:
        0 = <sos>
        1 = I
        2 = love
        3 = transformers
        4 = <eos>

        Therefore, the input for this example is as follows.
        Encoder input = [I, love]
        Decoder input = [<sos>, transformers]

        Meaning:
        The Encoder understands "I love", and the Decoder predicts the next token
        after seeing "<sos> transformers".
    */
    const int dModel = 4;
    const int numHeads = 2; // Number of parallel attention heads in Multi-Head Attention

    std::vector<std::string> vocab = {
        "<sos>", "I", "love", "transformers", "<eos>"
    };

    Matrix embeddingTable = {
        {0.12f, 0.10f, 0.08f, 0.05f}, // <sos>
        {0.95f, 0.15f, 0.20f, 0.05f}, // I
        {0.20f, 0.85f, 0.25f, 0.15f}, // love
        {0.25f, 0.20f, 0.95f, 0.82f}, // transformers
        {0.02f, 0.04f, 0.12f, 0.92f}  // <eos>
    };

    // The attention, FFN, and output weights below
    // are not model parameters obtained through actual training.
    //
    // They are also not completely random numbers.
    // They are small toy weights manually chosen
    // so that this example works stably and the computation flow is easy to explain.
    //
    // Why are these numbers used?
    // 1. Most values are kept around 0 to 1 so the computations do not grow excessively.
    // 2. Larger near-diagonal values help preserve the original dimensional information to some extent.
    // 3. Small off-diagonal values allow a small amount of information mixing across dimensions.
    // 4. The final example output remains interpretable and naturally leads to the <eos> prediction.
    //
    // Therefore, values such as 0.90f and 0.85f are not "answers found by training";
    // they are manually assigned values meaning "this dimension should pass through relatively strongly".

    // Weights for Encoder self-attention.
    // In order, these are Wq, Wk, Wv, and Wo, all with shape 4x4.
    //
    // Interpretation:
    // - Large diagonal values: each dimension strongly preserves its own information.
    // - Small off-diagonal values: information from other dimensions is weakly mixed.
    // - 0.00 values: some connections are effectively removed for simplicity.
    //
    // In other words, encoderSelf defines the linear transformation rules
    // that create Query, Key, and Value from the encoder input "I love".
    AttentionWeights encoderSelf {
        { // encoderSelf[0] → Encoder self-attention - Q(Q_enc)
            {0.90f, 0.10f, 0.00f, 0.10f},
            {0.10f, 0.85f, 0.05f, 0.00f},
            {0.00f, 0.15f, 0.80f, 0.10f},
            {0.05f, 0.00f, 0.20f, 0.75f}
        },
        { // encoderSelf[1] → Encoder self-attention - K(K_enc)
            {0.80f, 0.00f, 0.10f, 0.10f},
            {0.00f, 0.90f, 0.10f, 0.00f},
            {0.10f, 0.10f, 0.85f, 0.05f},
            {0.00f, 0.05f, 0.15f, 0.85f}
        },
        { // encoderSelf[2] → Encoder self-attention - V(V_enc)
            {0.85f, 0.10f, 0.00f, 0.00f},
            {0.10f, 0.80f, 0.15f, 0.00f},
            {0.00f, 0.10f, 0.90f, 0.10f},
            {0.05f, 0.00f, 0.10f, 0.95f}
        },
        { // encoderSelf[3] -> Converts the concatenated heads into the final MHA_enc.
            {0.95f, 0.05f, 0.00f, 0.00f},
            {0.05f, 0.95f, 0.00f, 0.00f},
            {0.00f, 0.10f, 0.85f, 0.05f},
            {0.00f, 0.05f, 0.10f, 0.90f}
        }
    };

    // Weights for Decoder masked self-attention.
    // The structure is the same as encoderSelf, but separate values are used for the decoder block.
    //
    // These are also manually assigned example values, not trained values.
    // The reason for using values similar in scale to encoderSelf but slightly different is
    // to make it clear in the code that encoder self-attention and decoder self-attention
    // are different blocks.
    //
    // The actual attention result is affected not only by these weights but also by the causal mask.
    AttentionWeights decoderSelf {
        {
            {0.88f, 0.08f, 0.00f, 0.12f},
            {0.10f, 0.82f, 0.10f, 0.00f},
            {0.00f, 0.10f, 0.90f, 0.05f},
            {0.05f, 0.00f, 0.18f, 0.78f}
        },
        {
            {0.82f, 0.00f, 0.12f, 0.08f},
            {0.00f, 0.88f, 0.10f, 0.00f},
            {0.10f, 0.12f, 0.82f, 0.08f},
            {0.00f, 0.04f, 0.18f, 0.88f}
        },
        {
            {0.84f, 0.12f, 0.00f, 0.00f},
            {0.10f, 0.82f, 0.12f, 0.00f},
            {0.00f, 0.12f, 0.88f, 0.08f},
            {0.04f, 0.00f, 0.10f, 0.94f}
        },
        {
            {0.92f, 0.08f, 0.00f, 0.00f},
            {0.08f, 0.90f, 0.02f, 0.00f},
            {0.00f, 0.10f, 0.86f, 0.04f},
            {0.00f, 0.04f, 0.12f, 0.92f}
        }
    };

    // Weights for Decoder cross-attention.
    //
    // Query comes from the decoder state,
    // while Key and Value come from the encoder output.
    // Therefore, these weights define the rules
    // for how the decoder queries the encoder memory.
    //
    // These are also human-made example values, not trained values.
    // The reason for not using exactly the same values as self-attention is
    // to separate the role difference between "attention to itself"
    // and "attention that refers to the encoder".
    AttentionWeights crossAttention {
        {
            {0.90f, 0.05f, 0.00f, 0.10f},
            {0.05f, 0.88f, 0.12f, 0.00f},
            {0.00f, 0.12f, 0.88f, 0.08f},
            {0.05f, 0.00f, 0.15f, 0.85f}
        },
        {
            {0.78f, 0.00f, 0.10f, 0.12f},
            {0.00f, 0.92f, 0.08f, 0.00f},
            {0.12f, 0.10f, 0.82f, 0.06f},
            {0.00f, 0.04f, 0.18f, 0.88f}
        },
        {
            {0.86f, 0.08f, 0.00f, 0.02f},
            {0.08f, 0.84f, 0.12f, 0.00f},
            {0.00f, 0.12f, 0.90f, 0.08f},
            {0.04f, 0.00f, 0.10f, 0.96f}
        },
        {
            {0.92f, 0.08f, 0.00f, 0.00f},
            {0.08f, 0.88f, 0.04f, 0.00f},
            {0.00f, 0.12f, 0.84f, 0.04f},
            {0.00f, 0.04f, 0.12f, 0.90f}
        }
    };

    // Encoder FFN weights.
    //
    // Structure:
    // - First matrix W1: 4x8
    // - First bias b1: length 8
    // - Second matrix W2: 8x4
    // - Second bias b2: length 4
    //
    // These are also manually assigned values, not values obtained through training.
    //
    // Why use the 4 -> 8 -> 4 structure?
    // - This demonstrates the structure of expanding the intermediate hidden dimension
    //   and then reducing it back to d_model, as in a real Transformer FFN.
    //
    // Why are many values positive?
    // - This prevents too much information from disappearing after passing through ReLU.
    //
    // Why are small negative values mixed into the bias?
    // - This gives a weak reference point, similar to a threshold,
    //   so that not all hidden units are always activated.
    FFNWeights encoderFFN {
        {
            {0.40f, 0.10f, 0.30f, 0.00f, 0.20f, 0.00f, 0.35f, 0.10f},
            {0.10f, 0.45f, 0.05f, 0.25f, 0.15f, 0.30f, 0.00f, 0.20f},
            {0.00f, 0.15f, 0.50f, 0.20f, 0.00f, 0.25f, 0.35f, 0.10f},
            {0.20f, 0.00f, 0.10f, 0.45f, 0.35f, 0.00f, 0.10f, 0.40f}
        },
        {0.05f, -0.02f, 0.03f, 0.01f, 0.00f, -0.01f, 0.02f, 0.04f},
        {
            {0.35f, 0.05f, 0.00f, 0.10f},
            {0.00f, 0.40f, 0.08f, 0.05f},
            {0.10f, 0.05f, 0.38f, 0.00f},
            {0.00f, 0.10f, 0.12f, 0.36f},
            {0.25f, 0.00f, 0.10f, 0.08f},
            {0.05f, 0.22f, 0.08f, 0.02f},
            {0.10f, 0.05f, 0.25f, 0.05f},
            {0.00f, 0.08f, 0.10f, 0.30f}
        },
        {0.01f, 0.02f, 0.01f, 0.03f}
    };

    // Decoder FFN weights.
    //
    // The structure is the same 4 -> 8 -> 4 structure as encoderFFN.
    // These values are also manually assigned example values, not trained values.
    //
    // Why are the numbers slightly different from encoderFFN?
    // - The decoder handles representations that have passed through masked self-attention and cross-attention,
    //   so this shows that it does not use exactly the same transformation as the encoder.
    // - Another purpose is to transform the decoder state
    //   in a direction suitable for next-token prediction before the final vocabulary projection.
    FFNWeights decoderFFN {
        {
            {0.42f, 0.08f, 0.28f, 0.00f, 0.24f, 0.00f, 0.30f, 0.12f},
            {0.08f, 0.44f, 0.08f, 0.22f, 0.14f, 0.32f, 0.02f, 0.22f},
            {0.00f, 0.18f, 0.52f, 0.18f, 0.00f, 0.28f, 0.30f, 0.12f},
            {0.22f, 0.00f, 0.12f, 0.48f, 0.32f, 0.00f, 0.12f, 0.42f}
        },
        {0.03f, -0.01f, 0.04f, 0.02f, 0.01f, -0.02f, 0.02f, 0.05f},
        {
            {0.36f, 0.05f, 0.00f, 0.08f},
            {0.00f, 0.38f, 0.10f, 0.05f},
            {0.12f, 0.05f, 0.36f, 0.00f},
            {0.00f, 0.12f, 0.12f, 0.34f},
            {0.22f, 0.00f, 0.10f, 0.08f},
            {0.05f, 0.20f, 0.10f, 0.02f},
            {0.10f, 0.05f, 0.22f, 0.06f},
            {0.00f, 0.08f, 0.12f, 0.28f}
        },
        {0.02f, 0.03f, 0.02f, 0.03f}
    };

    // Final vocabulary projection weights.
    //
    // The shape is 4x5.
    // - Input: final decoder state h_last (length 4)
    // - Output: logits for the 5 vocabulary tokens
    //
    // Formula:
    // z = h_last * W_vocab
    //
    // These are also manually assigned example values, not trained output-layer parameters.
    // In particular, in this example, some columns are adjusted
    // so that <eos> is likely to receive the highest final probability.
    //
    // Therefore, each number can be understood as a coefficient
    // that determines "how much a word score should increase
    // when a specific dimension of the decoder state becomes large".
    Matrix outputWeight = {
        {0.10f, 0.20f, 0.12f, 0.85f, 0.30f},
        {0.12f, 0.10f, 0.15f, 0.70f, 0.28f},
        {0.00f, 0.10f, 0.18f, 1.20f, 0.25f},
        {0.00f, 0.05f, 0.10f, 0.95f, 0.85f}
    };


    // ******************************************************************************************************************
    // ** ENCODER **
    // ******************************************************************************************************************
    std::vector<int> encoderTokenIds = {1, 2};
    std::vector<int> decoderTokenIds = {0, 3};

    // STEP 1. Inputs
    // X_id = [1, 2]
    // Y_id = [0, 3]
    //
    // The Encoder receives the input sentence "I love".
    // The Decoder receives "<sos> transformers" as a teacher-forcing example
    // and predicts the next token at the final position.
    stage(1, "Inputs");
    std::cout << "Input sentence: I love\n";
    std::cout << "Example output: transformers <eos>\n";
    std::cout << "Decoder input uses teacher forcing with <sos> transformers.\n";
    printTokenIds("Encoder token IDs", encoderTokenIds);
    printTokenIds("Decoder token IDs", decoderTokenIds);

    // STEP 2. Embedding Lookup
    // Converts integer token IDs into semantic vectors of length d_model = 4.
    //
    // Shape of the embedding table E:
    // |V| x d_model = 5 x 4
    //
    // Example:
    // I    -> [0.9500, 0.1500, 0.2000, 0.0500]
    // love -> [0.2000, 0.8500, 0.2500, 0.1500]
    stage(2, "Embedding Lookup");
    Matrix encoderInput = tokenEmbedding(encoderTokenIds, embeddingTable);
    Matrix decoderInput = tokenEmbedding(decoderTokenIds, embeddingTable);
    printMatrix("Encoder embedding", encoderInput);
    printMatrix("Decoder embedding", decoderInput);

    // STEP 3. Positional Encoding
    // Creates sine/cosine vectors for each position (pos) and adds them to the embeddings.
    //
    // For example, the positional encoding for position 0 is
    // [0.0000, 1.0000, 0.0000, 1.0000],
    // and when it is added to the embedding of "I", [0.9500, 0.1500, 0.2000, 0.0500],
    // the result becomes [0.9500, 1.1500, 0.2000, 1.0500].
    stage(3, "Sinusoidal Positional Encoding");
    Matrix encoderPE = positionalEncoding(static_cast<int>(encoderInput.size()), dModel);
    Matrix decoderPE = positionalEncoding(static_cast<int>(decoderInput.size()), dModel);
    printMatrix("Encoder positional encoding", encoderPE);
    printMatrix("Decoder positional encoding", decoderPE);

    encoderInput = addMatrix(encoderInput, encoderPE);
    decoderInput = addMatrix(decoderInput, decoderPE);
    printMatrix("Encoder input + position", encoderInput);
    printMatrix("Decoder input + position", decoderInput);

    // STEP 4. Encoder Self-Attention
    // In Encoder self-attention, Query, Key, and Value all come from the encoder input.
    //
    // Q = XW^Q
    // K = XW^K
    // V = XW^V
    //
    // Each encoder token attends to the entire input sequence
    // and computes how "I" and "love" are related to each other.
    stage(4, "Encoder Self-Attention");
    std::cout << "각 encoder 토큰이 전체 입력 시퀀스를 본다.\n";
    Matrix encoderAttention = multiHeadAttention(
        encoderInput, encoderInput, encoderSelf, numHeads, false, "Encoder self-attention"
    );

    // STEP 5. Encoder Add & Norm
    // X_tilde = LayerNorm(X^(0) + MHA(X^(0)))
    //
    // Residual Add:
    // Combines the original input information with the attention result.
    //
    // LayerNorm:
    // Normalizes the mean and variance of each token vector to stabilize training and computation.
    stage(5, "Encoder Add & Norm");
    Matrix encoderNorm1 = addNorm(encoderInput, encoderAttention, "Encoder add&norm 1");

    // STEP 6. Encoder Feed Forward
    // Multi-Head Attention(STEP4) -> Add&Norm -> FFN
    // The FFN nonlinearly transforms the context information mixed by attention at each token position.
    // Then another Add & Norm is performed to produce the final encoder output.
    //
    // X_enc = LayerNorm(X_tilde + FFN(X_tilde))
    stage(6, "Encoder Feed Forward");
    Matrix encoderFFNOut = feedForwardBlock(encoderNorm1, encoderFFN, "Encoder FFN");
    Matrix encoderOutput = addNorm(encoderNorm1, encoderFFNOut, "Encoder add&norm 2");


    // ******************************************************************************************************************
    // ** DECODER **
    // ******************************************************************************************************************

    // STEP 7. Decoder Masked Self-Attention
    // Decoder self-attention also creates Q, K, and V from the decoder input,
    // but applies a causal mask because future tokens must not be visible.
    //
    // Mask:
    // M(i, j) = 0      if j <= i
    // M(i, j) = -inf   if j > i
    //
    // Therefore, the first position <sos> cannot see the later token "transformers",
    // while the second position can see <sos> and itself.
    stage(7, "Decoder Masked Self-Attention");
    std::cout << "Decoder는 미래 토큰을 보지 못하도록 causal mask를 적용\n";
    Matrix decoderMaskedAttention = multiHeadAttention(
        decoderInput, decoderInput, decoderSelf, numHeads, true, "Decoder masked self-attention"
    );

    // STEP 8. Decoder Add & Norm After Masked Attention
    // Y_tilde_1 = LayerNorm(Y^(0) + MaskedMHA(Y^(0)))
    //
    // Combines the token information seen by the decoder so far
    // with the masked self-attention result to create the current decoder state.
    stage(8, "Decoder Add & Norm After Masked Attention");
    Matrix decoderNorm1 = addNorm(decoderInput, decoderMaskedAttention, "Decoder add&norm 1");

    // ******************************************************************************************************************
    // ** DECODER **
    // ******************************************************************************************************************

    // STEP 9. Decoder Cross-Attention
    // Here, the sources are separated.
    //
    // Query  = decoder-side state
    // Key    = encoder output
    // Value  = encoder output
    //
    // Q = YW^Q
    // K = X_enc W^K
    // V = X_enc W^V
    //
    // In other words, when the decoder predicts the next word,
    // it computes which part of the encoder-understood "I love" should be referenced.
    stage(9, "Decoder Cross-Attention");
    std::cout << "Decoder가 encoder output 전체를 memory로 참고한다.\n";
    Matrix crossAttentionOut = multiHeadAttention(
        decoderNorm1, encoderOutput, crossAttention, numHeads, false, "Decoder cross-attention"
    );

    // STEP 10. Decoder Add & Norm After Cross-Attention
    // Y_tilde_2 = LayerNorm(Y_tilde_1 + CrossMHA(Y_tilde_1, X_enc))
    //
    // Combines the decoder internal context with the encoder memory
    // and updates it into a decoder representation that refers to the input sentence.
    stage(10, "Decoder Add & Norm After Cross-Attention");
    Matrix decoderNorm2 = addNorm(decoderNorm1, crossAttentionOut, "Decoder add&norm 2");

    // STEP 11. Decoder Feed Forward
    // Transforms the input-output relationship representation obtained through cross-attention
    // into a form that is more suitable for next-token prediction.
    //
    // Y_dec = LayerNorm(Y_tilde_2 + FFN(Y_tilde_2))
    stage(11, "Decoder Feed Forward");
    Matrix decoderFFNOut = feedForwardBlock(decoderNorm2, decoderFFN, "Decoder FFN");
    Matrix decoderOutput = addNorm(decoderNorm2, decoderFFNOut, "Decoder add&norm 3");

    // STEP 12. Linear Projection To Vocabulary
    // decoderOutput.back() means the vector at the "last position" of the decoder sequence.
    //
    // Since the current decoder input is [<sos>, transformers],
    // the last position is the "transformers" position.
    // One next token is predicted from this position.
    //
    // Formula:
    // z = h_last W_vocab
    stage(12, "Linear Projection To Vocabulary");
    Vec finalDecoderState = decoderOutput.back();
    printVec("Final decoder state (last position)", finalDecoderState);

    Vec logits(vocab.size(), 0.0f);
    for (size_t vocabIdx = 0; vocabIdx < vocab.size(); ++vocabIdx) {
        for (int d = 0; d < dModel; ++d) {
            // Multiplies the final decoder state by the vocabulary projection weights
            // to produce a score (logit) for each token (<sos>, I, love, transformers, <eos>).
            logits[vocabIdx] += finalDecoderState[d] * outputWeight[d][vocabIdx];
        }
    }

    printVec("Logits", logits);

    // Softmax:
    // P(v) = exp(z_v) / sum_j exp(z_j)
    //
    // The token with the highest probability becomes the predicted next token.
    Vec probs = softmax(logits);
    std::cout << "\nFinal Probabilities:\n";
    for (size_t i = 0; i < vocab.size(); ++i) {
        std::cout << std::setw(12) << vocab[i] << " : "
                  << std::fixed << std::setprecision(6) << probs[i] << "\n";
    }

    int predicted = argmax(probs);
    std::cout << "\nPredicted next token: " << vocab[predicted] << "\n";
    std::cout << "In this example, the last decoder position predicts the next token candidate (<eos>, etc.).\n";

    // STEP 13. Summary
    // 1. Creates input token IDs.
    // 2. Converts token IDs into embedding vectors.
    // 3. Adds positional encoding.
    // 4. Performs encoder self-attention.
    // 5. Performs encoder Add & Norm.
    // 6. Creates encoder output through encoder FFN and Add & Norm.
    // 7. Performs decoder masked self-attention.
    // 8. Performs decoder Add & Norm.
    // 9. Refers to the encoder output through decoder cross-attention.
    // 10. Performs decoder Add & Norm.
    // 11. Creates decoder output through decoder FFN and Add & Norm.
    // 12. Projects the final decoder position into vocabulary logits.
    // 13. Computes next-token probabilities using softmax.
    //
    // Final one-line conclusion:
    // This code is a Transformer encoder-decoder example in which the Encoder converts "I love"
    // into context vectors, and the Decoder, after seeing "<sos> transformers",
    // predicts <eos> as the next token.
    stage(13, "Summary");
    std::cout << "1. forward.cpp processes the entire sequence with attention, not just a single token.\n";
    std::cout << "2. Creates Q, K, and V as separate projection matrices\n";
    std::cout << "3. Splits the multi-head attention into two heads and merges them back together\n";
    std::cout << "4. The decoder self-attention includes a causal mask.\n";
    std::cout << "5. The FFN has been extended to a 2-layer structure, and the final state has been replaced with vocab logits\n";

    return 0;
}
