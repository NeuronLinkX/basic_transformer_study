#include "forward.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Prints a computation stage as a STEP title block.
void stage(int n, const std::string& title) {
    std::cout << "\n====================================================\n";
    std::cout << "STEP " << n << ". " << title << "\n";
    std::cout << "====================================================\n";
}

// Prints a one-dimensional vector in fixed-point format.
void printVec(const std::string& name, const Vec& v) {
    std::cout << name << " = [ ";
    for (float x : v) {
        std::cout << std::fixed << std::setprecision(4) << x << " ";
    }
    std::cout << "]\n";
}

// Prints a two-dimensional matrix row by row in a readable format.
void printMatrix(const std::string& name, const Matrix& m) {
    std::cout << name << ":\n";
    for (const auto& row : m) {
        std::cout << "  [ ";
        for (float x : row) {
            std::cout << std::fixed << std::setprecision(4) << x << " ";
        }
        std::cout << "]\n";
    }
}

// Throws an exception when the condition is false to stop invalid computation immediately.
void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Computes the dot product of two vectors.
float dot(const Vec& a, const Vec& b) {
    check(a.size() == b.size(), "dot: vector size mismatch");

    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Converts a one-dimensional score vector into a probability distribution.
Vec softmax(const Vec& logits) {
    check(!logits.empty(), "softmax: empty logits");

    Vec result(logits.size());
    float maxValue = *std::max_element(logits.begin(), logits.end());

    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        result[i] = std::exp(logits[i] - maxValue);
        sum += result[i];
    }

    for (float& x : result) {
        x /= sum;
    }

    return result;
}

// Applies softmax to each row of a matrix.
Matrix softmaxRows(const Matrix& logits) {
    Matrix result = logits;
    for (auto& row : result) {
        row = softmax(row);
    }
    return result;
}

// Creates a zero-filled matrix with the given size.
Matrix zeros(size_t rows, size_t cols) {
    return Matrix(rows, Vec(cols, 0.0f));
}

// Creates the transpose of a matrix by swapping rows and columns.
Matrix transpose(const Matrix& m) {
    check(!m.empty(), "transpose: empty matrix");

    Matrix result(m[0].size(), Vec(m.size(), 0.0f));
    for (size_t r = 0; r < m.size(); ++r) {
        check(m[r].size() == m[0].size(), "transpose: ragged matrix");
        for (size_t c = 0; c < m[r].size(); ++c) {
            result[c][r] = m[r][c];
        }
    }
    return result;
}

// Performs matrix multiplication.
Matrix matMul(const Matrix& a, const Matrix& b) {
    check(!a.empty() && !b.empty(), "matMul: empty matrix");
    check(a[0].size() == b.size(), "matMul: incompatible shapes");

    Matrix result(a.size(), Vec(b[0].size(), 0.0f));
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t k = 0; k < a[0].size(); ++k) {
            for (size_t j = 0; j < b[0].size(); ++j) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

// Adds two matrices with the same shape element by element.
Matrix addMatrix(const Matrix& a, const Matrix& b) {
    check(a.size() == b.size(), "addMatrix: row mismatch");
    check(!a.empty() && a[0].size() == b[0].size(), "addMatrix: col mismatch");

    Matrix result = a;
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[i].size(); ++j) {
            result[i][j] += b[i][j];
        }
    }
    return result;
}

// Adds two vectors with the same length element by element.
Vec addVec(const Vec& a, const Vec& b) {
    check(a.size() == b.size(), "addVec: size mismatch");

    Vec result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

// Adds the same bias vector to each row of a matrix.
Matrix addBias(const Matrix& x, const Vec& bias) {
    check(!x.empty(), "addBias: empty matrix");
    check(x[0].size() == bias.size(), "addBias: size mismatch");

    Matrix result = x;
    for (auto& row : result) {
        for (size_t i = 0; i < row.size(); ++i) {
            row[i] += bias[i];
        }
    }
    return result;
}

// Computes the linear transformation xW (+ b).
Matrix linear(const Matrix& x, const Matrix& w, const Vec& bias) {
    Matrix y = matMul(x, w);
    if (!bias.empty()) {
        y = addBias(y, bias);
    }
    return y;
}

// Applies the ReLU activation function to every element of a matrix.
Matrix relu(const Matrix& x) {
    Matrix result = x;
    for (auto& row : result) {
        for (float& v : row) {
            v = std::max(0.0f, v);
        }
    }
    return result;
}

// Applies mean/variance-based LayerNorm to a single vector.
Vec layerNormOne(const Vec& x) {
    float mean = 0.0f;
    for (float v : x) {
        mean += v;
    }
    mean /= static_cast<float>(x.size());

    float var = 0.0f;
    for (float v : x) {
        float diff = v - mean;
        var += diff * diff;
    }
    var /= static_cast<float>(x.size());

    float invStd = 1.0f / std::sqrt(var + 1e-5f);

    Vec result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = (x[i] - mean) * invStd;
    }
    return result;
}

// Applies LayerNorm to each row of a matrix.
Matrix layerNorm(const Matrix& x) {
    Matrix result = x;
    for (auto& row : result) {
        row = layerNormOne(row);
    }
    return result;
}

// Looks up a token ID sequence in the embedding table and converts it into a vector sequence.
Matrix tokenEmbedding(const std::vector<int>& tokenIds, const Matrix& embeddingTable) {
    // A token ID is only an integer index, so semantic computation cannot be performed on it directly.
    // Therefore, the corresponding row is retrieved from the embedding table,
    // and each token is converted into a d_model-dimensional real-valued vector.
    //
    // Formula:
    // X = E[X_id]
    // Y = E[Y_id]
    Matrix result;
    for (int tokenId : tokenIds) {
        check(tokenId >= 0 && static_cast<size_t>(tokenId) < embeddingTable.size(),
              "tokenEmbedding: token id out of range");
        result.push_back(embeddingTable[tokenId]);
    }
    return result;
}

// Generates a sine/cosine-based positional encoding matrix.
Matrix positionalEncoding(int seqLen, int dModel) {
    // Since a Transformer does not pass hidden states sequentially like an RNN,
    // it must be explicitly told which position each token occupies.
    //
    // Formula:
    // PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    // PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    //
    // Final input:
    // X^(0) = X + PE
    // Y^(0) = Y + PE
    Matrix pe(seqLen, Vec(dModel, 0.0f));

    for (int pos = 0; pos < seqLen; ++pos) {
        for (int i = 0; i < dModel; ++i) {
            float denom = std::pow(10000.0f, static_cast<float>(2 * (i / 2)) / dModel);
            float angle = static_cast<float>(pos) / denom;
            pe[pos][i] = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
    }

    return pe;
}

// Extracts a contiguous range of columns from a matrix and creates a new matrix.
Matrix sliceColumns(const Matrix& m, int start, int width) {
    check(!m.empty(), "sliceColumns: empty matrix");
    check(start >= 0 && width > 0, "sliceColumns: invalid slice");
    check(static_cast<size_t>(start + width) <= m[0].size(), "sliceColumns: out of range");

    Matrix result(m.size(), Vec(width, 0.0f));
    for (size_t r = 0; r < m.size(); ++r) {
        for (int c = 0; c < width; ++c) {
            result[r][c] = m[r][start + c];
        }
    }
    return result;
}

// Concatenates multiple head outputs along the column dimension into one matrix.
Matrix concatColumns(const std::vector<Matrix>& parts) {
    check(!parts.empty(), "concatColumns: empty parts");

    size_t rows = parts[0].size();
    size_t totalCols = 0;
    for (const auto& part : parts) {
        check(part.size() == rows, "concatColumns: row mismatch");
        totalCols += part[0].size();
    }

    Matrix result(rows, Vec(totalCols, 0.0f));
    for (size_t r = 0; r < rows; ++r) {
        size_t offset = 0;
        for (const auto& part : parts) {
            for (size_t c = 0; c < part[r].size(); ++c) {
                result[r][offset + c] = part[r][c];
            }
            offset += part[r].size();
        }
    }
    return result;
}

// Computes scaled dot-product attention and applies a causal mask if needed.
Matrix scaledDotAttention(
    const Matrix& q,
    const Matrix& k,
    const Matrix& v,
    bool causalMask,
    Matrix* scoresOut,
    Matrix* probsOut
) {
    // Core computation of attention:
    // 1. Compute the similarity between each Query and Key.
    //    score(i, j) = (Q_i dot K_j) / sqrt(d_k)
    // 2. Mask future positions if needed.
    // 3. Convert the scores into a probability distribution using softmax.
    // 4. Use those probabilities to compute a weighted sum of Values and create context vectors.
    //
    // Formula:
    // Score = (QK^T) / sqrt(d_k)
    // A = softmax(Score)
    // Attention(Q, K, V) = AV
    check(!q.empty() && !k.empty() && !v.empty(), "scaledDotAttention: empty matrix");
    check(k.size() == v.size(), "scaledDotAttention: K/V length mismatch");
    check(q[0].size() == k[0].size(), "scaledDotAttention: Q/K depth mismatch");
    check(k[0].size() == v[0].size(), "scaledDotAttention: K/V depth mismatch");

    const float scale = std::sqrt(static_cast<float>(q[0].size()));
    Matrix scores(q.size(), Vec(k.size(), 0.0f));

    for (size_t i = 0; i < q.size(); ++i) {
        for (size_t j = 0; j < k.size(); ++j) {
            float score = dot(q[i], k[j]) / scale;
            if (causalMask && j > i) {
                // In decoder masked self-attention,
                // the current position i must not attend to a future position j > i.
                // Therefore, the score is set to a very small value (-1e9) so it becomes almost zero after softmax.
                score = -1e9f;
            }
            scores[i][j] = score;
        }
    }

    Matrix probs = softmaxRows(scores);
    Matrix context(q.size(), Vec(v[0].size(), 0.0f));

    for (size_t i = 0; i < q.size(); ++i) {
        for (size_t j = 0; j < v.size(); ++j) {
            for (size_t d = 0; d < v[j].size(); ++d) {
                context[i][d] += probs[i][j] * v[j][d];
            }
        }
    }

    if (scoresOut != nullptr) {
        *scoresOut = scores;
    }
    if (probsOut != nullptr) {
        *probsOut = probs;
    }

    return context;
}

// Performs the full multi-head attention process, from Q/K/V generation to head concatenation and output projection.
Matrix multiHeadAttention(
    const Matrix& queryInput,
    const Matrix& keyValueInput,
    const AttentionWeights& weights,
    int numHeads,
    bool causalMask,
    const std::string& name
) {
    check(!queryInput.empty() && !keyValueInput.empty(), "multiHeadAttention: empty sequence");
    const int dModel = static_cast<int>(queryInput[0].size());
    check(dModel % numHeads == 0, "multiHeadAttention: d_model must be divisible by num_heads");
    const int dHead = dModel / numHeads;

    // First, linearly project the input into the Query, Key, and Value spaces.
    //
    // For encoder self-attention:
    // Q = XW^Q, K = XW^K, V = XW^V
    //
    // For decoder masked self-attention:
    // Q = YW^Q, K = YW^K, V = YW^V
    //
    // For cross-attention:
    // Q = YW^Q
    // K = X_enc W^K
    // V = X_enc W^V
    Matrix Q = linear(queryInput, weights.Wq);
    Matrix K = linear(keyValueInput, weights.Wk);
    Matrix V = linear(keyValueInput, weights.Wv);

    printMatrix(name + " - Q", Q);
    printMatrix(name + " - K", K);
    printMatrix(name + " - V", V);

    std::vector<Matrix> headOutputs;

    for (int head = 0; head < numHeads; ++head) {
        // Split d_model across num_heads so that each head handles only part of the dimensions.
        // In this example, since d_model = 4 and num_heads = 2,
        // each head processes d_head = 2 dimensions.
        Matrix qHead = sliceColumns(Q, head * dHead, dHead);
        Matrix kHead = sliceColumns(K, head * dHead, dHead);
        Matrix vHead = sliceColumns(V, head * dHead, dHead);

        Matrix headScores;
        Matrix headProbs;
        Matrix headContext = scaledDotAttention(
            qHead, kHead, vHead, causalMask, &headScores, &headProbs
        );

        printMatrix(name + " - head " + std::to_string(head) + " scores", headScores);
        printMatrix(name + " - head " + std::to_string(head) + " probs", headProbs);
        printMatrix(name + " - head " + std::to_string(head) + " context", headContext);

        headOutputs.push_back(headContext);
    }

    // Concatenate the outputs of all heads side by side,
    // then apply the final output projection W^O,
    // and merge the result back into the original d_model dimension.
    Matrix concatenated = concatColumns(headOutputs);
    printMatrix(name + " - concatenated heads", concatenated);

    Matrix projected = linear(concatenated, weights.Wo);
    printMatrix(name + " - output projection", projected);

    return projected;
}

// Performs the position-wise FFN consisting of two linear layers and ReLU.
Matrix feedForwardBlock(const Matrix& x, const FFNWeights& weights, const std::string& name) {
    // Position-wise FFN:
    // If attention mixes relationships between tokens,
    // the FFN nonlinearly transforms the internal features of each token vector.
    //
    // Formula:
    // FFN(x) = ReLU(xW1 + b1)W2 + b2
    Matrix hidden = linear(x, weights.W1, weights.b1);
    printMatrix(name + " - hidden before ReLU", hidden);

    Matrix activated = relu(hidden);
    printMatrix(name + " - hidden after ReLU", activated);

    Matrix output = linear(activated, weights.W2, weights.b2);
    printMatrix(name + " - output", output);

    return output;
}

// Applies a residual connection and LayerNorm to stabilize the sublayer output.
Matrix addNorm(const Matrix& residual, const Matrix& sublayerOutput, const std::string& name) {
    // Standard Transformer block:
    // 1. Preserve the original input information through the residual connection. In deep networks, vanishing gradients can cause information loss and unstable learning.
    // 2. Stabilize the value distribution with LayerNorm.
    //
    // Formula:
    // LayerNorm(residual + sublayerOutput)
    Matrix added = addMatrix(residual, sublayerOutput);
    Matrix normalized = layerNorm(added);

    printMatrix(name + " - residual + sublayer", added);
    printMatrix(name + " - normalized", normalized);

    return normalized;
}

// Prints a token ID sequence on one line.
void printTokenIds(const std::string& name, const std::vector<int>& tokenIds) {
    std::cout << name << " = [ ";
    for (int tokenId : tokenIds) {
        std::cout << tokenId << " ";
    }
    std::cout << "]\n";
}

// Returns the index of the position with the largest value.
int argmax(const Vec& v) {
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}
