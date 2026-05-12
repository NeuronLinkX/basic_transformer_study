#include "forward.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// 계산 단계를 STEP 제목 형태로 구분해 출력
void stage(int n, const std::string& title) {
    std::cout << "\n====================================================\n";
    std::cout << "STEP " << n << ". " << title << "\n";
    std::cout << "====================================================\n";
}

// 1차원 벡터를 고정 소수점 형식으로 출력
void printVec(const std::string& name, const Vec& v) {
    std::cout << name << " = [ ";
    for (float x : v) {
        std::cout << std::fixed << std::setprecision(4) << x << " ";
    }
    std::cout << "]\n";
}

// 2차원 행렬을 행 단위로 보기 좋게 출력
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

// 조건이 거짓이면 예외를 던져 잘못된 계산을 즉시 중단한다.
void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// 두 벡터의 내적(dot product)을 계산한다.
float dot(const Vec& a, const Vec& b) {
    check(a.size() == b.size(), "dot: vector size mismatch");

    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// 1차원 점수 벡터를 확률 분포로 변환한다.
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

// 행렬의 각 행마다 softmax를 적용
Matrix softmaxRows(const Matrix& logits) {
    Matrix result = logits;
    for (auto& row : result) {
        row = softmax(row);
    }
    return result;
}

// 주어진 크기의 0으로 채운 행렬을 만듬
Matrix zeros(size_t rows, size_t cols) {
    return Matrix(rows, Vec(cols, 0.0f));
}

// 행렬의 행과 열을 뒤집은 전치행렬을 만듬
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

// 두 행렬의 곱셈을 수행
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

// 같은 shape의 두 행렬을 원소별로 더함
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

// 같은 길이의 두 벡터를 원소별로 더함
Vec addVec(const Vec& a, const Vec& b) {
    check(a.size() == b.size(), "addVec: size mismatch");

    Vec result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

// 행렬의 각 행에 동일한 bias 벡터를 더함
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

// 선형변환 xW (+ b)를 계산한다.
Matrix linear(const Matrix& x, const Matrix& w, const Vec& bias) {
    Matrix y = matMul(x, w);
    if (!bias.empty()) {
        y = addBias(y, bias);
    }
    return y;
}

// 행렬의 모든 원소에 ReLU 활성화 함수를 적용
Matrix relu(const Matrix& x) {
    Matrix result = x;
    for (auto& row : result) {
        for (float& v : row) {
            v = std::max(0.0f, v);
        }
    }
    return result;
}

// 벡터 하나에 대해 평균/분산 기반 LayerNorm을 적용
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

// 행렬의 각 행마다 LayerNorm을 적용
Matrix layerNorm(const Matrix& x) {
    Matrix result = x;
    for (auto& row : result) {
        row = layerNormOne(row);
    }
    return result;
}

// 토큰 ID 시퀀스를 embedding table에서 찾아 벡터 시퀀스로 바꿈
Matrix tokenEmbedding(const std::vector<int>& tokenIds, const Matrix& embeddingTable) {
    // Token ID는 단지 정수 인덱스일 뿐이므로 바로 의미 연산을 할 수 없다.
    // 따라서 embedding table에서 해당 행(row)을 꺼내
    // 각 토큰을 d_model 차원의 실수 벡터로 바꿈
    //
    // 수식:
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

// 사인/코사인 기반 positional encoding 행렬을 생성한다.
Matrix positionalEncoding(int seqLen, int dModel) {
    // Transformer는 RNN처럼 순서대로 hidden state를 넘기지 않기 때문에
    // "몇 번째 위치의 토큰인지"를 별도로 알려줘야 한다.
    //
    // 수식:
    // PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    // PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    //
    // 최종 입력:
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

// 행렬에서 연속된 일부 열만 잘라내 새 행렬로 만듬
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

// 여러 head 결과를 열 방향으로 이어 붙여 하나의 행렬로 합침
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

// scaled dot-product attention을 계산하고 필요하면 causal mask를 적용
Matrix scaledDotAttention(
    const Matrix& q,
    const Matrix& k,
    const Matrix& v,
    bool causalMask,
    Matrix* scoresOut,
    Matrix* probsOut
) {
    // Attention의 핵심 계산:
    // 1. 각 Query와 Key의 유사도를 구한다.
    //    score(i, j) = (Q_i dot K_j) / sqrt(d_k)
    // 2. 필요하면 미래 위치를 mask 처리한다.
    // 3. softmax로 확률 분포로 바꿈
    // 4. 그 확률로 Value를 가중합해 문맥 벡터를 만듬
    //
    // 수식:
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
                // Decoder masked self-attention에서는
                // 현재 위치 i가 미래 위치 j > i를 보면 안 된다.
                // 그래서 매우 작은 값(-1e9)으로 보내 softmax 후 거의 0이 되게 만듬
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

// Q/K/V 생성부터 head 결합과 output projection까지 multi-head attention 전체를 수행
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

    // 먼저 입력을 각각 Query, Key, Value 공간으로 선형 변환한다.
    //
    // Encoder self-attention이면:
    // Q = XW^Q, K = XW^K, V = XW^V
    //
    // Decoder masked self-attention이면:
    // Q = YW^Q, K = YW^K, V = YW^V
    //
    // Cross-attention이면:
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
        // d_model을 num_heads 개수만큼 나눠 각 head가 일부 차원만 담당하게 한다.
        // 이 예제에서는 d_model = 4, num_heads = 2 이므로
        // 각 head는 d_head = 2 차원씩 처리한다.
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

    // 모든 head의 출력을 옆으로 이어 붙인 뒤(concatenate),
    // 최종 출력 projection W^O를 한 번 더 적용해
    // 다시 원래 d_model 차원으로 합침
    Matrix concatenated = concatColumns(headOutputs);
    printMatrix(name + " - concatenated heads", concatenated);

    Matrix projected = linear(concatenated, weights.Wo);
    printMatrix(name + " - output projection", projected);

    return projected;
}

// 두 개의 선형층과 ReLU로 이루어진 position-wise FFN을 수행
Matrix feedForwardBlock(const Matrix& x, const FFNWeights& weights, const std::string& name) {
    // Position-wise FFN:
    // attention이 토큰 간 관계를 섞어줬다면,
    // FFN은 각 토큰 벡터 내부 특징을 비선형 변환한다.
    //
    // 수식:
    // FFN(x) = ReLU(xW1 + b1)W2 + b2
    Matrix hidden = linear(x, weights.W1, weights.b1);
    printMatrix(name + " - hidden before ReLU", hidden);

    Matrix activated = relu(hidden);
    printMatrix(name + " - hidden after ReLU", activated);

    Matrix output = linear(activated, weights.W2, weights.b2);
    printMatrix(name + " - output", output);

    return output;
}

// residual connection과 LayerNorm을 적용해 sublayer 출력을 안정화한다.
Matrix addNorm(const Matrix& residual, const Matrix& sublayerOutput, const std::string& name) {
    // Transformer의 표준 블록:
    // 1. Residual connection으로 원래 입력 정보를 보존하고 (deeplearning 층이 깊어질수록 : gradient vanishing 발생 --> 정보 손실 및 학습 불안정)
    // 2. LayerNorm으로 값 분포를 안정화한다.
    //
    // 수식:
    // LayerNorm(residual + sublayerOutput)
    Matrix added = addMatrix(residual, sublayerOutput);
    Matrix normalized = layerNorm(added);

    printMatrix(name + " - residual + sublayer", added);
    printMatrix(name + " - normalized", normalized);

    return normalized;
}

// 토큰 ID 시퀀스를 한 줄로 출력
void printTokenIds(const std::string& name, const std::vector<int>& tokenIds) {
    std::cout << name << " = [ ";
    for (int tokenId : tokenIds) {
        std::cout << tokenId << " ";
    }
    std::cout << "]\n";
}

// 가장 큰 값을 가진 위치의 인덱스를 반환한다.
int argmax(const Vec& v) {
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}
