#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

/*
    프로젝트 개요
    이 프로그램은 Transformer Encoder-Decoder의 순전파(forward pass)를 아주 작은 예제 문장으로 직접 계산해 보는 C++ 학습용 코드다.

    입력 예시:
    - Encoder 입력: "I love"
    - Decoder 입력: "<sos> transformers"

    최종 목표:
    - Encoder가 입력 문장을 문맥 벡터로 바꾸고
    - Decoder가 현재까지 본 토큰을 바탕으로
    - 다음 토큰의 확률을 계산하여 예측하는 과정을 단계별로 출력한다.

    전체 처리 흐름
    1. Token ID 준비
    2. Embedding Lookup
    3. Positional Encoding 추가
    4. Encoder Self-Attention
    5. Encoder Add & Norm
    6. Encoder FFN
    --- output ---> Encoder의 context vector
    7. Decoder Masked Self-Attention
    8. Decoder Add & Norm
    9. Decoder Cross-Attention
    10. Decoder Add & Norm
    11. Decoder FFN
    12. Linear Projection + Softmax
    13. 다음 토큰 예측

    주요 자료형
    - Vec: 1차원 벡터(float 배열)
    - Matrix: 2차원 행렬(벡터들의 배열)
    - AttentionWeights: Wq, Wk, Wv, Wo를 묶은 구조체
    - FFNWeights: W1, b1, W2, b2를 묶은 구조체

    함수별 기능 설명
    - stage: 각 계산 단계를 콘솔에 STEP 형식으로 구분해 출력한다.
    - printVec, printMatrix: 벡터와 행렬 내용을 사람이 읽기 좋은 형태로 출력한다.
    - check: shape mismatch 같은 잘못된 입력이 들어오면 예외를 발생시킨다.
    - dot: 두 벡터의 내적을 계산한다.
    - softmax: 1차원 점수 벡터를 확률 분포로 바꾼다.
    - softmaxRows: 행렬의 각 행마다 softmax를 적용한다.
    - zeros: 0으로 채운 행렬을 만든다.
    - transpose: 행과 열을 뒤집은 전치행렬을 만든다.
    - matMul: 두 행렬의 곱셈을 수행한다.
    - addMatrix, addVec: 같은 shape의 행렬 또는 벡터를 원소별로 더한다.
    - addBias: 행렬의 각 행에 bias 벡터를 더한다.
    - linear: 선형변환 xW (+ b)를 계산한다.
    - relu: 음수는 0으로 바꾸는 ReLU 활성화 함수를 적용한다.
    - layerNormOne: 벡터 1개에 대해 평균/분산 기준 정규화를 수행한다.
    - layerNorm: 행렬의 각 행마다 LayerNorm을 적용한다.
    - tokenEmbedding: token id를 embedding table에서 찾아 벡터로 바꾼다.
    - positionalEncoding: 사인/코사인 기반 위치 인코딩을 만든다.
    - sliceColumns: multi-head attention을 위해 행렬의 일부 열만 잘라낸다.
    - concatColumns: 여러 head의 결과를 열 방향으로 다시 합친다.
    - scaledDotAttention: attention score 계산, mask 적용, softmax, value 가중합을 수행한다.
    - multiHeadAttention: Q/K/V 생성, head 분할, 각 head attention, head 결합, output projection을 수행한다.
    - feedForwardBlock: FFN의 두 개 선형층과 ReLU를 수행한다.
    - addNorm: residual connection 후 LayerNorm을 적용한다.
    - printTokenIds: 토큰 ID 시퀀스를 출력한다.
    - argmax: 가장 큰 확률을 가진 인덱스를 찾아 최종 예측 토큰을 결정한다.
    - main: 전체 Transformer forward 예제를 순서대로 실행하고 결과를 출력한다.

    참고
    이 파일의 가중치들은 실제 학습된 모델 파라미터가 아니라,
    계산 흐름을 설명하기 쉽게 사람이 직접 정한 toy weight들이다.
*/

using Vec = std::vector<float>;
using Matrix = std::vector<Vec>;

struct AttentionWeights {
    Matrix Wq;
    Matrix Wk;
    Matrix Wv;
    Matrix Wo;
};

struct FFNWeights {
    Matrix W1;
    Vec b1;
    Matrix W2;
    Vec b2;
};

// 계산 단계를 STEP 제목 형태로 구분해 출력한다.
void stage(int n, const std::string& title) {
    std::cout << "\n====================================================\n";
    std::cout << "STEP " << n << ". " << title << "\n";
    std::cout << "====================================================\n";
}

// 1차원 벡터를 고정 소수점 형식으로 출력한다.
void printVec(const std::string& name, const Vec& v) {
    std::cout << name << " = [ ";
    for (float x : v) {
        std::cout << std::fixed << std::setprecision(4) << x << " ";
    }
    std::cout << "]\n";
}

// 2차원 행렬을 행 단위로 보기 좋게 출력한다.
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

// 행렬의 각 행마다 softmax를 적용한다.
Matrix softmaxRows(const Matrix& logits) {
    Matrix result = logits;
    for (auto& row : result) {
        row = softmax(row);
    }
    return result;
}

// 주어진 크기의 0으로 채운 행렬을 만든다.
Matrix zeros(size_t rows, size_t cols) {
    return Matrix(rows, Vec(cols, 0.0f));
}

// 행렬의 행과 열을 뒤집은 전치행렬을 만든다.
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

// 두 행렬의 곱셈을 수행한다.
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

// 같은 shape의 두 행렬을 원소별로 더한다.
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

// 같은 길이의 두 벡터를 원소별로 더한다.
Vec addVec(const Vec& a, const Vec& b) {
    check(a.size() == b.size(), "addVec: size mismatch");

    Vec result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

// 행렬의 각 행에 동일한 bias 벡터를 더한다.
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
Matrix linear(const Matrix& x, const Matrix& w, const Vec& bias = {}) {
    Matrix y = matMul(x, w);
    if (!bias.empty()) {
        y = addBias(y, bias);
    }
    return y;
}

// 행렬의 모든 원소에 ReLU 활성화 함수를 적용한다.
Matrix relu(const Matrix& x) {
    Matrix result = x;
    for (auto& row : result) {
        for (float& v : row) {
            v = std::max(0.0f, v);
        }
    }
    return result;
}

// 벡터 하나에 대해 평균/분산 기반 LayerNorm을 적용한다.
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

// 행렬의 각 행마다 LayerNorm을 적용한다.
Matrix layerNorm(const Matrix& x) {
    Matrix result = x;
    for (auto& row : result) {
        row = layerNormOne(row);
    }
    return result;
}

// 토큰 ID 시퀀스를 embedding table에서 찾아 벡터 시퀀스로 바꾼다.
Matrix tokenEmbedding(const std::vector<int>& tokenIds, const Matrix& embeddingTable) {
    // Token ID는 단지 정수 인덱스일 뿐이므로 바로 의미 연산을 할 수 없다.
    // 따라서 embedding table에서 해당 행(row)을 꺼내
    // 각 토큰을 d_model 차원의 실수 벡터로 바꾼다.
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

// 행렬에서 연속된 일부 열만 잘라내 새 행렬로 만든다.
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

// 여러 head 결과를 열 방향으로 이어 붙여 하나의 행렬로 합친다.
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

// scaled dot-product attention을 계산하고 필요하면 causal mask를 적용한다.
Matrix scaledDotAttention(
    const Matrix& q,
    const Matrix& k,
    const Matrix& v,
    bool causalMask,
    Matrix* scoresOut = nullptr,
    Matrix* probsOut = nullptr
) {
    // Attention의 핵심 계산:
    // 1. 각 Query와 Key의 유사도를 구한다.
    //    score(i, j) = (Q_i dot K_j) / sqrt(d_k)
    // 2. 필요하면 미래 위치를 mask 처리한다.
    // 3. softmax로 확률 분포로 바꾼다.
    // 4. 그 확률로 Value를 가중합해 문맥 벡터를 만든다.
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
                // 그래서 매우 작은 값(-1e9)으로 보내 softmax 후 거의 0이 되게 만든다.
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

// Q/K/V 생성부터 head 결합과 output projection까지 multi-head attention 전체를 수행한다.
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
    // 다시 원래 d_model 차원으로 합친다.
    Matrix concatenated = concatColumns(headOutputs);
    printMatrix(name + " - concatenated heads", concatenated);

    Matrix projected = linear(concatenated, weights.Wo);
    printMatrix(name + " - output projection", projected);

    return projected;
}

// 두 개의 선형층과 ReLU로 이루어진 position-wise FFN을 수행한다.
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
    // 1. Residual connection으로 원래 입력 정보를 보존하고
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

// 토큰 ID 시퀀스를 한 줄로 출력한다.
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

// Transformer encoder-decoder 예제 전체 순전파를 단계별로 실행한다.
int main() {
    /*
        Vocabulary:
        0 = <sos>
        1 = I
        2 = love
        3 = transformers
        4 = <eos>

        따라서 이번 예제의 입력은 다음과 같다.
        Encoder 입력 = [I, love]
        Decoder 입력 = [<sos>, transformers]

        의미:
        Encoder는 "I love"를 이해하고, Decoder는 "<sos> transformers"까지 본 뒤 다음 토큰을 예측
    */
    const int dModel = 4;
    const int numHeads = 2;

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

    // 이 아래의 attention/FFN/output 가중치들은
    // 실제 학습(train)으로 얻은 모델 파라미터가 아니다.
    //
    // 또한 완전 랜덤으로 뽑은 숫자도 아니다.
    // 이 예제가 안정적으로 동작하고, 계산 흐름을 설명하기 쉽도록
    // 사람이 직접 작게 정한 toy weight들이다.
    //
    // 왜 이런 숫자를 썼는가?
    // 1. 대체로 0~1 근처의 작은 값으로 두어 계산이 과하게 커지지 않게 한다.
    // 2. 대각선 부근 값을 크게 두어 원래 차원 정보가 어느 정도 유지되게 한다.
    // 3. 비대각선에 작은 값을 넣어 차원 간 정보가 조금 섞이게 한다.
    // 4. 최종적으로 예제 출력이 해석 가능하고, <eos> 예측까지 자연스럽게 이어지게 한다.
    //
    // 따라서 0.90f, 0.85f 같은 값은 "학습이 찾아낸 정답"이 아니라
    // "이 차원을 비교적 강하게 통과시키겠다"는 의도를 담은 수동 설정값이다.

    // Encoder self-attention용 가중치.
    // 순서대로 Wq, Wk, Wv, Wo 이며 모두 shape 4x4 이다.
    //
    // 해석:
    // - 큰 대각선 값: 각 차원이 자기 정보는 강하게 유지
    // - 작은 비대각선 값: 다른 차원 정보는 약하게 섞음
    // - 0.00 값: 일부 연결은 단순화를 위해 사실상 끊어 둠
    //
    // 즉 encoderSelf는 encoder 입력 "I love"를 가지고
    // Query/Key/Value를 만드는 선형변환 규칙이다.
    AttentionWeights encoderSelf {
        {
            {0.90f, 0.10f, 0.00f, 0.10f},
            {0.10f, 0.85f, 0.05f, 0.00f},
            {0.00f, 0.15f, 0.80f, 0.10f},
            {0.05f, 0.00f, 0.20f, 0.75f}
        },
        {
            {0.80f, 0.00f, 0.10f, 0.10f},
            {0.00f, 0.90f, 0.10f, 0.00f},
            {0.10f, 0.10f, 0.85f, 0.05f},
            {0.00f, 0.05f, 0.15f, 0.85f}
        },
        {
            {0.85f, 0.10f, 0.00f, 0.00f},
            {0.10f, 0.80f, 0.15f, 0.00f},
            {0.00f, 0.10f, 0.90f, 0.10f},
            {0.05f, 0.00f, 0.10f, 0.95f}
        },
        {
            {0.95f, 0.05f, 0.00f, 0.00f},
            {0.05f, 0.95f, 0.00f, 0.00f},
            {0.00f, 0.10f, 0.85f, 0.05f},
            {0.00f, 0.05f, 0.10f, 0.90f}
        }
    };

    // Decoder masked self-attention용 가중치.
    // 구조는 encoderSelf와 같지만, decoder 블록용으로 별도 값을 둔 것이다.
    //
    // 이것도 학습된 값이 아니라 예제용 수동 설정값이다.
    // encoderSelf와 비슷한 크기의 숫자를 쓰되 약간 다르게 둔 이유는
    // encoder self-attention과 decoder self-attention이
    // 서로 다른 블록이라는 점을 코드상에서 드러내기 위해서다.
    //
    // 실제 attention 결과는 이 가중치뿐 아니라 causal mask의 영향도 함께 받는다.
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

    // Decoder cross-attention용 가중치.
    //
    // Query는 decoder 상태에서 오고,
    // Key와 Value는 encoder output에서 온다.
    // 따라서 이 가중치들은 decoder가 encoder memory를
    // 어떤 방식으로 조회할지 정하는 규칙이다.
    //
    // 이 역시 학습된 값이 아니라 사람이 만든 예제용 숫자다.
    // self-attention과 완전히 같은 값을 쓰지 않은 이유는
    // "자기 자신을 보는 attention"과 "encoder를 참고하는 attention"의 역할 차이를
    // 분리해서 보여주기 위해서다.
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

    // Encoder FFN 가중치.
    //
    // 구조:
    // - 첫 번째 행렬 W1: 4x8
    // - 첫 번째 bias b1: 길이 8
    // - 두 번째 행렬 W2: 8x4
    // - 두 번째 bias b2: 길이 4
    //
    // 이것도 학습으로 얻은 값이 아니라 수동 설정값이다.
    //
    // 왜 4 -> 8 -> 4 구조인가?
    // - 실제 Transformer의 FFN처럼 중간 hidden 차원을 확장한 뒤
    //   다시 d_model로 줄이는 구조를 보여주기 위해서다.
    //
    // 왜 양수 값이 많은가?
    // - ReLU 통과 후 정보가 너무 많이 사라지지 않게 하기 위해서다.
    //
    // 왜 bias에 작은 음수가 섞여 있는가?
    // - 모든 hidden unit이 무조건 활성화되지 않도록
    //   약한 기준점(threshold 비슷한 역할)을 주기 위해서다.
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

    // Decoder FFN 가중치.
    //
    // 구조는 encoderFFN과 동일한 4 -> 8 -> 4 이다.
    // 이 값들도 학습된 값이 아니라 예제용 수동 설정값이다.
    //
    // encoderFFN과 숫자가 조금 다른 이유:
    // - decoder는 masked self-attention과 cross-attention을 거친 표현을 다루므로
    //   encoder와 완전히 같은 변환을 쓰지 않는다는 점을 보여주기 위해서다.
    // - 마지막 vocabulary projection 전에 decoder 상태가
    //   다음 토큰 예측에 적합한 방향으로 변형되게 하려는 목적도 있다.
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

    // 최종 vocabulary projection 가중치.
    //
    // shape은 4x5 이다.
    // - 입력: 마지막 decoder 상태 h_last (길이 4)
    // - 출력: vocab 5개 단어의 logit
    //
    // 수식:
    // z = h_last * W_vocab
    //
    // 이것도 학습된 출력층 파라미터가 아니라 예제용 수동 설정값이다.
    // 특히 이 예제에서는 최종적으로 <eos>가 가장 높은 확률로 나오도록
    // 일부 열이 그 방향으로 반응하기 쉽게 조정되어 있다.
    //
    // 따라서 각 숫자는
    // "decoder 상태의 특정 차원이 커졌을 때 어떤 단어 점수를 얼마나 올릴지"
    // 를 정하는 계수라고 보면 된다.
    Matrix outputWeight = {
        {0.10f, 0.20f, 0.12f, 0.85f, 0.30f},
        {0.12f, 0.10f, 0.15f, 0.70f, 0.28f},
        {0.00f, 0.10f, 0.18f, 1.20f, 0.25f},
        {0.00f, 0.05f, 0.10f, 0.95f, 0.85f}
    };

    std::vector<int> encoderTokenIds = {1, 2};
    std::vector<int> decoderTokenIds = {0, 3};

    // STEP 1. Inputs
    // X_id = [1, 2]
    // Y_id = [0, 3]
    //
    // Encoder는 입력 문장 "I love"를 받는다.
    // Decoder는 teacher forcing 예제로 "<sos> transformers"를 입력받고
    // 마지막 위치에서 다음 토큰을 예측한다.
    stage(1, "Inputs");
    std::cout << "입력 문장: I love\n";
    std::cout << "출력 정답 예시: transformers <eos>\n";
    std::cout << "Decoder 입력은 teacher forcing 형태로 <sos> transformers 를 사용한다.\n";
    printTokenIds("Encoder token IDs", encoderTokenIds);
    printTokenIds("Decoder token IDs", decoderTokenIds);

    // STEP 2. Embedding Lookup
    // 정수 토큰 ID를 길이 d_model = 4 인 의미 벡터로 바꾼다.
    //
    // Embedding table E의 shape:
    // |V| x d_model = 5 x 4
    //
    // 예:
    // I    -> [0.9500, 0.1500, 0.2000, 0.0500]
    // love -> [0.2000, 0.8500, 0.2500, 0.1500]
    stage(2, "Embedding Lookup");
    Matrix encoderInput = tokenEmbedding(encoderTokenIds, embeddingTable);
    Matrix decoderInput = tokenEmbedding(decoderTokenIds, embeddingTable);
    printMatrix("Encoder embedding", encoderInput);
    printMatrix("Decoder embedding", decoderInput);

    // STEP 3. Positional Encoding
    // 각 위치(pos)에 대한 사인/코사인 벡터를 만들어 embedding에 더한다.
    //
    // 예를 들어 position 0의 positional encoding은
    // [0.0000, 1.0000, 0.0000, 1.0000] 이고,
    // "I"의 embedding [0.9500, 0.1500, 0.2000, 0.0500]에 더하면
    // [0.9500, 1.1500, 0.2000, 1.0500]이 된다.
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
    // Encoder self-attention에서는 Query, Key, Value가 모두 encoder 입력에서 나온다.
    //
    // Q = XW^Q
    // K = XW^K
    // V = XW^V
    //
    // 각 encoder 토큰은 입력 문장 전체를 보며
    // "I"와 "love"가 서로 어떤 관련이 있는지 계산한다.
    stage(4, "Encoder Self-Attention");
    std::cout << "각 encoder 토큰이 전체 입력 시퀀스를 본다.\n";
    Matrix encoderAttention = multiHeadAttention(
        encoderInput, encoderInput, encoderSelf, numHeads, false, "Encoder self-attention"
    );

    // STEP 5. Encoder Add & Norm
    // X_tilde = LayerNorm(X^(0) + MHA(X^(0)))
    //
    // Residual Add:
    // 원래 입력 정보와 attention 결과를 합친다.
    //
    // LayerNorm:
    // 각 토큰 벡터의 평균/분산을 정규화해 학습과 계산을 안정화한다.
    stage(5, "Encoder Add & Norm");
    Matrix encoderNorm1 = addNorm(encoderInput, encoderAttention, "Encoder add&norm 1");

    // STEP 6. Encoder Feed Forward
    // FFN은 attention으로 섞인 문맥 정보를 각 토큰 위치마다 비선형 변환한다.
    // 그 뒤 Add & Norm을 한 번 더 수행해 encoder 최종 출력을 만든다.
    //
    // X_enc = LayerNorm(X_tilde + FFN(X_tilde))
    stage(6, "Encoder Feed Forward");
    Matrix encoderFFNOut = feedForwardBlock(encoderNorm1, encoderFFN, "Encoder FFN");
    Matrix encoderOutput = addNorm(encoderNorm1, encoderFFNOut, "Encoder add&norm 2");

    // STEP 7. Decoder Masked Self-Attention
    // Decoder self-attention도 Q, K, V를 decoder 입력에서 만들지만,
    // 미래 토큰을 보면 안 되므로 causal mask를 적용한다.
    //
    // Mask:
    // M(i, j) = 0      if j <= i
    // M(i, j) = -inf   if j > i
    //
    // 따라서 첫 번째 위치 <sos>는 뒤의 "transformers"를 볼 수 없고,
    // 두 번째 위치는 <sos>와 자기 자신까지 볼 수 있다.
    stage(7, "Decoder Masked Self-Attention");
    std::cout << "Decoder는 미래 토큰을 보지 못하도록 causal mask를 적용한다.\n";
    Matrix decoderMaskedAttention = multiHeadAttention(
        decoderInput, decoderInput, decoderSelf, numHeads, true, "Decoder masked self-attention"
    );

    // STEP 8. Decoder Add & Norm After Masked Attention
    // Y_tilde_1 = LayerNorm(Y^(0) + MaskedMHA(Y^(0)))
    //
    // 지금까지 decoder가 본 토큰 정보와
    // masked self-attention 결과를 합쳐 현재 decoder 상태를 만든다.
    stage(8, "Decoder Add & Norm After Masked Attention");
    Matrix decoderNorm1 = addNorm(decoderInput, decoderMaskedAttention, "Decoder add&norm 1");

    // STEP 9. Decoder Cross-Attention
    // 여기서는 출처가 갈린다.
    //
    // Query  = decoder 쪽 상태
    // Key    = encoder output
    // Value  = encoder output
    //
    // Q = YW^Q
    // K = X_enc W^K
    // V = X_enc W^V
    //
    // 즉 decoder가 다음 단어를 예측할 때
    // encoder가 이해한 "I love" 중 무엇을 참고할지 계산한다.
    stage(9, "Decoder Cross-Attention");
    std::cout << "Decoder가 encoder output 전체를 memory로 참고한다.\n";
    Matrix crossAttentionOut = multiHeadAttention(
        decoderNorm1, encoderOutput, crossAttention, numHeads, false, "Decoder cross-attention"
    );

    // STEP 10. Decoder Add & Norm After Cross-Attention
    // Y_tilde_2 = LayerNorm(Y_tilde_1 + CrossMHA(Y_tilde_1, X_enc))
    //
    // Decoder 내부 문맥과 encoder memory를 합쳐
    // 입력 문장을 참고한 decoder 표현으로 업데이트한다.
    stage(10, "Decoder Add & Norm After Cross-Attention");
    Matrix decoderNorm2 = addNorm(decoderNorm1, crossAttentionOut, "Decoder add&norm 2");

    // STEP 11. Decoder Feed Forward
    // Cross-attention으로 얻은 입력-출력 관계 표현을
    // 다음 토큰 예측에 더 적합한 형태로 다시 변환한다.
    //
    // Y_dec = LayerNorm(Y_tilde_2 + FFN(Y_tilde_2))
    stage(11, "Decoder Feed Forward");
    Matrix decoderFFNOut = feedForwardBlock(decoderNorm2, decoderFFN, "Decoder FFN");
    Matrix decoderOutput = addNorm(decoderNorm2, decoderFFNOut, "Decoder add&norm 3");

    // STEP 12. Linear Projection To Vocabulary
    // decoderOutput.back()은 decoder 시퀀스의 "마지막 위치" 벡터를 뜻한다.
    //
    // 현재 decoder 입력이 [<sos>, transformers] 이므로
    // 마지막 위치는 "transformers" 위치다.
    // 이 위치에서 다음 토큰 1개를 예측한다.
    //
    // 수식:
    // z = h_last W_vocab
    stage(12, "Linear Projection To Vocabulary");
    Vec finalDecoderState = decoderOutput.back();
    printVec("Final decoder state (last position)", finalDecoderState);

    Vec logits(vocab.size(), 0.0f);
    for (size_t vocabIdx = 0; vocabIdx < vocab.size(); ++vocabIdx) {
        for (int d = 0; d < dModel; ++d) {
            // 마지막 decoder 상태와 vocabulary projection weight를 곱해
            // 각 단어(<sos>, I, love, transformers, <eos>)의 점수(logit)를 만든다.
            logits[vocabIdx] += finalDecoderState[d] * outputWeight[d][vocabIdx];
        }
    }

    printVec("Logits", logits);

    // Softmax:
    // P(v) = exp(z_v) / sum_j exp(z_j)
    //
    // 가장 확률이 큰 토큰이 다음 토큰 예측 결과가 된다.
    Vec probs = softmax(logits);
    std::cout << "\n최종 확률:\n";
    for (size_t i = 0; i < vocab.size(); ++i) {
        std::cout << std::setw(12) << vocab[i] << " : "
                  << std::fixed << std::setprecision(6) << probs[i] << "\n";
    }

    int predicted = argmax(probs);
    std::cout << "\n예측된 다음 토큰: " << vocab[predicted] << "\n";
    std::cout << "이 예시에서는 마지막 decoder 위치가 다음 토큰 후보(<eos> 등)를 예측한다.\n";

    // STEP 13. Summary
    // 1. 입력 token ID를 만든다.
    // 2. token ID를 embedding vector로 바꾼다.
    // 3. positional encoding을 더한다.
    // 4. encoder self-attention을 수행한다.
    // 5. encoder Add & Norm을 수행한다.
    // 6. encoder FFN과 Add & Norm으로 encoder output을 만든다.
    // 7. decoder masked self-attention을 수행한다.
    // 8. decoder Add & Norm을 수행한다.
    // 9. decoder cross-attention으로 encoder output을 참고한다.
    // 10. decoder Add & Norm을 수행한다.
    // 11. decoder FFN과 Add & Norm으로 decoder output을 만든다.
    // 12. 마지막 decoder 위치를 vocab logits로 projection한다.
    // 13. softmax로 다음 token 확률을 구한다.
    //
    // 최종 한 줄 결론:
    // 이 코드는 "I love"를 Encoder가 문맥 벡터로 만들고,
    // Decoder가 "<sos> transformers"까지 본 상태에서
    // 다음 token으로 <eos>를 예측하는 Transformer Encoder-Decoder 예제다.
    stage(13, "Summary");
    std::cout << "1. forward.cpp는 토큰 1개가 아니라 시퀀스 전체를 attention으로 처리한다.\n";
    std::cout << "2. Q, K, V를 projection matrix로 따로 만든다.\n";
    std::cout << "3. Multi-head attention을 2개 head로 분리해 다시 합친다.\n";
    std::cout << "4. Decoder self-attention에는 causal mask가 들어간다.\n";
    std::cout << "5. FFN은 2-layer 구조로 확장했고, 마지막 상태를 vocab logits로 바꾼다.\n";

    return 0;
}
