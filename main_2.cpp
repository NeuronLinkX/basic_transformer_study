#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "forward.h"

namespace {

Matrix randomMatrix(int rows, int cols, std::mt19937& rng, float minValue, float maxValue) {
    std::uniform_real_distribution<float> dist(minValue, maxValue);
    Matrix result(rows, Vec(cols, 0.0f));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            result[r][c] = dist(rng);
        }
    }
    return result;
}

AttentionWeights randomAttentionWeights(
    int dModel,
    std::mt19937& rng,
    float minValue,
    float maxValue
) {
    return AttentionWeights{
        randomMatrix(dModel, dModel, rng, minValue, maxValue),
        randomMatrix(dModel, dModel, rng, minValue, maxValue),
        randomMatrix(dModel, dModel, rng, minValue, maxValue),
        randomMatrix(dModel, dModel, rng, minValue, maxValue)
    };
}

void printAttentionWeights(const std::string& name, const AttentionWeights& weights) {
    printMatrix(name + " - Wq", weights.Wq);
    printMatrix(name + " - Wk", weights.Wk);
    printMatrix(name + " - Wv", weights.Wv);
    printMatrix(name + " - Wo", weights.Wo);
}

} // namespace

// 수동 attention weight 대신 랜덤 weight를 넣었을 때 attention이 어떻게 동작하는지 보여주는 실험용 프로그램
int main() {
    const int dModel = 4;
    const int numHeads = 2;
    const unsigned int seed = 20260512;

    std::vector<std::string> vocab = {
        "<sos>", "I", "love", "transformers", "<eos>"
    };

    Matrix embeddingTable = {
        {0.12f, 0.10f, 0.08f, 0.05f},
        {0.95f, 0.15f, 0.20f, 0.05f},
        {0.20f, 0.85f, 0.25f, 0.15f},
        {0.25f, 0.20f, 0.95f, 0.82f},
        {0.02f, 0.04f, 0.12f, 0.92f}
    };

    std::vector<int> encoderTokenIds = {1, 2};
    std::vector<int> decoderTokenIds = {0, 3};

    // 실제 학습 대신 재현 가능한 랜덤 초기값을 넣는다.
    std::mt19937 rng(seed);
    AttentionWeights encoderSelf = randomAttentionWeights(dModel, rng, -0.5f, 0.5f);
    AttentionWeights decoderSelf = randomAttentionWeights(dModel, rng, -0.5f, 0.5f);

    stage(1, "Inputs");
    printTokenIds("Encoder token IDs", encoderTokenIds);
    printTokenIds("Decoder token IDs", decoderTokenIds);
    std::cout << "Vocabulary size = " << vocab.size() << "\n";
    std::cout << "Random seed = " << seed << "\n";

    stage(2, "Embedding Lookup");
    Matrix encoderInput = tokenEmbedding(encoderTokenIds, embeddingTable);
    Matrix decoderInput = tokenEmbedding(decoderTokenIds, embeddingTable);
    printMatrix("Encoder embedding", encoderInput);
    printMatrix("Decoder embedding", decoderInput);

    stage(3, "Sinusoidal Positional Encoding");
    Matrix encoderPE = positionalEncoding(static_cast<int>(encoderInput.size()), dModel);
    Matrix decoderPE = positionalEncoding(static_cast<int>(decoderInput.size()), dModel);
    encoderInput = addMatrix(encoderInput, encoderPE);
    decoderInput = addMatrix(decoderInput, decoderPE);
    printMatrix("Encoder input + position", encoderInput);
    printMatrix("Decoder input + position", decoderInput);

    stage(4, "Random Attention Weights");
    std::cout << "encoderSelf, decoderSelf를 수동 toy weight 대신 랜덤으로 생성한다.\n";
    std::cout << "범위는 [-0.5, 0.5]이며 시드를 고정했기 때문에 실행할 때마다 같은 값이 나온다.\n";
    printAttentionWeights("encoderSelf", encoderSelf);
    printAttentionWeights("decoderSelf", decoderSelf);

    stage(5, "Encoder Self-Attention With Random Weights");
    Matrix encoderAttention = multiHeadAttention(
        encoderInput, encoderInput, encoderSelf, numHeads, false, "Encoder self-attention"
    );
    printMatrix("Encoder attention output", encoderAttention);

    stage(6, "Decoder Masked Self-Attention With Random Weights");
    Matrix decoderAttention = multiHeadAttention(
        decoderInput, decoderInput, decoderSelf, numHeads, true, "Decoder masked self-attention"
    );
    printMatrix("Decoder masked attention output", decoderAttention);

    stage(7, "Interpretation");
    std::cout << "랜덤 weight여도 Q/K/V와 attention 계산 자체는 가능하다.\n";
    std::cout << "다만 이 값들은 학습된 의미가 없으므로 출력은 임의적이며 해석 가능한 예제가 되지 않는다.\n";
    std::cout << "즉 수동 toy weight의 목적은 계산을 가능하게 만드는 것뿐 아니라,\n";
    std::cout << "출력이 과하게 깨지지 않도록 교육용으로 통제하는 데 있다.\n";
    return 0;
}
