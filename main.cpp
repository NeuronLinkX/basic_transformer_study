#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "forward.h"

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
    const int numHeads = 2; // Multi-Head Attention의 병렬 Attention head 개수

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


    // ******************************************************************************************************************
    // ** ENCODER **
    // ******************************************************************************************************************
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
    // 정수 토큰 ID를 길이 d_model = 4 인 의미 벡터로 바꿈
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
    // 각 위치(pos)에 대한 사인/코사인 벡터를 만들어 embedding에 더함
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
    // 원래 입력 정보와 attention 결과를 합침
    //
    // LayerNorm:
    // 각 토큰 벡터의 평균/분산을 정규화해 학습과 계산을 안정화한다.
    stage(5, "Encoder Add & Norm");
    Matrix encoderNorm1 = addNorm(encoderInput, encoderAttention, "Encoder add&norm 1");

    // STEP 6. Encoder Feed Forward
    // Multi-Head Attention(STEP4) -> Add&Norm -> FFN
    // FFN은 attention으로 섞인 문맥 정보를 각 토큰 위치마다 비선형 변환한다.
    // 그 뒤 Add & Norm을 한 번 더 수행해 encoder 최종 출력을 만듬
    //
    // X_enc = LayerNorm(X_tilde + FFN(X_tilde))
    stage(6, "Encoder Feed Forward");
    Matrix encoderFFNOut = feedForwardBlock(encoderNorm1, encoderFFN, "Encoder FFN");
    Matrix encoderOutput = addNorm(encoderNorm1, encoderFFNOut, "Encoder add&norm 2");

    // STEP 7. Decoder Masked Self-Attention
    // Decoder self-attention도 Q, K, V를 decoder 입력에서 만들지만,
    // 미래 토큰을 보면 안 되므로 causal mask를 적용
    //
    // Mask:
    // M(i, j) = 0      if j <= i
    // M(i, j) = -inf   if j > i
    //
    // 따라서 첫 번째 위치 <sos>는 뒤의 "transformers"를 볼 수 없고,
    // 두 번째 위치는 <sos>와 자기 자신까지 볼 수 있다.
    stage(7, "Decoder Masked Self-Attention");
    std::cout << "Decoder는 미래 토큰을 보지 못하도록 causal mask를 적용\n";
    Matrix decoderMaskedAttention = multiHeadAttention(
        decoderInput, decoderInput, decoderSelf, numHeads, true, "Decoder masked self-attention"
    );

    // STEP 8. Decoder Add & Norm After Masked Attention
    // Y_tilde_1 = LayerNorm(Y^(0) + MaskedMHA(Y^(0)))
    //
    // 지금까지 decoder가 본 토큰 정보와
    // masked self-attention 결과를 합쳐 현재 decoder 상태를 만듬
    stage(8, "Decoder Add & Norm After Masked Attention");
    Matrix decoderNorm1 = addNorm(decoderInput, decoderMaskedAttention, "Decoder add&norm 1");

    // ******************************************************************************************************************
    // ** DECODER **
    // ******************************************************************************************************************

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
            // 각 단어(<sos>, I, love, transformers, <eos>)의 점수(logit)를 만듬
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
    // 1. 입력 token ID를 만듬
    // 2. token ID를 embedding vector로 바꿈
    // 3. positional encoding을 더함
    // 4. encoder self-attention을 수행
    // 5. encoder Add & Norm을 수행
    // 6. encoder FFN과 Add & Norm으로 encoder output을 만듬
    // 7. decoder masked self-attention을 수행
    // 8. decoder Add & Norm을 수행
    // 9. decoder cross-attention으로 encoder output을 참고한다.
    // 10. decoder Add & Norm을 수행
    // 11. decoder FFN과 Add & Norm으로 decoder output을 만듬
    // 12. 마지막 decoder 위치를 vocab logits로 projection한다.
    // 13. softmax로 다음 token 확률을 구한다.
    //
    // 최종 한 줄 결론:
    // 이 코드는 "I love"를 Encoder가 문맥 벡터로 만들고,
    // Decoder가 "<sos> transformers"까지 본 상태에서
    // 다음 token으로 <eos>를 예측하는 Transformer Encoder-Decoder 예제다.
    stage(13, "Summary");
    std::cout << "1. forward.cpp는 토큰 1개가 아니라 시퀀스 전체를 attention으로 처리한다.\n";
    std::cout << "2. Q, K, V를 projection matrix로 따로 만듬\n";
    std::cout << "3. Multi-head attention을 2개 head로 분리해 다시 합침\n";
    std::cout << "4. Decoder self-attention에는 causal mask가 들어간다.\n";
    std::cout << "5. FFN은 2-layer 구조로 확장했고, 마지막 상태를 vocab logits로 바꿈\n";

    return 0;
}