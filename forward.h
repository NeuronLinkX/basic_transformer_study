#ifndef FORWARD_H
#define FORWARD_H

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
    - 다음 토큰의 확률을 계산하여 예측하는 과정을 단계별로 출력

    전체 처리 흐름
        1. Token ID 시퀀스 생성
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
        - stage: 각 계산 단계를 콘솔에 STEP 형식으로 구분해 출력
        - printVec, printMatrix: 벡터와 행렬 내용을 사람이 읽기 좋은 형태로 출력
        - check: shape mismatch 같은 잘못된 입력이 들어오면 예외를 발생
        - dot: 두 벡터의 내적을 계산한다.
        - softmax: 1차원 점수 벡터를 확률 분포로 바꿈
        - softmaxRows: 행렬의 각 행마다 softmax를 적용
        - zeros: 0으로 채운 행렬을 만듬
        - transpose: 행과 열을 뒤집은 전치행렬을 만듬
        - matMul: 두 행렬의 곱셈을 수행
        - addMatrix, addVec: 같은 shape의 행렬 또는 벡터를 원소별로 더함
        - addBias: 행렬의 각 행에 bias 벡터를 더함
        - linear: 선형변환 xW (+ b)를 계산한다.
        - relu: 음수는 0으로 바꾸는 ReLU 활성화 함수를 적용
        - layerNormOne: 벡터 1개에 대해 평균/분산 기준 정규화를 수행
        - layerNorm: 행렬의 각 행마다 LayerNorm을 적용
        - tokenEmbedding: token id를 embedding table에서 찾아 벡터로 바꿈
        - positionalEncoding: 사인/코사인 기반 위치 인코딩을 만듬
        - sliceColumns: multi-head attention을 위해 행렬의 일부 열만 잘라낸다.
        - concatColumns: 여러 head의 결과를 열 방향으로 다시 합침
        - scaledDotAttention: attention score 계산, mask 적용, softmax, value 가중합을 수행
        - multiHeadAttention: Q/K/V 생성, head 분할, 각 head attention, head 결합, output projection을 수행
        - feedForwardBlock: FFN의 두 개 선형층과 ReLU를 수행
        - addNorm: residual connection 후 LayerNorm을 적용
        - printTokenIds: 토큰 ID 시퀀스를 출력
        - argmax: 가장 큰 확률을 가진 인덱스를 찾아 최종 예측 토큰을 결정한다.
    - main: 전체 Transformer forward 예제를 순서대로 실행하고 결과를 출력

    - 참고 : 이 파일의 가중치들은 실제 학습된 모델 파라미터가 아닌, 계산 흐름을 설명하기 쉽게 사람이 직접 정한 toy weight
*/

/*
    [English]
    Project Overview

    This program is C++ learning code that directly calculates the forward pass of a Transformer Encoder-Decoder using very small example statements.

    * Input Example:
    - Encoder Input: "I love"
    - Decoder Input: "<sos> transformers"

    Final Goal:
    - Output the step-by-step process where the Encoder converts the input sentence into a context vector
    - The Decoder calculates the probability of the next token based on the tokens seen so far
    - and predicts it

    [Overall Processing Flow]
        1. Generate Token ID Sequence
        2. Embedding Lookup
        3. Add Positional Encoding
        4. Encoder Self-Attention
        5. Encoder Add & Norm
        6. Encoder FFN
        --- output ---> Encoder's context vector
        7. Decoder Masked Self-Attention
        8. Decoder Add & Norm
        9. Decoder Cross-Attention
        10. Decoder Add & Norm
        11. Decoder FFN
        12. Linear Projection + Softmax
        13. Predict Next Token

    [Main Data Types]
        - Vec: 1-dimensional vector (float array)
        - Matrix: A 2D matrix (an array of vectors)
        - AttentionWeights: A structure grouping Wq, Wk, Wv, and Wo
        - FFNWeights: A structure grouping W1, b1, W2, and b2

    [Description of Functions]
        - stage: Prints each calculation step to the console in STEP format
        - printVec, printMatrix: Prints vector and matrix contents in a human-readable format
        - check: Raises an exception if incorrect input, such as a shape mismatch, is entered
        - dot: Calculates the inner product of two vectors. - softmax: Converts a 1D score vector into a probability distribution
        - softmaxRows: Applies softmax to each row of the matrix
        - zeros: Creates a matrix filled with zeros
        - transpose: Creates a transpose matrix by flipping the rows and columns
        - matMul: Performs the multiplication of two matrices
        - addMatrix, addVec: Adds elements of matrices or vectors of the same shape
        - addBias: Adds a bias vector to each row of the matrix
        - linear: Calculates the linear transformation xW (+ b).
        - relu: Applies the ReLU activation function, converting negative values ​​to zero
        - layerNormOne: Performs mean/variance-based normalization on a single vector
        - layerNorm: Applies LayerNorm to each row of the matrix
        - tokenEmbedding: Finds a token ID in the embedding table and converts it into a vector
        - positionalEncoding: Creates a sine/cosine-based positional encoding
        - sliceColumns: Cuts out only a subset of columns from the matrix for multi-head attention - concatColumns: Combines the results of multiple heads in a column direction
        - scaledDotAttention: Calculates attention scores, applies masks, performs softmax, and performs a weighted sum of values
        - multiHeadAttention: Generates Q/K/V, splits heads, focuses attention on each head, combines heads, and performs output projection
        - feedForwardBlock: Performs two linear layers of the FFN and ReLU
        - addNorm: Applies LayerNorm after residual connection
        - printTokenIds: Prints the sequence of token IDs
        - argmax: Finds the index with the highest probability to determine the final predicted token.
        - main: Executes the entire Transformer forward example in order and prints the results
        - Note: The weights in this file are not actual trained model parameters, but "toy weights" manually assigned to make it easier to explain the computation flow

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

void stage(int n, const std::string& title);
void printVec(const std::string& name, const Vec& v);
void printMatrix(const std::string& name, const Matrix& m);
void check(bool condition, const std::string& message);
float dot(const Vec& a, const Vec& b);
Vec softmax(const Vec& logits);
Matrix softmaxRows(const Matrix& logits);
Matrix zeros(size_t rows, size_t cols);
Matrix transpose(const Matrix& m);
Matrix matMul(const Matrix& a, const Matrix& b);
Matrix addMatrix(const Matrix& a, const Matrix& b);
Vec addVec(const Vec& a, const Vec& b);
Matrix addBias(const Matrix& x, const Vec& bias);
Matrix linear(const Matrix& x, const Matrix& w, const Vec& bias = {});
Matrix relu(const Matrix& x);
Vec layerNormOne(const Vec& x);
Matrix layerNorm(const Matrix& x);
Matrix tokenEmbedding(const std::vector<int>& tokenIds, const Matrix& embeddingTable);
Matrix positionalEncoding(int seqLen, int dModel);
Matrix sliceColumns(const Matrix& m, int start, int width);
Matrix concatColumns(const std::vector<Matrix>& parts);
Matrix scaledDotAttention(
    const Matrix& q,
    const Matrix& k,
    const Matrix& v,
    bool causalMask,
    Matrix* scoresOut = nullptr,
    Matrix* probsOut = nullptr
);
Matrix multiHeadAttention(
    const Matrix& queryInput,
    const Matrix& keyValueInput,
    const AttentionWeights& weights,
    int numHeads,
    bool causalMask,
    const std::string& name
);
Matrix feedForwardBlock(const Matrix& x, const FFNWeights& weights, const std::string& name);
Matrix addNorm(const Matrix& residual, const Matrix& sublayerOutput, const std::string& name);
void printTokenIds(const std::string& name, const std::vector<int>& tokenIds);
int argmax(const Vec& v);

#endif
