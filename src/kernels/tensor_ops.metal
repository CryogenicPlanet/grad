#include <metal_stdlib>
using namespace metal;


kernel void vector_add(const device float* in1 [[buffer(0)]],
                       const device float* in2 [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    out[id] = in1[id] + in2[id];
}

// this is single vector backwards which will handle A + B as two different A, B 
kernel void vector_add_backwards(const device float* in1 [[buffer(0)]],
                                 const device float* gradOutput [[buffer(1)]],
                                 device float* out [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
    out[id] = in1[id] + 1 * gradOutput[id];
}

kernel void vector_mul(const device float* in1 [[buffer(0)]],
                       const device float* in2 [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    out[id] = in1[id] * in2[id];
}

kernel void vector_mul_backwards(const device float* in1 [[buffer(0)]],
                                 const device float* in2 [[buffer(1)]],
                                 const device float* in3 [[buffer(2)]],
                                 device float* out [[buffer(3)]],
                                 uint id [[thread_position_in_grid]]) {
    out[id] = in1[id] + in2[id] * in3[id];
}



kernel void matrix_add(const device float* lhs [[buffer(0)]],
                       const device float* rhs [[buffer(1)]],
                       device float* outMatrix [[buffer(2)]],
                       constant uint* matrixSizes [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    // Extract numRows and colSize from matrixSizes buffer
    // uint numRows = matrixSizes[0];
    uint colSize = matrixSizes[1];
    uint isRowWise = matrixSizes[2] ? matrixSizes[2] : 0;

    // Calculate row and column this thread is responsible for
    uint row = id / colSize;
    uint col = id % colSize;

    if (isRowWise == 1) {
        outMatrix[row * colSize + col] = lhs[row * colSize + col] + rhs[col];
    } else {
        outMatrix[row * colSize + col] = lhs[row * colSize + col] + rhs[row * colSize + col];
    }
}

kernel void matrix_add_backwards(const device float* lhs_grad [[buffer(0)]],
                                 const device float* rhs_grad [[buffer(1)]],
                                 const device float* gradOutput [[buffer(2)]],
                                 device float* mut_lhs_grad [[buffer(3)]],
                                 device float* mut_rhs_grad [[buffer(4)]],
                                 constant uint* matrixSizes [[buffer(5)]],
                                 uint id [[thread_position_in_grid]]) {
    // Extract numRows, colSize, and isRowWise from matrixSizes buffer
    uint numRows = matrixSizes[0];
    uint colSize = matrixSizes[1];

    // Calculate row and column this thread is responsible for
    uint row = id / colSize;
    uint col = id % colSize;

   
    mut_lhs_grad[row * colSize + col] = gradOutput[row * colSize + col] + lhs_grad[row * colSize + col];
    mut_rhs_grad[row * colSize + col] = gradOutput[row * colSize + col] + rhs_grad[row * colSize + col];
}

kernel void matrix_mul(const device float* matrix1 [[buffer(0)]],
                       const device float* matrix2 [[buffer(1)]],
                       device float* outMatrix [[buffer(2)]],
                       constant uint* matrixSizes [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    uint numRowsMatrix1 = matrixSizes[0];
    uint numColsMatrix1 = matrixSizes[1];
    uint numColsMatrix2 = matrixSizes[2];

    // Calculate row and column this thread is responsible for
    uint row = id / numColsMatrix2;
    uint col = id % numColsMatrix2;

    float sum = 0.0;

    for (uint i = 0; i < numColsMatrix1; ++i) {
        sum += matrix1[row * numColsMatrix1 + i] * matrix2[i * numColsMatrix2 + col];
    }

    // Store the result in the output matrix
    outMatrix[row * numColsMatrix2 + col] = sum;
}


#include <metal_stdlib>
using namespace metal;

constant uint TILE_SIZE = 16;

kernel void matrix_mul_tiled(const device float* matrix1 [[buffer(0)]],
                             const device float* matrix2 [[buffer(1)]],
                             device float* outMatrix [[buffer(2)]],
                             constant uint* matrixSizes [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]],
                             uint2 tid [[thread_position_in_threadgroup]],
                             uint2 groupID [[threadgroup_position_in_grid]]) {
    uint numRowsMatrix1 = matrixSizes[0];
    uint numColsMatrix1 = matrixSizes[1];
    uint numColsMatrix2 = matrixSizes[2];

    threadgroup float Asub[TILE_SIZE][TILE_SIZE];
    threadgroup float Bsub[TILE_SIZE][TILE_SIZE];

    uint row = gid.y;
    uint col = gid.x;
    float sum = 0.0;

    for (uint tileIdx = 0; tileIdx < (numColsMatrix1 + TILE_SIZE - 1) / TILE_SIZE; ++tileIdx) {
        if (row < numRowsMatrix1 && (tileIdx*TILE_SIZE + tid.x) < numColsMatrix1) {
            Asub[tid.y][tid.x] = matrix1[row * numColsMatrix1 + tileIdx*TILE_SIZE + tid.x];
        } else {
            Asub[tid.y][tid.x] = 0.0;
        }

        if ((tileIdx*TILE_SIZE + tid.y) < numColsMatrix1 && col < numColsMatrix2) {
            Bsub[tid.y][tid.x] = matrix2[(tileIdx*TILE_SIZE + tid.y) * numColsMatrix2 + col];
        } else {
            Bsub[tid.y][tid.x] = 0.0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[tid.y][k] * Bsub[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < numRowsMatrix1 && col < numColsMatrix2) {
        outMatrix[row * numColsMatrix2 + col] = sum;
    }
}


kernel void matrix_mul_tiled_backwards(const device float* matrix1 [[buffer(0)]], // A
                                       const device float* matrix2 [[buffer(1)]], // B
                                       const device float* out_grad [[buffer(2)]], // dC
                                       device atomic_float* matrix1_grad [[buffer(3)]], // dA
                                       device atomic_float* matrix2_grad [[buffer(4)]], // dB
                                       constant uint* matrixSizes [[buffer(5)]],
                                       uint2 gid [[thread_position_in_grid]],
                                       uint2 tid [[thread_position_in_threadgroup]],
                                       uint2 groupID [[threadgroup_position_in_grid]]) {
    uint numRowsMatrix1 = matrixSizes[0];
    uint numColsMatrix1 = matrixSizes[1];
    uint numColsMatrix2 = matrixSizes[2];

    threadgroup float Asub[TILE_SIZE][TILE_SIZE];
    threadgroup float Bsub[TILE_SIZE][TILE_SIZE];
    threadgroup float dCsub[TILE_SIZE][TILE_SIZE];

    uint row = gid.y;
    uint col = gid.x;
    float sumA = 0.0;
    float sumB = 0.0;

    for (uint tileIdx = 0; tileIdx < (numColsMatrix1 + TILE_SIZE - 1) / TILE_SIZE; ++tileIdx) {
        if (row < numRowsMatrix1 && (tileIdx*TILE_SIZE + tid.x) < numColsMatrix1) {
            Asub[tid.y][tid.x] = matrix1[row * numColsMatrix1 + tileIdx*TILE_SIZE + tid.x];
        } else {
            Asub[tid.y][tid.x] = 0.0;
        }

        if ((tileIdx*TILE_SIZE + tid.y) < numColsMatrix1 && col < numColsMatrix2) {
            Bsub[tid.y][tid.x] = matrix2[(tileIdx*TILE_SIZE + tid.y) * numColsMatrix2 + col];
        } else {
            Bsub[tid.y][tid.x] = 0.0;
        }

        if (row < numRowsMatrix1 && col < numColsMatrix2) {
            dCsub[tid.y][tid.x] = out_grad[row * numColsMatrix2 + col];
        } else {
            dCsub[tid.y][tid.x] = 0.0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; ++k) {
            sumA += dCsub[tid.y][k] * Bsub[k][tid.x];
            sumB += Asub[tid.y][k] * dCsub[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < numRowsMatrix1 && col < numColsMatrix1) {
        atomic_fetch_add_explicit(&matrix1_grad[row * numColsMatrix1 + col], sumA, memory_order_relaxed);
    }

    if (row < numColsMatrix1 && col < numColsMatrix2) {
        atomic_fetch_add_explicit(&matrix2_grad[row * numColsMatrix2 + col], sumB, memory_order_relaxed);
    }
}


kernel void matrix_mul_backwards(const device float* lhs [[buffer(0)]], // A
                                 const device float* rhs [[buffer(1)]], // B
                                 const device float* out_grad [[buffer(2)]], // dC
                                 device float* lhs_grad [[buffer(3)]], // dA
                                 device float* rhs_grad [[buffer(4)]], // dB
                                 constant uint* matrixSizes [[buffer(5)]],
                                 uint id [[thread_position_in_grid]]) {
    
    uint m = matrixSizes[0];
    uint n = matrixSizes[1];
    uint p = matrixSizes[2];

    // Calculate row and column this thread is responsible for
    uint row = id / p;
    uint col = id % p;

    // For d(rhs) = (lhs)^T * d(out)
    if (row < n && col < p) {
        float sum = 0.0;
        for (uint k = 0; k < m; ++k) {
            sum += lhs[k * n + row] * out_grad[k * p + col];
        }
        rhs_grad[row * p + col] += sum;
    }

    // For d(lhs) = d(out) * (rhs)^T
    if (row < m && col < n) {
        float sum = 0.0;
        for (uint i = 0; i < p; ++i) {
            sum += out_grad[row * p + i] * rhs[col * p + i];
        }
        lhs_grad[row * n + col] += sum;
    }
}




kernel void set_grad(device float* matrix [[buffer(0)]],
                     device float* gradVal [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    uint x = gradVal[0];
    matrix[id] = x;
}

kernel void matrix_tanh(const device float* inputMatrix [[buffer(0)]],
                        device float* outputMatrix [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
        float x = inputMatrix[id];
        // https://www.perplexity.ai/search/do-people-use-uIKc5CQjQciUBFPXOze6MA
        if (x > 20.0) {
            outputMatrix[id] = 0.9999; // tanh(x) approaches 1 for large x
        } else if (x < -20.0) {
            outputMatrix[id] = -0.9999; // tanh(x) approaches -1 for large negative x
        } else {
            float e2x = exp(2 * x);
            outputMatrix[id] = (e2x - 1) / (e2x + 1);
        }
}

kernel void matrix_tanh_backwards(const device float* outputMatrix [[buffer(0)]],
                                  const device float* inGrad [[buffer(1)]],
                                  const device float* outGrad [[buffer(2)]],
                                  device float* outputGradMatrix [[buffer(3)]],
                                  uint id [[thread_position_in_grid]]) {
    float y = outputMatrix[id];
    float d = outGrad[id]; 
    // outputGradMatrix[id] = d * (1.0 - y * y);
    outputGradMatrix[id] = inGrad[id] + d * (1.0 - y * y);
}

kernel void apply_learning_rate(const device float* inputMatrix [[buffer(0)]],
                           const device float* gradMatrix [[buffer(1)]],
                           device float* outputMatrix [[buffer(2)]],
                           const device float* learningRate [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
    float base = inputMatrix[id];
    float grad = gradMatrix[id];
    float ltr = learningRate[0];

    outputMatrix[id] = base + (ltr * grad);
}

kernel void matrix_scalar_mul(const device float* inputMatrix [[buffer(0)]],
                              const device float* scalar [[buffer(1)]],
                              device float* outputMatrix [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    outputMatrix[id] = inputMatrix[id] * scalar[0];
}

kernel void matrix_scalar_mul_backwards(const device float* outputGrad [[buffer(0)]],
                                        const device float* scalar [[buffer(1)]],
                                        device float* resultGrad [[buffer(2)]],
                                        uint id [[thread_position_in_grid]]) {
    resultGrad[id] = outputGrad[id] * scalar[0];
}

kernel void matrix_pow(const device float* inputMatrix [[buffer(0)]],
                       const device float* exponent [[buffer(1)]],
                       device float* outputMatrix [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    outputMatrix[id] = pow(inputMatrix[id], exponent[0]);
}

kernel void matrix_pow_backwards(const device float* outputGrad [[buffer(0)]],
                                 const device float* inputMatrix [[buffer(1)]],
                                 const device float* exponent [[buffer(2)]],
                                 device float* resultGrad [[buffer(3)]],
                                 uint id [[thread_position_in_grid]]) {
    float y = inputMatrix[id];
    float d = outputGrad[id];
    float e = exponent[0];
    // Derivative of y^e with respect to y is e * y^(e - 1)
    float derivative = e * pow(y, e - 1.0);
    resultGrad[id] = d * derivative;
}

kernel void matrix_add_scalar(const device float* inputMatrix [[buffer(0)]],
                              const device float* scalar [[buffer(1)]],
                              device float* outputMatrix [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    outputMatrix[id] = inputMatrix[id] + scalar[0];
}

kernel void matrix_add_scalar_backwards(const device float* outputGrad [[buffer(0)]],
                                        const device float* inGrad [[buffer(1)]],
                                        device float* resultGrad [[buffer(2)]],
                                        uint id [[thread_position_in_grid]]) {
    resultGrad[id] = inGrad[id] + 1 * outputGrad[id];
}

kernel void matrix_relu(const device float* inputMatrix [[buffer(0)]],
                        device float* outputMatrix [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    outputMatrix[id] = max(inputMatrix[id], 0.0);
}

kernel void matrix_relu_backwards(const device float* outputGrad [[buffer(0)]],
                                  const device float* inputMatrix [[buffer(1)]],
                                  device float* resultGrad [[buffer(2)]],
                                  uint id [[thread_position_in_grid]]) {
    float y = inputMatrix[id];
    float d = outputGrad[id];
    resultGrad[id] = d * (y > 0.0 ? 1.0 : 0.0);
}