package utils;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Random;

public class SimdMatrix {
    public final int rows;
    public final int cols;
    public final float[] data;

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public SimdMatrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new float[rows * cols];
    }

    public SimdMatrix(int rows, int cols, float[] data) {
        this.rows = rows;
        this.cols = cols;
        this.data = data;
    }

    /**
     * Matrix Multiplication (C = A * B).
     * This method multiplies this matrix (A) with another matrix (B) and returns the result (C).
     * It uses SIMD vectorization for the innermost loop to optimize performance.
     * In the innermost loop, we always walk across contiguous memory of B’s row and C’s row. This is cache-friendly.
     * @param other The matrix to multiply with this matrix. Must have shape (cols of this, any).
     * @return A new SimdMatrix that is the result of multiplying this matrix with the other matrix.
     */
    public SimdMatrix matmul(SimdMatrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("Shape mismatch: " + shape() + " vs " + other.shape());
        }
        SimdMatrix result = new SimdMatrix(this.rows, other.cols);

        // Cache local variables for speed
        int m = this.rows;
        int n = this.cols;
        int p = other.cols;
        int loopBound = SPECIES.loopBound(p);

        // Loop i: Picking row of A and C (Accessing C sequentially)
        for (int i = 0; i < m; i++) {
            int rowOffsetC = i * p; // Row offset for C (result)
            int rowOffsetA = i * n; // Row offset for A (this)

            // Loop k: Iterating over columns of A / rows of B (Accessing A sequentially, B in row chunks)
            // Take one element from row i of A, and multiply it with the corresponding row k of B, accumulating into row i of C.
            // Take A[i][k] and multiply with B[k][j...] to accumulate into C[i][j...]
            for (int k = 0; k < n; k++) {
                float valA = this.data[rowOffsetA + k];

                // Optimization: If A[i][k] is zero, skip the entire row operation for this k (sparse optimization)
                if (valA == 0.0f) continue;

                int rowOffsetB = k * p; // Row offset for B (other) corresponding to the k-th row which we will multiply with valA
                int j = 0;

                // Loop j: Iterating over columns of B and C (Accessing B and C sequentially in chunks)
                // Multiply the scalar valA with the k-th row of B and add it to the i-th row of C. We will do this in vectorized chunks.
                // We will process p columns in chunks of the vector size (loopBound) for SIMD, and then handle any remaining columns in a scalar loop.
                for (; j < loopBound; j += SPECIES.length()) {
                    // Load Accumulator (C)
                    // We load the current values of C[i][j...] into a vector register to accumulate the results. This allows us to perform the FMA operation in-place.
                    var vc = FloatVector.fromArray(SPECIES, result.data, rowOffsetC + j);
                    // Load Weights (B)
                    // We load a vector from the k-th row of B starting at column j. This gives us a vector of values that we will multiply by the scalar valA.
                    var vb = FloatVector.fromArray(SPECIES, other.data, rowOffsetB + j);

                    // FMA: vc = vc + (vb * valA)
                    // We perform a fused multiply-add operation where we multiply the vector vb by the scalar valA and then add it to the accumulator vc. This is more efficient than separate multiply and add operations.
                    vc = vb.mul(valA).add(vc);

                    // Store back
                    vc.intoArray(result.data, rowOffsetC + j);
                }

                // Cleanup Scalar Loop (Tail)
                for (; j < p; j++) {
                    result.data[rowOffsetC + j] += valA * other.data[rowOffsetB + j];
                }
            }
        }
        return result;
    }

    /**
     * Broadcast Add (Bias Addition).
     * Adds a bias vector (1 x cols) to every row of this matrix.
     * @param vector The bias vector to add. Must have shape (1, cols).
     * @return A new SimdMatrix where the bias vector has been added to each row of this matrix.
     */
    public SimdMatrix addRowVector(SimdMatrix vector) {
        if (vector.rows != 1 || vector.cols != this.cols) {
            throw new IllegalArgumentException("Shape mismatch for bias add");
        }
        SimdMatrix result = new SimdMatrix(this.rows, this.cols);
        int loopBound = SPECIES.loopBound(this.cols);

        for (int i = 0; i < this.rows; i++) {
            int offset = i * this.cols;
            int j = 0;
            // Vectorized addition
            for (; j < loopBound; j += SPECIES.length()) {
                var vData = FloatVector.fromArray(SPECIES, this.data, offset + j);
                var vBias = FloatVector.fromArray(SPECIES, vector.data, j);
                vData.add(vBias).intoArray(result.data, offset + j);
            }
            // Tail loop
            for (; j < this.cols; j++) {
                result.data[offset + j] = this.data[offset + j] + vector.data[j];
            }
        }
        return result;
    }

    /**
     * Element-wise Multiplication (Hadamard Product).
     * @param other The other matrix to multiply element-wise with this matrix. Must have the same shape.
     * @return A new SimdMatrix where each element is the product of the corresponding elements in this and the other matrix.
     */
    public SimdMatrix elementMult(SimdMatrix other) {
        // Hadamard Product (A * B element-wise)
        if (this.rows != other.rows || this.cols != other.cols) throw new IllegalArgumentException("Shape mismatch");
        SimdMatrix result = new SimdMatrix(this.rows, this.cols);

        int len = this.data.length;
        int loopBound = SPECIES.loopBound(len);
        int i = 0;

        for (; i < loopBound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, this.data, i);
            var vb = FloatVector.fromArray(SPECIES, other.data, i);
            va.mul(vb).intoArray(result.data, i);
        }
        for (; i < len; i++) {
            result.data[i] = this.data[i] * other.data[i];
        }
        return result;
    }

    /**
     * Element-wise Subtraction.
     * @param other The other matrix to subtract from this matrix. Must have the same shape.
     * @return A new SimdMatrix where each element is the difference of the corresponding elements in this and the other matrix (this - other).
     */
    public SimdMatrix sub(SimdMatrix other) {
        // Subtraction (Prediction - Target)
        if (this.rows != other.rows || this.cols != other.cols) throw new IllegalArgumentException("Shape mismatch");
        SimdMatrix result = new SimdMatrix(this.rows, this.cols);

        int len = this.data.length;
        int loopBound = SPECIES.loopBound(len);
        int i = 0;

        for (; i < loopBound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, this.data, i);
            var vb = FloatVector.fromArray(SPECIES, other.data, i);
            va.sub(vb).intoArray(result.data, i);
        }
        for (; i < len; i++) {
            result.data[i] = this.data[i] - other.data[i];
        }
        return result;
    }

    /**
     * Scalar Multiplication.
     * @param alpha The scalar value to multiply each element of this matrix by.
     * @return A new SimdMatrix where each element is the product of the corresponding element in this matrix and the scalar alpha.
     */
    public SimdMatrix scale(float alpha) {
        // Scalar multiplication (Weights -= lr * gradients)
        SimdMatrix result = new SimdMatrix(this.rows, this.cols);
        int len = this.data.length;
        int loopBound = SPECIES.loopBound(len);
        int i = 0;

        for (; i < loopBound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, this.data, i)
                    .mul(alpha)
                    .intoArray(result.data, i);
        }
        for (; i < len; i++) result.data[i] = this.data[i] * alpha;
        return result;
    }

    /**
     * Transpose of the matrix.
     * @return A new SimdMatrix that is the transpose of this matrix (rows and columns swapped).
     */
    public SimdMatrix transpose() {
        SimdMatrix t = new SimdMatrix(this.cols, this.rows);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                t.data[j * this.rows + i] = this.data[i * this.cols + j];
            }
        }
        return t;
    }

    /**
     * Computes the sum of all elements in the matrix using SIMD vectorization for performance.
     * @return The sum of all elements in the matrix.
     */
    public float sum() {
        // Sum all elements (for Loss)
        int len = this.data.length;
        int loopBound = SPECIES.loopBound(len);
        var vSum = FloatVector.zero(SPECIES);
        int i = 0;

        for (; i < loopBound; i += SPECIES.length()) {
            vSum = vSum.add(FloatVector.fromArray(SPECIES, this.data, i));
        }
        float sum = vSum.reduceLanes(VectorOperators.ADD);

        for (; i < len; i++) sum += this.data[i];
        return sum;
    }

    /**
     * Finds the maximum element in the matrix using SIMD vectorization for performance.
     * @return The maximum element in the matrix.
     */
    public float max() {
        // Max element (for Softmax stability)
        int len = this.data.length;
        int loopBound = SPECIES.loopBound(len);
        var vMax = FloatVector.broadcast(SPECIES, -Float.MAX_VALUE);
        int i = 0;

        for (; i < loopBound; i += SPECIES.length()) {
            vMax = vMax.max(FloatVector.fromArray(SPECIES, this.data, i));
        }
        float max = vMax.reduceLanes(VectorOperators.MAX);

        for (; i < len; i++) max = Math.max(max, this.data[i]);
        return max;
    }

    // For debugging and testing purposes, we can implement a method to get the shape of the matrix as a string.
    public String shape() { return "(" + rows + ", " + cols + ")"; }

    public static SimdMatrix random(int rows, int cols) {
        SimdMatrix m = new SimdMatrix(rows, cols);
        Random r = new Random();
        for (int i = 0; i < m.data.length; i++) {
            m.data[i] = (float) (r.nextGaussian() * 0.1f);
        }
        return m;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Matrix ").append(shape()).append("\n");
        int rMax = Math.min(rows, 6);
        int cMax = Math.min(cols, 6);
        for(int i=0; i<rMax; i++) {
            sb.append("[ ");
            for(int j=0; j<cMax; j++) sb.append(String.format("%.4f ", data[i*cols+j]));
            if(cols > cMax) sb.append("... ");
            sb.append("]\n");
        }
        return sb.toString();
    }
}