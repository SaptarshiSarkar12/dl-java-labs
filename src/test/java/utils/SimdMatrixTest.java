package utils;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("SIMD-Optimized Matrix Multiplication Tests")
public class SimdMatrixTest {

    /**
     * Helper method: Naive Matrix Multiplication (Standard Triple Loop).
     * Used as the "Ground Truth" to verify the SIMD implementation.
     * We use float[] here to match the precision of SimdMatrix.
     */
    private float[] naiveMatrixMultiply(float[] A, float[] B, int m, int n, int p) {
        float[] C = new float[m * p];

        // Standard i-j-k loop (Slow, but easy to verify correctness)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) {
                    // A[i][k] * B[k][j]
                    // Row-major index: row * width + col
                    sum += A[i * n + k] * B[k * p + j];
                }
                C[i * p + j] = sum;
            }
        }
        return C;
    }

    @Test
    @DisplayName("Verify SIMD MatMul correctness against Naive Loop")
    public void testSimdMatrixMultiplicationCorrectness() {
        int m = 1024;
        int n = 512;
        int p = 1024;

        SimdMatrix matA = SimdMatrix.random(m, n);
        SimdMatrix matB = SimdMatrix.random(n, p);

        // SIMD Implementation
        long startSimd = System.nanoTime();
        SimdMatrix resultSimd = matA.matmul(matB);
        long endSimd = System.nanoTime();

        // Naive Implementation
        long startNaive = System.nanoTime();
        float[] resultNaiveData = naiveMatrixMultiply(matA.data, matB.data, m, n, p);
        long endNaive = System.nanoTime();

        // Verification
        // Check shapes
        Assertions.assertEquals(m, resultSimd.rows);
        Assertions.assertEquals(p, resultSimd.cols);

        // Check Data Content
        // We use an epsilon (tolerance) because SIMD FMA instructions handle rounding
        // slightly differently (and often more accurately) than scalar loops.
        float epsilon = 1e-4f;

        for (int i = 0; i < resultNaiveData.length; i++) {
            float expected = resultNaiveData[i];
            float actual = resultSimd.data[i];

            Assertions.assertEquals(expected, actual, epsilon, "Mismatch at index " + i + ". SIMD logic may be incorrect.");
        }

        // Output Timing (Just for visibility)
        System.out.println("--- Test Passed ---");
        System.out.printf("Matrix Size: [%d x %d] * [%d x %d]%n", m, n, n, p);
        System.out.printf("SIMD Time:  %.3f ms%n", (endSimd - startSimd) / 1e6);
        System.out.printf("Naive Time: %.3f ms%n", (endNaive - startNaive) / 1e6);
        System.out.printf("Speedup:    %.1fx%n", (double)(endNaive - startNaive) / (endSimd - startSimd));
    }
}