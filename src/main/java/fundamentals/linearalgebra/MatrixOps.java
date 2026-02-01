package fundamentals.linearalgebra;

public class MatrixOps {
    static void main() {
        // Initialization of two matrices
        int[][] x = {{12, 45, 67}, {23, 89, 34}, {58, 23, 89}};
        int[][] y = {{34, 58, 23}, {89, 12, 45}, {67, 23, 58}};

        System.out.println("Matrix a: ");
        displayMatrix(x);
        System.out.println("Matrix b: ");
        displayMatrix(y);

        // Element-wise addition of two matrices
        int[][] sumMatrix = addMatrices(x, y);
        System.out.println("Element-wise addition (a + b): ");
        displayMatrix(sumMatrix);

        // Scalar multiplication
        int scalar = 5;
        int[][] scaledMatrix = scalarMultiply(x, scalar);
        System.out.println("Scalar multiplication (a * " + scalar + "): ");
        displayMatrix(scaledMatrix);

        // Matrix multiplication
        int[][] productMatrix = multiplyMatrices(x, y);
        System.out.println("Matrix multiplication (a * b): ");
        displayMatrix(productMatrix);

        // Transpose of a matrix
        int[][] transposedMatrix = transposeMatrix(x);
        System.out.println("Transpose of matrix a: ");
        displayMatrix(transposedMatrix);

        // Row-reduced echelon form
        double[][] z = {{1, 2}, {3, 8}};
        System.out.println("Original matrix z: ");
        displayMatrix(z);
        toEchelonForm(z);
        System.out.println("Echelon form of matrix z: ");
        displayMatrix(z);
        System.out.println("Row-Reduced Echelon form of matrix z: ");
        toRREF(z);
        displayMatrix(z);
    }

    static void toRREF(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        int row = 0;

        // Forward elimination (REF)
        for (int i = 0; i < cols && row < rows; i++) {
            // Find pivot
            int pivotRow = -1;
            for (int j = row; j < rows; j++) {
                if (matrix[j][i] != 0) {
                    pivotRow = j;
                    break;
                }
            }
            if (pivotRow == -1) continue;

            // Swap pivot row into position
            if (pivotRow != row) {
                exchangeRows(matrix, row, pivotRow);
            }

            // Scale pivot row to make pivot = 1
            double pivotVal = matrix[row][i];
            scaleRow(matrix, row, 1.0 / pivotVal);

            // Eliminate below
            for (int j = row + 1; j < rows; j++) {
                double factor = matrix[j][i];
                if (factor != 0) {
                    addScaledRows(matrix[j], matrix[row], -factor);
                }
            }

            row++;
        }

        // Backward elimination (to RREF)
        for (int i = row - 1; i >= 0; i--) {
            // Find pivot column
            int pivotCol = -1;
            for (int j = 0; j < cols; j++) {
                if (matrix[i][j] == 1) {
                    pivotCol = j;
                    break;
                }
            }
            if (pivotCol == -1) continue;

            // Eliminate above
            for (int k = i - 1; k >= 0; k--) {
                double factor = matrix[k][pivotCol];
                if (factor != 0) {
                    addScaledRows(matrix[k], matrix[i], -factor);
                }
            }
        }
    }

    static void toEchelonForm(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        int row = 0;

        for (int i = 0; i < cols && row < rows; i++) {
            // Find pivot
            int pivotRow = -1;
            for (int j = row; j < rows; j++) {
                if (matrix[j][i] != 0) {
                    pivotRow = j;
                    break;
                }
            }

            if (pivotRow == -1) continue; // No pivot in this column

            // Swap pivot row into position
            if (pivotRow != row) {
                exchangeRows(matrix, row, pivotRow);
            }

            // Scale pivot row to make pivot = 1
            double pivotVal = matrix[row][i];
            scaleRow(matrix, row, 1.0 / pivotVal);

            // Eliminate below
            reduceOtherRows(matrix, row, i);

            row++;
        }
    }

    static void scaleRow(double[][] matrix, int row, double scale) {
        for (int j = 0; j < matrix[row].length; j++) {
            matrix[row][j] *= scale;
        }
    }

    static void exchangeRows(double[][] matrix, int row, int pivotRow) {
        double[] temp = matrix[row];
        matrix[row] = matrix[pivotRow];
        matrix[pivotRow] = temp;
    }

    static void reduceOtherRows(double[][] matrix, int currentRow, int currentCol) {
        for (int i = currentRow + 1; i < matrix.length; i++) {
            double factor = matrix[i][currentCol];
            if (factor != 0) {
                addScaledRows(matrix[i], matrix[currentRow], -factor);
            }
        }
    }

    static void addScaledRows(double[] targetRow, double[] sourceRow, double scale) {
        for (int j = 0; j < targetRow.length; j++) {
            targetRow[j] += sourceRow[j] * scale;
        }
    }

    static int[][] transposeMatrix(int[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        int[][] transposed = new int[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }

    static int[][] multiplyMatrices(int[][] a, int[][] b) {
        if (a[0].length != b.length) {
            throw new IllegalArgumentException("Number of columns in the first matrix must be equal to the number of rows in the second matrix for multiplication.");
        }
        int rows = a.length;
        int cols = b[0].length;
        int commonDim = a[0].length;
        int[][] result = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < commonDim; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    }

    static int[][] scalarMultiply(int[][] matrix, int scalar) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        int[][] result = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = matrix[i][j] * scalar;
            }
        }
        return result;
    }

    static int[][] addMatrices(int[][] a, int[][] b) {
        if (a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException("Matrices must be of the same dimensions for addition.");
        }
        int rows = a.length;
        int cols = a[0].length;
        int[][] result = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }

    static void displayMatrix(int[][] matrix) {
        for (int[] rows : matrix) {
            for (int elements : rows) {
                System.out.print(elements + " ");
            }
            System.out.println();
        }
    }

    static void displayMatrix(double[][] matrix) {
        for (double[] row : matrix) {
            for (double element : row) {
                System.out.print(element + " ");
            }
            System.out.println();
        }
    }
}
