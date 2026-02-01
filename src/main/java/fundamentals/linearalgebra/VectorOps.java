package fundamentals.linearalgebra;

import java.util.Vector;

public class VectorOps {
    static void main() {
        // Initialization of two vectors
        Vector<Integer> v1 = new Vector<>();
        Vector<Integer> v2 = new Vector<>();
        int[] x = {12, 45, 67, 23, 89};
        int[] y = {34, 58, 23, 89, 12};

        addElements(v1, x);
        addElements(v2, y);

        System.out.println("Vector v1: " + v1);
        System.out.println("Vector v2: " + v2);

        // Element-wise addition of two vectors
        Vector<Integer> v3 = addVectors(v1, v2);
        System.out.println("Element-wise addition (v1 + v2): " + v3);

        // Scalar multiplication
        int scalar = 3;
        Vector<Integer> v4 = scalarMultiply(v1, scalar);
        System.out.println("Scalar multiplication (v1 * " + scalar + "): " + v4);

        // Dot product
        int dotProduct = dotProduct(v1, v2);
        System.out.println("Dot product (v1 . v2): " + dotProduct);
    }

    static int dotProduct(Vector<Integer> v1, Vector<Integer> v2) {
        if (v1.size() != v2.size()) {
            throw new IllegalArgumentException("Vectors must be of the same size for dot product.");
        }
        int result = 0;
        for (int i = 0; i < v1.size(); i++) {
            result += v1.get(i) * v2.get(i);
        }
        return result;
    }

    static Vector<Integer> scalarMultiply(Vector<Integer> v1, int scalar) {
        Vector<Integer> result = new Vector<>();
        for (int value : v1) {
            result.add(value * scalar);
        }
        return result;
    }

    static Vector<Integer> addVectors(Vector<Integer> v1, Vector<Integer> v2) {
        if (v1.size() != v2.size()) {
            throw new IllegalArgumentException("Vectors must be of the same size for addition.");
        }
        Vector<Integer> result = new Vector<>();
        for (int i = 0; i < v1.size(); i++) {
            result.add(v1.get(i) + v2.get(i));
        }
        return result;
    }

    static void addElements(Vector<Integer> v, int[] elements) {
        for (int element : elements) {
            v.add(element);
        }
    }
}
