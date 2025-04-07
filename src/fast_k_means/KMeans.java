// Port courtesy of Gemini 2.5

package fast_k_means;

import java.util.Arrays;

import static fast_k_means.KMeansUtils.distanceSq;
import static fast_k_means.KMeansUtils.centroid;
import static fast_k_means.KMeansUtils.KMeansResult;

/**
 * Java implementation of the k-means clustering algorithm.
 * This class provides methods to perform k-means clustering using Lloyd's algorithm.
 * It includes methods for performing a single step of the algorithm and for running the full k-means process.
 */
public final class KMeans {

    // --- New/Translated Methods ---

    /**
     * Performs one step of the Lloyd's k-means algorithm.
     * Assigns points to nearest centers and updates center positions.
     *
     * @param dim         Dimension of points.
     * @param dataset     Input data points (flat array).
     * @param centers     Current cluster centers (flat array). Modified in place.
     * @param nextCenters Temporary storage for coordinate sums (must be same size as centers).
     * @param q           Temporary storage for point counts per cluster (must be size k).
     * @param a           Assignment array (size n). Stores the offset of the assigned center. Modified in place.
     * @return            {@code true} if any point assignment changed, {@code false} otherwise.
     */
    private static boolean stepLloyd(int dim,
                                     float[] dataset,
                                     float[] centers,        // Modifies this in-place
                                     float[] nextCenters,    // Used as temp buffer
                                     long[] q,               // Used as temp buffer (counts)
                                     long[] a) {             // Modifies this in-place

        boolean changed = false;
        int k = q.length; // Number of clusters
        int n = a.length; // Number of data points

        // Reset buffers for the current iteration
        Arrays.fill(nextCenters, 0.0f);
        Arrays.fill(q, 0L);

        // --- Assignment Step ---
        // Iterate through each point i in the dataset
        for (int i = 0, dataOffset = 0; i < n; ++i, dataOffset += dim) {
            long bestCenterOffset = 0; // Store the index/offset of the best center
            float minDsq = Float.POSITIVE_INFINITY;

            // Find the nearest center for point i
            for (int clusterIdx = 0, centerOffset = 0; clusterIdx < k; ++clusterIdx, centerOffset += dim) {
                // Ensure centerOffset does not exceed bounds (important if centers array was resized unexpectedly)
                if (centerOffset + dim <= centers.length) {
                   float dsq = distanceSq(dim, dataset, dataOffset, centers, centerOffset);
                   if (dsq < minDsq) {
                       minDsq = dsq;
                       bestCenterOffset = centerOffset;
                   }
                } else {
                     // This case should ideally not happen if k matches centers.length/dim
                     System.err.println("Warning: centerOffset out of bounds in stepLloyd assignment.");
                     // Decide how to handle: skip this center? Throw error?
                     // For now, we implicitly skip checking this center.
                }

            }

            // Check if assignment changed
            if (a[i] != bestCenterOffset) {
                changed = true;
            }
            a[i] = bestCenterOffset; // Update assignment (store offset)

            // Update count and sum for the assigned cluster
            int bestClusterIndex = (int)(bestCenterOffset / dim); // Calculate cluster index from offset
            if (bestClusterIndex >= 0 && bestClusterIndex < q.length) {
                q[bestClusterIndex]++;
                // Add point coordinates to the sum for this cluster
                // Loop candidate for JIT auto-vectorization
                for (int d = 0; d < dim; ++d) {
                    // Bounds check for safety, though should be fine if dataset/dim are correct
                    if (dataOffset + d < dataset.length && bestCenterOffset + d < nextCenters.length) {
                       nextCenters[(int)bestCenterOffset + d] += dataset[dataOffset + d];
                    }
                }
            } else {
                 // Should not happen with valid bestCenterOffset
                 System.err.println("Warning: bestClusterIndex out of bounds in stepLloyd update.");
            }
        }

        // --- Update Step ---
        // Iterate through each cluster and update its center
        for (int clusterIdx = 0, centerOffset = 0; clusterIdx < k; ++clusterIdx, centerOffset += dim) {
            if (q[clusterIdx] > 0) {
                float countF = (float) q[clusterIdx]; // Cast count to float once
                // Calculate new center by dividing sum by count
                // Loop candidate for JIT auto-vectorization
                for (int d = 0; d < dim; ++d) {
                     // Bounds check for safety
                     if (centerOffset + d < centers.length && centerOffset + d < nextCenters.length) {
                         centers[centerOffset + d] = nextCenters[centerOffset + d] / countF;
                     }
                }
            } else {
                // Empty cluster: Its position in 'centers' remains unchanged from the previous iteration.
                // Optionally, could implement logic here to handle empty clusters
                // (e.g., re-initialize, assign furthest point), but C++ code didn't.
                 System.err.println("Warning: Cluster " + clusterIdx + " became empty.");
            }
        }

        return changed; // Return whether any assignments changed
    }

    /**
     * Performs k-means clustering using Lloyd's algorithm.
     *
     * @param dim            Dimension of the data points.
     * @param dataset        Input data points (flat array: [p1_x, p1_y, ..., pN_dim]).
     * @param initialCenters Initial positions for the k cluster centers (flat array). This array is NOT modified.
     * @param k              The desired number of clusters.
     * @param maxIterations  The maximum number of iterations to perform.
     * @return A KMeansResult object containing final centers, assignments, iterations run, and convergence status.
     */
    public static KMeansResult kMeans(int dim,
                                      final float[] dataset,   // Mark final to emphasize it's not changed
                                      final float[] initialCenters, // Mark final, we'll copy it
                                      int k,
                                      int maxIterations) {

        if (dim <= 0 || dataset == null || dataset.length == 0 || k <= 0) {
            // Handle invalid input gracefully
            System.err.println("Warning: Invalid input to kMeans (dim, dataset, or k).");
            return new KMeansResult(0, new float[0], new long[0], 0, false);
        }
        if (initialCenters == null || initialCenters.length != k * dim) {
            throw new IllegalArgumentException("initialCenters must be non-null and have size k * dim");
        }
        if (dataset.length % dim != 0) {
            throw new IllegalArgumentException("dataset length must be a multiple of dim");
        }

        int n = dataset.length / dim; // Number of data points

        // --- Handle Edge Cases ---
        if (k == 1) {
            float[] center = new float[dim];
            centroid(dim, dataset, center); // Calculate the single centroid
            long[] assignments = new long[n]; // All points assigned to cluster 0 (offset 0)
            // Arrays.fill(assignments, 0L); // Default long value is 0, so fill is optional
            return new KMeansResult(1, center, assignments, 0, true); // 0 iterations, converged
        }

        if (k >= n) {
            k = n; // Cannot have more clusters than points
            long[] assignments = new long[n];
            float[] centers = new float[n * dim];
            // Each point is its own cluster
            System.arraycopy(dataset, 0, centers, 0, dataset.length); // Copy dataset points as centers
            for (int i = 0; i < n; ++i) {
                assignments[i] = (long)i * dim; // Assign point i to center i (offset i*dim)
            }
            return new KMeansResult(k, centers, assignments, 0, true); // 0 iterations, converged
        }

        // --- Standard k-means Initialization ---
        long[] assignments = new long[n]; // Point assignments (stores center offset)
        // Create a COPY of initial centers to avoid modifying the caller's array
        float[] centers = Arrays.copyOf(initialCenters, initialCenters.length);
        float[] nextCenters = new float[centers.length]; // Buffer for sums
        long[] counts = new long[k]; // Buffer for counts

        int iterationsRun = 0;
        boolean converged = false;

        // --- Iteration Loop ---
        for (iterationsRun = 0; iterationsRun < maxIterations; ++iterationsRun) {
            boolean changed = stepLloyd(dim, dataset, centers, nextCenters, counts, assignments);
            if (!changed) {
                converged = true;
                break; // Exit loop if no assignments changed
            }
        }

        if (!converged) {
            System.err.println("Warning: k-means did not converge after " + maxIterations + " iterations.");
        }

        // Create the result object - the constructor takes ownership of centers and assignments
        return new KMeansResult(k, centers, assignments, iterationsRun, converged);
    }
}