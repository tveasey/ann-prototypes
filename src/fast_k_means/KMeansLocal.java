// Port courtesy of Gemini 2.5

package fast_k_means;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue; // For priority queue

import static fast_k_means.KMeansUtils.distanceSq;
import static fast_k_means.KMeansUtils.KMeansResult;

public final class KMeansLocal {
    /**
     * Helper class to store distance and offset for the priority queue in computeNeighborhoods.
     * Implements Comparable for use in a max-heap (keeps track of largest distances).
     */
    private static final class NeighborInfo implements Comparable<NeighborInfo> {
        final float distanceSq;
        final long offset; // Store offset directly

        NeighborInfo(float distanceSq, long offset) {
            this.distanceSq = distanceSq;
            this.offset = offset;
        }

        @Override
        public int compareTo(NeighborInfo other) {
            // Reverse order for max-heap behavior based on distance
            return Float.compare(other.distanceSq, this.distanceSq);
        }

        // Optional: equals and hashCode if used in Sets/Maps, though not strictly needed for PQ here
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            NeighborInfo that = (NeighborInfo) o;
            return Float.compare(that.distanceSq, distanceSq) == 0 && offset == that.offset;
        }

        @Override
        public int hashCode() {
            int result = Float.hashCode(distanceSq);
            result = 31 * result + Long.hashCode(offset);
            return result;
        }
    }

    /**
     * Computes the nearest neighbors for each cluster center.
     *
     * @param dim                     Dimension of points.
     * @param centers                 Flat array of center coordinates.
     * @param neighborhoods           Output List. neighborhoods.get(i) will be populated with the
     * offsets (in centers array) of the nearest neighbors to center i.
     * Assumes the outer list is initialized with 'k' null or empty entries.
     * @param clustersPerNeighborhood The maximum number of neighbors to find for each center.
     */
    private static void computeNeighborhoods(int dim,
                                             float[] centers,
                                             List<long[]> neighborhoods, // Modified in place
                                             int clustersPerNeighborhood) {

        int k = neighborhoods.size(); // Number of centers
        if (k == 0 || clustersPerNeighborhood <= 0) {
            return; // Nothing to compute
        }

        // List of max-priority queues (stores NeighborInfo, ordered by distance descending)
        List<PriorityQueue<NeighborInfo>> neighborQueues = new ArrayList<>(k);
        for (int i = 0; i < k; ++i) {
            // Max-heap using NeighborInfo's compareTo (or Comparator.reverseOrder())
            neighborQueues.add(new PriorityQueue<>());
        }

        // Lambda-like helper using a private method for clarity
        UpdateNeighborsHelper updateNeighborsHelper = new UpdateNeighborsHelper(clustersPerNeighborhood);

        // Compute pairwise distances and update neighborhood queues
        for (int i = 0, iOffset = 0; i < k; ++i, iOffset += dim) {
            for (int j = 0, jOffset = 0; j < i; ++j, jOffset += dim) {
                // Ensure offsets are within bounds
                if (iOffset + dim <= centers.length && jOffset + dim <= centers.length) {
                    float dsq = distanceSq(dim, centers, iOffset, centers, jOffset);

                    // Update neighborhood for center i (add neighbor j)
                    updateNeighborsHelper.update(jOffset, dsq, neighborQueues.get(i));
                    // Update neighborhood for center j (add neighbor i)
                    updateNeighborsHelper.update(iOffset, dsq, neighborQueues.get(j));
                } else {
                    System.err.println("Warning: Center offset out of bounds in computeNeighborhoods.");
                }
            }
        }

        // Extract neighbor offsets from queues and sort them
        for (int i = 0; i < k; ++i) {
            PriorityQueue<NeighborInfo> queue = neighborQueues.get(i);
            int neighborCount = queue.size();
            long[] neighbors = new long[neighborCount];
            int idx = 0;
            while (!queue.isEmpty()) {
                // Store the offset of the neighbor
                neighbors[idx++] = queue.poll().offset;
            }
            // Sort neighbors by offset (index in the centers array)
            Arrays.sort(neighbors);
            neighborhoods.set(i, neighbors); // Set the computed neighbors in the output list
        }
    }

    /** Helper class to encapsulate the logic for updating the priority queue */
    private static class UpdateNeighborsHelper {
        private final int maxSize;

        UpdateNeighborsHelper(int clustersPerNeighborhood) {
            this.maxSize = clustersPerNeighborhood;
        }

        void update(long neighborOffset, float distanceSq, PriorityQueue<NeighborInfo> queue) {
            if (queue.size() < maxSize) {
                queue.offer(new NeighborInfo(distanceSq, neighborOffset));
            } else {
                // Queue is full, check if new distance is smaller than the largest distance currently in the queue
                NeighborInfo largestNeighbor = queue.peek(); // Peek returns element with highest priority (max distance)
                if (largestNeighbor != null && distanceSq < largestNeighbor.distanceSq) {
                    queue.poll(); // Remove the neighbor with the largest distance
                    queue.offer(new NeighborInfo(distanceSq, neighborOffset)); // Add the new, closer neighbor
                }
            }
        }
    }

    /**
     * Performs one step of the Lloyd's k-means algorithm, using neighborhoods for assignment.
     * Assigns points to nearest centers within the neighborhood and updates center positions.
     *
     * @param dim           Dimension of points.
     * @param dataset       Input data points (flat array).
     * @param neighborhoods Precomputed neighborhoods for each center. neighborhoods.get(i) contains neighbor offsets for center i.
     * @param centers       Current cluster centers (flat array). Modified in place.
     * @param nextCenters   Temporary storage for coordinate sums (must be same size as centers).
     * @param q             Temporary storage for point counts per cluster (must be size k).
     * @param a             Assignment array (size n). Stores the offset of the assigned center. Modified in place.
     * @return              {@code true} if any point assignment changed, {@code false} otherwise.
     */
    private static boolean stepLloyd(int dim,
                                     float[] dataset,
                                     List<long[]> neighborhoods, // Use precomputed neighborhoods
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

        // --- Assignment Step (Localized) ---
        // Iterate through each point i in the dataset
        for (int i = 0, dataOffset = 0; i < n; ++i, dataOffset += dim) {
            long currentAssignmentOffset = a[i];
            long bestCenterOffset = currentAssignmentOffset; // Start assuming current center is best

            // Calculate distance to the *currently assigned* center first
            float minDsq = distanceSq(dim, dataset, dataOffset, centers, (int)currentAssignmentOffset);

            // Determine the index of the current cluster to fetch its neighborhood
            int currentClusterIndex = (int)(currentAssignmentOffset / dim);

            // Check neighborhood of the *current* cluster
            if (currentClusterIndex >= 0 && currentClusterIndex < neighborhoods.size()) {
                long[] neighborOffsets = neighborhoods.get(currentClusterIndex);
                if (neighborOffsets != null) {
                    for (long neighborOffset : neighborOffsets) {
                        // Ensure neighbor offset is valid before calculating distance
                        if (neighborOffset >= 0 && neighborOffset + dim <= centers.length) {
                            float dsq = distanceSq(dim, dataset, dataOffset, centers, (int)neighborOffset);
                            if (dsq < minDsq) {
                                minDsq = dsq;
                                bestCenterOffset = neighborOffset;
                            }
                        } else {
                            System.err.println("Warning: Invalid neighbor offset (" + neighborOffset + ") found in neighborhood for cluster " + currentClusterIndex);
                        }
                    }
                }
            } else {
                System.err.println("Warning: Invalid current cluster index (" + currentClusterIndex + ") derived from offset " + currentAssignmentOffset);
            }

            // Check if assignment changed
            if (a[i] != bestCenterOffset) {
                changed = true;
            }
            a[i] = bestCenterOffset; // Update assignment (store offset)

            // Update count and sum for the (potentially newly) assigned cluster
            // Ensure bestCenterOffset is valid before using it
            if (bestCenterOffset >= 0 && bestCenterOffset + dim <= centers.length) {
                int bestClusterIndex = (int)(bestCenterOffset / dim);
                if (bestClusterIndex >= 0 && bestClusterIndex < q.length) {
                    q[bestClusterIndex]++;
                    // Add point coordinates to the sum for this cluster
                    // Loop candidate for JIT auto-vectorization
                    for (int d = 0; d < dim; ++d) {
                        if (dataOffset + d < dataset.length) { // Check dataset bounds
                            nextCenters[(int)bestCenterOffset + d] += dataset[dataOffset + d];
                        }
                    }
                } else {
                    System.err.println("Warning: bestClusterIndex out of bounds after assignment for point " + i);
                }
            } else {
                System.err.println("Warning: Invalid bestCenterOffset (" + bestCenterOffset + ") assigned to point " + i + ". Skipping count/sum update.");
            }

        } // End loop over points

        // --- Update Step (Identical to original stepLloyd) ---
        // Iterate through each cluster and update its center
        for (int clusterIdx = 0, centerOffset = 0; clusterIdx < k; ++clusterIdx, centerOffset += dim) {
            if (q[clusterIdx] > 0) {
                float countF = (float) q[clusterIdx];
                // Calculate new center by dividing sum by count
                for (int d = 0; d < dim; ++d) {
                    if (centerOffset + d < centers.length && centerOffset + d < nextCenters.length) {
                        centers[centerOffset + d] = nextCenters[centerOffset + d] / countF;
                    }
                }
            } else {
                // Empty cluster: Its position in 'centers' remains unchanged.
                // System.err.println("Warning: Cluster " + clusterIdx + " is empty after step.");
            }
        }

        return changed; // Return whether any assignments changed
    }

    /**
     * Performs k-means clustering using a localized version of Lloyd's algorithm,
     * where each point is only compared against centers in the neighborhood
     * of its currently assigned center.
     *
     * @param dim                     Dimension of the data points.
     * @param dataset                 Input data points (flat array).
     * @param initialCenters          Initial positions for the k cluster centers (flat array). This array is NOT modified.
     * @param initialAssignments      Initial assignment for each point (flat array, stores center offset). This array is NOT modified.
     * @param clustersPerNeighborhood The number of neighboring centers to consider during assignment.
     * @param maxIterations           The maximum number of iterations to perform.
     * @return A KMeansResult object containing final centers, assignments, iterations run, and convergence status.
     */
    public static KMeansResult kMeansLocal(int dim,
                                           final float[] dataset,
                                           final float[] initialCenters,
                                           final long[] initialAssignments, // Takes initial assignments now
                                           int clustersPerNeighborhood,
                                           int maxIterations) {

        // --- Input Validation ---
        if (dim <= 0 || dataset == null || dataset.length == 0) {
             System.err.println("Warning: Invalid input to kMeansLocal (dim, dataset).");
             return new KMeansResult(0, new float[0], new long[0], 0, false);
        }
        if (initialCenters == null || initialCenters.length % dim != 0) {
            throw new IllegalArgumentException("initialCenters must be non-null and have length multiple of dim");
        }
        int k = initialCenters.length / dim; // Number of clusters
        if (k <= 0) {
            System.err.println("Warning: k must be positive in kMeansLocal.");
            return new KMeansResult(0, new float[0], new long[0], 0, false);
        }
        if (dataset.length % dim != 0) {
            throw new IllegalArgumentException("dataset length must be a multiple of dim");
        }
        int n = dataset.length / dim; // Number of points
        if (initialAssignments == null || initialAssignments.length != n) {
            throw new IllegalArgumentException("initialAssignments must be non-null and have size n (number of points)");
        }
        if (clustersPerNeighborhood <= 0) {
            System.err.println("Warning: clustersPerNeighborhood must be positive. Using k=" + k + " instead.");
            clustersPerNeighborhood = k; // Default to checking all clusters if input is invalid
        }

        // --- Create copies to avoid modifying caller's arrays ---
        float[] centers = Arrays.copyOf(initialCenters, initialCenters.length);
        long[] assignments = Arrays.copyOf(initialAssignments, initialAssignments.length);

        // --- Handle Edge Cases (using copied state) ---
        if (k == 1 || k >= n) {
            // No iterations needed, return the initial state (copied)
            boolean converged = true; // Already in final state
            return new KMeansResult(k, centers, assignments, 0, converged);
        }

        // --- Compute Neighborhoods ---
        List<long[]> neighborhoods = new ArrayList<>(k);
        // Initialize the list structure before passing it
        for(int i=0; i<k; ++i) neighborhoods.add(null); // Add placeholders
        computeNeighborhoods(dim, centers, neighborhoods, clustersPerNeighborhood);

        // --- Initialize Buffers and State ---
        int iterationsRun = 0;
        boolean converged = false;
        long[] q = new long[k];            // Buffer for counts
        float[] nextCenters = new float[centers.length]; // Buffer for sums

        // --- Iteration Loop ---
        for (iterationsRun = 0; iterationsRun < maxIterations; ++iterationsRun) {
            // Use the neighborhood-aware stepLloyd
            boolean changed = stepLloyd(dim, dataset, neighborhoods,
                                        centers, nextCenters, q, assignments);
            if (!changed) {
                converged = true;
                break; // Exit loop if no assignments changed
            }
            // Optional: Recompute neighborhoods periodically or if centers moved significantly?
            // The C++ code doesn't, so we won't either for direct translation.
            // Recomputing neighborhoods every iteration would be computationally expensive.
        }

        if (!converged) {
            System.err.println("Warning: k-means local did not converge after " + maxIterations + " iterations.");
        }

        // Create the result object - constructor takes ownership of the modified copies
        return new KMeansResult(k, centers, assignments, iterationsRun, converged);
    }
}