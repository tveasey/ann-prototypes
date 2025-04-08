// Port courtesy of Gemini 2.5

package fast_k_means;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.lang.Math; // For Math.sqrt

// Utility class for Pair-like functionality (can be replaced with a Record in Java 16+)
final class FloatPair {
    public final float first;
    public final float second;

    public FloatPair(float first, float second) {
        this.first = first;
        this.second = second;
    }
}

public final class KMeansUtils {

    // --- Type Aliases (Conceptual - implemented via parameter types) ---
    // Point/ConstPoint: Represented by (float[] array, int offset)
    // Dataset: float[]
    // Centers: float[]

    /**
     * Calculates the squared Euclidean distance between two points.
     * Assumes points referenced by (array, offset) have the same dimension.
     *
     * @param dim       The dimension of the points.
     * @param p1Array   The array containing the first point.
     * @param p1Offset  The starting offset of the first point in p1Array.
     * @param p2Array   The array containing the second point.
     * @param p2Offset  The starting offset of the second point in p2Array.
     * @return The squared Euclidean distance.
     */
    public static float distanceSq(int dim,
                                   float[] p1Array, int p1Offset,
                                   float[] p2Array, int p2Offset) {
        float dsq = 0.0f;
        // This loop is a good candidate for JIT auto-vectorization
        for (int i = 0; i < dim; ++i) {
            float diff = p1Array[p1Offset + i] - p2Array[p2Offset + i];
            dsq += diff * diff;
        }
        return dsq;
    }

    /**
     * Calculates the centroid of a dataset.
     *
     * @param dim           The dimension of the points.
     * @param dataset       The dataset array (all points concatenated).
     * @param centroidOut   The pre-allocated array to store the calculated centroid (size must be dim).
     */
    public static void centroid(int dim, float[] dataset, float[] centroidOut) {
        if (dataset == null || dataset.length == 0) {
            Arrays.fill(centroidOut, 0, dim, 0.0f);
            return;
        }
        if (centroidOut.length < dim) {
            throw new IllegalArgumentException("centroidOut array must have size at least dim");
        }

        Arrays.fill(centroidOut, 0, dim, 0.0f);
        long numPoints = dataset.length / dim; // Using long for safety if dataset is huge

        // Sum coordinates
        for (int offset = 0; offset < dataset.length; offset += dim) {
            // Inner loop candidate for JIT auto-vectorization
            for (int d = 0; d < dim; ++d) {
                centroidOut[d] += dataset[offset + d];
            }
        }

        // Divide by number of points
        if (numPoints > 0) {
            // Loop candidate for JIT auto-vectorization
            float numPointsF = (float) numPoints; // Cast once
            for (int d = 0; d < dim; ++d) {
                centroidOut[d] /= numPointsF; // Use float division
            }
        }
        // If numPoints is 0, centroidOut remains all zeros, which is correct.
    }

    // This class encapsulates the result of the k-means clustering algorithm.
    public static final class KMeansResult {
        private final long numClusters;
        // Store centers contiguously: [c1_x, c1_y, ..., c2_x, c2_y, ...]
        private final float[] finalCenters;
        // Stores the offset in finalCenters for the assigned cluster of each point
        private long[] assignments; // Not final as assignRemainingPoints modifies it
        private final long iterationsRun;
        private final boolean converged;

        public KMeansResult(long numClusters,
                            float[] centers, // Takes ownership/copy if needed outside
                            long[] assignments, // Takes ownership/copy if needed outside
                            long iterationsRun,
                            boolean converged) {
            this.numClusters = numClusters;
            // Consider Arrays.copyOf if the caller needs to keep the original arrays unmodified
            this.finalCenters = centers;
            this.assignments = assignments;
            this.iterationsRun = iterationsRun;
            this.converged = converged;

             // Basic validation
            if (centers != null && numClusters > 0 && centers.length % numClusters != 0) {
                 throw new IllegalArgumentException("Center array length must be a multiple of numClusters");
            }
            if (centers != null && numClusters == 0 && centers.length != 0) {
                throw new IllegalArgumentException("numClusters is 0 but centers array is not empty");
            }
            if (centers == null && numClusters != 0) {
                throw new IllegalArgumentException("centers array is null but numClusters is not 0");
            }
        }

        public long getNumClusters() { return numClusters; }
        public float[] getFinalCenters() { return finalCenters; } // Returns direct ref for performance
        public long[] getAssignments() { return assignments; }   // Returns direct ref for performance
        public long getIterationsRun() { return iterationsRun; }
        public boolean hasConverged() { return converged; }

        private int getDimension() {
             if (numClusters == 0 || finalCenters == null || finalCenters.length == 0) return 0;
             return finalCenters.length / (int)numClusters; // Safe cast due to constructor check
        }

        public long[] clusterSizes() {
            if (numClusters == 0) {
                return new long[0];
            }
            long[] sizes = new long[(int)numClusters]; // Safe cast
            int dim = getDimension();
            if (dim == 0) { // Handle case of 0 dimension (though unlikely)
                // If assignments exist, assign all to the first 'cluster' (index 0)
                if (assignments != null) {
                    sizes[0] = assignments.length;
                }
                return sizes;
            }
            if (assignments == null) return sizes; // No assignments yet

            for (long assignmentOffset : assignments) {
                // assignmentOffset is the index in finalCenters, divide by dim for cluster index
                int clusterIndex = (int)(assignmentOffset / dim);
                if (clusterIndex >= 0 && clusterIndex < sizes.length) {
                    sizes[clusterIndex]++;
                } else {
                    // Should not happen with valid assignments, but good to check
                    System.err.println("Warning: Invalid assignment offset encountered: " + assignmentOffset);
                }
            }
            return sizes;
        }

        /**
         * Assigns points starting from a given index and updates centroids.
         * Modifies the internal state (assignments and finalCenters).
         * Assumes the initial part of assignments array (before beginUnassigned / dim) is correct.
         */
        public void assignRemainingPoints(int dim,
                                          long beginUnassignedDataIndex, // Index in the dataset float array
                                          float[] dataset) { // The full dataset

            if (finalCenters == null || numClusters == 0 || dim == 0) return; // Nothing to assign to
            if (dataset == null || beginUnassignedDataIndex >= dataset.length) {
                return; // No points to assign or index out of bounds
            }

            long numExistingAssignments = beginUnassignedDataIndex / dim;
            long totalPoints = dataset.length / dim;

            // 1. Ensure assignments array is large enough
            if (assignments == null || assignments.length < totalPoints) {
                long[] oldAssignments = assignments;
                assignments = new long[(int)totalPoints]; // Need total size
                if(oldAssignments != null) {
                    System.arraycopy(oldAssignments, 0, assignments, 0, oldAssignments.length);
                }
            }

            // 2. Prepare for incremental centroid update
            // We need mutable centers scaled by current counts
            float[] scaledCenters = Arrays.copyOf(finalCenters, finalCenters.length);
            long[] currentSizes = clusterSizes(); // Get sizes based on existing assignments

            for (int c = 0; c < numClusters; ++c) {
                long size = currentSizes[c];
                int centerOffset = c * dim;
                // Loop candidate for JIT auto-vectorization
                for (int d = 0; d < dim; ++d) {
                    scaledCenters[centerOffset + d] *= size; // Scale by current count
                }
            }


            // 3. Assign remaining points and update scaled centers/sizes incrementally
            for (long i = numExistingAssignments, dataOffset = beginUnassignedDataIndex;
                 i < totalPoints;
                 ++i, dataOffset += dim)
            {
                long bestCenterOffset = 0; // Store offset directly
                float minDsq = Float.POSITIVE_INFINITY;

                // Find nearest center for point i (at dataOffset)
                for (int c = 0, centerOffset = 0; c < numClusters; ++c, centerOffset += dim) {
                    float dsq = distanceSq(dim, dataset, (int)dataOffset, finalCenters, centerOffset);
                    if (dsq < minDsq) {
                        minDsq = dsq;
                        bestCenterOffset = centerOffset;
                    }
                }

                // Assign point i
                assignments[(int)i] = bestCenterOffset;
                int bestClusterIndex = (int)(bestCenterOffset / dim);

                // Incrementally update the winning scaled center and its size count
                // Loop candidate for JIT auto-vectorization
                for (int d = 0; d < dim; ++d) {
                    scaledCenters[(int)bestCenterOffset + d] += dataset[(int)dataOffset + d];
                }
                currentSizes[bestClusterIndex]++;
            }


            // 4. Recalculate final centers from the updated scaled centers and sizes
            // Loop candidate for JIT auto-vectorization (outer)
            for (int c = 0, centerOffset = 0; c < numClusters; ++c, centerOffset += dim) {
                long size = currentSizes[c];
                if (size > 0) {
                    float sizeF = (float) size; // Cast once
                    // Loop candidate for JIT auto-vectorization (inner)
                    for (int d = 0; d < dim; ++d) {
                        finalCenters[centerOffset + d] = scaledCenters[centerOffset + d] / sizeF;
                    }
                } else {
                    // Handle empty cluster - behavior might depend on specific k-means variant
                    // Option 1: Keep its last known position (already in finalCenters)
                    // Option 2: Re-initialize (e.g., to 0 or a random point) - C++ keeps old position implicitly
                    // We'll keep the old position to match C++ behavior here.
                    // If it was newly created and never assigned, it might be 0 or uninitialized.
                    // The original C++ code doesn't explicitly handle re-initializing empty clusters here.
                }
            }
        }


        /** Computes the average squared distance of points to their assigned cluster center. */
        public float computeDispersion(int dim, float[] dataset) {
            if (assignments == null || assignments.length == 0 || dim == 0 || dataset == null || finalCenters == null) {
                return 0.0f;
            }

            double totalDispersion = 0.0; // Use double for summation precision
            long numAssignments = assignments.length;

            for (int i = 0, dataOffset = 0; i < numAssignments; ++i, dataOffset += dim) {
                long centerOffset = assignments[i]; // Already the correct offset
                totalDispersion += distanceSq(dim, dataset, dataOffset, finalCenters, (int)centerOffset);
            }

            return (float)(totalDispersion / numAssignments);
        }

        /** Calculates the mean and standard deviation of cluster sizes. */
        public FloatPair clusterSizeMoments() {
            long[] sizes = clusterSizes();
            if (sizes.length == 0) {
                return new FloatPair(0.0f, 0.0f);
            }

            double sum = 0.0;
            for (long size : sizes) {
                sum += size;
            }
            double mean = sum / sizes.length;

            double varianceSum = 0.0;
            for (long size : sizes) {
                double diff = (double)size - mean;
                varianceSum += diff * diff;
            }
            double variance = varianceSum / sizes.length;
            double stddev = Math.sqrt(variance);

            return new FloatPair((float)mean, (float)stddev);
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            int dim = getDimension();
            sb.append("\nConverged: ").append(converged ? "Yes" : "No");
            sb.append("\nIterations Run: ").append(iterationsRun);
            sb.append("\nNumber of clusters: ").append(numClusters);
            if (numClusters > 0) {
                sb.append("\nDimension: ").append(dim);
                FloatPair moments = clusterSizeMoments();
                sb.append("\nCluster size moments: mean = ").append(moments.first)
                  .append(" sd = ").append(moments.second);
            } else {
                sb.append("\nCluster size moments: mean = 0.0 sd = 0.0");
            }
            // Optionally add printing centers/assignments if needed (can be very long)
            return sb.toString();
        }
    }

    // This class encapsulates the result of the hierarchical k-means clustering algorithm.
    public static final class HierarchicalKMeansResult {
        // Each element is a center (float[dim])
        private final List<float[]> finalCenters;
        // Each element is the list of *original dataset indices* belonging to that cluster
        private final List<long[]> assignments; // List of arrays of original point indices

        HierarchicalKMeansResult() {
            this.finalCenters = new ArrayList<>();
            this.assignments = new ArrayList<>();
        }

        // Construct from a flat KMeansResult
        public HierarchicalKMeansResult(KMeansResult result) {
            long k = result.getNumClusters();
            if (k == 0) {
                this.finalCenters = new ArrayList<>(0);
                this.assignments = new ArrayList<>(0);
                return;
            }

            int dim = result.getDimension();
             if (dim == 0 && k > 0) {
                // Handle zero dimension case if needed, though less common
                this.finalCenters = new ArrayList<>((int)k);
                this.assignments = new ArrayList<>((int)k);
                for(int i = 0; i < k; ++i) {
                    this.finalCenters.add(new float[0]); // Add empty center
                }
                // Distribute all assignments to the first cluster if assignments exist
                if (result.getAssignments() != null) {
                    long[] allIndices = new long[result.getAssignments().length];
                    for (int i = 0; i < allIndices.length; ++i) allIndices[i] = i;
                    this.assignments.add(allIndices);
                    // Add empty assignment lists for other clusters
                    for (int i = 1; i < k; ++i) this.assignments.add(new long[0]);
                } else {
                    for (int i = 0; i < k; ++i) this.assignments.add(new long[0]);
                }
                return;
            }

            this.finalCenters = new ArrayList<>((int)k);
            this.assignments = new ArrayList<>((int)k);
            List<List<Long>> tempAssignments = new ArrayList<>((int)k); // Use List<Long> temporarily

            for (int i = 0; i < k; ++i) {
                assignments.add(null); // Placeholder, will be replaced
                tempAssignments.add(new ArrayList<>()); // Initialize temporary lists

                // Copy center for cluster i
                float[] center = new float[dim];
                int sourceOffset = i * dim;
                System.arraycopy(result.getFinalCenters(), sourceOffset, center, 0, dim);
                this.finalCenters.add(center);
            }

            // Group original point indices by cluster
            long[] flatAssignments = result.getAssignments();
            if (flatAssignments != null) {
                for (int i = 0; i < flatAssignments.length; ++i) {
                    long centerOffset = flatAssignments[i];
                    int clusterIndex = (int)(centerOffset / dim);
                    if (clusterIndex >= 0 && clusterIndex < k) {
                        tempAssignments.get(clusterIndex).add((long)i); // Add original index i
                    } else {
                        System.err.println("Warning: Invalid assignment offset during hierarchical construction: " + centerOffset);
                    }
                }
            }


            // Convert temporary List<Long> to long[] for efficiency
            for (int i = 0; i < k; ++i) {
                List<Long> clusterIndices = tempAssignments.get(i);
                long[] indicesArray = new long[clusterIndices.size()];
                for (int j = 0; j < clusterIndices.size(); ++j) {
                    indicesArray[j] = clusterIndices.get(j); // Auto-unboxing
                }
                this.assignments.set(i, indicesArray); // Replace placeholder
            }
        }

        // Private constructor for internal use (e.g., recursive splitting)
        private HierarchicalKMeansResult(List<float[]> centers, List<long[]> assignments) {
            this.finalCenters = centers;
            this.assignments = assignments;
        }

        public int getNumClusters() { return finalCenters.size(); }
        // Returns direct refs for performance - be careful if modifying externally
        public List<float[]> getFinalCenters() { return finalCenters; }
        public List<long[]> getAssignments() { return assignments; }

        public long[] clusterSizes() {
            long[] sizes = new long[assignments.size()];
            for (int i = 0; i < assignments.size(); ++i) {
                sizes[i] = (assignments.get(i) != null) ? assignments.get(i).length : 0;
            }
            return sizes;
        }

        /** Flattens the centers into a single float array. */
        public float[] finalCentersFlat() {
            if (finalCenters.isEmpty()) {
                return new float[0];
            }
            int dim = finalCenters.get(0).length;
            int numClusters = finalCenters.size();
            float[] flatCenters = new float[numClusters * dim];
            for (int i = 0, offset = 0; i < numClusters; ++i, offset += dim) {
                System.arraycopy(finalCenters.get(i), 0, flatCenters, offset, dim);
            }
            return flatCenters;
        }

        /**
         * Flattens the assignments into a single array where assignments[original_point_index]
         * gives the starting offset of the assigned center in the *flattened* centers array.
         */
        public long[] assignmentsFlat() {
            if (assignments.isEmpty()) {
                return new long[0];
            }

            // Determine total number of points
            long n = 0;
            for (long[] clusterAssigns : assignments) {
                 if (clusterAssigns != null) n += clusterAssigns.length;
            }
            if (n == 0) {
                return new long[0];
            }

            // Determine dimension (handle empty centers case)
            int dim = finalCenters.isEmpty() ? 0 : finalCenters.get(0).length;

            // Create the flat array (needs size of the original dataset)
            // We infer the size from the max index + 1, assuming indices are contiguous from 0
            long maxIndex = -1;
            for (long[] clusterAssigns : assignments) {
                if (clusterAssigns != null) {
                    for (long pointIndex : clusterAssigns) {
                        if (pointIndex > maxIndex) maxIndex = pointIndex;
                    }
                }
            }
            if (maxIndex < 0) return new long[0]; // No points assigned

            long[] flatAssignments = new long[(int)(maxIndex + 1)];
             // Initialize with a sentinel if needed, or assume all points are covered
             // Arrays.fill(flatAssignments, -1L); // Example initialization

            for (int i = 0, centerOffset = 0; i < assignments.size(); ++i, centerOffset += dim) {
                long[] clusterAssigns = assignments.get(i);
                if (clusterAssigns != null) {
                    for (long pointIndex : clusterAssigns) {
                        // Ensure pointIndex is within bounds before assigning
                        if (pointIndex >= 0 && pointIndex < flatAssignments.length) {
                            flatAssignments[(int)pointIndex] = centerOffset;
                        } else {
                            System.err.println("Warning: Point index " + pointIndex + " out of bounds for flat assignments array (size " + flatAssignments.length + ")");
                        }
                    }
                }
            }
            return flatAssignments;
        }

        /**
         * Copies the data points belonging to a specific cluster into a new array.
         *
         * @param dim       Dimension of points.
         * @param cluster   The index of the cluster to copy.
         * @param dataset   The original dataset array.
         * @return A new float[] containing only the points from the specified cluster,
         * or null if the cluster index is invalid or the cluster is empty.
         */
        public float[] copyClusterPoints(int dim, int cluster, float[] dataset) {
            if (cluster < 0 || cluster >= assignments.size() || dim <= 0 || dataset == null) {
                return null;
            }

            long[] indices = assignments.get(cluster);
            if (indices == null || indices.length == 0) {
                return new float[0]; // Empty cluster
            }

            long n = indices.length;
            float[] copy = new float[(int)(n * dim)]; // Create the destination array
            int copyOffset = 0;

            for (long originalIndex : indices) {
                int dataOffset = (int)(originalIndex * dim);
                // Basic bounds check for safety
                if (dataOffset >= 0 && dataOffset + dim <= dataset.length) {
                    System.arraycopy(dataset, dataOffset, copy, copyOffset, dim);
                    copyOffset += dim;
                } else {
                    System.err.println("Warning: Original index " + originalIndex + " leads to out-of-bounds access in dataset during copy.");
                    // Skip this point or handle error appropriately
                }
            }
             // If some points were skipped due to errors, the copy array might be too large.
             // We could trim it, but for performance, returning the potentially oversized array is often done.
             // Alternatively, check bounds *before* creating the copy array to get the exact size.
            return copy;
        }

        /**
         * Updates this result by replacing a cluster with the results of a recursive split.
         * Modifies the internal state (finalCenters and assignments).
         *
         * @param dim            Dimension of points.
         * @param clusterToSplit The index of the cluster in *this* result that was split.
         * @param splitResult    The HierarchicalKMeansResult obtained by splitting clusterToSplit.
         */
        public void updateAssignmentsWithRecursiveSplit(int dim, // Unused in current logic, but kept for signature match
                                                         int clusterToSplit,
                                                         HierarchicalKMeansResult splitResult) {

            if (clusterToSplit < 0 || clusterToSplit >= finalCenters.size() || splitResult == null || splitResult.getNumClusters() == 0) {
                // Invalid input, do nothing or throw exception
                System.err.println("Warning: Invalid arguments for updateAssignmentsWithRecursiveSplit.");
                return;
            }

            // Get the original point indices that belonged to the cluster we just split
            long[] originalIndices = assignments.get(clusterToSplit);
            if (originalIndices == null) {
                System.err.println("Warning: Cluster to split has null assignments.");
                return; // Cannot proceed
            }

            // --- Update Centers ---
            // Replace the center of the split cluster with the first center from the split result
            finalCenters.set(clusterToSplit, splitResult.finalCenters.get(0));
            // Add the remaining centers from the split result to the end
            for (int i = 1; i < splitResult.finalCenters.size(); ++i) {
                finalCenters.add(splitResult.finalCenters.get(i));
            }

            // --- Update Assignments ---
            List<long[]> newAssignments = new ArrayList<>(splitResult.assignments.size());

             // Map the *local* indices (0 to N-1 within the split subset) from splitResult
             // back to the *original* dataset indices stored in originalIndices.
            for (long[] splitClusterAssignments : splitResult.assignments) {
                if (splitClusterAssignments == null) {
                    newAssignments.add(new long[0]); // Add empty assignment if split resulted in one
                    continue;
                }
                long[] remappedAssignments = new long[splitClusterAssignments.length];
                for (int j = 0; j < splitClusterAssignments.length; ++j) {
                    long localIndex = splitClusterAssignments[j];
                    // Check bounds for safety
                    if (localIndex >= 0 && localIndex < originalIndices.length) {
                        remappedAssignments[j] = originalIndices[(int)localIndex]; // Map local back to original
                    } else {
                        System.err.println("Warning: Local index " + localIndex + " out of bounds for original indices array (size " + originalIndices.length + ") during split update.");
                        // Handle error: maybe assign a sentinel, skip, or throw
                        // For now, potentially results in incorrect mapping if error occurs.
                        remappedAssignments[j] = -1; // Example: mark as invalid
                    }
                }
                newAssignments.add(remappedAssignments);
            }

            // Replace the assignments of the split cluster with the first new assignment list
            assignments.set(clusterToSplit, newAssignments.get(0));
            // Add the remaining new assignment lists to the end
            for (int i = 1; i < newAssignments.size(); ++i) {
                assignments.add(newAssignments.get(i));
            }
        }


        /** Computes the average squared distance of points to their assigned cluster center. */
        public float computeDispersion(int dim, float[] dataset) {
             if (assignments.isEmpty() || dim <= 0 || dataset == null) {
                 return 0.0f;
             }

            double totalDispersion = 0.0;
            long totalPoints = 0;

            for (int i = 0; i < assignments.size(); ++i) {
                long[] clusterIndices = assignments.get(i);
                float[] center = finalCenters.get(i);
                if (clusterIndices != null && center != null) {
                    totalPoints += clusterIndices.length;
                    for (long pointIndex : clusterIndices) {
                        int dataOffset = (int)(pointIndex * dim);
                        // Basic bounds check
                        if (dataOffset >= 0 && dataOffset + dim <= dataset.length && center.length == dim) {
                            totalDispersion += distanceSq(dim, dataset, dataOffset, center, 0);
                        } else {
                            System.err.println("Warning: Invalid index or dimension mismatch during dispersion calculation.");
                        }
                    }
                }
            }

            return (totalPoints == 0) ? 0.0f : (float)(totalDispersion / totalPoints);
        }

        /** Calculates the mean and standard deviation of cluster sizes. */
        public FloatPair clusterSizeMoments() {
            long[] sizes = clusterSizes();
            if (sizes.length == 0) {
                return new FloatPair(0.0f, 0.0f);
            }

            double sum = 0.0;
            for (long size : sizes) {
                sum += size;
            }
            double mean = sum / sizes.length;

            double varianceSum = 0.0;
            for (long size : sizes) {
                double diff = (double)size - mean;
                varianceSum += diff * diff;
            }
            double variance = varianceSum / sizes.length;
            double stddev = Math.sqrt(variance);

            return new FloatPair((float)mean, (float)stddev);
        }


        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            int numClusters = getNumClusters();
            sb.append("\nHierarchical Result:");
            sb.append("\nNumber of clusters: ").append(numClusters);
            if (numClusters > 0) {
                int dim = finalCenters.get(0).length;
                sb.append("\nDimension: ").append(dim);
                FloatPair moments = clusterSizeMoments();
                sb.append("\nCluster size moments: mean = ").append(moments.first)
                  .append(" sd = ").append(moments.second);
            } else {
                sb.append("\nCluster size moments: mean = 0.0 sd = 0.0");
            }
            return sb.toString();
        }
    }
}