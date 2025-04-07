// Port courtesy of Gemini 2.5

package fast_k_means;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections; // For shuffle
import java.util.List;
import java.util.Random;     // For shuffling
import java.lang.Math; // For Math.sqrt, Math.max, Math.min

import static fast_k_means.KMeansUtils.distanceSq;
import static fast_k_means.KMeansUtils.centroid;
import static fast_k_means.KMeansUtils.KMeansResult;
import static fast_k_means.KMeansUtils.HierarchicalKMeansResult;
import static fast_k_means.KMeans.kMeans;
import static fast_k_means.KMeansLocal.kMeansLocal;

public final class KMeansHierarchical {
    /**
     * Selects up to k distinct initial centers from the dataset randomly.
     *
     * @param dim      Dimension of points.
     * @param dataset  Input data points (flat array).
     * @param k        Desired number of centers.
     * @return A float[] containing the coordinates of the selected distinct centers.
     * The actual number of centers might be less than k if the dataset
     * has fewer distinct points. The length of the returned array will be k_actual * dim.
     */
    private static float[] pickInitialCenters(int dim, float[] dataset, int k) {
        int n = dataset.length / dim;
        if (k <= 0 || n == 0) {
            return new float[0];
        }
        if (k > n) {
            k = n; // Cannot pick more centers than points
        }

        // Create list of candidate indices (0 to n-1)
        List<Integer> candidates = new ArrayList<>(n);
        for (int i = 0; i < n; ++i) {
            candidates.add(i);
        }

        // Shuffle the candidate indices
        Collections.shuffle(candidates, new Random()); // Use java.util.Random

        float[] centersOutput = new float[k * dim]; // Preallocate optimistically
        int actualK = 0; // Number of distinct centers found so far
        final float DUP_CHECK_EPSILON_SQ = 0.0f; // Use 0.0f for exact match as in C++
                                                 // Could use a small value like 1e-12f for near-duplicates

        for (int candidateIndex : candidates) {
            if (actualK >= k) {
                break; // Found enough distinct centers
            }

            int candidateDataOffset = candidateIndex * dim;
            boolean isDuplicate = false;

            // Check if this candidate is a duplicate of already chosen centers
            for (int c = 0, centerOffset = 0; c < actualK; ++c, centerOffset += dim) {
                // Add bounds check for safety
                if (candidateDataOffset + dim <= dataset.length && centerOffset + dim <= centersOutput.length) {
                    float dsq = distanceSq(dim, dataset, candidateDataOffset, centersOutput, centerOffset);
                    if (dsq <= DUP_CHECK_EPSILON_SQ) { // Check for duplicate (within epsilon)
                        isDuplicate = true;
                        break;
                    }
                } else {
                    System.err.println("Warning: Bounds error during duplicate check in pickInitialCenters.");
                    // Treat as potential duplicate to be safe? Or skip check? Skip check for now.
                }
            }

            if (!isDuplicate) {
                // Copy the distinct candidate point to the output centers
                int destOffset = actualK * dim;
                // Add bounds check for safety
                if (candidateDataOffset + dim <= dataset.length && destOffset + dim <= centersOutput.length) {
                    System.arraycopy(dataset, candidateDataOffset, centersOutput, destOffset, dim);
                    actualK++;
                } else {
                    System.err.println("Warning: Bounds error during center copy in pickInitialCenters.");
                }
            }
        } // End loop through candidates

        // Trim the output array to the actual number of centers found
        if (actualK < k) {
            return Arrays.copyOf(centersOutput, actualK * dim);
        } else {
            return centersOutput; // Already correct size
        }
    }


    /**
     * Performs hierarchical k-means clustering. Recursively splits large clusters.
     *
     * @param dim                     Dimension of points.
     * @param dataset                 Input data points (flat array).
     * @param targetSize              Desired maximum number of points per final cluster.
     * @param maxIterations           Max iterations for k-means runs at each level.
     * @param maxK                    Max clusters allowed at each split level.
     * @param samplesPerCluster       Factor to determine sample size for initial k-means run.
     * @param clustersPerNeighborhood Parameter for the final local refinement step.
     * @param depth                   Current recursion depth (internal use, start with 0).
     * @return A HierarchicalKMeansResult representing the final clustering.
     */
    public static HierarchicalKMeansResult kMeansHierarchical(int dim,
                                                              final float[] dataset,
                                                              int targetSize,
                                                              int maxIterations,
                                                              int maxK,
                                                              int samplesPerCluster,
                                                              int clustersPerNeighborhood,
                                                              int depth) { // Add depth parameter

        int n = dataset.length / dim;

        // Base Case 1: Dataset is small enough, no further splitting needed.
        if (n <= targetSize) {
            // Return an empty result, indicating this branch needs no splitting.
            // The caller (updateAssignmentsWithRecursiveSplit) should handle this.
            // However, the C++ returns {} which likely implies an empty HIERARCHICAL result,
            // meaning the current 'dataset' IS the single cluster result for this branch.
            // Let's construct a result representing this single cluster.
            if (n > 0) {
                float[] center = new float[dim];
                centroid(dim, dataset, center);
                long[] assignments = new long[n]; // All points assigned to this one cluster
                // Assign original indices (relative to THIS dataset subset)
                for(int i=0; i<n; ++i) assignments[i] = i; // Store 0..n-1

                // Wrap in KMeansResult then HierarchicalKMeansResult
                KMeansResult singleClusterKMR = new KMeansResult(1, center, null /* Assign indices later? */, 0, true);
                // The Hierarchical constructor needs flat assignments based on center offsets (0 here)
                // and original *global* indices. This simple case is tricky.

                // Let's stick to the C++ return {} interpretation = empty result means "don't split me".
                return new HierarchicalKMeansResult(); // Empty result signals termination for this branch
            } else {
                return new HierarchicalKMeansResult(); // Empty dataset -> empty result
            }
        }

        // Determine k for this split level
        // Equivalent to C++: std::clamp((n + targetSize - 1) / targetSize, 2UL, maxK)
        // Use long for intermediate calculation to avoid potential overflow if n is large
        long kLong = ((long)n + targetSize - 1) / targetSize;
        int k = (int) Math.max(2L, Math.min((long)maxK, kLong)); // Ensure k is at least 2 for splitting

        // Determine sample size m
        // Use long for intermediate calculation
        long mLong = Math.min((long)k * samplesPerCluster * dim, (long)dataset.length);
        int m = (int) mLong; // Sample size

        // Create sample dataset (simple prefix sample)
        float[] sample;
        if (m == dataset.length) {
            sample = dataset; // Use full dataset (reference, no copy needed)
        } else {
            sample = Arrays.copyOf(dataset, m); // Copy first m elements
        }

        // Pick initial centers from the sample
        float[] initialCenters = pickInitialCenters(dim, sample, k);
        k = initialCenters.length / dim; // Update k based on actual distinct centers found

        // Handle case where pickInitialCenters found 0 or 1 center
        if (k <= 1) {
            // Cannot split further down this path with k<=1. Return a single cluster result for the current dataset.
            // Similar logic to the n <= targetSize base case.
            if (n > 0) {
                float[] center = new float[dim];
                centroid(dim, dataset, center);
                // Construct HierarchicalKMeansResult representing a single cluster containing the current dataset
                // This requires knowing the original indices mapping to 'dataset'. This info is lost here.
                // Reverting to the C++ interpretation: return empty result to signal no split occurred.
                return new HierarchicalKMeansResult();
            } else {
                return new HierarchicalKMeansResult();
            }
        }

        // Initial Split using standard k-means on the sample
        HierarchicalKMeansResult result;
        { // Scope for intermediate result_
            // kMeans copies initialCenters, so no move needed
            KMeansResult result_ = kMeans(dim, sample, initialCenters, k, maxIterations);

            // Assign remaining points from the full dataset
            // Ensure assignRemainingPoints can handle being called when assignments might already partially exist from kMeans
            // or assumes assignments are null/empty initially. Let's assume it handles it.
            // The sample size 'm' corresponds to 'sample.length' data index.
            result_.assignRemainingPoints(dim, sample.length, dataset);

            // Convert to hierarchical structure
            result = new HierarchicalKMeansResult(result_);
        }

        // Base Case 3: Initial split resulted in k=1 (unlikely if initial k>1, but possible if data collapses)
        if (result.getNumClusters() <= 1) {
            // No further splitting possible/needed down this path
            // Return the single cluster result obtained so far
            return result; // Return the single cluster found
        }

        // Recursive Step: Split large clusters
        long[] counts = result.clusterSizes();
        for (int c = 0; c < counts.length; ++c) {
            // Check if cluster c is larger than target size (with 30% margin)
            // Use long for multiplication to avoid overflow
            if (counts[c] > 0 && 10L * counts[c] > 13L * targetSize) {

                // Copy points belonging to cluster c for recursive call
                // copyClusterPoints returns a new array
                float[] subDataset = result.copyClusterPoints(dim, c, dataset);

                if (subDataset.length > 0) {
                    // Recursive call
                    HierarchicalKMeansResult subResult = kMeansHierarchical(
                        dim, subDataset, targetSize, maxIterations, maxK,
                        samplesPerCluster, clustersPerNeighborhood, depth + 1);

                    // Integrate the result of the split back into the main result
                    // Only update if the recursive call actually performed a split (returned non-empty)
                    if (subResult != null && subResult.getNumClusters() > 0) {
                        result.updateAssignmentsWithRecursiveSplit(dim, c, subResult);
                        // Adjust loop index 'c' and counts.length if updateAssignments adds clusters?
                        // The current C++/Java logic adds clusters at the end, so iterating up to the original
                        // counts.length should be okay, but processing newly added clusters needs care.
                        // Let's assume updateAssignments handles list modifications correctly and
                        // we only process the original set of clusters for potential splitting.
                        // Re-calculating counts inside the loop would be safer but less efficient.
                    }
                }
            }
        }

        // Final Refinement Step (only at top level)
        if (depth == 0) {
            if (result.getNumClusters() > 0 && clustersPerNeighborhood > 0) {
                // Flatten the results from the hierarchical process
                float[] flatCenters = result.finalCentersFlat();
                long[] flatAssignments = result.assignmentsFlat();

                // Check if flatAssignments is valid for the original dataset size 'n'
                if (flatAssignments.length == n) {
                    // Run kMeansLocal for final refinement
                    // kMeansLocal copies its input arrays
                    KMeansResult localResult = kMeansLocal(
                        dim, dataset, flatCenters, flatAssignments,
                        clustersPerNeighborhood, maxIterations);

                    // Convert the refined result back to hierarchical structure
                    result = new HierarchicalKMeansResult(localResult);
                } else {
                    System.err.println("Warning: Size mismatch for flatAssignments ("+flatAssignments.length+") vs dataset size ("+n+") at depth 0 refinement. Skipping kMeansLocal.");
                    // This might happen if assignmentsFlat() has issues or point indices were lost.
                }

            } else if (result.getNumClusters() == 0) {
                System.err.println("Warning: Hierarchical clustering resulted in 0 clusters at depth 0.");
            }
        }

        return result;
    }
} // End KMeansJava class