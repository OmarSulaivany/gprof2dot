#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include "knn.h"
#include <stdio.h>
// #include <stdlib.h>


// Function prototypes
void initialize_3_nearest(BestPoint *best_points);
void updating_3_nearest(BestPoint *best_points, DATA_TYPE distance, CLASS_ID_TYPE classification_id);
void merge_best_3_points(BestPoint *best_points_thread_0, BestPoint *best_points_thread_1, BestPoint *best_points);

#if SPECIALIZED == 1 && K == 3

/**
 * @brief Classifies a given point using **parallelized KNN (K=3)**.
 *
 * @param new_point The point to classify.
 * @param known_points Array of training points.
 * @param num_points Number of training points.
 * @param num_features Number of features per point.
 * @return The predicted class ID.
 */
CLASS_ID_TYPE knn_classifyinstance_3(Point *new_point, Point *known_points, int num_points, int num_features) {
    BestPoint best_points[3]; // Stores the final 3 nearest neighbors

    get_3_NN(new_point, known_points, num_points, best_points, num_features); // Find nearest neighbors

    return plurality_voting_3(best_points); // Perform voting and return class
}

/**
 * @brief Finds the 3 nearest neighbors using **OpenMP with 2 threads**.
 */
void get_3_NN(Point *__restrict new_point, Point *__restrict known_points, int num_points,
	BestPoint *__restrict best_points, int num_features) {

	int i, j;
	DATA_TYPE distance;
	BestPoint best_points_thread_0[3]; // Thread 0's nearest neighbors
	BestPoint best_points_thread_1[3]; // Thread 1's nearest neighbors

	// Initialize each thread's best points
	initialize_3_nearest(best_points_thread_0);
	initialize_3_nearest(best_points_thread_1);

	int nthreads = 2; // Using 2 threads

	// Parallelized distance computation
	#pragma omp parallel for private(i, j, distance) num_threads(nthreads)
	for (i = 0; i < num_points; i++) {
	distance = 0.0; // Reset distance for each training point

	// Compute the Euclidean distance
	#pragma omp simd reduction(+:distance)
	for (j = 0; j < num_features; j++) {
	DATA_TYPE diff = new_point->features[j] - known_points[i].features[j];
	distance += diff * diff;
	}
	distance = sqrt(distance);

	// Determine which thread is processing this iteration
	int id = omp_get_thread_num();

	// Each thread updates its own best points
	if (id == 0) {
	updating_3_nearest(best_points_thread_0, distance, known_points[i].classification_id);
	} else {
	updating_3_nearest(best_points_thread_1, distance, known_points[i].classification_id);
	}
	}

	// // 🔹 **DEBUG: Print Best Points Before Merging**
	// printf("Thread 0 best points BEFORE merging:\n");
	// for (int i = 0; i < 3; i++) {
	// 	printf("Dist: %f, Class: %d\n", best_points_thread_0[i].distance,
	// 		   best_points_thread_0[i].classification_id);
	// }
	
	// printf("Thread 1 best points BEFORE merging:\n");
	// for (int i = 0; i < 3; i++) {
	// 	printf("Dist: %f, Class: %d\n", best_points_thread_1[i].distance, 
	// 		best_points_thread_1[i].classification_id);
	// }

	//🔹 Merge the best 3 nearest neighbors from both threads
	merge_best_3_points(best_points_thread_0, best_points_thread_1, best_points);

	// // 🔹 **DEBUG: Print Final Best Points After Merging**
	// printf("Final best points AFTER merging:\n");
	// for (int i = 0; i < 3; i++) {
	// 	printf("Dist: %f, Class: %d\n", best_points[i].distance, 
	// 		best_points[i].classification_id);
	// }

}

// Function to compare two BestPoint structures based on distance
int compare_best_points(const void *a, const void *b) {
    /*
	The problem: A void * pointer does not know what type of data it points to!
	Solution: We cast the void * back to a BestPoint *, so we can access its fields (distance, classification_id).
	
	Casting is a way to convert one data type into another. 
	In this case, we are converting a generic pointer (void *) into a specific pointer type (BestPoint *). 

*/
	// Hey, treat 'a' as a pointer to a BestPoint struct.
    BestPoint *pointA = (BestPoint *)a;
    BestPoint *pointB = (BestPoint *)b;

    /*  
        The return statement determines the sorting order:
        
        - If `pointA->distance` is **smaller** than `pointB->distance`, return `-1`
          → This means `pointA` should come **before** `pointB` in the sorted list.

        - If `pointA->distance` is **greater** than `pointB->distance`, return `1`
          → This means `pointA` should come **after** `pointB`.

        - If both distances are **equal**, return `0`, meaning their order remains unchanged.
        
        The logic `(X > Y) - (X < Y)` is a **shortcut** for:
        
        ```c
        if (X < Y) return -1;
        if (X > Y) return 1;
        return 0;
        ```
        
        This avoids using multiple if-else statements.
    */
    return (pointA->distance > pointB->distance) - (pointA->distance < pointB->distance);
}


void merge_best_3_points(BestPoint *best_points_thread_0, BestPoint *best_points_thread_1, BestPoint *best_points) {
    BestPoint merged[6]; // Hold all 6 points from both threads

    // Copy the best points from both threads into the merged array
    for (int i = 0; i < 3; i++) {
        merged[i] = best_points_thread_0[i];
        merged[i + 3] = best_points_thread_1[i];
    }

    // Sort the merged array based on distance (ascending order)
	//this sorting function ensures that the first 3 elements in merged are the smallest distances.
    qsort(merged, 6, sizeof(BestPoint), compare_best_points);

    // Select the top 3 smallest distances
	//simply copy the first 3 elements (smallest distances) to the final output
    for (int i = 0; i < 3; i++) {
        best_points[i] = merged[i];
    }
}

/**
 * @brief Determines the most frequent class among the 3 nearest neighbors.
 */
CLASS_ID_TYPE plurality_voting_3(BestPoint *best_points) {
    CLASS_ID_TYPE ids0 = best_points[0].classification_id;
    CLASS_ID_TYPE ids1 = best_points[1].classification_id;
    CLASS_ID_TYPE ids2 = best_points[2].classification_id;

    CLASS_ID_TYPE classification_id = ids0; // Default to first neighbor

    if (ids0 == ids2)
        classification_id = ids0;
    else if (ids1 == ids2)
        classification_id = ids1;

    return classification_id;
}

/**
 * @brief Updates the 3 nearest neighbors if a closer point is found.
 */
void updating_3_nearest(BestPoint *best_points, DATA_TYPE distance, CLASS_ID_TYPE classification_id) {
    DATA_TYPE max_distance = best_points[0].distance;
    int max_index = 0;

    // Find the farthest neighbor among the 3 stored points.
    for (int i = 1; i < 3; i++) {
        if (distance < best_points[i].distance) {
            max_distance = best_points[i].distance;
            max_index = i;
        }
    }

    // Replace the farthest neighbor if the new point is closer.
    if (distance < max_distance) {
        best_points[max_index].distance = distance;
        best_points[max_index].classification_id = classification_id;
    }
}

/**
 * @brief Initializes the 3 nearest neighbors with max possible distance.
 */
void initialize_3_nearest(BestPoint *best_points) {
    for (int i = 0; i < 3; i++) {
        best_points[i].distance = MAX_FP_VAL; // Set to infinity
        best_points[i].classification_id = -1; // Mark as "unknown"
    }
}


#endif
