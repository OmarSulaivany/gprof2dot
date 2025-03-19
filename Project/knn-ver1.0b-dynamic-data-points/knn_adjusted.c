#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include "knn.h"

/*
- Function declarations tell the compiler what to expect before it sees the function definition.
- This ensures that when get_3_NN() calls initialize_3_nearest(), the compiler already knows its correct type.*/
void initialize_3_nearest(BestPoint *best_points);
void updating_3_nearest(BestPoint *best_points, DATA_TYPE distance, CLASS_ID_TYPE classification_id);


#if SPECIALIZED == 1 && K == 3


/**
 * First Function called.
 * @brief This function is called first when we want to classify a given point using the K=3 Nearest Neighbors algorithm (KNN).
🔹 It initializes best_points (an array to store the 3 closest neighbors).
🔹 It then calls get_3_NN() to find the 3 nearest points.
🔹 Once the nearest neighbors are found, it calls plurality_voting_3() to determine the final class.
 * 
 * @param new_point The point to classify.
 * @param known_points Array of training points.
 * @param num_points Number of training points.
 * @param num_features Number of features per point.
 * @return The predicted class ID.
 */

 CLASS_ID_TYPE knn_classifyinstance_3(Point *new_point, Point *known_points, int num_points, int num_features) {
    BestPoint best_points[3]; // Step1: Array storing the 3 nearest points

    get_3_NN(new_point, known_points, num_points, best_points, num_features); // Step 2: Find the 3 nearest neighbors

	// Step 3: Return the class with the highest frequency
    return plurality_voting_3(best_points); 
}

/** Second function called.
 * 
 * @brief Finds the 3 nearest neighbors of a given point.
 * This function computes the **distance from the new point to every known point**
 * and updates the list of the 3 nearest neighbors dynamically.
🔹 Calls initialize_3_nearest() to set initial best values to "infinity" so that any real distance will be smaller.
🔹 Loops through every known training point (num_points times).
🔹 and, Calculates the Euclidean distance between new_point and each known_points[i].
🔹 Calls updating_3_nearest() to check if this new point should replace one of the 3 nearest neighbors.
🔹 The 3 nearest neighbors are updated dynamically.
 * 
 * @param new_point The new data point being classified.
 * @param known_points Array of training points.
 * @param num_points Total number of known points.
 * @param best_points Output array storing the 3 nearest points.
 * @param num_features Number of features per point.
 */
void get_3_NN(Point * __restrict new_point, Point * __restrict known_points, int num_points,
	BestPoint * __restrict best_points, int num_features) {

	initialize_3_nearest(best_points); //Initialize with max values.

	// This loop is to iterate through every training point (known_points[i]).
	for (int i = 0; i < num_points; i++) { // Compare with all training points
		/* Initialize Distance for Each Training Point
		This is the starting value before we sum the squared differences. in other words,
		before calculating the Euclidean distance, the variable distance is set to 0.0:*/
		DATA_TYPE distance = 0.0; 

		/* this loop is to go over all 90 features (num_features). And,
		   Compute the difference between the test sample (new_point) and the training sample (known_points[i]) for feature j.*/
		for (int j = 0; j < num_features; j++) 
			{ // Compute the Euclidean Distance . 
			 //The formula for Euclidean distance in two dimensions is D = sqrt {(x_2 - x_1)^2 + (y_2 - y_1)^2}
			DATA_TYPE diff = new_point->features[j] - known_points[i].features[j];
			distance += diff * diff;
			}
		distance = sqrt(distance);

		/* This function is responsible for maintaining the top 3 nearest neighbors as the algorithm 
		iterates through all training points. It ensures that only the 3 closest training points are stored, replacing the farthest 
		one if a closer neighbor is found. 
		best_points:	The array storing the top 3 closest neighbors found so far.
		distance:	The computed Euclidean distance between new_point and known_points[i].
		known_points[i].classification_id:	The class label of the training point known_points[i].*/
		updating_3_nearest(best_points, distance, known_points[i].classification_id); // Step 5: Update the best 3 points
	}
}

/** Third function called.
 * @brief Initializes the 3 nearest neighbors with max distance.
 * This function ensures that any real distance will be smaller and thus be stored.
 * 
 * @param best_points Array storing the top 3 nearest neighbors.
 */
void initialize_3_nearest(BestPoint *best_points) {
    for(int i = 0; i < 3; i++) {
        best_points[i].distance = MAX_FP_VAL; // Set initial distance to infinity
        best_points[i].classification_id = -1;  // Mark as "unknown"
    }
}

/** Fourth function called.
 * 
 * @brief Updates the list of the 3 nearest neighbors if a closer point is found.
 * This function finds the **current farthest** point among the 3 stored points. 
 * If a new point is closer, it **replaces the farthest one**.
 * 
 * @param best_points Array storing the top 3 nearest neighbors.
 * @param distance Distance of the new point being checked.
 * @param classification_id Class ID of the new point.
 */
void updating_3_nearest(BestPoint *best_points, DATA_TYPE distance, CLASS_ID_TYPE classification_id) {

	//Assume the first stored neighbor (best_points[0]) is the farthest.
    //Store its distance and its index.
    DATA_TYPE max_distance = best_points[0].distance;
    int max_index = 0;

	// Find the farthest neighbor among the 3 stored points.
    for(int i = 1; i < 3; i++) {
        if(distance < best_points[i].distance) {
            max_distance = best_points[i].distance;
            max_index = i;
        }
    }
	// If the new point is closer than the farthest neighbor, replace it.
	// This is done by updating the distance and classification ID of the farthest neighbor.
    if (distance < max_distance) {
        best_points[max_index].distance = distance;
        best_points[max_index].classification_id = classification_id;
    }
}


/** Fifth function called.
 * @brief This function is called after the 3 nearest neighbors are found.
 * It uses majority voting to determine the class of the new point.
 * Since we are using K=3, this function picks the **most frequent class** among the 3 nearest neighbors.
 * If two or more of them share the same class, that class is chosen.
 *
 * @param best_points Array containing the 3 nearest neighbors.
 * @return The most common class ID.
 */
CLASS_ID_TYPE plurality_voting_3(BestPoint *best_points) {

    CLASS_ID_TYPE ids0 = best_points[0].classification_id;
    CLASS_ID_TYPE ids1 = best_points[1].classification_id;
    CLASS_ID_TYPE ids2 = best_points[2].classification_id;

    CLASS_ID_TYPE classification_id = ids0; // Assume first class is the majority

    //If two of the three have the same class, return that class.
    if(ids0 == ids2)
        classification_id = ids0;
    else if(ids1 == ids2)
        classification_id = ids1;
    
    return classification_id;
}

// /** Helper Function
//  * @brief Copies the 3 nearest points from `dist_points` to `best_points`.
//  * 
//  * This function moves the top 3 nearest neighbors into the `best_points` array.
//  * 
//  * @param dist_points Source array (contains distances and class IDs).
//  * @param best_points Destination array (stores the top 3 nearest points).
//  */
// void copy_3_nearest(BestPoint *dist_points, BestPoint *best_points) {
//     best_points[0].classification_id = dist_points[0].classification_id;
//     best_points[0].distance = dist_points[0].distance;
    
//     best_points[1].classification_id = dist_points[1].classification_id;
//     best_points[1].distance = dist_points[1].distance;
    
//     best_points[2].classification_id = dist_points[2].classification_id;
//     best_points[2].distance = dist_points[2].distance;
// }


#endif
