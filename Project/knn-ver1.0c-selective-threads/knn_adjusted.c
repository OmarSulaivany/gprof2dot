#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include "knn.h"


CLASS_ID_TYPE plurality_voting_3(BestPoint *best_points) {

	CLASS_ID_TYPE ids0 = best_points[0].classification_id;
    CLASS_ID_TYPE ids1 = best_points[1].classification_id;
    CLASS_ID_TYPE ids2 = best_points[2].classification_id;
	
	CLASS_ID_TYPE classification_id = ids0; // assuming they are sorted then the first one is the class
		
    if(ids0 == ids2)
        classification_id = ids0;
    else if(ids1 == ids2)
        classification_id = ids1;
	
    return classification_id;
}



//__restrict assigning pointers to each value in order to make sure they dont overlap and are eligible for vectorization
void updating_3_nearest(BestPoint *best_points, DATA_TYPE distance, CLASS_ID_TYPE classification_id) {

	DATA_TYPE max_distance = best_points[0].distance;
	int max_index = 0;

	
	for(int i = 1; i < 3; i++) {
		if(distance < best_points[i].distance) {
			max_distance = best_points[i].distance;
			max_index = i;

		}
	}
	if (distance < max_distance)
	{
	best_points[max_index].distance = distance;
	best_points[max_index].classification_id = classification_id;
	}

}

void initialize_3_nearest(BestPoint *best_points) {
	for(int i = 0; i < 3; i++) {
		best_points[i].distance = MAX_FP_VAL;
		best_points[i].classification_id = -1;
	}
}


//__restrict assigning pointers to each value in order to make sure they dont overlap and are eligible for vectorization
void get_3_NN(Point * __restrict new_point, Point * __restrict known_points, int num_points,
	BestPoint * __restrict best_points, int num_features) {

	int i,j;
	DATA_TYPE distance;

	BestPoint best_points_thread_0[3];
	BestPoint best_points_thread_1[3];

	// initialize the best_points array
	initialize_3_nearest(best_points_thread_0);
	initialize_3_nearest(best_points_thread_1);
	// int nthreads = omp_get_max_threads();
	int nthreads = 2;
	// Parallelized distance computation
	//TODO: need to put private for variables that are local.
    #pragma omp parallel for private(i,j,distance) num_threads (nthreads)
    for ( i = 0; i < num_points; i++) {
        distance = (DATA_TYPE) 0.0;

		#if DIST_METHOD == 1 // calculate the Euclidean distance
		//Help to vectorize the loop in omp.
		#pragma omp simd reduction(+:distance)
        for ( j = 0; j < num_features; j++) {
            DATA_TYPE diff = (DATA_TYPE) new_point->features[j] - (DATA_TYPE) known_points[i].features[j];
            distance += diff * diff;
        }
		#if USE_SQRT == 1
			#if MATH_TYPE == 1 && DT == 2 // float
				distance = sqrtf(distance);
			#else // double
				distance = sqrt(distance);
			#endif
		#endif
	
		#elif DIST_METHOD == 2 // calculate the Manhattan distance
		for (int j = 0; j < num_features; j++) {
			#if MATH_TYPE == 1 && DT == 2 // float
            DATA_TYPE absdiff = fabsf((DATA_TYPE) new_point->features[j] - (DATA_TYPE) known_points[i].features[j]);
            #else // double
            DATA_TYPE absdiff = fabs((DATA_TYPE) new_point->features[j] - (DATA_TYPE) known_points[i].features[j]);
			#endif
			distance += absdiff;
        }
		#endif
		int id = omp_get_thread_num ();
		if(id == 0)
			updating_3_nearest(best_points_thread_0, distance, known_points[i].classification_id);
		else
			updating_3_nearest(best_points_thread_1, distance, known_points[i].classification_id);
	}
		// A function is needed to merge the two arrays of best points.
		// The function will be called here.
		merge_best_3_points(best_points_thread_0, best_points_thread_1, best_points);
		
		// these lines stores the distances.
        // dist_points[i].classification_id = known_points[i].classification_id;
        // dist_points[i].distance = distance;
    }

	// select the k nearest Points: 3 first elements of dist_points
    // select_3_nearest(dist_points, num_points);
	
	// copy the 3 first elements of dist_points to the best_points data structure
    // copy_3_nearest(dist_points, best_points);
	
	#if DIMEM != 0
	free(dist_points);
	#endif
} // end of get_3_NN

/*
* Classify a given Point (instance).
* It returns the classified class ID.
*/
CLASS_ID_TYPE knn_classifyinstance_3(Point *new_point, Point *known_points, int num_points, int num_features) {

	//BestPoint *best_points = (BestPoint *) calloc(k, sizeof(BestPoint)) ;
    BestPoint best_points[3]; // Array with the k nearest points to the Point to classify

    // calculate the distances of the new point to each of the known points and get
    // the k nearest points
    get_3_NN(new_point, known_points, num_points, best_points, num_features);

	// use plurality voting of 3 classes to return the class inferred for the new point
	CLASS_ID_TYPE classID = plurality_voting_3(best_points);
	
	return classID;
}

#endif
