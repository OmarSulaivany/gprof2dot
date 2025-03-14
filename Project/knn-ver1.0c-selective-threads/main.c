/**
*
*  JACA Benchmark Suite v1.0
*
*	Human Activity Recognition (HAR) using a kNN for inference.
*
*	FEUP.DEI,
*	v1.0 January 2025
*	Author: João MP Cardoso
*	Email: jmpc@fe.up.pt
*
*	Scenario A, WISDM dataset
*	Files for scenario must be copied from scenario-wisdm/
*	READ 2 // default 4
*	NUM_TRAINING_SAMPLES 4336
*	NUM_TESTING_SAMPLES 1082
*	NUM_FEATURES 43 
*	NUM_CLASSES 6
*
*	Scenario B, PAMAP2 dataset
*	Files for scenario must be copied from scenario-pamap2/
*	READ 2 // default 4
*	NUM_TRAINING_SAMPLES 6186
*	NUM_TESTING_SAMPLES 1008
*	NUM_FEATURES 90 
*	NUM_CLASSES 22 // 21 activities + 1 activity representing transient activities
*
*	Scenario C, Generated dataset
*	Files for scenario must be copied from scenario-gen100x8x50000/
*	READ 4 // default 4
*	NUM_TRAINING_SAMPLES 40002
*	NUM_TESTING_SAMPLES 9998
*	NUM_FEATURES 100
*	NUM_CLASSES 8
*
* 	Scenario D, Generated dataset
*	Files for scenario must be copied from scenario-gen100x8x10000/
*	READ 2 // default 4
*	NUM_TRAINING_SAMPLES 8004
*	NUM_TESTING_SAMPLES 1996
*	NUM_FEATURES 100
*	NUM_CLASSES 8
*
*	TIMMING 0 or *1* // 0: w/o timing measurements; 1: w/ timing measurements; default: 1
*	ACCURACY 0 or *1* // 0: w/o calculating andd reporting; 1: w/ calculation and reporting; default: 1
*	K = *3*, 5, 7 or 20 // any value >= 1 but these 4 were the ones considered for the scenarios; default: 3
* 	DATA_TYPE = float or double // default: double
*	DIMEM 0 or *1* // 0: w/o dynamic memory allocation; 1: w/ dynamic memory allocation; default: 0
*	STREAMING 0 or *1* // 0: w/o considering a possible streaming scheme; 1: considering streaming; default: 1
*
*	Other code versions:
*	SPECIALIZED 0 or 1 // 0: w/o specialization based on K value; 1: use specialized functions according to, for now, K=3; default: 1
*	MATH_TYPE 0 or 1 // // 0: use the math.h functions in the code; 1: use the math.h functions according to the type float or double, e.g., sqrt or sqrtf; default: 1
*	DIST_METHOD 0 or 1 // 1: Euclidean distance; 2: Manhattan distance; default: 1
*	USE_SQRT 0 or 1 // 0: w/o sqrt in Enclidean distance; 1: w/ sqrt in Euclidean distance; default: 1
*
*	All use minmax normalization of features
*
**/

/*
* Inputs
*	READ = 1			READ = 2			READ = 3			READ = 4
*	Static				static				From file			From file
*	-----------------------------------------------------------------------------
*	train1_notnorm.dat 	train1_norm.dat 	train2_notnorm.dat 	train2_norm.dat 
*	-----------------------------------------------------------------------------
*	test1_notnorm.dat	test1_notnorm.dat	test2_notnorm.dat	test2_notnorm.dat
*/

#ifndef READ
	#define READ 4 // 1 and 3: static initalization; 2 and 4: read from .dat
#endif

#ifndef TIMMING
	#define TIMMING 1 // 0: without; 1: Linux/Windows, 2: specific timers (not implemented)
#endif

#ifndef VERIFY
	#define VERIFY 0 	// 0: none verification;
						// 1: to verify if the results are according to the ones expected
#endif

#ifndef ACCURACY
	#define ACCURACY 1 	// 0: no; 1: to report the accuracy of the classification
#endif

#ifndef STREAMING
	#define STREAMING 1 
	// 0: all the points are first normalized and only then the classification occurs; 
	// 1: each point is normalized in the loop of the classification
#endif

#include <stdio.h>

#include "params.h"
#include "types.h"
#include "utils.h"
#include "io.h"
#include "knn.h"
#include "features.h"

#if TIMMING == 1
	#include "timer.h"
#endif


#if READ == 1 // data embedded in program, arrays statically filled
	
	Point known_points1[NUM_KNOWN_POINTS] = {
					#include "train1_notnorm.dat"
				}; //{{{1,2,3,4,5},1}};
	Point *known_points = known_points1;

	Point new_points1[NUM_NEW_POINTS] = {
					#include "test1_notnorm.dat"
				}; //{{{2.0,1.0, 1.0},0},{{2.0,3.0,2.0},0},...};
	Point *new_points = new_points1;

#elif READ == 2 // data embedded in program, arrays statically filled
	
	Point known_points1[NUM_KNOWN_POINTS] = {
					#include "train1_norm.dat"
				}; //{{{1,2,3,4,5},1}};
	Point *known_points = known_points1;

	Point new_points1[NUM_NEW_POINTS] = {
					#include "test1_notnorm.dat"
				}; //{{{2.0,1.0, 1.0},0},{{2.0,3.0,2.0},0},...};
	Point *new_points = new_points1;

#elif READ == 3 || READ == 4 // data read from files 

	#if DIMEM == 0
	Point known_points1[NUM_KNOWN_POINTS];
	Point *known_points = known_points1;

	Point new_points1[NUM_NEW_POINTS];
	Point *new_points = new_points1;
	
    #else
	Point *known_points
	Point *new_points;
	
	#endif
#endif


#if VERIFY == 1  // verify if classifications are still as the golden ones
	CLASS_ID_TYPE key[NUM_NEW_POINTS] = {
		#if K == 20
			#include "key-READ4-k20.dat"
		#elif K == 3
			#include "key-READ4-k3.dat"
		#elif K == 5
			#include "key-READ4-k5.dat"
		#elif K == 7
			#include "key-READ4-k7.dat"
		#endif
	};
#endif

int main() {

	const int k = K;
	const int num_features = NUM_FEATURES;
    const int num_classes = NUM_CLASSES;
    const int num_known_points = NUM_KNOWN_POINTS;
	const int num_new_points = NUM_NEW_POINTS;
	
	#if DIMEM == 1 && (READ == 3 || READ == 4)
	known_points = (Point *) malloc(NUM_KNOWN_POINTS*sizeof(Point));
	new_points = (Point *) malloc(NUM_NEW_POINTS*sizeof(Point));
	#endif
	
	#if READ == 1 || READ == 2 // data embedded in program
		printf("All statically initialized from data in files\n");
	#elif READ == 3  // data read from .dat files
		printf("Initializing non-normalized training vectors of features from .dat ...\n");
		read_data_points("train2_notnorm.dat", num_features, num_known_points, known_points); 
		printf("Initializing testing vectors of features from .dat ...\n");
		read_data_points("test2_notnorm.dat", num_features, num_new_points, new_points);
		printf("Initialization done.\n\n");
	#elif READ == 4  // data read from .dat files
		printf("Initializing normalized training vectors of features from .dat ...\n");
		read_data_points("train2_norm.dat", num_features, num_known_points, known_points); 
		printf("Initializing testing vectors of features from .dat ...\n");
		read_data_points("test2_notnorm.dat", num_features, num_new_points, new_points);
		printf("Initialization done.\n\n");
	#endif
	
	#if READ == 2 || READ == 4 // load min and max feature values for normalization
		DATA_TYPE min[NUM_FEATURES] = {
				#include "min.dat"
			};
		DATA_TYPE max[NUM_FEATURES] = {
				#include "max.dat"
			};
	#elif READ == 1 || READ == 3 // calculate min and max feature values for normalization
		DATA_TYPE min[NUM_FEATURES];
		DATA_TYPE max[NUM_FEATURES];
		
		// determine min and max from known points
		minmax(min, max, num_known_points, known_points, num_features);
	/*	
	printf("\nmin\n");
	for(int i=0; i< num_features; i++) {
		printf("%.4f,",min[i]);
	}
	printf("\nmax\n");
	for(int i=0; i< num_features; i++) {
		printf("%.4f,",max[i]);
	}*/
		
		// normalize known points
		minmax_normalize(min, max, num_known_points, known_points, num_features);
		
		//printf("known normalized:\n");
		//output_points(num_known_points, known_points, num_features, 1);
		//printf("known normalized:\n");
		//output_points(num_known_points, known_points, num_features, 2);
	#endif
	
	//printf("new nonnormalized:\n");
	//output_points(num_new_points, new_points, num_features, 1);
	//printf("new nonnormalized:\n");
	//output_points(num_new_points, new_points, num_features, 2);
		
	#if STREAMING == 0 // no streaming: normalize all features of test vectors before 
		minmax_normalize(min, max, num_new_points, new_points, num_features);
	#endif
	
    printf("Executing kNN...\n");

	#if TIMMING == 1
	Timer *timer = timer_init();
	timer_start(timer);
	#endif
	
	#if ACCURACY == 1
	int fail = 0; // count the number of test instances incorrectly classified
	#endif
		
	// loop over the input instances to classify.
	// Note that depending on the application this can be
	// instances arriving as streaming data.
	// Here assume that the loop below needs to run in serial mode and the
	// value of num_new_points is just to test
    for (int i = 0; i < num_new_points; i++) {
		
		// get instance to classify
		// in a streaming implementation this might be something like:
		// get(new_point);
		Point *new_point = &new_points[i];	
		
		//printf("{{");
		//output_point(*new_point, num_features, 1);  // 1 for spaces
		//printf("},\n");
		
		#if STREAMING == 1 
		// normalize the point to classify
		minmax_normalize_point(min, max, new_point, num_features);
		#endif
		
		// classify
		#if SPECIALIZED == 1 && K == 3
        CLASS_ID_TYPE instance_class = knn_classifyinstance_3(new_point,
                                       known_points, num_known_points, num_features);
		#else
        CLASS_ID_TYPE instance_class = knn_classifyinstance(new_point, k, num_classes,
                                       known_points, num_known_points, num_features);
		#endif
	
		#if ACCURACY
			if(new_point->classification_id != instance_class) fail++;
		#endif

		//printf("class id: %d\n", instance_class);
		//printf("%d,", instance_class);
		
        // The following store the inferred class in the point structure
        // In practice and especially in streaming operation, this
        // may not be done and the class is output to the subsequent
        // stages of the application
        // For now this is used to verify the results by comparing
        // the class obtained for each point to a golden class
		// for now: output the inferred class of the instance
		put(&instance_class, 0, "class id:"); // 0 is the id used for char
		
		#if STREAMING == 0 // store the inferred class: we don't need the label anymore
		new_point->classification_id = instance_class;
		#endif
    }

	#if TIMMING == 1
		timer_stop(timer);
	#endif

    printf("kNN done.\n\n");

    printf("HAR: number of features = %d\n", num_features);
    printf("HAR: number of classes = %d\n", num_classes);
	
	printf("kNN: k = %d\n", k);
    printf("kNN: number of training instances = %d\n", num_known_points);
    printf("kNN: number of testing instances = %d\n", num_new_points);

	#if DT == 2
      printf("kNN: data type used = float\n");
    #elif DT == 1
      printf("kNN: data type used = double\n");
	#endif
	
    printf("kNN: number of classified instances = %d\n", num_new_points);

	#if ACCURACY == 1
		printf("kNN: number of classifications wrong = %d\n", fail);
		printf("kNN: number of classifications right = %d\n", num_new_points-fail);
		printf("kNN: accuracy = %.2f %s\n\n", 100*((float)(num_new_points-fail)/(float) num_new_points),"%");
	#endif

	#if VERIFY == 1
		verify_results(num_new_points, new_known_points, key);
	#endif

	#if TIMMING == 1
		const double time = timer_get_s(timer);
		timer = timer_destroy(timer);
		
		printf("\nTotal execution time:  %.4f s\n", time);
		
		printf("Inference time: %.4f ms\n", (time/num_new_points*1000));
		printf("Throughput: %.2f inferences/s\n", num_new_points/time);
	#endif

	#if DIMEM != 0
		free(new_points);
		free(known_points);
	#endif

    return 0;
}