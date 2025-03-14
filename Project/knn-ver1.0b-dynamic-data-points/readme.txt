gcc -D K=3 -D SPECIALIZED=1 -D MATH_TYPE=1 -D DIST_METHOD=1 -D USE_SQRT=1 -D SCENARIO_FEATURES=2 -D READ=4 -D DIMEM=0 -D VERIFY=0 -D STREAMING=1 -O3 -Wall -Wextra -std=gnu99  knn.c utils.c timer.c io.c features.c main.c -lm -o knn -pg
./knn
gprof knn
