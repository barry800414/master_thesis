#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>
#include <tgmath.h>
#include "data_structure.h"

//==============================================================================================
typedef struct MATRIX_FACTORIZATION{
	int latentFactorCount;					// K
	double learningRate;					// alpha
	double userRegularizationRate;			// lambda_1
	double itemRegularizationRate;			// lambda_2
	double userSocialRegularizationRate;	// beta_1
	double itemSocialRegularizationRate;	// beta_2
	double convergenceThreshold;
	int maxSGDIterationCount;
	int userCount;
	int itemCount;
	Matrix *userMatrix;						// N * K matrix
	Matrix *itemMatrix;						// M * K matrix
} MatrixFactorization;

double matrixFactorizationEvaluateRMSE(MatrixFactorization *model, List *ratings);

void matrixFactorizationRunSGDStep(MatrixFactorization *model, List *ratings, Matrix *userMatrix, Matrix *itemMatrix, List *userSocialNetwork, List *itemSocialNetwork, double learningRate){
	double* userFactorDifferences = (double*)malloc(sizeof(double) * model -> latentFactorCount);
	double* itemFactorDifferences = (double*)malloc(sizeof(double) * model -> latentFactorCount);
	for(int user = 0; user < ratings -> rowCount; user ++){
		for(int j = 0; j < ratings -> columnCounts[user]; j ++){
			int item = ratings -> entries[user][j].key;
			double trueRating = ratings -> entries[user][j].value;
			double predictedRating = vectorCalculateDotProduct(userMatrix -> entries[user], itemMatrix -> entries[item], model -> latentFactorCount);
			double ratingError = trueRating - predictedRating;					// e_ij = r_ij - \sum(u_ki * v_kj)
			
			// friendship regularization ========
			
			for(int k = 0; k < model -> latentFactorCount; k ++){
				userFactorDifferences[k] = 0.0;
				itemFactorDifferences[k] = 0.0;
			}
			for(int f = 0; f < userSocialNetwork -> columnCounts[user]; f ++){
				int friend = userSocialNetwork -> entries[user][f].key;
				double similarity = userSocialNetwork -> entries[user][f].value;

				for(int k = 0; k < model -> latentFactorCount; k ++){
					double userFactor = userMatrix -> entries[user][k];
					double friendFactor = userMatrix -> entries[friend][k];
					userFactorDifferences[k] += similarity * (userFactor - friendFactor);
				}
			}
			for(int f = 0; f < itemSocialNetwork -> columnCounts[item]; f ++){
				int friend = itemSocialNetwork -> entries[item][f].key;
				double similarity = itemSocialNetwork -> entries[item][f].value;

				for(int k = 0; k < model -> latentFactorCount; k ++){
					double itemFactor = itemMatrix -> entries[item][k];
					double friendFactor = itemMatrix -> entries[friend][k];
					itemFactorDifferences[k] += similarity * (itemFactor - friendFactor);
				}
			}

			// SGD updates ========

			for(int k = 0; k < model -> latentFactorCount; k ++){
				double userFactor = userMatrix -> entries[user][k];
				double itemFactor = itemMatrix -> entries[item][k];
				
				double userUpdateValue = learningRate 
												  * (ratingError * itemFactor 
													- model -> userRegularizationRate * userFactor 
													- model -> userSocialRegularizationRate * userFactorDifferences[k]);
				double itemUpdateValue = learningRate 
												  * (ratingError * userFactor 
													- model -> itemRegularizationRate * itemFactor
													- model -> itemSocialRegularizationRate * itemFactorDifferences[k]);

				userMatrix -> entries[user][k] += userUpdateValue;
				itemMatrix -> entries[item][k] += itemUpdateValue;
			}
		}

	}

	free(userFactorDifferences);
	free(itemFactorDifferences);
}

void matrixFactorizationLearn(MatrixFactorization *model, List *trainingRatings, List *validationRatings, List *userSocialNetwork, List *itemSocialNetwork){
	int trainingRatingCount = listCountEntries(trainingRatings);
	int validationRatingCount = listCountEntries(validationRatings);

	printf("Total %d training ratings, %d validation ratings\n", trainingRatingCount, validationRatingCount);
	
	double lastValidationCost = DBL_MAX;
	
	matrixAssignRandomValues(model -> userMatrix, 0, 1);
	matrixAssignRandomValues(model -> itemMatrix, 0, 1);

	for(int iteration = 0; iteration < model -> maxSGDIterationCount; iteration ++){
		matrixFactorizationRunSGDStep(model, trainingRatings, model -> userMatrix, model -> itemMatrix, userSocialNetwork, itemSocialNetwork, model -> learningRate);

		double validationCost = matrixFactorizationEvaluateRMSE(model, validationRatings);
		printf("Iteration %d\tValidCost %f\tCostDescent %f\n", iteration + 1, validationCost, (lastValidationCost < DBL_MAX) ? lastValidationCost - validationCost : 0.0);
		if(iteration > 0 && lastValidationCost - validationCost < model -> convergenceThreshold){
			break;
		}

		lastValidationCost = validationCost;
	}
}

double matrixFactorizationPredict(MatrixFactorization *model, int user, int item){
	double predictedRating = vectorCalculateDotProduct(model -> userMatrix -> entries[user], model -> itemMatrix -> entries[item], model -> latentFactorCount);
	return predictedRating;
}

double matrixFactorizationEvaluateRMSE(MatrixFactorization *model, List *ratings){
	double rmse = 0.0;
	int totalRatingCount = listCountEntries(ratings);

	for(int user = 0; user < ratings -> rowCount; user ++){
		for(int j = 0; j < ratings -> columnCounts[user]; j ++){
			int item = ratings -> entries[user][j].key;
			double trueRating = ratings -> entries[user][j].value;

			double predictedRating = matrixFactorizationPredict(model, user, item);

			double difference = predictedRating - trueRating;
			rmse += difference * difference;
		}
	}

	rmse = sqrt(rmse / totalRatingCount);
	return rmse;
}

double matrixFactorizationEvaluateMAE(MatrixFactorization *model, List *ratings){
	double mae = 0.0;
	int totalRatingCount = listCountEntries(ratings);

	for(int user = 0; user < ratings -> rowCount; user ++){
		for(int j = 0; j < ratings -> columnCounts[user]; j ++){
			int item = ratings -> entries[user][j].key;
			double trueRating = ratings -> entries[user][j].value;

			double predictedRating = matrixFactorizationPredict(model, user, item);
			
			mae += fabs(predictedRating - trueRating);
		}
	}

	mae /= totalRatingCount;
	return mae;
}

double matrixFactorizationEvaluate(MatrixFactorization *model, List *ratings, int evaluationType){
	switch(evaluationType){
		case 1:
			return matrixFactorizationEvaluateRMSE(model, ratings);
		case 2:
			return matrixFactorizationEvaluateMAE(model, ratings);
	}
}

void matrixFactorizationSaveModel(MatrixFactorization *model, char *modelFilePath){
	FILE *modelFile = fopen(modelFilePath, "w");

	matrixPrint(model -> userMatrix, modelFile);

	fprintf(modelFile, "\n");

	matrixPrint(model -> itemMatrix, modelFile);

	fclose(modelFile);
}

void matrixFactorizationLoadModel(char *modelFilePath, MatrixFactorization *model){
	FILE *modelFile = fopen(modelFilePath, "r");

	matrixScan(modelFile, model -> userMatrix);

	matrixScan(modelFile, model -> itemMatrix);

	fclose(modelFile);
}
//==============================================================================================
typedef struct CROSS_VALIDATION{
	int foldCount;
	int evaluationTypeCount;
	int *evaluationTypes;
	int trainingFoldCount;
} CrossValidation;

// Group index [0, groupCount - 1]
int crossValidationDetermineGroup(int groupCount){
	double uniform;
	do{
		uniform = (double)rand() / RAND_MAX;
	}while(uniform == 1.0);
	return (int)(uniform * groupCount);
}

void crossValidationGroupRatings(List *ratings, int foldCount, List *groupMarkers){
	for(int user = 0; user < ratings -> rowCount; user ++){
		groupMarkers -> entries[user] = (Dict*)realloc(groupMarkers -> entries[user], sizeof(Dict) * ratings -> columnCounts[user]);
		groupMarkers -> columnCounts[user] = ratings -> columnCounts[user];

		for(int j = 0; j < ratings -> columnCounts[user]; j ++){
			int item = ratings -> entries[user][j].key;
			groupMarkers -> entries[user][j].key = item;
			groupMarkers -> entries[user][j].value = crossValidationDetermineGroup(foldCount);
		}
	}
}

void crossValidationSplitRatings(List *ratings, List *groupMarkers, List *trainingRatings, List *validationRatings, int validationGroup){
	for(int user = 0; user < ratings -> rowCount; user ++){
		for(int j = 0; j < ratings -> columnCounts[user]; j ++){
			int item = ratings -> entries[user][j].key;
			double rating = ratings -> entries[user][j].value;
			int group = groupMarkers -> entries[user][j].value;

			if(group == validationGroup){
				listAddEntry(validationRatings, user, item, rating);
			}
			else{
				listAddEntry(trainingRatings, user, item, rating);
			}
		}
	}
}

double crossValidationRun(CrossValidation *validation, MatrixFactorization *model, List *ratings, List *userSocialNetwork, List *itemSocialNetwork, double *performanceMeans){
	List groupMarkers;
	listInitialize(&groupMarkers, ratings -> rowCount);
	crossValidationGroupRatings(ratings, validation -> foldCount, &groupMarkers);
	
	bool performanceMeansDeclared = false;
	if(performanceMeans == NULL){
		performanceMeansDeclared = true;
		performanceMeans = (double*)malloc(sizeof(double) * validation -> evaluationTypeCount);
	}
	for(int e = 0; e < validation -> evaluationTypeCount; e ++){
		performanceMeans[e] = 0.0;
	}

	for(int validedFold = 0; validedFold < validation -> foldCount; validedFold ++){
		List trainingRatings, validationRatings;
		listInitialize(&trainingRatings, ratings -> rowCount);
		listInitialize(&validationRatings, ratings -> rowCount);	
		crossValidationSplitRatings(ratings, &groupMarkers, &trainingRatings, &validationRatings, validedFold);
		
		List trainingGroupMarkers, trainingTrainRatings, trainingValidRatings;
		listInitialize(&trainingGroupMarkers, ratings -> rowCount);
		listInitialize(&trainingTrainRatings, ratings -> rowCount);
		listInitialize(&trainingValidRatings, ratings -> rowCount);
		crossValidationGroupRatings(&trainingRatings, validation -> trainingFoldCount, &trainingGroupMarkers);
		crossValidationSplitRatings(&trainingRatings, &trainingGroupMarkers, &trainingTrainRatings, &trainingValidRatings, 0);

		matrixFactorizationLearn(model, &trainingTrainRatings, &trainingValidRatings, userSocialNetwork, itemSocialNetwork);

		printf("Cross validation %d\n", validedFold + 1);
		for(int e = 0; e < validation -> evaluationTypeCount; e ++){
			double performance = matrixFactorizationEvaluate(model, &validationRatings, validation -> evaluationTypes[e]);
			performanceMeans[e] += performance;
			printf("\tPerformance %d %f\n", validation -> evaluationTypes[e], performance);
		}

		listReleaseSpace(&trainingRatings);
		listReleaseSpace(&validationRatings);
		listReleaseSpace(&trainingGroupMarkers);
		listReleaseSpace(&trainingTrainRatings);
		listReleaseSpace(&trainingValidRatings);
	}

	double wholePerformanceValue = 0.0;
	for(int e = 0; e < validation -> evaluationTypeCount; e ++){
		performanceMeans[e] /= validation -> foldCount;
		printf("Average performance %d %f\n", validation -> evaluationTypes[e], performanceMeans[e]);
		
		switch(validation -> evaluationTypes[e]){
			case 2:
				wholePerformanceValue += - performanceMeans[e];
				break;
		}
	}

	listReleaseSpace(&groupMarkers);
	if(performanceMeansDeclared){
		free(performanceMeans);
	}

	return wholePerformanceValue;	
}

void crossValidationRunWithParameters(CrossValidation *validation, MatrixFactorization *model, List *ratings, List *userSocialNetwork, List *itemSocialNetwork){
	bool isFirstValue = true;
	double bestRegularizationRate = 0;
	double bestSocialRegularizationRate = 0;

	double bestPerformanceValue = 0;
	double *bestPerformanceMeans = (double*)malloc(sizeof(double) * validation -> evaluationTypeCount);
	double *performanceMeans = (double*)malloc(sizeof(double) * validation -> evaluationTypeCount);

	for(int power1 = -1; power1 >= -3; power1 --){
		double regularizationRate = pow(10, power1);
		for(int power2 = -1; power2 >= -3; power2 --){
			double socialRegularizationRate = pow(10, power2);
			model -> userRegularizationRate = regularizationRate;
			model -> itemRegularizationRate = regularizationRate;
			model -> userSocialRegularizationRate = socialRegularizationRate;
			model -> itemSocialRegularizationRate = socialRegularizationRate;

			double performanceValue = crossValidationRun(validation, model, ratings, userSocialNetwork, itemSocialNetwork, performanceMeans);
			if(isFirstValue || performanceValue > bestPerformanceValue){
				isFirstValue = false;
				bestPerformanceValue = performanceValue;
				bestRegularizationRate = regularizationRate;
				bestSocialRegularizationRate = socialRegularizationRate;
				for(int e = 0; e < validation -> evaluationTypeCount; e ++){
					bestPerformanceMeans[e] = performanceMeans[e];
				}
			}
		}
	}

	printf("========\n");
	printf("Best regularization rate %f\n", bestRegularizationRate);
	printf("Best social regularization rate %f\n", bestSocialRegularizationRate);
	for(int e = 0; e < validation -> evaluationTypeCount; e ++){	
		printf("Average performance %d %f\n", validation -> evaluationTypes[e], bestPerformanceMeans[e]);
	}

	free(bestPerformanceMeans);
	free(performanceMeans);
}

//==============================================================================================
int main(int argc, char *argv[]){
	if(argc < 4){
        fprintf(stderr, "%s RatingFile EvalType(1:RMSE/2:MAE) ModelFile [-options parameter] .... \n", argv[0]);
        fprintf(stderr, "\t[-k LatentDim] [-rate learning rate] [-uReg userReguraization] [-iReg itemRegularization]\n");
        fprintf(stderr, "\t[-iter maxIteration] [-userNetwork file reg] [-itemNetwork file reg] [-seed seed]\n");
        return 0;
    }

    // reading arguments
	char *ratingFilePath = argv[1];
	int evaluationType = atoi(argv[2]); //1: RMSE, 2:MAE
	char *modelFilePath = argv[3];
    int k = 10; //latent dimension
    double learnRate = 0.005, uReg = 0.1, iReg = 0.1, uSocialReg = 0.01, iSocialReg = 0.01;
    int maxIter = 1000, seed=1;
    char *userNetworkFile = NULL, *itemNetworkFile = NULL, *ignoreUserIdFile = NULL;
    for(int i=4; i<argc; i++){
        if(strcmp(argv[i], "-k") == 0 && argc > i){
            k = atoi(argv[i+1]);
            printf("k: %d\n", k);
            i = i + 1;
        }
        else if(strcmp(argv[i], "-rate") == 0 && argc > i){
            learnRate = atof(argv[i+1]);
            printf("learning rate: %f\n", learnRate);
            i = i + 1;
        }
        else if(strcmp(argv[i], "-uReg") == 0 && argc > i){
            uReg = atof(argv[i+1]);
            printf("uReg: %f\n", uReg); 
            i = i + 1;
        }
        else if(strcmp(argv[i], "-iReg") == 0 && argc > i){
            iReg = atof(argv[i+1]);
            printf("iReg: %f\n", iReg);
            i = i + 1;
        }
        else if(strcmp(argv[i], "-iter") == 0 && argc > i){
            maxIter = atoi(argv[i+1]);
            printf("maxIter: %d\n", maxIter);
            i = i + 1;
        }
        else if(strcmp(argv[i], "-userNetwork") == 0 && argc > (i+1)){
            userNetworkFile = argv[i+1];
            uSocialReg = atof(argv[i+2]);
            printf("UserNetworkFile:%s uSocialReg: %f\n", userNetworkFile, uSocialReg);
            i = i + 2;
        }
        else if(strcmp(argv[i], "-itemNetwork") == 0 && argc > (i+1)){
            itemNetworkFile = argv[i+1];
            iSocialReg = atof(argv[i+2]);
            printf("ItemNetworkFile:%s iSocialReg: %f\n", itemNetworkFile, iSocialReg);
            i = i + 2;
        }
        else if(strcmp(argv[i], "-seed") == 0 && argc > i){
            seed = atoi(argv[i+1]);
            printf("Random Seed: %d\n", seed);
        }
        else if(strcmp(argv[i], "-ignoreUserIdFile") == 0 && argc > i){
            ignoreUserIdFile = argv[i+1];
            printf("IgnoreUserIdFile: %s\n", ignoreUserIdFile);
        }
    }

    srand(seed);
	int userCount, itemCount;
	ratingFetchUserItemCount(ratingFilePath, &userCount, &itemCount);

	// Read rating data
	List ratings;
	listInitialize(&ratings, userCount);
	ratingReadFromFile(ratingFilePath, &ratings);
	listSortRows(&ratings);
	printf("%d users, %d items\n", userCount, itemCount);

	// Set user social network
	List userSocialNetwork;
	listInitialize(&userSocialNetwork, userCount);
	if(userNetworkFile != NULL){
		FILE *inFile = fopen(userNetworkFile, "r");
        if(ignoreUserIdFile != NULL){
            FILE *ignoreInFile = fopen(ignoreUserIdFile, "r");
            listScanWithIgnore(inFile, ignoreInFile, &userSocialNetwork);
            fclose(ignoreInFile);
        }
		listScan(inFile, &userSocialNetwork);
		fclose(inFile);
	}

	// Set item social network	
	List itemSocialNetwork;
	listInitialize(&itemSocialNetwork, itemCount);
	if(itemNetworkFile != NULL){
		FILE *inFile = fopen(itemNetworkFile, "r");
		listScan(inFile, &itemSocialNetwork);
		fclose(inFile);
	}
	
	// Set matrix factorization
	MatrixFactorization mf = {
		.latentFactorCount = k,
		.learningRate = learnRate,
		.userRegularizationRate = uReg,
		.itemRegularizationRate = iReg,
		.userSocialRegularizationRate = uSocialReg,
		.itemSocialRegularizationRate = iSocialReg,
		.convergenceThreshold = 1e-4,
		.maxSGDIterationCount = maxIter,
		.userCount = userCount,
		.itemCount = itemCount
	};
	Matrix userMatrix;
	Matrix itemMatrix;
	matrixInitialize(&userMatrix, userCount, mf.latentFactorCount);
	matrixInitialize(&itemMatrix, itemCount, mf.latentFactorCount);
	mf.userMatrix = &userMatrix;
	mf.itemMatrix = &itemMatrix;

	printf("User matrix %d %d\n", userMatrix.rowCount, userMatrix.columnCount);
	printf("Item matrix %d %d\n", itemMatrix.rowCount, itemMatrix.columnCount);
	
	matrixFactorizationLearn(&mf, &ratings, &ratings, &userSocialNetwork, &itemSocialNetwork);
	matrixFactorizationSaveModel(&mf, modelFilePath);

	// Release space
	listReleaseSpace(&ratings);
	listReleaseSpace(&userSocialNetwork);
	listReleaseSpace(&itemSocialNetwork);
	matrixReleaseSpace(&userMatrix);
	matrixReleaseSpace(&itemMatrix);

	printf("OK\n");
	return 0;
}
