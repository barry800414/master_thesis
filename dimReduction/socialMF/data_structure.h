#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>
#ifndef __DATA_STRUCTURE_H__
#define __DATA_STRUCTURE_H__

//==============================================================================================
typedef struct DICT{
	int key;
	double value;
} Dict;

int dictCompareAscendingKeys(const void *a, const void *b);
double dictVectorCalculateValueMean(Dict *vector, int length);
// The two vectors should be sorted by keys
double dictVectorCalculatePearsonCorrelationCoefficient(Dict* vector1, Dict* vector2, int length1, int length2, double mean1, double mean2, int intersectionLowerBound);
//==============================================================================================
double vectorCalculateEuclideanDistanceSquare(double *vector1, double *vector2, int length);
double vectorCalculateCorrelationCoefficient(double *vector1, double *vector2, int length);
double vectorCalculateDotProduct(double *vector1, double *vector2, int length);
double vectorCalculateMean(double *vector, int length);
void vectorRunElimination(double *vector1, double *vector2, int length, int factorIndex);
//==============================================================================================
typedef struct LIST{
	int rowCount;
	int *columnCounts;
	Dict **entries;
} List;

void listInitialize(List *list, int rowCount);
void listReleaseSpace(List *list);
void listAddEntry(List *list, int row, int key, double value);
void listCopyEntries(List *source, List *target);
void listSortRows(List *list);
int listCountEntries(List *list);
// returns a row vector sorted by keys (column index)
void listGetRowVector(List *list, int row, Dict **vectorPointer, int *vectorLength);
// returns a column vectors sorted by keys (row index)
void listGetColumnVector(List *list, int column, Dict **vectorPointer, int *vectorLength);
// rowVectors should contain at least the number of rows that list contains
void listGetAllRowVectors(List *list, List *rowVectors);
// columnVectors should contain at least the number of columns that list contains
void listGetAllColumnVectors(List *list, List *columnVectors);
void listCountRowEntries(List *list, int *entryCounts);
void listCountColumnEntries(List *list, int *entryCounts, int columnCount);
void listPrint(List *list, FILE *outputStream);
void listScan(FILE *inputStream, List *list);
void listScanWithIgnore(FILE *inputStream, FILE *ignoreInputStream, List *list);
//==============================================================================================
typedef struct MATRIX{
	int rowCount;
	int columnCount;
	double **entries;
} Matrix;

void matrixInitialize(Matrix *matrix, int rowCount, int columnCount);
void matrixReleaseSpace(Matrix *matrix);
void matrixAssignRandomValues(Matrix *matrix, double minValue, double maxValue);
// Source and target should contain the same rows and columns
void matrixCopyEntries(Matrix *source, Matrix *target);
void matrixPrint(Matrix *matrix, FILE *outputStream);
void matrixScan(FILE *inputStream, Matrix *matrix);
void matrixSetIdentity(Matrix *matrix);
void matrixSetValue(Matrix *matrix, double value);
void matrixMultiplyScalar(Matrix *matrix, double scalar);
void matrixAddScalar(Matrix *matrix, double scalar);
double matrixCalculateSquareSum(Matrix *matrix);
void matrixGetTranspose(Matrix *matrix, Matrix *transpose);
bool matrixGetInverse(Matrix *matrix, Matrix *inverse);
double matrixCalculatePositiveDefiniteLogDeterminant(Matrix *matrix);
//==============================================================================================
/*
typedef struct SOCIAL_NETWORK_SETTING{
	double similarityThreshold;
	double dissimilarityThreshold;
	int vectorKeyIntersectionLowerBound;
	int friendCountThreshold;
} SocialNetworkSetting;

void socialNetworkSettingFetchImplicitFriendship(SocialNetworkSetting *socialNetworkSetting, List *ratings, List *socialNetwork, bool userOrItem, bool similarOnly);
void socialNetworkSettingFetchImplicitUserFriendship(SocialNetworkSetting *socialNetworkSetting, List *ratings, List *socialNetwork, bool similarOnly);
void socialNetworkSettingFetchImplicitItemFriendship(SocialNetworkSetting *socialNetworkSetting, List *ratings, List *socialNetwork, bool similarOnly);
*/
//==============================================================================================
void ratingFetchUserItemCount(char *ratingFilePath, int *userCount, int *itemCount);
void ratingReadFromFile(char *ratingFilePath, List *ratings);
#endif
