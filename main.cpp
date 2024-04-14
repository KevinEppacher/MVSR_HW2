#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>

// Owner: Kevin Eppacher
// https://github.com/KevinEppacher/MVSR_HW2.git


/// @struct Data
/// @brief Holds pointers to training and testing data.
struct Data 
{
    cv::Ptr<cv::ml::TrainData> total; ///< Total dataset loaded from file.
    cv::Ptr<cv::ml::TrainData> train; ///< Filtered training data.
    cv::Ptr<cv::ml::TrainData> test;  ///< Filtered testing data.
};

/// @struct Matrices
/// @brief Contains matrices of samples and labels for training or testing.
struct Matrices 
{
    cv::Mat sample; ///< Matrix of data samples.
    cv::Mat label;  ///< Matrix of corresponding labels.
};

/// @struct FilterRows
/// @brief Defines the range of rows to filter from the dataset.
struct FilterRows 
{
    int startRow; ///< Starting row index for filtering.
    int endRow;   ///< Ending row index for filtering.
};


/// @brief Computes the accuracy of predictions against actual labels.
/// @param predicted Matrix of predicted labels.
/// @param actual Matrix of actual labels.
/// @return Proportion of correct predictions.
float computeAccuracy(const cv::Mat& predicted, const cv::Mat& actual) 
{
    int correct = 0;
    for (int i = 0; i < predicted.rows; ++i) {
        int predLabel = predicted.at<float>(i, 0) >= 0.5 ? 1 : 0;
        int trueLabel = static_cast<int>(actual.at<float>(i, 0));
        if (predLabel == trueLabel) {
            correct++;
        }
    }
    return static_cast<float>(correct) / predicted.rows;
}


/// @brief Maps labels to binary values based on specified targets.
/// @param originalLabel Matrix of original labels.
/// @param desiredLabels Vector of desired binary labels.
/// @return Matrix of mapped binary labels.
cv::Mat mapLabelToBinary(const cv::Mat& originalLabel, const std::vector<float>& desiredLabels) 
{
    cv::Mat binaryLabels = originalLabel.clone();

    for (int i = 0; i < binaryLabels.rows; ++i) {
        float& label = binaryLabels.at<float>(i, 0);
        label = (label == 2) ? desiredLabels.at(0) : (label == 5 ? desiredLabels.at(1) : label);
    }
    return binaryLabels;
}

class DataPreprocessor {
private:
    cv::Mat samples;
    cv::Mat label;
    cv::Mat filteredDigits;
    cv::Mat filteredLabel;
    Matrices train, test;
    Data data;


public:
    DataPreprocessor(){};

    ~DataPreprocessor(){};

    /// @brief Filters data by specified digit labels within a row range.
    /// @param originalData The complete dataset from which to filter.
    /// @param digit1 First digit to filter by.
    /// @param digit2 Second digit to filter by.
    /// @param startRow Starting row index for filtering.
    /// @param endRow Ending row index for filtering.
    /// @return A pointer to the filtered training data.
    cv::Ptr<cv::ml::TrainData> filter(const cv::Ptr<cv::ml::TrainData>& originalData, int digit1, int digit2, int startRow, int endRow) 
    {
        int totalRows = originalData->getSamples().rows;
        startRow = std::max(0, startRow);
        endRow = std::min(endRow, totalRows);

        cv::Mat samples = originalData->getSamples().rowRange(startRow, endRow);
        cv::Mat responses = originalData->getResponses().rowRange(startRow, endRow);

        cv::Mat mask = (responses == digit1) | (responses == digit2);
        cv::Mat filteredSamples, filteredResponses;

        for (size_t i = 0; i < mask.total(); ++i) {
            if (mask.at<uint8_t>(i)) {
                filteredSamples.push_back(samples.row(i));
                filteredResponses.push_back(responses.row(i));
            }
        }

        return cv::ml::TrainData::create(filteredSamples, cv::ml::ROW_SAMPLE, filteredResponses);
    }

    cv::Mat getFilteredDigits() const { return filteredDigits; }

    cv::Mat getFilteredLabel() const { return filteredLabel; }

    void displayImagesInTerminal(int numOfImages, cv::Mat& samples, cv::Mat& label)
    {
        for (int i = 0; i < numOfImages && i < samples.rows; i++) {
            int currentLabel = static_cast<int>(label.at<float>(i, 0));
            std::cout << "Label: " << currentLabel << std::endl;

            for (int row = 0; row < 28; row++) {
                for (int col = 0; col < 28; col++) {
                    float pixelValue = samples.at<float>(i, row * 28 + col);
                    std::cout << (pixelValue > 0.5 ? "#" : " ");
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    /// @brief Standardizes the data by subtracting the mean and dividing by the standard deviation.
    /// @param data The data matrix to be standardized.
    void standardizeData(cv::Mat& data) 
    {
        data.convertTo(data, CV_32F);

        cv::Scalar mean, stddev;

        cv::meanStdDev(data, mean, stddev);

        if (stddev[0] != 0) 
        {
            data = (data - mean[0]) / stddev[0];
        }
    }

    /// @brief Checks and prints the mean and standard deviation of the data.
    /// @param data The data matrix to be checked.
    /// @param name Description of the data being checked.
    void checkStandardization(const cv::Mat& data, const std::string& name) 
    {
        cv::Scalar mean, stdDev;

        cv::meanStdDev(data, mean, stdDev);

        std::cout << name << " Mean: " << mean[0] << ", Stddev: " << stdDev[0] << std::endl;

    }

    /// @brief Writes PCA-reduced data and labels to a CSV file.
    /// @param projectedData The matrix containing PCA-reduced data.
    /// @param filepath Path to the output CSV file.
    void writeCsvFile(const cv::Mat& projectedData, const std::string& filepath)
    {
        std::ofstream outputFile(filepath);

        outputFile << "Component1,Component2,Component3,Label\n";

        for (int i = 0; i < projectedData.rows; i++) 
        {
            for (int j = 0; j < projectedData.cols; j++) 
            {
                outputFile << projectedData.at<float>(i, j) << ",";
            }
            outputFile << label.at<float>(i, 0) << "\n";
        }
        outputFile.close();

        std::cout << "PCA-reduced data and labels have been written to '" << filepath << "'" << std::endl;
    }

    cv::Mat getTrainSamples(){ return train.sample; }
    
    cv::Mat getTestSamples(){ return train.sample; }

    cv::Mat getTrainLabels(){ return train.label; }
    
    cv::Mat getTestLabels(){ return test.label; }

};

/// @class LogisticRegression
/// @brief Implements logistic regression including training and prediction functionalities.
class LogisticRegression {
private:
    cv::Mat weights; ///< Weight matrix for logistic regression.

    /// @brief Adds a bias term to the dataset and returns the modified dataset.
    /// @param trainData The training data without the bias term.
    /// @return The training data with a bias term added.
    cv::Mat calculateTrainDataWithBias(const cv::Mat& trainData) 
    {
        cv::Mat trainDataWithBias;

        cv::hconcat(cv::Mat::ones(trainData.rows, 1, trainData.type()), trainData, trainDataWithBias);

        return trainDataWithBias;
    }

    /// @brief Applies the sigmoid function to each element in the matrix.
    /// @param z The input matrix.
    /// @return The matrix with the sigmoid function applied.
    cv::Mat sigmoid(const cv::Mat& z) 
    {
        cv::Mat output;

        cv::exp(-z, output);

        output = 1.0 / (1.0 + output);

        return output;
    }

    /// @brief Calculates the diagonal matrix R for logistic regression training.
    /// @param trainDataWithBias The training data including the bias term.
    /// @param trainPrediction Predictions made by the current model on the training data.
    /// @return The diagonal matrix R used in the Hessian calculation.
    cv::Mat calculateR(const cv::Mat& trainDataWithBias, const cv::Mat& trainPrediction) 
    {
        cv::Mat R = cv::Mat::zeros(trainDataWithBias.rows, trainDataWithBias.rows, CV_32F);

        for (int i = 0; i < trainDataWithBias.rows; ++i) 
        {
            double p = trainPrediction.at<float>(i, 0);
            R.at<float>(i, i) = p * (1 - p);
        }
        return R;
    }

    /// @brief Calculates the Hessian matrix for the logistic regression model.
    /// @param trainDataWithBias The training data including the bias term.
    /// @param R The diagonal matrix R from the logistic regression formula.
    /// @return The Hessian matrix.
    cv::Mat calculateHessian(const cv::Mat& trainDataWithBias, const cv::Mat& R) 
    {
        return trainDataWithBias.t() * R * trainDataWithBias;
    }

    /// @brief Calculates the gradient of the loss function.
    /// @param trainDataWithBias The training data including the bias term.
    /// @param predictions Predictions made by the current model.
    /// @param labels Actual labels for the training data.
    /// @return The gradient of the loss function.
    cv::Mat calculateGradient(const cv::Mat& trainDataWithBias, const cv::Mat& predictions, const cv::Mat& labels) 
    {
        cv::Mat errors = predictions - labels;
        return trainDataWithBias.t() * errors;
    }

public:
    /// @brief Constructor for the LogisticRegression class.
    /// @param quantityRows Number of rows in the training data, used to size the weights matrix.
    LogisticRegression(int quantityRows) : weights(cv::Mat::zeros(quantityRows + 1, 1, CV_32F)) {}

    /// @brief Destructor for the LogisticRegression class.
    ~LogisticRegression() {}

    /// @brief Makes predictions on the provided data using the logistic regression model.
    /// @param trainDataWithBias The training data including the bias term.
    /// @return Predictions made by the model.
    cv::Mat predict(const cv::Mat& trainDataWithBias) 
    {
        cv::Mat exponent = trainDataWithBias * weights;
        return sigmoid(exponent);
    }

    /// @brief Trains the logistic regression model using the provided data.
    /// @param trainData Training data.
    /// @param labelData Labels for the training data.
    /// @param testData Test data used for evaluating the model.
    /// @param testLabel Labels for the test data.
    /// @param maxIterations Maximum number of iterations to perform.
    void train(const cv::Mat& trainData, const cv::Mat& labelData, const cv::Mat& testData, const cv::Mat& testLabel, int maxIterations) 
    {
        cv::Mat trainDataWithBias = calculateTrainDataWithBias(trainData);

        for (int i = 0; i < maxIterations; ++i) 
        {
            cv::Mat predictions = predict(trainDataWithBias);

            cv::Mat R = calculateR(trainDataWithBias, predictions);

            cv::Mat hessian = calculateHessian(trainDataWithBias, R);

            cv::Mat gradient = calculateGradient(trainDataWithBias, predictions, labelData);

            cv::Mat hessianInv;

            cv::invert(hessian, hessianInv, cv::DECOMP_SVD);

            weights = weights - hessianInv * gradient;

            cv::Mat testDataWithBias = calculateTrainDataWithBias(testData);

            cv::Mat testPredictions = predict(testDataWithBias);

            cv::Mat predictedTestLabels;

            cv::threshold(testPredictions, predictedTestLabels, 0.5, 1, cv::THRESH_BINARY);

            if (predictedTestLabels.rows == testLabel.rows) 
            {
                float accuracy = computeAccuracy(predictedTestLabels, testLabel);
                std::cout << "Accuracy at Iteration " << i + 1 << ": " << accuracy << std::endl;
            } else {
                std::cerr << "Mismatch in the number of rows between predicted labels and actual labels." << std::endl;
            }
        }
    }
};


int main() 
{
    DataPreprocessor preprocessor;
    FilterRows filterRowsTrain{0, 1000}, filterRowsTest{1000, 6000};

    Data data;
    data.total = cv::ml::TrainData::loadFromCSV("./mnist_test.csv", 0, 0, 1);

    data.train = preprocessor.filter(data.total, 2, 5, filterRowsTrain.startRow, filterRowsTrain.endRow);
    data.test = preprocessor.filter(data.total, 2, 5, filterRowsTest.startRow, filterRowsTest.endRow);

    Matrices train{data.train->getTrainSamples(), data.train->getTrainResponses()},
             test{data.test->getTrainSamples(), data.test->getTrainResponses()};

    preprocessor.standardizeData(train.sample);
    preprocessor.standardizeData(test.sample);

    preprocessor.checkStandardization(train.sample, "Train Data");
    preprocessor.checkStandardization(test.sample, "Test Data");

    int PCADim = 3;
    cv::PCA pcaTrain(train.sample, cv::Mat(), cv::PCA::DATA_AS_ROW, PCADim),
           pcaTest(test.sample, cv::Mat(), cv::PCA::DATA_AS_ROW, PCADim);

    cv::Mat projectedTrainSamples = pcaTrain.project(train.sample),
            projectedTestSamples = pcaTest.project(test.sample);

    std::vector<float> desiredLabels = {0, 1};
    cv::Mat binaryTrainLabel = mapLabelToBinary(train.label, desiredLabels),
            binaryTestLabel = mapLabelToBinary(test.label, desiredLabels);

    LogisticRegression model(projectedTrainSamples.cols);
    model.train(projectedTrainSamples, binaryTrainLabel, projectedTestSamples, binaryTestLabel, 8);

    return 0;
}