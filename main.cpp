#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>

struct Data
{
    cv::Ptr<cv::ml::TrainData> total;
    cv::Ptr<cv::ml::TrainData> train;
    cv::Ptr<cv::ml::TrainData> test;
};

struct Matrices
{
    cv::Mat sample, label;
};

struct FilterRows
{
    int startRow, endRow;
};

float computeAccuracy(const cv::Mat& predicted, const cv::Mat& actual) {
    int correct = 0;
    for (int i = 0; i < predicted.rows; ++i) {
        // Rundet die Vorhersage auf 0 oder 1, basierend auf einem Schwellenwert von 0.5
        int predLabel = predicted.at<float>(i, 0) >= 0.5 ? 1 : 0;
        int trueLabel = static_cast<int>(actual.at<float>(i, 0));
        if (predLabel == trueLabel) {
            correct++;
        }
    }
    return static_cast<float>(correct) / predicted.rows;
}


cv::Mat mapLabelToBinary(const cv::Mat& originalLabel, const std::vector<float>& desiredLabels) {
    cv::Mat binaryLabels = originalLabel.clone();

    for(int i = 0; i < binaryLabels.rows; i++) 
    {
        float& label = binaryLabels.at<float>(i, 0);
        if (label == 2) {
            label = desiredLabels.at(0);
        } 
        else if (label == 5) 
        {
            label = desiredLabels.at(1);
        }
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

    //anders schreiben !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    cv::Ptr<cv::ml::TrainData> filter(const cv::Ptr<cv::ml::TrainData> &originalData, int digit1, int digit2, int startRow, int endRow)
    {
        startRow = std::max(0, startRow);
        endRow = std::min(endRow, originalData->getNSamples()); 

        cv::Mat samples = originalData->getSamples().rowRange(startRow, endRow);
        cv::Mat responses = originalData->getResponses().rowRange(startRow, endRow);

        cv::Mat mask = (responses == digit1) | (responses == digit2);

        cv::Mat filteredSamples, filteredResponses;

        for (size_t i = 0; i < mask.total(); ++i)
        {
            if (mask.at<uint8_t>(i))
            {
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

    void standardizeData(cv::Mat& data) {
        // Sicherstellen, dass die Daten im richtigen Format sind
        data.convertTo(data, CV_32F);

        cv::Scalar mean, stddev;
        // Mittelwert und Standardabweichung für die aktuelle Spalte berechnen
        cv::meanStdDev(data, mean, stddev);

        // Prüfen, ob die Standardabweichung zu klein ist, um Division durch Null zu vermeiden
        if (stddev[0] != 0) 
        {
            // Standardisierung: (x - Mittelwert) / Standardabweichung für jedes Element in der Spalte
            data = (data - mean[0]) / stddev[0];
        }
    }

    void checkStandardization(const cv::Mat& data, const std::string& name) 
    {
        cv::Scalar mean, stdDev;
        cv::meanStdDev(data, mean, stdDev); // Calculate mean and standard deviation for the training samples

        std::cout<< name << " Train Mean: " << mean[0] << " Train Stddev: " << stdDev[0] << std::endl;
    }

    void writeCsvFile(const cv::Mat& projectedData, const std::string& filepath)
    {
        std::ofstream outputFile(filepath);
        outputFile << "Component1,Component2,Component3,Label\n";

        for (int i = 0; i < projectedData.rows; i++) {
            for (int j = 0; j < projectedData.cols; j++) {
                outputFile << projectedData.at<float>(i, j) << ",";
            }
            outputFile << label.at<float>(i, 0) << "\n"; 
        }

        outputFile.close();
        std::cout << "PCA-reduced data and labels have been written to 'pca_reduced_data.csv'" << std::endl;
    }

    cv::Mat getTrainSamples(){ return train.sample; }
    
    cv::Mat getTestSamples(){ return train.sample; }

    cv::Mat getTrainLabels(){ return train.label; }
    
    cv::Mat getTestLabels(){ return test.label; }

    void loadDataset(const std::string& csvPath, const int& filterDigit1, const int& filterDigit2, const FilterRows& filterRowsTrain, const FilterRows& filterRowsTest) 
    {
        std::cout << "Loading MNIST Test Dataset from " << csvPath << "..." << std::endl;

        data.total = cv::ml::TrainData::loadFromCSV(csvPath, 0, 0, 1);

        data.train = filter(data.total, filterDigit1, filterDigit2, filterRowsTrain.startRow, filterRowsTrain.endRow);
        data.test = filter(data.total, filterDigit1, filterDigit2, filterRowsTest.startRow, filterRowsTest.endRow);

        if (data.train->getSamples().empty()) 
        {
            std::cerr << "Train data is empty after filtering." << std::endl;
        } 
        else 
        {
            std::cout << "Train Samples Rows: " << data.train->getSamples().rows << std::endl;
            std::cout << "Test Samples Rows: " << data.test->getSamples().rows << std::endl;

            train.sample = data.train->getTrainSamples();
            train.label = data.train->getTrainResponses();
            test.sample = data.test->getTrainSamples();
            test.label = data.test->getTrainResponses();
        }
    }

};

class LogisticRegression
{
    private:
        cv::Mat weights;

        cv::Mat calculateTrainDataWithBias(const cv::Mat& trainData)
        {
            cv::Mat trainDataWithBias;
            cv::hconcat(cv::Mat::ones(trainData.rows, 1, trainData.type()), trainData, trainDataWithBias);
            return trainDataWithBias;
        }

        cv::Mat sigmoid(const cv::Mat& z) 
        {
            cv::Mat output;
            cv::exp(-z, output);
            output = 1.0 / (1.0 + output);
            return output;
        }


        cv::Mat calculateR(const cv::Mat& trainDataWithBias)
        {
            cv::Mat R = cv::Mat::zeros(trainDataWithBias.rows, trainDataWithBias.rows, CV_32F);

            for (int i = 0; i < trainDataWithBias.rows; ++i) 
            {
                double p = trainDataWithBias.at<float>(i, 0);
                R.at<float>(i, i) = p * (1 - p); // y_n (1 - y_n)
            }
            return R;
        }

        cv::Mat calculateHessian(const cv::Mat& trainDataWithBias, const cv::Mat& R)
        {
            return trainDataWithBias.t() * R * trainDataWithBias;
        }

        cv::Mat calculateGradient(const cv::Mat& trainDataWithBias, const cv::Mat& predictions, const cv::Mat& labels)
        {
            cv::Mat errors = predictions - labels;
            return trainDataWithBias.t() * errors;  // ∇E(w) = Φ^T (y - t)
        }

    public:
        LogisticRegression(int quantityRows): weights(cv::Mat::zeros(quantityRows + 1, 1, CV_32F)) {
            std::cout << "Initial weights (including bias): " <<std::endl<< weights << std::endl;
        }

        ~LogisticRegression(){};
        
        cv::Mat predict(const cv::Mat& trainDataWithBias) 
        {
            cv::Mat exponent = trainDataWithBias * weights;
            return sigmoid(exponent);
        }

        void train(const cv::Mat& trainData, const cv::Mat& labelData, int maxIterations = 10)
        {
            cv::Mat trainDataWithBias = calculateTrainDataWithBias(trainData);

            for(int i=0; i<maxIterations; i++)
            {
                cv::Mat R = calculateR(trainDataWithBias);
                cv::Mat predictions = predict(trainDataWithBias);
                cv::Mat hessian = calculateHessian(trainDataWithBias, R);
                cv::Mat gradient = calculateGradient(trainDataWithBias, predictions, labelData);
                //std::cout <<"Hessian: "<<std::endl<<hessian<<std::endl;

                cv::Mat hessianInv;
                cv::invert(hessian, hessianInv, cv::DECOMP_SVD);

                weights = weights - hessianInv * gradient;

                std::cout << "Iteration " << i + 1 << " complete. Weights updated." <<std::endl<<weights<< std::endl;
            }
        }

};




int main() 
{
    /*
    Preproccessing:
        Extract Data from dataset with the matrikel number digits
        Standardize dataset
        PCA

    Training:
        Train
        Predict
        Plot Accuracy
        Select iteration quantity
    
    Formatting:


    */


    DataPreprocessor preprocessor;

    FilterRows filterRowsTrain, filterRowsTest;
    filterRowsTrain.startRow = 0; filterRowsTrain.endRow = 1000; filterRowsTest.startRow = 1000; filterRowsTest.endRow = 6000;

    preprocessor.loadDataset("./mnist_test.csv", 2, 5, filterRowsTrain, filterRowsTest);

    //preprocessor.filterData(1, 3);

    Matrices train, test;
    
    train.sample = preprocessor.getTrainSamples();
    train.label = preprocessor.getTrainLabels();
    test.sample = preprocessor.getTestSamples();
    test.label = preprocessor.getTestLabels();
    
    //preprocessor.displayImagesInTerminal(10, data, label);
    

    /*
    cv::namedWindow("data", 1);
    cv::imshow("data", data);
    cv::waitKey(0);
    */ 

    preprocessor.standardizeData(train.sample);
    preprocessor.standardizeData(test.sample);

    preprocessor.checkStandardization(train.sample, "Train Data");
    preprocessor.checkStandardization(test.sample, "Test Data");



    int PCADim = 3; // Beispiel: Reduzieren auf 50 Dimensionen
    cv::PCA pcaTrain(train.sample, cv::Mat(), cv::PCA::DATA_AS_ROW, PCADim);
    cv::PCA pcaTest(test.sample, cv::Mat(), cv::PCA::DATA_AS_ROW, PCADim);

    // Projizieren Sie die Daten auf die PCA-Hauptkomponenten
    cv::Mat projectedTrainSamples = pcaTrain.project(train.sample);
    cv::Mat projectedTestSamples = pcaTest.project(test.sample);


    std::cout<<"projectedTrainSamples.cols: "<<projectedTrainSamples.cols<<std::endl;
    std::cout<<"projectedTrainSamples.rows: "<<projectedTrainSamples.rows<<std::endl;

    std::cout<<"projectedTestSamples.cols: "<<projectedTestSamples.cols<<std::endl;
    std::cout<<"projectedTestSamples.rows: "<<projectedTestSamples.rows<<std::endl;

    //preprocessor.writeCsvFile(projectedData, "./pca_reduced_data.csv");

    //Map Target Labels of the Train and Test Dataset from 2 to 0 and from 5 to 1
    std::vector<float> desiredLabels = {0, 1}; // Korrekte Initialisierung
    cv::Mat binaryTrainLabel = mapLabelToBinary(train.label, desiredLabels);
    cv::Mat binaryTestLabel = mapLabelToBinary(test.label, desiredLabels);
    
    LogisticRegression model(projectedTrainSamples.cols);

    model.train(projectedTrainSamples, binaryTrainLabel, 100);





    return 0;
}
