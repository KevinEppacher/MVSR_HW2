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


class DataPreprocessor {
public:
    DataPreprocessor(){};

    DataPreprocessor(const std::string& csvPath)
    {
        loadDataset(csvPath);
    }

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

private:
    cv::Mat samples;
    cv::Mat label;
    cv::Mat filteredDigits;
    cv::Mat filteredLabel;
    Matrices train, test;
    Data data;

    void loadDataset(const std::string& csvPath) 
    {
        std::cout << "Loading MNIST Test Dataset from " << csvPath << "..." << std::endl;

        data.total = cv::ml::TrainData::loadFromCSV(csvPath, 0, 0, 1);

        if (data.total->getSamples().empty()) 
        {
            std::cerr << "data.total is empty after filtering." << std::endl;
        } 

        data.train = filter(data.total, 1, 3, 0, 5000);
        data.test = filter(data.total, 1, 3, 5000, 10000);

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
            std::cout<<"geh scheißen"<<std::endl;

        }



    }

};

class LogisticRegression
{
    private:

    public:
        LogisticRegression(){};
        ~LogisticRegression(){};
        
        
        
        void train()
        {

        }

        void predict()
        {

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


    DataPreprocessor preprocessor("./mnist_test.csv");

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


    std::cout<<"projectedData.cols"<<projectedTrainSamples.cols<<std::endl;
    std::cout<<"projectedData.rows"<<projectedTrainSamples.rows<<std::endl;

    std::cout<<"projectedData.cols"<<projectedTestSamples.cols<<std::endl;
    std::cout<<"projectedData.rows"<<projectedTestSamples.rows<<std::endl;

    //preprocessor.writeCsvFile(projectedData, "./pca_reduced_data.csv");













    return 0;
}
