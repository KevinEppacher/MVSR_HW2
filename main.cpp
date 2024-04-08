#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

class DataPreprocessor {
public:
    DataPreprocessor(){};
    DataPreprocessor(const std::string& csvPath)
    {
        loadDataset(csvPath);
    }

    void filterDigits(int digit1, int digit2)
    {
        std::vector<int> indices;

        for (int i = 0; i < label.rows; ++i) 
        {
            float currentLabel = label.at<float>(i, 0);
            if (currentLabel == digit1 || currentLabel == digit2) 
            {
                indices.push_back(i);
            }
        }

        filteredDigits = cv::Mat(indices.size(), samples.cols, samples.type());
        filteredLabel = cv::Mat(indices.size(), 1, label.type());

        for (size_t i = 0; i < indices.size(); ++i) 
        {
            samples.row(indices[i]).copyTo(filteredDigits.row(i));
            label.row(indices[i]).copyTo(filteredLabel.row(i));
        }

        std::cout << "Filtered dataset to include only digits " << digit1 << " and " << digit2 << "." << std::endl;
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

    void standardizeData(cv::Mat& data) 
    {
        // Mittelwert und Standardabweichung für jede Spalte berechnen
        for (int i = 0; i < data.cols; i++) 
        {
            cv::Scalar mean, stddev;
            //cv::meanStdDev (InputArray src, OutputArray mean, OutputArray stddev, InputArray mask=noArray())
            cv::meanStdDev(data.col(i), mean, stddev);

            // Prüfen, ob die Standardabweichung zu klein ist, um Division durch Null zu vermeiden
            if (stddev[0] < 1e-6) stddev[0] = 1;

            // Standardisierung: (x - Mittelwert) / Standardabweichung
            data.col(i) = (data.col(i) - mean[0]) / stddev[0];
        }
    }

    void checkStandardization(const cv::Mat& data) 
    {
        for (int i = 0; i < data.cols; i++) 
        {
            cv::Scalar mean, stddev;
            cv::meanStdDev(data.col(i), mean, stddev);

            std::cout << "Feature " << i << ": Mean = " << mean[0] << ", StdDev = " << stddev[0] << std::endl;

            // Überprüfen, ob die Werte den Erwartungen entsprechen
            if (std::abs(mean[0]) > 1e-3 || (stddev[0] < 0.9 || stddev[0] > 1.1)) 
            {
                std::cout << "Warnung: Feature " << i << " ist möglicherweise nicht korrekt standardisiert." << std::endl;
            }
        }
    }

    

private:
    cv::Mat samples;
    cv::Mat label;
    cv::Mat filteredDigits;
    cv::Mat filteredLabel;

    void loadDataset(const std::string& csvPath) 
    {
        std::cout << "Loading MNIST Test Dataset from " << csvPath << "..." << std::endl;
        cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::loadFromCSV(csvPath, 0, 0, 1);
        samples = tdata->getTrainSamples();
        label = tdata->getTrainResponses();
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

    preprocessor.filterDigits(1, 3);

    cv::Mat data = preprocessor.getFilteredDigits();

    preprocessor.standardizeData(data);

    preprocessor.checkStandardization(data);

    //cv::Mat label = preprocessor.getFilteredLabel();

    //preprocessor.displayImagesInTerminal(10, data, label);


    return 0;
}
