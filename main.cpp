#include<iostream>
#include<fstream>
#include<vector>

using namespace std;

#include"mnist_reader.h"
#include"network.h"

/* ENTER LOCATION OF MNIST TRAINING AND TEST DATA */
#define training_set_location "NULL"
#define training_label_location "NULL"
#define test_set_location "NULL"
#define test_label_location "NULL"

#define training_set_size 60000
#define test_set_size 10000

int main()
{
    cout << "******************************************\nDigit Recogniser - Created by Sayan Bakshi\n******************************************\n\n";

    /* PAIR.FIRST WILL BE STORED WITH THE NUMBER ON THE IMAGE AND PAIR.SECOND CONTAINS MATRIX WHICH WILL BE STORED WITH VALUES OF BRIGHTNESS OF EACH PIXEL OF THE IMAGE */
    vector<pair<int,vector<vector<float>>>> dataset;

    /* READS TRAINING DATA FROM THE MNIST DATASET AND STORES THE VALUES INTO THE VECTOR */
    read_mnist(training_set_location, training_label_location, training_set_size, dataset);

    /* SET SIZE WILL CONTAIN SIZE OF THE BATCH AND NUMBER OF SET CONTAINS TOTAL NUMBER OF BATCHES */
    int set_size;
    int number_of_set;

    cout << "Enter the size of batch you want to train with ( recommended 15 ) : ";
    cin >> set_size;

    cout << "Enter the number of batches you want to train with ( max is " << training_set_size / set_size << " ) : ";
    cin >> number_of_set;

    if(number_of_set > training_set_size / set_size)
    {
        cout << "Max possible value exceeded! Value set to : " << training_set_size / set_size << "\n";
        number_of_set = training_set_size / set_size;
    }

    /* NUMBER OF HIDDEN LAYERS AND NEURONS PER LAYER */
    int hidden_layers;
    int neurons_per_layer;

    cout << "Enter the number of hidden layers ( recommended 2 ) : ";
    cin >> hidden_layers;

    cout << "Enter the number of neurons per layer ( recommended 20 ) : ";
    cin >> neurons_per_layer;

    /* CONTAINS NUMBER OF NEURONS AND NUMBER OF LAYERS FOR THE NETWORK - INCLUSIVE OF OUTPUT, INPUT AND HIDDEN LAYER */
    vector<int> sizes( 2 + hidden_layers, neurons_per_layer );

    /* INITIALIZING OUPUT (LAST LAYER) AND INPUT LAYER (FIRST LAYER) */
    sizes[0] = dataset[0].second.size() * dataset[0].second[0].size();
    sizes[sizes.size()-1] = 10;

    /* CREATES THE NETWORK */
    network model(sizes);

    cout << "\n\n*****************\nTraining Started\n*****************\n\n";

    for(int l = 0; l < number_of_set; l++)
    {
        /* WILL CONTAIN GRADIENT OF COST FUNCTION FOR CURRENT BATCH */
        vector<layer> __layer_set_vals;

        /* __layer_set_vals IS INITIALIZED WITH THE STRUCTURE OF NETWORK */
        model.set_values_initialize(__layer_set_vals);

        /* FOR EACH BATCH, IT CALCULATES THE GRADIENT OF COST FUNCTION AND STORES IT IN __layer_set_vals */
        for(int m = 0; m < set_size; m++)
        {
            int i = l * set_size + m;
            for(int r = 0; r < dataset[i].second.size(); r++)
            {
                for(int c = 0; c < dataset[i].second[r].size(); c++)
                {
                    model.__layers[0].node[r * dataset[i].second[r].size() + c] = dataset[i].second[r][c];
                }
            }
            model.calculate_output();
            model.train_layers(dataset[i].first, __layer_set_vals);

            cout << "Under training... [ " << i + 1 << " / " <<  number_of_set * set_size << " ]\n";
        }

        /* UPDATES THE VALUES OF WEIGHTS AND BIASES AFTER TRAINGING OF BATCH IS OVER */
        model.set_value_w_b(__layer_set_vals, set_size);
    }

    cout << "\n\n*****************\nTraining Over\n*****************\n\n";

    /* READS TEST DATA FROM THE MNIST DATASET AND STORES THE VALUES INTO THE VECTOR */
    read_mnist(test_set_location, test_label_location, test_set_size, dataset);

    /* STORES CURRENT TEST NUMBER, PREDICTION FOR CURRENT TEST AND TOTAL CORRECT PREDICTIONS THIS FAR */
    float correct_predictions = 0;
    int prediction, test_no = 0;

    cout << "\n\n*****************\nTesting Started\n*****************\n\n";

    /* TESTING THE NETWORK */
    while(test_no < dataset.size())
    {
        int c;

        cout << "Enter 1 to test on test number " << test_no + 1 << " \nEnter 2 to test remaining images at once. \nEnter -1 to exit. \n";
        cin >> c;

        if(c == -1)
        {
            cout << "\n\nExited!\n";
            exit(0);
        }
        else if(c == 1)
        {
            /* TESTING FOR THE GIVEN TEST ON THE NETWORK */
            for(int r = 0; r < dataset[test_no].second.size(); r++)
            {
                for(int c = 0; c < dataset[test_no].second[r].size(); c++)
                {
                    model.__layers[0].node[r * dataset[test_no].second[r].size() + c] = dataset[test_no].second[r][c];
                    if(dataset[test_no].second[r][c] > 0.7)
                        cout << "o";
                    else if(dataset[test_no].second[r][c] > 0.3)
                        cout << ".";
                    else
                        cout << " ";
                }
                cout << "\n";
            }
            cout << "\n";
            model.calculate_output();
            model.show_output(prediction);

            cout << "Actual value = " << dataset[test_no].first << "\n\n";

            if(prediction == dataset[test_no].first)
            {
                correct_predictions++;
            }
        }
        else if(c == 2)
        {
            cout << "Testing remaining dataset\n";

            for(test_no; test_no < dataset.size(); test_no++)
            {
                /* TESTING FOR THE GIVEN TEST ON THE NETWORK */
                for(int r = 0; r < dataset[test_no].second.size(); r++)
                {
                    for(int c = 0; c < dataset[test_no].second[r].size(); c++)
                    {
                        model.__layers[0].node[r * dataset[test_no].second[r].size() + c] = dataset[test_no].second[r][c];
                    }
                }
                model.calculate_output();
                model.show_output(prediction, false);

                if(prediction == dataset[test_no].first)
                {
                    correct_predictions++;
                }
                cout << "Under testing... [ " << test_no + 1 << " / " <<  dataset.size() << " ]\n";
            }
        }
        else
        {
            cout << "Invalid entry!\n";
            continue;
        }
        test_no++;
    }

    cout << "\n\n*****************\nTesting Over\n*****************\n\n";

    cout << "Accuracy of model is = " << (correct_predictions / dataset.size()) * 100 <<"%\n";
}
