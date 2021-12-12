/* FILLS VECTOR WITH RANDOM VALUES */
void fill_rand(vector<float> &v, int value = 100, int range_dec = 0)
{
    for(auto &c: v)
    {
        c = (rand()%value)/100.0 + range_dec;
    }
}

/* FILLS VECTOR WITH RANDOM VALUES */
void fill_rand(vector<vector<float>> &v, int value = 1000, int range_dec = 0)
{
    for(auto &c: v)
    {
        fill_rand(c, value, range_dec);
    }
}

/* RETURNS ABSOLUTE VALUE OF INTEGER */
int __abs(int x)
{
    if(x < 0)
    {
        x *= -1;
    }
    return x;
}


/* FINDS x^n */
float __pow(float x, int n)
{
    if (__abs(n) == 0) return 1.0;
    float u = __pow(x,__abs(n)/2);
    u = (u*u);
    if (__abs(n)%2 == 1) u = (u*x);
    if(n < 0)
        u = 1.0 / u;
    return u;
}

/* TAKES A NUMBER AND RETURNS A VALUE BETWEEN 0 AND 1 INCLUSIVE */
float sigmoid(float value)
{
    if(value > 600)
    {
        value = 600;
    }
    else if(value < -600)
    {
        value = -600;
    }
    value = 1.0 / (1.0 + (__pow(1.11, -1*(int)value)));
    return value;
}

/* TAKES A NUMBER AND RETURNS THE DIFFERENTIATED VALUE OF sigmoid(..) function */
float sigmoid_diff(float value)
{
    value = sigmoid(value) * (1 - sigmoid(value));
    return value;
}

/* CLASS DENOTING LAYER OF THE NETWORK */
class layer
{
public:

    /* NODE CONTAINS ACTIVATION VALUES FOR THIS LAYER - WHICH IS BETWEEN 0 AND 1 AND IS ACHIEVED BY SQUASHING THE Z_VALUE BETWEEN SIGMOID FUNCTION*/
    vector<float> node;

    /* BIAS CONTAINS BIAS ASSOCIATED WITH THE ACTIVATION VALUE FOR THIS LAYER */
    vector<float> bias;

    /* Z_VALUE CONTAINS WEIGHTED SUM OF ACTIVATION VALUES OF PREVIOUS LAYER ASSOCIATED WITH THE ACTIVATION VALUE OF THIS LAYER WITH THE BIAS */
    vector<float> z_value;

    /* WEIGHT CONTAINS WEIGHTS ASSOCIATED WITH NODES OF NEXT LAYER FOR EACH NODE OF THIS LAYER */
    vector<vector<float>> weight;

    /* CONSTRUCTOR FOR A LAYER TYPE OBJECT WHICH INITIALIZES WEIGHTS AND BIASES WITH RANDOM VALUES */
    layer(const int &number_of_nodes = 0, const int &number_of_weights = 0)
    {
        node.resize(number_of_nodes, 0);
        bias.resize(number_of_nodes);
        z_value.resize(number_of_nodes);
        if(number_of_weights > 0)
        {
            weight.resize(number_of_nodes, vector<float> (number_of_weights));
        }
        fill_rand(bias, 1000, -5);
        fill_rand(weight, 2000, -10);
    }
};

/* CLASS CONTAINING THE WHOLE NETWORK i.e LAYERS INCLUSIVE OF INPUT AND OUTPUT LAYER */
class network
{
public:

    /* VECTOR OF LAYERS - FIRST BEING INPUT - SECOND BEING OUTPUT - REST ARE HIDDEN */
    vector<layer> __layers;

    /* CONSTRUCTOR FOR LAYERS OBJECT WHICH INITIALIZES THE LAYER WITH SIZES GIVEN BY VECTOR PASSES AS PARAMETER */
    network(vector<int> &sizes)
    {
        __layers.resize(sizes.size());
        for(int i=0; i<sizes.size()-1; i++)
        {
            __layers[i] = layer(sizes[i], sizes[i+1]);
        }
        __layers[__layers.size()-1] = layer(sizes[sizes.size()-1]);
    }

    /* INITIALIZES THE VECTOR WITH SAME STRUCTURE AS THE NETWORK */
    void set_values_initialize(vector<layer> &__layer_set_vals)
    {
        __layer_set_vals = __layers;
        for(int i=0; i<__layer_set_vals.size(); i++)
        {
            __layer_set_vals[i].node = vector<float>(__layer_set_vals[i].node.size(), 0);
            __layer_set_vals[i].z_value = vector<float>(__layer_set_vals[i].z_value.size(), 0);
            __layer_set_vals[i].bias = vector<float>(__layer_set_vals[i].bias.size(), 0);
            if(i==__layer_set_vals.size()-1)
                continue;
            __layer_set_vals[i].weight = vector<vector<float>>(__layer_set_vals[i].weight.size(), vector<float>(__layer_set_vals[i].weight[0].size(),0));
        }

    }

    /* TRAINS THE LAYER WITH BACKWARD PROPOGATION - value CONTAINS THE NUMBER ON IMAGE - __layer_set_vals CONTAINS THE GRADIENT OF COST FUNCTION FOR CURRENT BATCH */
    void train_layers(const int &value, vector<layer> &__layer_set_vals)
    {
        /* WILL CONTAIN THE GRADIENT OF COST FUNCTION FOR CURRENT TEST IMAGE */
        /* ASSUMES: NODE CONTAINS dC/dA || WEIGHT CONTAINS dC/dW || BIAS CONTAINS dC/dB || Z_VALUE CONTAINS DIFFERNTIATED VALUE OF SIGMOID FUNCTION FOR THE Z_VALUE */
        vector<layer> __layers_diff_values;

        /* INITIALIZES __layers_diff_values WITH SAME STRUCTURE AS THE NETWORK */
        set_values_initialize(__layers_diff_values);

        /* CALCULATING GRADIENT VALUES FOR LAST LAYER */
        for(int i=0; i < __layers[__layers.size()-1].node.size(); i++)
        {
            __layers_diff_values[__layers_diff_values.size()-1].node[i] = 2*(__layers[__layers.size()-1].node[i]);
            if(i==(value))
            {
                __layers_diff_values[__layers_diff_values.size()-1].node[i] -= 2;
            }
            __layers_diff_values[__layers_diff_values.size()-1].z_value[i] = sigmoid_diff(__layers[__layers.size()-1].z_value[i]);
            __layers_diff_values[__layers_diff_values.size()-1].bias[i] = __layers_diff_values[__layers_diff_values.size()-1].z_value[i] * __layers_diff_values[__layers_diff_values.size()-1].node[i];
        }

        /* BACKPROPOGATING */
        for(int i = __layers.size()-2; i>=0; i--)
        {
            for(int j=0; j<__layers[i].node.size(); j++)
            {
                float temp = 0;
                for(int k=0; k<__layers[i+1].node.size(); k++)
                {
                    temp += __layers[i].weight[j][k] * __layers_diff_values[i+1].z_value[k] * __layers_diff_values[i+1].node[k];
                    __layers_diff_values[i].weight[j][k] = __layers[i].node[j] * __layers_diff_values[i+1].z_value[k] * __layers_diff_values[i+1].node[k];
                }
                __layers_diff_values[i].node[j] = temp;
                __layers_diff_values[i].z_value[j] = sigmoid_diff(__layers[i].z_value[j]);
                __layers_diff_values[i].bias[j] = __layers_diff_values[i].z_value[j] * __layers_diff_values[i].node[j];
            }
        }

        /* SUMMING THE VALUES OF GRADIENT OF COST FUNCTION FOR THIS SET WITH VALUES FOR THE CURRENT BATCH */
        for(int i=0; i<__layer_set_vals.size(); i++)
        {
            for(int j=0; j<__layer_set_vals[i].node.size(); j++)
            {
                __layer_set_vals[i].bias[j] += __layers_diff_values[i].bias[j];
                if(i == __layer_set_vals.size()-1)
                    continue;
                for(int k=0; k<__layer_set_vals[i+1].node.size(); k++)
                {
                    __layer_set_vals[i].weight[j][k] += __layers_diff_values[i].weight[j][k];
                }
            }
        }
    }

    /* UPDATEING VALUES OF WEIGHTS AND BIASES OF THE NETWORK FROM THE GRADIENT OF COST FUNCTION */
    void set_value_w_b(vector<layer> &__layers_diff_values, const int &set_size)
    {
        for(int i=0; i<__layers.size(); i++)
        {
            for(int j=0; j<__layers[i].node.size(); j++)
            {
                __layers[i].bias[j] -= (__layers_diff_values[i].bias[j])/set_size;
                if(i == __layers.size()-1)
                    continue;
                for(int k=0; k<__layers[i+1].node.size(); k++)
                {
                    __layers[i].weight[j][k] -= (__layers_diff_values[i].weight[j][k])/set_size;
                }
            }
        }
    }

    /* UPDATING THE ACTIVATION VALUES OF EACH LAYER */
    void calculate_output()
    {
        for(int i=1; i<__layers.size(); i++)
        {
            for(int j=0; j<__layers[i].node.size(); j++)
            {
                float value = 0;
                for(int k=0; k<__layers[i-1].node.size(); k++)
                {
                    value += __layers[i-1].node[k]*__layers[i-1].weight[k][j];
                }
                value += __layers[i].bias[j];
                __layers[i].z_value[j] = value;
                value = sigmoid(value);
                __layers[i].node[j] = value;
            }
        }
    }

    /* SHOWING OUTPUT FOR THE GIVEN IMAGE - REQUIRES YOU TO CALCULATE OUTPUT FIRST */
    void show_output(int &prediction, bool show = true)
    {
        float guess = -1, guess_per = -1;
        for(int i=0; i<10; i++)
        {
            if(__layers[__layers.size()-1].node[i] > guess_per)
            {
                guess_per = __layers[__layers.size()-1].node[i];
                guess = i;
            }
        }

        prediction = guess;
        if(show)
            cout << "Guessed output is = " << guess << " with confidence of " << guess_per * 100 << "%\n";
    }
};
