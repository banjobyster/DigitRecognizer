/* MNIST viewer, self+(code by mrgloom||https://stackoverflow.com/a/10409376/14728221) */

/* INTEGER IN FILE IS STORED IN MSB FIRST FORMAT THUS IS CONVERTED FIRST BEFORE USING */
int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

/* READS CHARACTER FROM FILE INTO THE VARIABLE */
template<typename T>
void file_read(ifstream &file, T &var)
{
    file.read((char*)&var,sizeof(var));
}

void rev_read(ifstream &file, int &var)
{
    file_read(file,var);
    var = reverseInt(var);
}

/* READS MNIST DATABASE. PAIR.FIRST WILL BE STORED WITH THE NUMBER ON THE IMAGE AND PAIR.SECOND CONTAINS MATRIX WILL BE STORED WITH VALUES OF BRIGHTNESS OF EACH PIXEL OF THE IMAGE */
void read_mnist(string t_set, string t_label, int number_of_tests, vector<pair<int,vector<vector<float>>>> &v)
{
    /* OPENS FILE */
    ifstream file, label;
    file.open(t_set,ios::binary);
    label.open(t_label,ios::binary);

    /* READS THE FILE AND UPDATES THE VECTOR v WITH VALUES */
    if (file.is_open() && label.is_open())
    {
        int magic_number=0, number_of_images=0, n_rows=0, n_cols=0;
        int l_magic_number=0, number_of_labels=0;

        rev_read(file,magic_number);
        rev_read(label,l_magic_number);
        rev_read(file,number_of_images);
        rev_read(label,number_of_labels);
        rev_read(file,n_rows);
        rev_read(file,n_cols);

        if(number_of_tests>number_of_images)
        {
            number_of_tests = number_of_images;
        }

        v.resize(number_of_tests);

        for(int i=0; i<number_of_tests; ++i)
        {
            unsigned char number;
            file_read(label,number);

            vector<vector<float>> img(n_rows,vector<float>(n_cols));
            for(int r=0; r<n_rows; ++r)
            {
                for(int c=0; c<n_cols; ++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    float t=temp/255.0;
                    img[r][c]=t;
                }
            }
            v[i]= {number,img};
        }
    }
    else
    {
        cout << "Could not open file!\n";
        exit(0);
    }
}
