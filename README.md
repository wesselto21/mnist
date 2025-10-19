# mnist


1. Download Datset via terminal cmd

MAC:
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

2. Unzip the files via terminal cmd

gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz

3. Run dataset.py to convert into csv format

4. Run plot.py for graphical output


Deep Neural Network Architecture in dnn.py

Input Layer: 784 Nodes

First Hidden Layer: 128 Nodes

Second Hidden Layer: 64 Nodes

Output Layer: 10 Nodes (Numbers 0 to 9)

5. Run dnn.py to train the Neural Network

After finishing a Traning curve will be plotted

6. Test Trained Model in test.py:
    Run dnn.py once to train and save the model
    Run plot.py as many times as you want to test different images
    Change image_index in plot.py to test different images
