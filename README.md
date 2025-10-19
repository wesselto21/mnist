# 🧠 MNIST Neural Network  

A simple **Deep Neural Network (DNN)** built from scratch to classify handwritten digits from the **MNIST dataset**.  

---

## 📦 Setup  

### 1. Download the MNIST Dataset  
Use the following commands in your terminal (macOS):  

```bash
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
```

### 2. Unzip the Files  

```bash
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
```

### 3. Convert Dataset to CSV  
Run the following script to convert the binary MNIST files into CSV format:  

```bash
python dataset.py
```

---

## 📊 Visualization  

Run the plotting script to visualize example digits from the dataset:  

```bash
python plot.py
```

---

## 🧩 Deep Neural Network Architecture (`dnn.py`)  

| Layer              | Nodes | Description |
|--------------------|--------|--------------|
| Input Layer        | 784    | Flattened 28×28 pixel images |
| Hidden Layer 1     | 128    | ReLU activation |
| Hidden Layer 2     | 64     | ReLU activation |
| Output Layer       | 10     | Softmax output (digits 0–9) |

---

## 🚀 Training  

Train the neural network by running:  

```bash
python dnn.py
```

After training, a **training curve** will be plotted automatically showing loss and accuracy progression.

---

## 🧪 Testing  

Once your model is trained and saved, you can test it on individual images:  

```bash
python test.py
```

Or visualize predictions interactively:  

```bash
python plot.py
```

> 💡 Tip: Change the `image_index` variable inside `plot.py` to test different images from the dataset.

---

## 🧰 Requirements  

- Python 3.x  
- NumPy  
- Matplotlib  

---

## 📚 Description  

This project demonstrates a minimal implementation of a **feedforward deep neural network** trained on the **MNIST handwritten digit dataset** using only **NumPy** — no deep learning frameworks required.  

It’s designed for educational purposes to help understand:  
- How data flows through layers  
- How forward and backward propagation work  
- How gradient descent updates weights  
- How to visualize model performance  

---

