import numpy as np
import time
import matplotlib.pyplot as plt

class DNN:
    def __init__(self, sizes=[784, 128, 64, 10], epochs=10, lr=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.lr = lr

        input_layer = sizes[0]
        hidden_layer1 = sizes[1]
        hidden_layer2 = sizes[2]
        output_layer = sizes[3]

        self.params = {
            "W1": np.random.randn(hidden_layer1, input_layer)*np.sqrt(1/hidden_layer1),#128*784
            "W2": np.random.randn(hidden_layer2, hidden_layer1)*np.sqrt(1/hidden_layer2),#64*128
            "W3": np.random.randn(output_layer, hidden_layer2)*np.sqrt(1/output_layer),#10*64
        }

    def sigmoid(self, x, derivative=False):
        if derivative:
            return np.exp(-x)/((1+np.exp(-x))**2)
        return 1/(1+np.exp(-x))

    def softmax(self, x, derivative=False):
        exps = np.exp(x-x.max())
        if derivative:
            return exps/np.sum(exps, axis=0)*(1-exps/np.sum(exps, axis=0))
        return exps/np.sum(exps, axis=0)

    def forward_pass(self, x_train):
        params = self.params
        params["A0"] = x_train

        #input layer to hidden layer 1
        params["Z1"] = np.dot(params["W1"], params["A0"])
        params["A1"] = self.sigmoid(params["Z1"])

        #hidden layer 1 to hidden layer 2
        params["Z2"] = np.dot(params["W2"], params["A1"])
        params["A2"] = self.sigmoid(params["Z2"])

        #hidden layer 2 to output layer
        params["Z3"] = np.dot(params["W3"], params["A2"])
        params["A3"] = self.softmax(params["Z3"])

        return params["Z3"]

    
    def backward_pass(self, y_train, output):
        params = self.params

        change_w = {}

        #calculate W3 update
        error = 2*(output-y_train) /output.shape[0] * self.softmax(params["Z3"], derivative=True)
        change_w["W3"] = np.outer(error, params["A2"])

        #calculate W2 update
        error = np.dot(params["W3"].T, error) * self.sigmoid(params["Z2"], derivative=True)
        change_w["W2"] = np.outer(error, params["A1"])

        #calculate W1 update
        error = np.dot(params["W2"].T, error) * self.sigmoid(params["Z1"], derivative=True)
        change_w["W1"] = np.outer(error, params["A0"])

        return change_w

    def update_weights(self, change_w):
        for key, value in change_w.items():
            self.params[key] -= self.lr * value #W_t+1=W_t-lr*change_w

            
    def train(self, train_list, test_list):
        start_time = time.time()
        accuracies = []  # Liste für Accuracy-Werte
        
        for i in range(self.epochs):
            for x in train_list:
                value = x.split(",")
                input = np.asfarray(value[1:])/255.0*0.99+0.01
                target = np.zeros(10)+0.01
                target[int(value[0])] = 0.99
                output = self.forward_pass(input)
                change_w = self.backward_pass(target, output)
                self.update_weights(change_w)

            accuracy = self.compute_accuracy(test_list)
            accuracies.append(accuracy)  # Accuracy speichern
            print(f"Epoch {i+1:2d}, Time: {time.time()-start_time:.2f}s, Accuracy: {accuracy:.2%}")

        end_time = time.time()
        print(f"Training time: {end_time-start_time:.2f} seconds")
        
        # Trainingskurve plotten
        self.plot_training_curve(accuracies)


    
    def compute_accuracy(self, test_data):
        predictions = []
        for x in test_data:
            values = x.split(",")
            inputs = np.asfarray(values[1:])/255.0*0.99+0.01
            targets = np.zeros(10)+0.01
            targets[int(values[0])] = 0.99
            output = self.forward_pass(inputs)
            pred = np.argmax(output)
            predictions.append(pred==np.argmax(targets))
        
        return np.mean(predictions)
    
    def plot_training_curve(self, accuracies):
        """Plottet die Trainingskurve (Accuracy über Epochs)"""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(accuracies) + 1)
        
        plt.plot(epochs, accuracies, 'b-o', linewidth=2, markersize=6)
        plt.title('Training Accuracy over Epochs', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Formatierung der y-Achse als Prozent
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Beste Accuracy markieren
        best_epoch = np.argmax(accuracies) + 1
        best_accuracy = max(accuracies)
        plt.annotate(f'Best: {best_accuracy:.2%} (Epoch {best_epoch})', 
                    xy=(best_epoch, best_accuracy), 
                    xytext=(best_epoch + 1, best_accuracy - 0.1),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10, color='red', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

    def save_model(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.params, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            self.params = pickle.load(f)
        print(f"Model loaded from {filename}")

if __name__ == "__main__":
    train_list = open("/Users/wesselto/Desktop/GitHub/MNIST/train.csv", "r").readlines()
    test_list = open("/Users/wesselto/Desktop/GitHub/MNIST/test.csv", "r").readlines()

    dnn = DNN(sizes=[784, 128, 64, 10], epochs=10, lr=0.001)
    dnn.train(train_list, test_list)

    # Save the trained model
    dnn.save_model("trained_dnn.pkl")


