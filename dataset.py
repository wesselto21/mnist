x_train = "/Users/wesselto/Desktop/GitHub/MNIST/train-images-idx3-ubyte"
y_train = "/Users/wesselto/Desktop/GitHub/MNIST/train-labels-idx1-ubyte"
x_test = "/Users/wesselto/Desktop/GitHub/MNIST/t10k-images-idx3-ubyte"
y_test = "/Users/wesselto/Desktop/GitHub/MNIST/t10k-labels-idx1-ubyte"

def convert(imgs, labels, outline, n):
    imgf = open(imgs, "rb")
    labf = open(labels, "rb")
    outf = open(outline, "w")
    imgf.read(16)
    labf.read(8)
    images = []
    for i in range(n):
        image = [ord(labf.read(1))]
        for j in range(28*28):
            image.append(ord(imgf.read(1)))
        images.append(image)
    
    for image in images:
        outf.write(",".join(str(pixel) for pixel in image) + "\n")
    outf.close()
    imgf.close()
    labf.close()

convert(x_train, y_train, "/Users/wesselto/Desktop/GitHub/MNIST/train.csv", 60000)
convert(x_test, y_test, "/Users/wesselto/Desktop/GitHub/MNIST/test.csv", 10000)
