# Cat Identification Neural Network

Cat-Identification-NN is a simple neural network built using logistic regression that classifies whether an image contains a cat or not. It currently achieves an accuracy of approximately 70% on a basic dataset. This project serves as a foundational step toward building more complex image classification systems.


## Detailed Overview
This Project includes a CLI interface for:
a. Training the model using .h5 datasets.
b. Testing any custom image from a folder.
There are two datasets currently, train_catvnoncat.h5 and test_catvnoncat.h5 to train and test the neural network
The trained params "w" (weight) and "b" (bias) are stored in `model_params.npz`
#### How to Use
1. Clone the repo

   ```
   git clone https://github.com/silliKate/Cat-Identification-NN
   ```
   
2. Install the requirements

   ```
   pip install -r requirements.txt
   ```
   
3. Place the images in the `images/` folder
2. Run the program using
   
   ```
   python main.py
   ```

## Future Goals
### Short-Term:
1. Add more nodes and hidden layers to improve performance
2. Expand and balance the training dataset for better generalization
3. Deploy the model via a web interface where users can upload an image and get predictions

### Long-Term:
1. Extend classification to multiple animal species (and possibly plants too!)
2. Integrate with an API to fetch factual information about the identified animal
3. Add support fro AR and Android/iOS


## Contributing
Currently a personal learning project. Contributions, suggestions, and improvements are welcome via issues or pull requests.
