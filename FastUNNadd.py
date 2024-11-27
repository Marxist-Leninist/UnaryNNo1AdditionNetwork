# Import necessary libraries
import random
import time

# UnaryNumber class with zero representation
class UnaryNumber:
    def __init__(self, count):
        self.count = int(count)  # Ensure count is an integer

    def __str__(self):
        return '1' * self.count if self.count > 0 else '0'

    def __repr__(self):
        return f"UnaryNumber({self.count})"

    def __eq__(self, other):
        return self.count == other.count

# Unary arithmetic operations
def unary_add(a, b):
    return UnaryNumber(a.count + b.count)

def unary_subtract(a, b):
    return UnaryNumber(max(a.count - b.count, 0))  # Ensure non-negative counts

def unary_multiply(a, b):
    # In unary, multiplication is repeated addition, but we'll cap it to prevent large counts
    max_count = 1000  # Adjust as needed
    return UnaryNumber(min(a.count * b.count, max_count))

# Neuron class for addition with unary weights and biases
class UnaryNeuron:
    def __init__(self):
        # Initialize weights and bias as unary numbers with counts
        self.weights = [UnaryNumber(random.randint(0, 2)) for _ in range(2)]
        self.bias = UnaryNumber(random.randint(0, 2))

    def forward(self, inputs):
        # Compute weighted sum using unary addition and multiplication
        weighted_sum = UnaryNumber(0)
        for w, i in zip(self.weights, inputs):
            product = unary_multiply(w, i)
            weighted_sum = unary_add(weighted_sum, product)
        # Add bias
        weighted_sum = unary_add(weighted_sum, self.bias)
        return weighted_sum

    def predict(self, inputs):
        output = self.forward(inputs)
        return output

    def train(self, inputs, expected_output):
        # Forward pass
        actual_output = self.forward(inputs)
        error = expected_output.count - actual_output.count

        # Update weights and bias using integer increments
        for idx, input_value in enumerate(inputs):
            if input_value.count > 0:
                # Calculate integer adjustment
                if error > 0:
                    delta = UnaryNumber(1)
                elif error < 0:
                    delta = UnaryNumber(-1)
                else:
                    delta = UnaryNumber(0)

                # Update weight
                self.weights[idx].count += delta.count
                # Ensure non-negative weights
                self.weights[idx].count = max(self.weights[idx].count, 0)

        # Update bias
        if error > 0:
            self.bias.count += 1
        elif error < 0:
            self.bias.count -= 1
        # Ensure non-negative bias
        self.bias.count = max(self.bias.count, 0)

        # Cap weights and bias to prevent them from growing too large
        self.cap_parameters()

    def cap_parameters(self, max_value=20):  # Adjust as needed
        # Cap weights and bias to prevent counts from growing too large
        for w in self.weights:
            w.count = min(w.count, max_value)
        self.bias.count = min(self.bias.count, max_value)

# Prepare dataset including zero
def prepare_dataset(max_value):
    dataset = []
    for a in range(0, max_value + 1):  # Including zero up to max_value
        for b in range(0, max_value + 1):
            inputs = [UnaryNumber(a), UnaryNumber(b)]
            output = UnaryNumber(a + b)
            dataset.append((inputs, output))
    return dataset

# Training function with timer
def train_neuron(neuron, dataset, epochs):
    start_time = time.time()
    for epoch in range(epochs):
        random.shuffle(dataset)
        for inputs, expected_output in dataset:
            neuron.train(inputs, expected_output)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.4f} seconds")

# Testing function with timer
def test_neuron(neuron, max_value):
    print("\nTesting the neuron:")
    start_time = time.time()
    correct_predictions = 0
    total_tests = 0
    for a in range(0, max_value + 1):
        for b in range(0, max_value + 1):
            inputs = [UnaryNumber(a), UnaryNumber(b)]
            predicted_output = neuron.predict(inputs)
            actual_output = a + b
            if predicted_output.count == actual_output:
                correct_predictions += 1
            total_tests += 1
    testing_time = time.time() - start_time
    accuracy = (correct_predictions / total_tests) * 100
    print(f"Testing completed in {testing_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.2f}%")

# Random testing function
def random_testing(neuron, num_tests, max_input_value):
    start_time = time.time()
    correct_predictions = 0
    for _ in range(num_tests):
        a = random.randint(0, max_input_value)
        b = random.randint(0, max_input_value)
        inputs = [UnaryNumber(a), UnaryNumber(b)]
        predicted_output = neuron.predict(inputs)
        actual_output = a + b

        if predicted_output.count == actual_output:
            correct_predictions += 1

    accuracy = (correct_predictions / num_tests) * 100
    testing_time = time.time() - start_time
    print(f"\nRandom Testing Results:")
    print(f"Total Tests: {num_tests}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Random testing completed in {testing_time:.4f} seconds")

# Main function
def main():
    # Define the maximum input value for scaling up
    max_input_value = 100  # Adjust this value to scale up or down
    epochs = 20  # Increase epochs for larger datasets

    # Initialize neuron
    neuron = UnaryNeuron()

    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(max_input_value)
    print(f"Dataset prepared with {len(dataset)} samples")

    # Train the neuron
    print("Starting training...")
    train_neuron(neuron, dataset, epochs)

    # Test the neuron
    test_neuron(neuron, max_input_value)

    # Perform random testing
    num_tests = 1000
    random_testing(neuron, num_tests, max_input_value)

if __name__ == "__main__":
    main()