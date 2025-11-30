import time
import argparse
from cnn_function_monothread import cnn_function_monothread

# Parse command line arguments
parser = argparse.ArgumentParser(description="Benchmark CNN training time")
parser.add_argument('--epoch', type=int, default=3, help='Number of epochs to train')
args = parser.parse_args()

print(f"Training CNN for {args.epoch} epochs...")

# Start timer
start_time = time.time()

# Call the training function
test_loss, test_accuracy = cnn_function_monothread(epochs=args.epoch)

# End timer
end_time = time.time()
elapsed_time = end_time - start_time

a=elapsed_time/60

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("------------------------------------------")

print(f"\nTraining completed in {elapsed_time:.2f} seconds")
print(f"\nTraining completed in {a:.2f} min")

