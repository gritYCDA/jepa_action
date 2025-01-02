import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

def get_last_available_gpu():
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("No available GPU found")
    
    print("\n=== Available GPU List ===")
    for i in range(gpu_count):
        gpu_properties = torch.cuda.get_device_properties(i)
        total_memory = gpu_properties.total_memory / 1024**2
        allocated_memory = torch.cuda.memory_allocated(i) / 1024**2
        print(f"GPU {i}: {gpu_properties.name}")
        print(f"   Total Memory: {total_memory:.0f}MB")
        print(f"   Used Memory: {allocated_memory:.0f}MB")
        print(f"   Available Memory: {(total_memory - allocated_memory):.0f}MB\n")
    
    last_gpu = gpu_count - 1
    print(f"Selected GPU {last_gpu}\n")
    return last_gpu

# GPU setup
device = torch.device(f"cuda:{get_last_available_gpu()}")
torch.cuda.set_device(device)

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Hyperparameters
input_size = 10
hidden_size = 20
output_size = 2
batch_size = 32
learning_rate = 0.001
log_interval = 100    # Log every 100 batches

# Time settings
iterations_per_epoch = 23700
seconds_per_epoch = 3600     # 1 hour = 3600 seconds
sleep_time = seconds_per_epoch / iterations_per_epoch

def generate_batch():
    x = torch.randn(batch_size, input_size, device=device)
    y = torch.randint(0, output_size, (batch_size,), device=device)
    return x, y

# Initialize model and move to GPU
model = SimpleNet(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Memory optimization
torch.cuda.empty_cache()
with torch.cuda.device(device):
    torch.cuda.memory.set_per_process_memory_fraction(0.1)

# Training loop
try:
    iteration = 0
    epoch = 1
    running_loss = 0.0
    epoch_start_time = time.time()
    
    print("Starting training...")
    while True:
        iteration += 1
        current_iter = iteration % iterations_per_epoch
        if current_iter == 0:
            current_iter = iterations_per_epoch
            
        model.train()
        
        # Generate data and train
        x, y = generate_batch()
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()
        
        # Memory cleanup
        del x, y, outputs
        torch.cuda.empty_cache()
        
        # Log progress
        if iteration % log_interval == 0:
            avg_loss = running_loss / log_interval
            elapsed_time = time.time() - epoch_start_time
            remaining_time = (seconds_per_epoch - elapsed_time) / 60
            
            print(f'Epoch {epoch} | Progress [{current_iter}/{iterations_per_epoch}] | '
                  f'Loss: {avg_loss:.4f} | GPU Memory: {torch.cuda.memory_allocated(device)/1024**2:.1f}MB | '
                  f'Time Left: {remaining_time:.1f}min')
            running_loss = 0.0
        
        # Check epoch completion
        if current_iter == iterations_per_epoch:
            print(f"\nEpoch {epoch} completed! (Time taken: {elapsed_time/60:.1f}min)")
            epoch += 1
            epoch_start_time = time.time()
        
        # Control speed
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nTraining interrupted!")
    elapsed_time = time.time() - epoch_start_time
    print(f"Current epoch progress: {elapsed_time/60:.1f}min")

except Exception as e:
    print(f"Error occurred: {e}")