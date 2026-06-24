from network import FM
import torch
from config import Config
import time
import matplotlib.pyplot as plt
num_steps = Config["num_steps"]
batch_size = Config["batch_size"]
lr = Config["lr"]
input_dim = Config["input_dim"]
state_dim = Config["state_dim"]
num_points = Config["num_points"]
hidden_dim = Config["hidden_dim"]
tent_sigma = Config["tent_sigma"]
grid_size = Config["grid_size"]

model = FM(input_dim=input_dim, num_points=num_points, state_dim=state_dim, hidden_dim=hidden_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
start_time = time.time()
total_loss = 0
loss_list = []
for step in range(num_steps):
    noise = torch.randn(batch_size, num_points, state_dim, device=device)
    t = torch.rand(batch_size, 1, device=device)
    start = torch.rand(batch_size, input_dim // 2, device=device) * grid_size
    end = torch.rand(batch_size, input_dim // 2, device=device) * grid_size
    input = torch.cat([start, end], dim=1)

    alphas = torch.linspace(0, 1, num_points, device=device).view(1, num_points, 1)
    x_1 = start.unsqueeze(1) + alphas * (end - start).unsqueeze(1)
    
    i = torch.arange(num_points, device=device, dtype=torch.float32)
    tent = (1.0 - torch.abs(2*i/(num_points-1) - 1)).view(1, num_points, 1)
    x_1 = x_1 + tent  * (grid_size/tent_sigma) * torch.randn(batch_size, num_points, state_dim, device=device)
    
    target = x_1 - noise
    output = model(noise, t, input)
    loss = torch.mean((output - target) ** 2)
    total_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if (step + 1) % 500 == 0:
        avg_loss = total_loss / 500 
        print(f"Step {step + 1}, Loss {avg_loss}")
        total_loss = 0
        if step + 1 != 500:
            loss_list.append(avg_loss)
        if step + 1 == 35000:
            torch.save(model.state_dict(), "model_35000.pth")
            print(f"Model saved at step 35000: model_35000.pth")
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
torch.save(model.state_dict(), "model_final.pth")
print(f"Model saved at step {num_steps}: model_final.pth")
plt.plot(loss_list)
plt.savefig("loss.png")
plt.show()
