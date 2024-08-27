from sympy import factor
import config
import factory
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import DataModel
import logging
logging.basicConfig(filename='training.log', level=logging.INFO)

population = []
for _ in range(population_size):
    model = LSTMBallModel(input_size, num_classes, output_size, hidden_size, num_layers, dropout)
    population.append(model)

def fitness_function(model, train_loader, val_loader):
    model.eval()
    train_loss = 0.0
    train_acc = 0.0
    val_loss = 0.0
    val_acc = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = model.process_inputs(data, target)
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        pred = output.argmax(dim=1)
        train_acc += (pred == target).sum().item()

    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = model.process_inputs(data, target)
        output = model(data)
        loss = criterion(output, target)
        val_loss += loss.item()
        pred = output.argmax(dim=1)
        val_acc += (pred == target).sum().item()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader.dataset)
    val_loss /= len(val_loader)
    val_acc /= len(val_loader.dataset)

    return val_acc

# Selection
def selection(population, fitness_scores):
    selected_indices = np.random.choice(
        population_size, population_size, replace=False, p=fitness_scores / np.sum(fitness_scores)
    )
    selected_population = [population[i] for i in selected_indices]
    return selected_population

# Crossover
def crossover(parent1, parent2):
    child = LSTMBallModel(input_size, num_classes, output_size, hidden_size, num_layers, dropout)
    for name, param in child.named_parameters():
        if random.random() < crossover_rate:
            param.data = parent1.state_dict()[name].clone()
        else:
            param.data = parent2.state_dict()[name].clone()
    return child

# Mutation
def mutation(model):
    for name, param in model.named_parameters():
        if random.random() < mutation_rate:
            param.data += torch.randn_like(param) * 0.01
    return model

if __name__ == "__main__":
    model_names = factory.model_list(config.models)
    parser = argparse.ArgumentParser(description="Train arguments")
    parser.add_argument("-n", "--epoch_num", type=int, help="Train Epoch Number", default=500)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size", default=64)
    parser.add_argument("-l", "--learning_rate", type=float, help="Learning Rate", default=1e-3)
    parser.add_argument(
        "-w",
        "--window_sizes",
        type=int,
        nargs="+",
        help="Window Sizes",
        default=[12],
    )
    parser.add_argument(
        "-m",
        "--models",
        type=str,
        nargs="+",
        choices=model_names,
        default=model_names,
        help="Model name to train.",
    )
    args = parser.parse_args()
    models = [mode for mode in config.models if mode["name"] in args.models]

    Train(models = models, epoch_num=args.epoch_num, batch_size=args.batch_size, learning_rate=args.learning_rate, window_sizes=args.window_sizes)

