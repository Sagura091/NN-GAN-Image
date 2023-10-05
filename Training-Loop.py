
import pickle
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import json
import vonage
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torchvision.models import resnet50, ResNet50_Weights
from scipy.linalg import sqrtm
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import entropy
client = vonage.Client(key="bc3cecbc", secret="D2oTKsYdBaQUEbKl")
sms = vonage.Sms(client)





    
def save_imgs(epoch, generator, text_data, noise, text_descriptions):
    print("Saving Image From epoch: ", epoch)
    r, c = 5, 5  # Grid size

    print("Getting the generated_images")
    generated_images  = generator(text_data, noise).detach().cpu().numpy()
    print("The Generated Images: ", generated_images)

    # Adjust for your image data format
    generated_images = generated_images.transpose(0, 2, 3, 1)
    print("Transposed the Generated images: ", generated_images)
    
    # Rescale images 0 - 1
    generated_images = 0.5 * generated_images + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            if cnt < len(generated_images):  # Check if the counter is less than the number of images
                axs[i,j].imshow(generated_images[cnt])
                axs[i,j].axis('off')
                cnt += 1
    fig.savefig("D:/Text-To-Image/epoch_{}.png".format(epoch))

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            if cnt < len(generated_images):  # Check if the counter is less than the number of images
                axs[i,j].imshow(generated_images[cnt])
                axs[i,j].set_title(text_descriptions[cnt], fontsize=6, wrap=True)  # Increase fontsize and allow text wrapping
                axs[i,j].axis('off')
                cnt += 1
    fig.savefig(f"D:/Text-To-Image/epoch_{epoch}_with_text.png")
    plt.close()

    
def display_generated_images(generator,text_embedding, num_samples=10, noise_dim=100):
    # Create random noise
    noise = torch.randn(num_samples, noise_dim).to(device)
    
    # Use the generator to produce images
    generated_images = generator(text_embedding, noise).detach().cpu()
    
    # Normalize the images from [-1, 1] to [0, 1]
    generated_images = (generated_images + 1) / 2.0
    
    # Plot the images
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 3))
    for ax, img in zip(axes, generated_images):
        ax.imshow(np.transpose(img, (1, 2, 0)))  # Convert from [C, H, W] to [H, W, C]
        ax.axis('off')
    plt.show()
    
def tokenize_text(text, word_to_id):
    return [word_to_id.get(token, word_to_id["<UNK>"]) for token in text]

def texts_to_tensor(texts, word_to_id, max_length=None):
    tokenized_texts = [tokenize_text(text, word_to_id) for text in texts]
    if not max_length:
        max_length = max(len(tokens) for tokens in tokenized_texts)
    padded_texts = [tokens + [0] * (max_length - len(tokens)) for tokens in tokenized_texts]
    return torch.tensor(padded_texts, dtype=torch.long)


def save_losses_to_json(epoch, g_loss, d_loss, fid_score, inception_scores, precisions, recalls, f1_scores, training_times, json_file_path="D:\\Ai-Clip-Art-Learner\\loss_data.json"):
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {
            'epoch': [],
            'g_loss': [],
            'd_loss': [],
            'fid_scores': [],
            'inception_scores':  [],
            'precisions': [],
            'recalls': [],
            'f1_scores':  [],
            'training_times': []
        }

    # Append new loss data
    data['epoch'].append(int(epoch))
    data['g_loss'].append(float(g_loss))
    data['d_loss'].append(float(d_loss))
    
    # Convert the complex number to a string if fid_score is complex
    data['fid_scores'].append(float(fid_score) if isinstance(fid_score, complex) else float(fid_score))
    
    data['inception_scores'].append(float(inception_scores))
    data['precisions'].append(float(precisions))
    data['recalls'].append(float(recalls))
    data['f1_scores'].append(float(f1_scores))
    data['training_times'].append(float(training_times))

    with open(json_file_path, 'w') as f:
        json.dump(data, f)


def custom_collate_fn(batch):
    # Unzip the batch into separate lists for images and texts
    print("Starting Collate function")
    images, texts = zip(*batch)

    # Use the default collate functions to collate the images and texts
    collated_images = torch.stack(images, 0)
    collated_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)

    # Check if all text tensors have the same shape after padding
    text_shapes = [text.shape[0] for text in collated_texts]
    if len(set(text_shapes)) > 1:
        print(f"Mismatched text tensor shapes in batch: {text_shapes}")


    print("Ending function")
    return collated_images, collated_texts

def compute_gradient_penalty(D, real_samples, fake_samples, text_embedding):
    """Compute gradient penalty for WGAN-GP."""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, text_embedding)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def save_training_graph(g_losses, d_losses, epoch, save_path="Image-gLoss-dloss-graph"):
    plt.figure(figsize=(10, 5))
    plt.title(f"Generator and Discriminator Loss During Training - Epoch {epoch}")
    plt.plot(g_losses, label="G")  # No detach() or cpu() or numpy() needed
    plt.plot(d_losses, label="D")  # No detach() or cpu() or numpy() needed
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"D:/{save_path}/training_graph_epoch_{epoch}.png")
    plt.close()
    
    
def calculate_fid(real_images, fake_images):
    # Initialize Inception model with appropriate weights
    model = inception_v3(pretrained=True)
    model = model.eval()
    
    # Move the model to the same device as your images
    device = real_images.device
    model = model.to(device)

    # Calculate activations for real and fake images
    with torch.no_grad():
        real_activations = model(real_images)
        fake_activations = model(fake_images)

    # Calculate mean and covariance statistics
    mu1, mu2 = real_activations.mean(dim=0), fake_activations.mean(dim=0)
    sigma1 = torch_cov(real_activations, rowvar=False)
    sigma2 = torch_cov(fake_activations, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = torch.sum((mu1 - mu2)**2.0).item()

    # Calculate sqrt of product between covariances
    covmean, _ = sqrtm((sigma1.cpu().numpy()).dot(sigma2.cpu().numpy()), disp=False)
    
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding epsilon to diagonal of cov estimates"
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = sqrtm((sigma1.cpu().numpy() + offset).dot(sigma2.cpu().numpy() + offset))

    # Calculate score
    fid = ssdiff + np.trace(sigma1.cpu().numpy() + sigma2.cpu().numpy() - 2.0 * covmean)

    return fid

# Function to compute covariance in PyTorch
def torch_cov(m, rowvar=False):
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    m = m.type(torch.double)
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def plot_quality_diversity_graph(fid_scores, inception_scores, epoch, save_path="Image-quality-diversity-graph"):
    plt.figure(figsize=(10, 5))
    plt.title(f"FID and Inception Score - Epoch {epoch}")
    plt.plot(fid_scores, label="FID Score")
    plt.plot(inception_scores, label="Inception Score")
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(f"D:/{save_path}/quality_diversity_graph_epoch_{epoch}.png")
    plt.close()

def plot_classification_metrics_graph(precisions, recalls, f1_scores, epoch, save_path="Image-metrics-graph"):
    plt.figure(figsize=(10, 5))
    plt.title(f"Classification Metrics - Epoch {epoch}")
    plt.plot(precisions, label="Precision")
    plt.plot(recalls, label="Recall")
    plt.plot(f1_scores, label="F1 Score")
    plt.xlabel("Iterations")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f"D:/{save_path}/classification_metrics_graph_epoch_{epoch}.png")
    plt.close()

def plot_training_time_graph(training_times, epoch, save_path="Image-time-graph"):
    plt.figure(figsize=(10, 5))
    plt.title(f"Training Time per Epoch - Up to Epoch {epoch}")
    plt.plot(training_times)
    plt.xlabel("Epochs")
    plt.ylabel("Time (seconds)")
    plt.savefig(f"D:/{save_path}/training_time_graph_epoch_{epoch}.png")
    plt.close()

def calculate_inception_score(probabilities):
    # Calculate the inception score using the probabilities from InceptionV3 model
    p_yx = probabilities
    p_y = np.mean(p_yx, axis=0)
    entropy_y = entropy(p_y)
    conditional_entropy_yx = np.mean(entropy(p_yx, axis=1))
    inception_score = np.exp(entropy_y - conditional_entropy_yx)
    return inception_score

def log_metrics(fid_scores, inception_scores, precisions, recalls, f1_scores, filename="D:/metrics_log.txt"):
    with open(filename, "w") as f:
        f.write("Epoch, FID Score, Inception Score, Precision, Recall, F1 Score\n")
        for i, (fid, inc, prec, rec, f1) in enumerate(zip(fid_scores, inception_scores, precisions, recalls, f1_scores)):
            f.write(f"{i+1}, {fid}, {inc}, {prec}, {rec}, {f1}\n")

# Use this function in your training loop to log metrics.


# TextEncoder
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, 128)
        self.linear = nn.Linear(128, embed_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # Change this line
        x = self.linear(x)
        return x




class Generator(nn.Module):
    def __init__(self, text_embedding_size, noise_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_embedding_size + noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            
            nn.Linear(1024, image_flat_size),
            nn.Tanh()
        )

    def forward(self, text_data, noise):
        combined = torch.cat((text_data, noise), 1)
        img = self.fc(combined)
        return img.view(img.size(0), 3, 256, 256)

# Updated Discriminator
class Discriminator(nn.Module):
    def __init__(self, text_embedding_size):
        super(Discriminator, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(3 * 256 * 256 + 1, 1024),  # Adjusted the input size
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, text_data):
        text_data_avg = text_data.mean(dim=1, keepdim=True)
        combined = torch.cat((img.view(img.size(0), -1), text_data_avg), 1)
        validity = self.fc(combined)
        return validity
    
class CustomDataset(Dataset):
    def __init__(self, images, texts, transform=None):
        self.images = images
        self.texts = texts
        self.transform = transform
        
        # 1. Create a vocabulary from all texts.
        self.vocab = self.build_vocab(texts)
        self.vocab_size = len(self.vocab)

    def build_vocab(self, texts):
    # Flatten the list of texts and create a set of all unique tokens
        all_tokens = set(token for text in texts for token in text) # No need to split again
    # Create a dictionary mapping each token to a unique integer ID
        return {token: idx for idx, token in enumerate(all_tokens)}

    def tokenize_text(self, text):
        return [self.vocab[token] for token in text]  # No need to split again  

    def __len__(self):
        return len(self.images) 

    def __getitem__(self, idx):
        image = self.images[idx]
        # Resize images to a consistent size
        image = Image.fromarray((image * 255).astype('uint8')).resize((256, 256))
    
        if self.transform:
            image = self.transform(image)

    # Check if the image is not in the expected format
        if image.shape != (3, 256, 256): # or whatever the expected shape is
         print(f"Invalid image shape at index {idx}. Shape: {image.shape}")

    # Convert the text into a sequence of integer IDs.
        tokenized_text = self.tokenize_text(self.texts[idx])
    # Convert this sequence of integer IDs to a tensor.
        text = torch.tensor(tokenized_text, dtype=torch.long)

    # Check if the text tensor is not in the expected format
        if len(text.shape) != 1:  # Assuming we expect a 1D tensor for text
            print(f"Invalid text tensor shape at index {idx}. Shape: {text.shape}. Text: {self.texts[idx]}")

        return image, text




with open(r'D:\Ai-Clip-Art-Learner\config.json', 'r') as f:
    config = json.load(f)

with open("D:\\new-vocab-real.pickle", 'rb') as f:
    word_to_id = pickle.load(f)
    
vocab_size = len(word_to_id)
print(f"The vocabulary size is: {vocab_size}")

images = []
texts =  []
if __name__ == '__main__':
    with open('D:\\new-data-real.pickle', 'rb') as f:
        data = pickle.load(f)
    print(data.keys())
    images = data["preprocessed_images"]
    texts = data["extracted_texts"]

filtered_images = []
filtered_texts = []

print("Filtering the images and texts")
for img, txt in zip(images, texts):
    if txt:  # Check if the text is not empty
        filtered_images.append(img)
        filtered_texts.append(txt)

images = filtered_images
texts = filtered_texts
print("Finished Filtering the images and texts")




transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("Loading The Dataset")
dataset = CustomDataset(images, texts, transform=transform)
print("Loaded The Dataset")

max_text_length = max(len(text) for text in texts)



print("Loading the DataLoader")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0,collate_fn=custom_collate_fn)
print("Loaded the DataLoader")

vocab_size = config['vocab_size']
text_embedding = config['text_embedding']
noise = config['noise']
epochs = config['epochs']
batch_size = config['batch_size']
image_flat_size = config['image_flat_size']
if config['best_g_loss'] == 1e10:
    best_g_loss = float('inf')
else:
    best_g_loss = config['best_g_loss']

learning_rate = config["learning_rate"]
epochs_without_improvement = config['epochs_without_improvement']
early_stopping_threshold = config['early_stopping_threshold']
lambda_gp = config['lambda_gp']

g_losses = []
d_losses = []
fid_scores = []
inception_scores = []
precisions = []
recalls = []
f1_scores = []
training_times = []

# Define models
print("Setting Text Encoder")
text_encoder = TextEncoder(vocab_size, text_embedding)

print("Setting Generator")
generator = Generator(text_embedding, noise)
print("Setting discriminator")
discriminator = Discriminator(text_embedding)

# Check if GPU is available and move models to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Checking GPU or CPU")
text_encoder.to(device)
generator.to(device)
discriminator.to(device)

criterion = nn.BCELoss()
weight_decay = config['weight_decay']  # Adjust this value based on your needs
optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=weight_decay)


if config['resume']:
    
    state_dict = torch.load(r'D:\Ai-Clip-Art-Learner\Art-text_encoder_weights.pth')
    state_dict['embedding.weight'] = state_dict['embedding.weight'][:9198, :]  # Truncate to fit new size
    text_encoder.load_state_dict(state_dict)
    generator.load_state_dict(torch.load(r'D:\Ai-Clip-Art-Learner\Art-generator_weights.pth'))
    discriminator.load_state_dict(torch.load(r'D:\Ai-Clip-Art-Learner\Art-discriminator_weights.pth'))

    optimizer_g.load_state_dict(torch.load(r'D:\Ai-Clip-Art-Learner\Art-optimizer_g.pth'))
    optimizer_d.load_state_dict(torch.load(r'D:\Ai-Clip-Art-Learner\Art-optimizer_d.pth'))
    
    try:
        with open('D:/metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
            g_losses = metrics['g_losses']
            d_losses = metrics['d_losses']
            fid_scores = metrics['fid_scores']
            inception_scores = metrics['inception_scores']
            precisions = metrics['precisions']
            recalls = metrics['recalls']
            f1_scores = metrics['f1_scores']
            training_times = metrics['training_times']
            
    except FileNotFoundError:
        print('No metrics found. Starting from scratch.')
    starting_epoch = config['last_completed_epoch'] + 1
    print("Resuming training from epoch:", starting_epoch)
else:
    starting_epoch = 0
    print("Starting training from scratch.")
   
# Training loop with early stopping


inception_model = inception_v3(pretrained=True, transform_input=False)
inception_model = inception_model.to(device)
inception_model.eval()
for epoch in range(starting_epoch, epochs):
    start_time = time.time()
    for batch_images, batch_texts in dataloader:
        batch_texts_tensor = texts_to_tensor(batch_texts, word_to_id, max_text_length)
        text_embedding = text_encoder(batch_texts_tensor)

        # Train discriminator
        optimizer_d.zero_grad()
        batch_size_current = text_embedding.shape[0]
        noise_tensor = torch.randn(batch_size_current, noise)
        generated_images = generator(text_embedding, noise_tensor)
        
        real_validity = discriminator(batch_images, text_embedding)
        fake_validity = discriminator(generated_images.detach(), text_embedding)

        gradient_penalty = compute_gradient_penalty(discriminator, batch_images.data, generated_images.data, text_embedding)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward(retain_graph=True)
        optimizer_d.step()

        # Train generator
        optimizer_g.zero_grad()
        fake_validity = discriminator(generated_images, text_embedding)
        g_loss = criterion(fake_validity, torch.ones_like(fake_validity))
        g_loss.backward(retain_graph=True)
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        optimizer_g.step()
        id_to_word = {id_: word for word, id_ in word_to_id.items()}

        with torch.no_grad():
            generated_images = generated_images.to(device)
            logits = inception_model(generated_images)
            probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

            inception_score = calculate_inception_score(probabilities)
            inception_scores.append(inception_score)

            # Precision, Recall, and F1 Score Calculation
            real_labels = torch.ones(real_validity.shape[0], 1).to(device)
            fake_labels = torch.zeros(fake_validity.shape[0], 1).to(device)

            predictions = torch.cat([real_validity, fake_validity])
            labels = torch.cat([real_labels, fake_labels])

            predictions = predictions.cpu().detach().numpy()
            labels = labels.cpu().numpy()

            predicted_labels = np.round(predictions)

            precision = precision_score(labels, predicted_labels)
            recall = recall_score(labels, predicted_labels)
            f1 = f1_score(labels, predicted_labels)

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        
        
        text_descriptions = []
        for text_tensor in batch_texts:
            description = [id_to_word[id_.item()] for id_ in text_tensor if id_.item() in id_to_word]
            text_descriptions.append(' '.join(description))
        print(f"Epoch [{epoch}/{epochs}] | d_loss: {d_loss.item()} | g_loss: {g_loss.item()}")
        end_time = time.time()
        training_times.append(end_time - start_time)
        
        
    
    with torch.no_grad():
        some_text_embedding = torch.randn(batch_size, text_embedding.shape[1])  # Replace text_embedding.shape[1] with the actual embedding size if needed
        some_noise = torch.randn(batch_size, noise) 
        some_real_images = next(iter(dataloader))[0]  #
        fake_images = generator(some_text_embedding, some_noise)  
        real_images = some_real_images

    # Move them to CPU and detach
        fake_images = fake_images.cpu().detach()
        real_images = real_images.cpu().detach()

    # Calculate FID
        fid_score = calculate_fid(real_images, fake_images)
        if isinstance(fid_score, complex):
            fid_score = fid_score.real
        print(f"FID score at epoch {epoch}: {fid_score}")
        
                
                
                
                
    save_losses_to_json(epoch, g_loss.item(), d_loss.item(), fid_score, inception_score, precision, recall, f1, end_time - start_time)
    save_imgs(epoch, generator, text_embedding, noise_tensor, text_descriptions)
    save_training_graph(g_losses, d_losses, epoch)
    plot_quality_diversity_graph(fid_scores, inception_scores, epoch)
    plot_classification_metrics_graph(precisions, recalls, f1_scores, epoch)
    plot_training_time_graph(training_times, epoch)
    log_metrics(fid_scores, inception_scores, precisions, recalls, f1_scores)
    torch.save(text_encoder.state_dict(), r'D:\Ai-Clip-Art-Learner\Art-text_encoder_weights.pth')
    torch.save(generator.state_dict(), r'D:\Ai-Clip-Art-Learner\Art-generator_weights.pth')
    torch.save(discriminator.state_dict(), r'D:\Ai-Clip-Art-Learner\Art-discriminator_weights.pth')
    torch.save(optimizer_g.state_dict(), r'D:\Ai-Clip-Art-Learner\Art-optimizer_g.pth')
    torch.save(optimizer_d.state_dict(), r'D:\Ai-Clip-Art-Learner\Art-optimizer_d.pth')
    with open('D:/metrics.pkl', 'wb') as f:
        pickle.dump({'g_losses': g_losses, 'd_losses': d_losses, 'training_times': training_times, 'fid_scores': fid_scores, 'inception_scores': inception_scores,
                     'precisions':precisions, 'recalls': recalls, 'f1_scores': f1_scores}, f)
    config['last_completed_epoch'] = epoch
    with open(r'D:\Ai-Clip-Art-Learner\config.json', 'w') as f:
        json.dump(config, f)

    # Early stopping check
    if g_loss.item() < best_g_loss:
        best_g_loss = g_loss.item()
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_threshold:
            print(f"Early stopping triggered after {epoch} epochs.")
            responseData2 = sms.send_message(
            {
                "from": "18332686604",
                "to": "16096028991",
                "text": "Text-To-Image-Gan Model Has Stopped Training due to Early Stopping",
            }
        )
            if responseData2["messages"][0]["status"] == "0":
                print("Message sent successfully.")
            else:
                print(f"Message failed with error: {responseData2['messages'][0]['error-text']}")
                break
