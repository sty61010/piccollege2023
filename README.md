# Take Home Quiz: Generative AI ML Engineer- George Tseng

## How well you can formulate the problem and devise a solution
1. Given 5000 points as a dataset. Each item in this dataset is a 5-dimensional array: [x,y,r,g,b] where [r,g,b] is the color of the pixel [x,y] in the image above. We have to build a generative model ”colored_pi_generator” whose output is a 5-dimensional point which comes from the same distribution as the points in the dataset.

2. To generate the the points from the same distribution, the output dimension is 5. The input may be a noise vector with arbitrary dimensions. Thus, we have to build a generative model:
    - input : noise vector with arbitrary dimensions
    - output : 5-dimensional array: [x,y,r,g,b]

3. I have tried generative models: Gaussion Mixture Model, VAE, Diffusion model, Transformer, and self-attention model.

4. The output performance (both qualitative and quantative results)
    - GMM > VAE > Diffusion > Self-attention > Transformer
    - I will introduce evaluation metrics later.


---
## How well your code is organized (you will need to share a link to your Github repo, Colab or wherever your code is located, or just a zip file containing your code)
1. Machine learning Models: GMM
    - I simply create numpy arrays as data points
        - ```python
            # Load the required data
            xs = np.load('pi_xs.npy')
            ys = np.load('pi_ys.npy')
            image_array = np.array(Image.open('sparse_pi_colored.jpg'))
            rgb_values = image_array[xs, ys]
            rgb_values = rgb_values.astype('float32') / 255.0

            # Concatenate the x, y coordinates and RGB values
            data = np.concatenate([xs.reshape(-1, 1), ys.reshape(-1, 1), rgb_values], axis=1)
    - And I train GMM from sklearn.mixture. and sample 5000 points from GMM
        - ```python
            # Train the GMM
            num_components = 128 # Number of mixture components in the GMM
            gmm = GaussianMixture(n_components=num_components, covariance_type='diag')
            gmm.fit(data)

            # Generate new samples
            num_samples = 5000 # Number of samples to generate
            samples = gmm.sample(num_samples)[0]
    - Finally, I visualize results by recreating the image array
        - ```python
            # Map the generated samples back to RGB values
            generated_rgb = samples[:, 2:]
            generated_rgb = np.clip(generated_rgb, 0, 1) * 255.0
            generated_rgb = generated_rgb.astype('uint8')

            # Map the generated samples back to x, y coordinates
            generated_xy = samples[:, :2]
            generated_xy = generated_xy.astype('uint8')

            # Visualize the generated image
            generated_image = np.zeros_like(image_array)
            generated_image[generated_xy[:, 0], generated_xy[:, 1]] = generated_rgb
            plt.imshow(generated_image, cmap='gray')
            plt.show()

2. Deep Learning Models: VAE, Diffusion Model, Self-attention, and Transformer
    - I use Dataset, Dataloader from torch.utils.data and build customize ColoredPiDataset, and normalize data points to [0, 1] or [-1, 1]:
        - ```python 
            # Define the dataset and dataloader
            class ColoredPiDataset(Dataset):
                def __init__(self, image_path, xs_path, ys_path):
                    self.xs = np.load(xs_path)
                    self.ys = np.load(ys_path)
                    self.image_array = np.array(Image.open(image_path))
                    self.rgb_values = self.image_array[self.xs, self.ys]
                    
                    # Normalize xy values to be between 0 and 1
                    self.xs, self.ys = self.xs / 299.0, self.ys / 299.0

                    # Normalize rgb values to be between 0 and 1
                    self.rgb_values = self.rgb_values / 255.0
                    
                    # Normalize xy values to be between 0 and 1
                    # self.xs, self.ys = (self.xs / 149.5) - 1.0, (self.ys / 149.5) - 1.0
                    
                    # # Normalize rgb values to be between -1 and 1
                    # self.rgb_values = (self.rgb_values / 127.5) - 1.0

                def __len__(self):
                    return len(self.xs)
                    # return 30000

                def __getitem__(self, idx):
                    if idx >= 5000:
                        return torch.zeros((5)).to(torch.float32)
                    return torch.tensor([self.xs[idx], self.ys[idx], self.rgb_values[idx][0], self.rgb_values[idx][1], self.rgb_values[idx][2]]).to(torch.float32)
    - Then, I build training function to conduct back-propagation in each epoch. Here we take the VAE training function as example.
        - ```python
            # Define training function
            def train_vae(model, optimizer, criterion, dataloader, device):
                model.train()
                running_loss = 0.0
                for batch in dataloader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    
                    noise = torch.randn(batch.shape[0], 5).to(device)
                    recon_batch, mu, logvar = model(batch)

                    loss = model.loss_function(batch, recon_batch, mu, logvar)

                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * batch.size(0)
                epoch_loss = running_loss / len(dataloader.dataset)
                return epoch_loss
    - Here I perform the training function and set the training parameters. I use tqdm to watch the training procedue. (we take VAE as example)
        - ```python
            # Set up device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Define hyperparameters
            input_dim = 5 # XYRGB values
            hidden_dim = 128
            latent_dim = 16
            num_layers = 2
            num_heads = 4
            dropout = 0.1

            batch_size = 128
            learning_rate = 3e-4
            num_epochs = 600
            num_samples = 500

            # Load the dataset
            dataset = ColoredPiDataset('sparse_pi_colored.jpg', 'pi_xs.npy', 'pi_ys.npy')
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            # Initialize model, optimizer, and loss function
            # model = ColoredPiGenerator().to(device)
            # model = ColoredPiGenerator(num_points=batch_size).to(device)
            # model = ColoredPiGenerator(input_dim, hidden_dim, num_layers, num_heads, dropout).to(device)
            model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            # Train model
            iteration = tqdm(range(num_epochs))
            for epoch in iteration:
                train_loss = train_vae(model, optimizer, criterion, dataloader, device)
                iteration.set_description('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss))
    - Finally, I performed the visualisation to see how the the distribution of model looks like by sampling the noise from gaussian distribution. I genrated the generated_image with the same shape of image_array. Also, I sample same number of points by iterating through the batchs in one epoch.
        - ```python
            # Generate some samples from the model
            generated_image = np.zeros(dataset.image_array.shape)
            xy  = np.zeros((len(dataloader)*batch_size, 2))
            rgb = np.zeros((len(dataloader)*batch_size, 3))
            for sample_idx, batch in enumerate(dataloader):
                with torch.no_grad():
                    # samples, _, _ = model(torch.randn(batch_size, 5).to(device))
                    # samples, _, _ = model(batch.to(device))
                    samples = model.decode(torch.randn(batch_size, latent_dim).to(device))

                    # Denomarlizing samples
                    # samples[:, :2] = (samples[:, :2] + 1) * 149.5
                    samples[:, :2] = torch.clip(samples[:, :2], 0, 1) * 299
                    
                    # Denomarlizing samples
                    # samples[:, 2:] = (samples[:, 2:] + 1) * 127.5
                    samples[:, 2:] = torch.clip(samples[:, 2:], 0, 1) * 255
                    
                    xy[sample_idx*batch_size:(sample_idx+1)*batch_size, :] = samples[:, :2].cpu().numpy()
                    rgb[sample_idx*batch_size:(sample_idx+1)*batch_size, :] = samples[:, 2:].cpu().numpy()

                    samples = samples.cpu().numpy().astype(np.uint8)
                    for i in range(batch_size):
                        x, y, r, g, b = samples[i]
                        generated_image[x, y] = [r, g, b]
                        
            print(f'xy mean: {np.mean(xy)}, xy std: {np.std(xy)}, xy max: {np.max(xy)}, xy min: {np.min(xy)}')
            print(f'rgb mean: {np.mean(rgb)}, rgb std: {np.std(rgb)}, rgb max: {np.max(rgb)}, rgb min: {np.min(rgb)}')
            print(f'Error: {np.mean(np.abs(generated_image - dataset.image_array))}')

            # Save the output image
            # Image.fromarray(generated_image).save('generated_pi_colored.jpg')
            plt.imshow(generated_image, cmap='gray')
---
## How familiar and comfortable you are using technologies like transformers (hence the suggestion to use it)

---
## The correctness of your solution (Evaluation Metrics)
- In order to evaluate the performance of generative models, I plot the results to show qualitative results. And I compute the error between the image array and generated image. To ensure the distribution of [x,y,r,g,b], I also compute the mean and standard deviation of the those values.
    - ```python
        print(f'xy mean: {np.mean(xy)}, xy std: {np.std(xy)}, xy max: {np.max(xy)}, xy min: {np.min(xy)}')
        print(f'rgb mean: {np.mean(rgb)}, rgb std: {np.std(rgb)}, rgb max: {np.max(rgb)}, rgb min: {np.min(rgb)}')
        print(f'Error: {np.mean(np.abs(generated_image - dataset.image_array))}')

        # Save the output image
        # Image.fromarray(generated_image).save('generated_pi_colored.jpg')
        plt.imshow(generated_image, cmap='gray')
---
## How well you can explain the problem and your solution
- The detail of each model is in the PicCollege_HW3_"model".ipynb file, please check it to evalute my solutions.
---
## How you demonstrate that the output points indeed come from the same distribution as the input points
1. GMM (From same distribution)
    - I used GMM to fit provided data points and thus the output points are sampled from the same distribution that GMM formulated in.
2. VAE (From same distribution)
    - I used VAE to fit provided data points, and use encoder to encode data points to embeddings. Those embeddings come from the specific gaussion distribution. And I used decoder to decode embeddings. During inference time, I directly sample noise from Gaussian distribution and use decoder to decode noise to data points. Thus output points are sampled from the same distribution.
    - In order to imporve the perfomance, I also used transformer encoder to encode data points and and decode embeddings.
3. Self-attention
    - According to the hints, it suggested to use self-attention to generate data points. However, it is unclear the compute the loss function if the input data points are noise vectors. Thus I used GT to compute the loss function and try to generate data points from the same distribution. It finally generated data points from the same distribution when input data points are from the same distribution. But the results are not stable when input data points are noise vectors. The model may be a function that maps data points to same distribution but not generative.
4. Transformer
    - Same result as above.
5. Diffusion model
    
---