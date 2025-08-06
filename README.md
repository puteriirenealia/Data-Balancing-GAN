# Data-Balancing-GAN
Data Balancing with Gen AI: Credit Card Fraud Detection
Data Balancing for Credit Card Fraud Detection using Generative Adversarial Networks (GANs)
This project demonstrates how to use a Generative Adversarial Network (GAN) to address the common problem of class imbalance in datasets. Specifically, it focuses on generating synthetic data for the minority class (fraudulent transactions) in a credit card fraud detection scenario.

The core idea is to train a Generator model to create realistic fraudulent transaction data that is indistinguishable from real data, and a Discriminator model to learn to tell the real and synthetic data apart. Through this adversarial process, the Generator becomes proficient at producing high-quality synthetic samples.

### 1. Model Architecture ###
The GAN is composed of two main neural networks: the Generator and the Discriminator.

Generator

The Generator's role is to take random noise as input and transform it into synthetic data that mimics the structure of the real fraudulent transactions.

Input: A vector of random noise (latent space).

Output: A vector representing a synthetic transaction with 29 features.

The model is built using a Sequential API with Dense layers, BatchNormalization to stabilize training, and relu activation functions, culminating in a linear activation for the output layer.

    def build_generator():
    model = Sequential()

    model.add(Dense(32, activation='relu', input_dim=29, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(29, activation='linear'))
    
    return model

Generator Summary:

Model: "sequential_2"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_12 (Dense)                │ (None, 32)             │           960 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 32)             │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_13 (Dense)                │ (None, 64)             │         2,112 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 64)             │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_14 (Dense)                │ (None, 128)            │         8,320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_2           │ (None, 128)            │           512 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_15 (Dense)                │ (None, 29)             │         3,741 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 16,029
 Trainable params: 15,581
 Non-trainable params: 448

Discriminator

The Discriminator acts as a binary classifier. It takes transaction data (both real and synthetic) as input and determines whether it is authentic or fake.

Input: A vector representing a transaction with 29 features.

Output: A single probability score (0 for fake, 1 for real).

This model uses a series of Dense layers with relu activation and a final sigmoid activation function to output a probability. It is compiled with the adam optimizer and binary_crossentropy loss function.

    def build_discriminator():
    model = Sequential()

    model.add(Dense(128, input_dim=29, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

Discriminator Summary:

Model: "sequential_3"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_16 (Dense)                │ (None, 128)            │         3,840 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_17 (Dense)                │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_18 (Dense)                │ (None, 32)             │         2,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_19 (Dense)                │ (None, 32)             │         1,056 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_20 (Dense)                │ (None, 16)             │           528 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_21 (Dense)                │ (None, 1)              │            17 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 15,777
 Trainable params: 15,777
 Non-trainable params: 0

### 2. The Combined GAN Model ###
The Generator and Discriminator are combined into a single GAN model. During the GAN's training, the Discriminator's weights are frozen so that only the Generator is updated. The Generator learns by trying to fool the Discriminator.



    def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(generator.input_shape[1],))
    X = generator(gan_input)
    gan_output = discriminator(X)
    gan = Model(gan_input, gan_output)
    return gan


GAN Summary:

Model: "functional_25"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_4 (InputLayer)      │ (None, 29)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ sequential_2 (Sequential)       │ (None, 29)             │        16,029 │
│ (generator)                     │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ sequential_3 (Sequential)       │ (None, 1)              │        15,777 │
│ (discriminator)                 │                        │               │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 31,806
 Trainable params: 15,581
 Non-trainable params: 16,225

### 3. Training Process ###
The GAN is trained in an iterative loop over a set number of epochs. In each epoch:

Train Discriminator: A half-batch of real fraudulent data and a half-batch of synthetic data (from the Generator) are used to train the Discriminator. The Discriminator learns to distinguish between the two.

Train Generator: The combined GAN model is trained. The Generator produces synthetic data and uses the Discriminator's feedback to improve. The goal is to generate data that the Discriminator classifies as real (label 1).

    num_epochs = 1000
    batch_size = 64
    half_batch = int(batch_size / 2)

    for epoch in range(num_epochs):
    # Train Discriminator
    x_fake = generate_synthetic_data(generator, half_batch)
    y_fake = np.zeros((half_batch, 1))
    
    x_real = data_fraud.drop('Class', axis=1).sample(half_batch)
    y_real = np.ones((half_batch, 1))
    
    discriminator.trainable = True
    discriminator.train_on_batch(x_real, y_real)
    discriminator.train_on_batch(x_fake, y_fake)
    
    # Train Generator (via GAN model)
    discriminator.trainable = False
    noise = np.random.normal(0, 1, (batch_size, 29))
    gan.train_on_batch(noise, np.ones((batch_size, 1)))
    
    # Monitor performance periodically
    if epoch % 10 == 0:
        monitor_generator(generator)

### 4. Monitoring and Evaluation ###
To visually assess the quality of the generated data during training, Principal Component Analysis (PCA) is used. By reducing the 29 features down to 2 principal components, we can plot both the real and synthetic data points on a 2D scatter plot.

As training progresses, the distribution of the 'fake' data points should converge and overlap with the distribution of the 'real' data points, indicating that the Generator is learning successfully.

    def monitor_generator(generator):
    pca = PCA(n_components=2)
    real_fraud_data = data_fraud.drop("Class", axis=1)
    transformed_data_real = pca.fit_transform(real_fraud_data.values)
    df_real = pd.DataFrame(transformed_data_real)
    df_real['label'] = "real"

    synthetic_fraud_data = generate_synthetic_data(generator, 492)
    transformed_data_fake = pca.fit_transform(synthetic_fraud_data)
    df_fake = pd.DataFrame(transformed_data_fake)
    df_fake['label'] = "fake"

    df_combined = pd.concat([df_real, df_fake])
    
    plt.figure()
    sns.scatterplot(data=df_combined, x=0, y=1, hue='label', s=10)
    plt.show()

### 5. Generating Final Synthetic Data ###
After training is complete, the trained Generator can be used to create any number of synthetic fraudulent data points. This new data can then be combined with the original dataset to create a balanced dataset for training a more robust fraud detection model.

)

    #Generate 1000 fraudulent data points
    synthetic_data = generate_synthetic_data(generator, 1000)

    # Compare distributions
    df = pd.DataFrame(synthetic_data)
    df['label'] = 'fake'

    df2 = data_fraud.drop('Class', axis=1)
    df2['label'] = 'real'

    #Ensure columns match before concatenating
    df2.columns = df.columns

    combined_df = pd.concat([df, df2])
