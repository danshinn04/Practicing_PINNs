
**Pressure and Velocity Prediction Using Physics-Informed Neural Networks (PINNs)**

This project aims to predict pressure and velocity fields in a cylindrical vessel using Physics-Informed Neural Networks (PINNs). The predictions are guided by the Hagen-Poiseuille equation and the model is trained on preoperative data.

Below are some image results from the PINNs prediction of velocity fields

![image](https://github.com/user-attachments/assets/d0cd9bd3-aa12-42cd-8462-0acfaec842bb)
![image](https://github.com/user-attachments/assets/9f7afba9-8c0a-4e92-8cba-b8cd958fa0a9)
![image](https://github.com/user-attachments/assets/8ec3ccb1-800b-473f-85d4-d6a9be821565)



**Dataset**

Download the dataset from the following link and place it in your Google Drive:
[Dataset Link]([url](https://figshare.com/articles/dataset/Figures_source_data_xlsx/13295915/1?file=25616711))

**Normalization**

The pressure and velocity data are normalized using the MinMaxScaler from scikit-learn. Normalization ensures that the data is scaled to a range between 0 and 1 for training neural networks.

'''
def normalize_data(Pressure, Velocity_u, Velocity_v, Velocity_w):
    pressure_scaler = MinMaxScaler()
    normalized_pressure = pressure_scaler.fit_transform(Pressure)
    velocity_scaler = MinMaxScaler()
    normalized_velocity_u = velocity_scaler.fit_transform(Velocity_u)
    normalized_velocity_v = velocity_scaler.fit_transform(Velocity_v)
    normalized_velocity_w = velocity_scaler.fit_transform(Velocity_w)
    return normalized_pressure, normalized_velocity_u, normalized_velocity_v, normalized_velocity_w
'''

**Custom Loss Function**

The custom loss function is based on the Hagen-Poiseuille equation, which describes the pressure drop in a cylindrical vessel. This physics-informed loss function helps the model learn faster and more accurately.

def hagen_poiseuille_loss(y_true, y_pred, radius, length, flow_rate):
    
    viscosity = 0.001  # Fluid viscosity
    
    pressure_drop = (8 * viscosity * length * flow_rate) / (np.pi * radius**4)
    
    mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred - pressure_drop)
    
    return mse_loss

**Model Architecture**

The neural network architecture for predicting pressure and velocity fields consists of multiple dense layers with ReLU activation functions.

def create_model():

    input_shape = 6  # X, Y, Z coordinates and the three velocity components: u, v, w
    
    model = keras.Sequential([
    
        keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        
        keras.layers.Dense(128, activation='relu'),
        
        keras.layers.Dense(64, activation='relu'),
        
        keras.layers.Dense(32, activation='relu'),
        
        keras.layers.Dense(1, activation='relu')  # Pressure output
    
    ])
    
    return model

**Training**
The model is trained using the Adam optimizer with a custom learning rate scheduler for exponential decay.

def exponential_decay_schedule(epoch, lr):

    initial_learning_rate = 0.01
    
    decay_rate = 0.1
    
    decay_step = 1
    
    new_lr = initial_learning_rate * (decay_rate ** (epoch // decay_step))
    
    return new_lr

lr_scheduler = LearningRateScheduler(exponential_decay_schedule)

model.compile(loss=custom_loss, optimizer=optimizer)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=8, batch_size=32, callbacks=[lr_scheduler])

**Prediction**

The model is used to predict pressure and velocity at given points using the trained neural network.

def predict_pressure(model, x, y, z, velocity_u, velocity_v, velocity_w):
    
    input_data = np.array([[x, y, z, normalized_velocity_u[0, 0], normalized_velocity_v[0, 0], normalized_velocity_w[0, 0]]])
    
    normalized_pressure_prediction = model.predict(input_data, verbose=0)
    
    max_pressure, min_pressure = maxmin(Pressure)
    
    pressure_prediction = denormalize_pressure(normalized_pressure_prediction, min_pressure, max_pressure)
    
    return pressure_prediction[0, 0]

**Visualization**

The pressure and velocity fields are visualized using heatmaps and contour plots.

def plot_velocity_heatmap(model, z, x_range, y_range, nx, ny, pressure_scaler, velocity_scaler):
    
    x_values = np.linspace(x_range[0], x_range[1], nx)
    
    y_values = np.linspace(y_range[0], y_range[1], ny)
    
    X_grid, Y_grid = np.meshgrid(x_values, y_values)
    
    predicted_velocities = np.zeros((ny, nx, 3))
    
    for i in range(nx):
    
        for j in range(ny):
        
            x = X_grid[j, i]
            
            y = Y_grid[j, i]
            
            pressure = 0.358
            
            predicted_velocities[j, i] = predict_velocity(model, x, y, z, pressure, pressure_scaler, velocity_scaler)
    
    plt.figure()
    
    plt.imshow(predicted_velocities[:, :, 0], extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower', aspect='auto')
    
    plt.colorbar(label='Velocity u (m/s)')
    
    plt.contour(X_grid, Y_grid, predicted_velocities[:, :, 0], colors='k', linewidths=0.5)
    
    plt.xlabel('X')
    
    plt.ylabel('Y')
    
    plt.title(f'Predicted Velocity u at Z = {z}')
    
    plt.show()
    
**Exporting Results**

The predicted pressure and velocity fields are exported to a CSV file.

def export_dataset(velocity_model, pressure_model, z_range, x_range, y_range, nx, ny, nz, pressure_scaler, velocity_scaler):
    
    x_values = np.linspace(x_range[0], x_range[1], nx)
    
    y_values = np.linspace(y_range[0], y_range[1], ny)
    
    z_values = np.linspace(z_range[0], z_range[1], nz)
    
    data = []
    
    for z in z_values:
    
        X_grid, Y_grid = np.meshgrid(x_values, y_values)
        
        predicted_velocities = np.zeros((ny, nx, 3))
        
        predicted_pressures = np.zeros((ny, nx))
        
        for i in range(nx):
        
            for j in range(ny):
            
                x = X_grid[j, i]
                
                y = Y_grid[j, i]
                
                pressure = 0.358
                
                predicted_velocities[j, i] = predict_velocity(velocity_model, x, y, z, pressure, pressure_scaler, velocity_scaler)
                
                predicted_pressure = pressure_model.predict(np.array([[x, y, z]]))
                
                predicted_pressures[j, i] = pressure_scaler.inverse_transform(predicted_pressure)
        
        z_values_2d = np.full_like(predicted_pressures, z)
        
        data.extend(np.stack((z_values_2d.flatten(), X_grid.flatten(), Y_grid.flatten(), predicted_velocities[:, :, 0].flatten(), predicted_velocities[:, :, 1].flatten(), predicted_velocities[:, :, 2].flatten(), predicted_pressures.flatten()), axis=1))

    columns = ['Z', 'X', 'Y', 'pred_vu', 'pred_vv', 'pred_vw', 'pred_pressure']
    
    df = pd.DataFrame(data, columns=columns)
    
    df.to_csv('z_variance_PINNS_NSE_Interpolation.csv', index=False)
    
Results
The model demonstrates the ability to predict pressure and velocity fields in a cylindrical vessel. The results are visualized using heatmaps and contour plots, providing insights into the flow characteristics within the vessel.

