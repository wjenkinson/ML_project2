# Frame Prediction with LAMMPS Simulation Data

This project focuses on predicting the next frame in a molecular dynamics simulation using machine learning. The goal is to learn the dynamics from LAMMPS simulation data.

## Project Structure

```
ML_project1/
├── data/                   # LAMMPS dump files
├── src/                    # Source code
│   └── explore_data.py     # Data exploration and visualization
│   └── visualize_grid.py   # Visualize grid data
│   └── train_cnn.py        # Train a CNN for frame prediction
│   └── predict_sequence.py # Predict a sequence of frames
│   └── make_video.py       # Make a video from the predicted frames
├── output/                 # Output files and visualizations
│   └── visualization_dump.0.png
│   └── visualization_dump.9000.png
│   └── grid_sample_train_0.png
│   └── grid_sample_train_9000.png
│   └── prediction_vs_gt.gif
├── requirements.txt        # Python dependencies
└── README.md               # This file
```


```mermaid
flowchart TD

    %% Data source
    A["**Raw simulation data**<br/>LAMMPS dump files"] 

    %% Split
    A --> B["**Train / Val split**<br/>Select timesteps & split into sequences"]

    %% Two parallel representations
    B --> C["**Grid representation**<br/>Particles → 2D density grids (t, t+1)"]
    B --> D["**Graph representation**<br/>Particles → nodes, neighbors → edges"]

    %% Models
    subgraph E[**Grid-based sequence models**]
        E1["CNN"]
        E2["RNN / GRU / LSTM"]
    end

    C --> E

    subgraph F[**Graph-based model**]
        F1["GNN<br/>(message passing on particles)"]
    end

    D --> F

    %% Unified prediction & evaluation
    E --> G["**Predictions on validation sequence**<br/>Next-frame grids (per model)"]
    F --> G

    G --> H["**Visual comparison**<br/>Side-by-side GIFs:<br/>ground truth vs prediction"]
    G --> I["**Quantitative comparison**<br/>Centerline profiles + MSE"]

    %% Reflection / docs
    H --> J["**Documentation & reflection**<br/>README, DEMO, postmortem"]
    I --> J
```


## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started

1. First, explore the data:
   ```bash
   python src/explore_data.py
   ```
   This will:
   - Scan the data directory for LAMMPS dump files
   - Display basic information about the first and last frames
   - Generate visualizations in the `output` directory

## Next Steps

1. **Inspect the visualizations** in the `output` directory to understand the data
2. **Modify `explore_data.py`** to explore different aspects of the data
3. **Create a data loader** to prepare the data for training
4. **Implement a simple model** for frame prediction

## Data Format

The data consists of LAMMPS dump files in the `data` directory. Each file represents a snapshot of the simulation at a specific timestep.

## Demo

This section provides a visual walkthrough of the project pipeline, from raw LAMMPS data to trained model predictions.

### 1. Raw LAMMPS Data (extrusion process)

We start with LAMMPS dump files containing particle positions over time.

![Simulation GIF](output/lammps_simulation.gif)



### 2. Data Preprocessing

#### Rasterization to 2D Grid
We convert the 3D particle positions into 2D density maps for processing with CNNs:

![Rasterized Grid](output/grid_sample_train_0.png)
*Left: Input frame (t), Right: Target frame (t+1)*

### 3. Training the CNN

We train a simple CNN to predict the next frame given the current one. The training progress looks like this:

```
Epoch 01 | train_loss=0.123456 | val_loss=0.098765
Epoch 02 | train_loss=0.098765 | val_loss=0.087654
...
```

### 4. Predictions

#### CNN

![Prediction](output/prediction_vs_gt_cnn.gif)
*Left: ground truth, Right: predicted frame*

Predicted frame resembles a fading shadow.

#### RNN

![Prediction](output/prediction_vs_gt_rnn.gif)
*Left: ground truth, Right: predicted frame*

Noise overwhelms the predictions.

#### GRU

![Prediction](output/prediction_vs_gt_gru.gif)
*Left: ground truth, Right: predicted frame*

Free surfaces become diffused.

#### LSTM

![Prediction](output/prediction_vs_gt_lstm.gif)
*Left: ground truth, Right: predicted frame*

Free surfaces become diffused.


#### GNN

![Prediction](output/prediction_vs_gt_gnn.gif)
*Left: ground truth, Right: predicted frame*

GNN directly models the atomic interactions, which allows it to capture the interfaces. The predictions are noisy, but the interfaces are preserved.

N.B. The GNN maps atom positions but the predictions are rasterized to a 2D grid to keep the comparison consistent.


#### Model Sensitivity

![Model Sensitivity](output/model_sensitivity_master.png)

The plots cut through the horizontal and vertical centrelines of the simulation. Being a multiphase system, the groundtruth shows either a void (density 0) or a dense phase (density > 0.75) with a sharp transition at the interface.

The groundtruth is compared with the 5 generated predicted frames. As a general comment, all models perform poorly for capturing the sharp interfaces, with CNN and basic RNN performing the worst.

GRU and LSTM performed marginally better, but failed to capture the motion of free surfaces, with the predicted frames showing a diffused free surface.

GNN captured the interfaces best, but the predictions are noisy and the density values are not consistent with the groundtruth.

See Postmortem for more details.

### How to Reproduce

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate visualizations:
   ```bash
   python src/explore_data.py
   python src/visualize_grid.py
   ```

3. Train the model:
   ```bash
   python src/train_cnn.py
   python src/train_gru.py
   python src/train_lstm.py
   python src/train_rnn.py
   python src/train_gnn.py
   ```

4. Generate predictions:
   ```bash
   python src/predict_sequence.py
   ```

5. Generate video:
   ```bash
   python src/make_video.py
   ```

6. Generate model sensitivity:
   ```bash
   python src/model_sensitivity.py
   ```

## License

This project is for educational purposes.
