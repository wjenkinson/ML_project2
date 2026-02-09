### Reflections on Project 1

#### General thoughts
Key similarities between ML and physics simulations workflows: 
 - Large data sets → Focus on pipeline automation using common data manip practices
 - Heavy use of parallelization protocols → Scalability more important than cleverness
 - Integrated validation steps → Immediate and continuous feedback.

Key differences:
 - "Training", as a concept, doesn't exist for simulation models → extra step in the total workflow
 - Physics-informed simulations usually have GT (e.g. discretizing conservation equations) → Absense of GT with ML models

#### Areas for personal improvement
 - Increase visibility of the ML landscape.
 - Dense vocabulary in the field, work on filling the gap.
 - Future emphasis on time series, alternative data structures, physical constraints 

#### Notes on project extension
The basic project was extended by swapping in additional neural network architectures: Recurrent and Graph (RNN (LSTM/GRU) & GNN)


##### RNNs
Designed to capture temporal dependencies.

New concepts:
 - backpropagation, variants like LSTM and GRU

##### GNNs
Captures relational structures in data, an essential feature of atomic interactions.

New concepts:
 - graph representation

 #### Comparison of architectures

Architectures were evaluated for their predictivity to ground truth data, with Mean Squared Error (MSE) serving as a baseline metric. LSTM and GRU showed similar performance, with LSTM slightly outperforming GRU. GNN performed slightly worse, but produced qualitatively sharper interfaces and less noise. This is likely due to the persistence of particle detail across time steps forcing mass constraints.

CNN and basic RNN architectures performed poorly in this task, lacking both temporal and physical constraints.

#### Next step

Physics-Informed Neural Nets (PINNs) incorporate physical constraints into the loss function, guiding more physically plausible solutions.

##### PINNs
Build physics constraints into the learning process e.g. to enforce conservation laws.

New concepts:
 - physics-based loss function