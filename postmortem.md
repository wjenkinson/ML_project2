### Reflections on Project 2

#### General thoughts
Going deeper on GNNs and physics-informed loss has exposed gaps in my understanding of the technique; the trained models performance has declined with the attempts to introduce additional constraints.

Most revealing is much of the implementation for physics-informed loss can be captured upstream by the GNN, so long and the data is preprocessed well.

#### Areas for personal improvement
 - More holistic understanding of the core GNN architecture and implementation.
 - Better development of the data pipeline and preprocessing steps. 

#### Notes on first project extension
Over indexing for physics-informed loss has created a complex codebase with a little leverage on the data:
 - All physics-informed loss checks can be stripped out with the exception of density constraint.
 - The density constraint will be off until we have improved the core GNN training steps.

 Once the codebase has been pruned, focus will be on:
 - Training the GNN with the relative positions of the atoms.
 - Incorporating velocities data.
 - Retrain the core GNN with additional meaningful data.

 ### Reflections on the first extension
Codebase is substantially simpler and GNN is not being used as intended. In general, the trained model is not performing sufficiently well to integrate into a production environment. However, there are opportunities to make additional improvements, including:
 - Optimized hyperparameters (epochs, layers)
 - Multistep training loss
 - Physics-informed loss (used surgically)

 #### Notes on second project extension
 Implementing multistep training loss could be a maor unlock as it stretches the overall usefulness of existing data and penalizes the model for compounded errors, not jjust next steps.

 In addition, we may find that the hyperparameters are simply not optimal for the current architecture, and it should be relatively straightforward to sweep.

 ### Reflections on second extension
 Diagnostics have been added the look into the neighbor distances and the velocity distributions of the nodes. In addition, the model has been modified to add more weight to the particles with interface interactions.

 Now phase 8 will be rolled out, a check on hyperparameter optimization, that will determine if this is a tuning problem, or a deeper problem in the implementation.

 #### Diagnostics, hyperparameter optimization & multistep roll out

More visializations have been created to analyse, for example, the neighbor distributions and velocity distributions of the system. These diagnostics reveal when the model is picking up essential physics such as treating the body of fluid as evenly spaces nodes, and identifying local velocities.

Hyperparameter sweeps have been implemented to determine if the model is simply not learning well enough, or if the model is not learning at all. To this end, the model looks qualitatively more effective with certain combinations, but none of the hyper paramater combinations lead to sensible predictions.

Likewise, multistep roll out has been introduced to help the model build stable solutions across multiple inference steps. (awaiting hyperparameter sweep results)


#### Future work (for Project 3)
I am trying to train a dynamic model using:
- displacements
- velocities

Future work should also train of forces/accelerations.

In addition, I am trying to one shot an extrusion model. Future training pipelines would benefit from training on general flows (canonical flows such as couette, dam breaks and perhaps a propeller type simulation).

Also, we should only focus on the fluid and solid phases. I do not think the motion of the pistons and walls need to be trained as they can remain clamped in all configurations.