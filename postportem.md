### Reflections on Project 2

#### General thoughts
Going deeper on GNNs and physics-informed loss has exposed gaps in my understanding of the technique; the trained models performance has declined with the attempts to introduce additional constraints.

Most revealing is much of the implementation for physics-informed loss can be captured upstream by the GNN, so long and the data is preprocessed well.

#### Areas for personal improvement
 - More holistic understanding of the core GNN architecture and implementation.
 - Better development of the data pipeline and preprocessing steps. 

#### Notes on project extension
Over indexing for physics-informed loss has created a complex codebase with a little leverage on the data:
 - All physics-informed loss checks can be stripped out with the exception of density constraint.
 - The density constraint will be off until we have improved the core GNN training steps.

 Once the codebase has been pruned, focus will be on:
 - Training the GNN with the relative positions of the atoms.
 - Incorporating velocities data.
 - Retrain the core GNN with additional meaningful data.