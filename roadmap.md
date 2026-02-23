### Project 2: Physics-Informed Frame Prediction

#### **Blurb: Scope and Non-Scope**
**Scope:** Improve the naive GNN architecture by incorporating physics constraints into the loss function. Learn how to blend domain knowledge with ML.

**Non-Scope:** Advanced optimization techniques, deployment, or user interface development.

---

#### (COMPLETE) **Phase 0: Project 1 Audit**
**Description:** Identify weak software architecture flows and inconsistencies from Project 1.

**Tasks:**
- Misuse of file strctures
- Redundant requirements
- Over complicated code

**Success Criteria:**
- Identified 3+ issues and documented them.
- Rebuilt core architecture making the necessary changes.

(notes: outcomes include predetermined file structure, a more minimal repo, a greater focus on tests)

**Failure Criteria:**
- No improvements made.

#### (COMPLETE) **Phase 1: Physics Analysis**
**Description:** Analyze the physics governing the simulation and identify constraints.

**Tasks:**
- Identify governing equations: Understand the physical laws relevant to the simulation.
- Implement physics checks: Develop methods to ensure physical consistency.

**Success Criteria:**
- Physical constraints are identified and documented. (See `physics.md`.)
- Initial physics checks (mass conservation and neighbour constraints) are implemented and functional. (See `physics_checks.py`.)

**Failure Criteria:**
- Physical constraints cannot be clearly defined or implemented.

---

#### (COMPLETE) **Phase 2: Hybrid Model Design**
**Description:** Combine a PDE solver with a graph neural network to predict frames.

**Tasks:**
- Design the hybrid model: Integrate a simple PDE solver with a graph neural network.
- Implement physics-informed loss: Add physics-based terms to the loss function.

**Success Criteria:**
- Hybrid model is implemented and runs without errors.
- Physics-informed loss is correctly applied and impacts training.

**Failure Criteria:**
- Hybrid model fails to integrate physics constraints effectively.

---

#### (COMPLETE) **Phase 3: Training and Evaluation**
**Description:** Train the model and evaluate its performance with physics constraints.

**Tasks:**
- Train the model: Use the training split with physics-informed loss.
- Evaluate predictions: Compare predictions to ground truth visually and with metrics.

**Success Criteria:**
- Model respects physical constraints (e.g., no negative matter, energy conservation).
- Improved accuracy over the naive model (quantified with metrics).

**Failure Criteria:**
- Model fails to incorporate physics or produces physically implausible results.
- No improvement over the naive model.

---

#### (COMPLETE) **Phase 4: Project Retrospective**
**Description:** Review the project to solidify understanding and identify next steps.

**Tasks:**
- Summarize findings: What did you learn about physics-informed ML?
- Identify gaps: Where did assumptions fail or knowledge fall short?
- Plan next steps: What would you change or explore further in a follow-up project?

**Success Criteria:**
- Clear, actionable insights are documented.
- Next steps are defined for continued learning.

**Failure Criteria:**
- No actionable insights or next steps are identified.

---
### Project 2 Extension: Core GNN improvements + tighter physics-informed loss

#### (COMPLETE) Phase 5: Rework the project into a minimal PINNs
**Description:** Over planned for the physics constraints without a solid foundation for the GNN architecture. Reduce the scope of the project to a minimal PINNs implementation with just one constraint (density).

**Tasks**
 - Remove rigid, interface, mass constraints, propagating changes through the code base
 - Go through physics check and remove any that are not needed
 - Update physics.md

**Success Criteria:**
 - Modules sizes are mostly below 200 lines, 300 line hard limit
 - More intuitive code structure, easier to read and skim

**Failure Criteria:**
 - Massive code files persist
 - Cannot navigate codebase

#### (COMPLETE) Phase 6: Improve upstream data usage
**Description:** Improve the upstream data usage to better leverage GNN architecture. Shrink the scope for physics-informed loss, focusing on a density constraint. 

**Tasks:**
- Implement relative positions 
- Implement velocities 
- Implement general atom labels

**Success Criteria:**
- Clear identifable improvements made to the trained model after implementing the above changes
- Unlock of the physics-informed loss
- Generality of the model demostrated (even to a limited capacity)

**Failure Criteria:**
- Predicted GIFs are not meaningfully different from the baseline
- Models fail to respect physical constraints
- no meaningful generality demonstrated

### Project 2 Further Extension: Multistep training loss and hyperparameter optimization

#### (COMPLETE) Phase 7: Diagnostics
**Description:** Identify and intergrate a series of tests demonstrate that the project is behaving as expected. Are Velocities physical, distribution of neighbours. How does validation compare to training?

This is about setting up the codebase for hyperparameter optimization. I need to be able to answer questions like: am I overfitting, am I underfitting, and is the model training the way I expect it to?

**Tasks:**
- Implement velocity/displacement distribution diagnostics
- Implement visualizations for convergence on the training data (alongside the existing validation visualization)
- Identify other properties to track.

**Success Criteria:**
- Diagnostics leads to correcting one or more ppreviously unidentified bugs in the project
- A clean set up makes it easy to understand and reason with the project, what it is doing and what could be going wrong

**Failure Criteria:**
- Project codebase gets more complicated, more messy
- No substantial insights garnered from the new diagnostics


#### Phase 8: Hyperparameter optimization
**Description:** Hyperparameter optimization to improve model performance.

**Tasks:**
- Implement hyperparameter optimization
- Retrain the core GNN

**Success Criteria:**
- Clear identifable improvements made to the trained model after implementing the above changes
- Generality of the model demostrated (even to a limited capacity)

**Failure Criteria:**
- Predicted GIFs are not meaningfully different from the baseline
- Models fail to respect physical constraints
- no meaningful generality demonstrated

#### Phase 9: Multistep training loss
**Description:** Implement multistep training loss performance.

**Tasks:**
- Implement multistep training loss
- Retrain the core GNN

**Success Criteria:**
- Clear identifable improvements made to the trained model after implementing the above changes
- Generality of the model demostrated (even to a limited capacity)

**Failure Criteria:**
- Predicted GIFs are not meaningfully different from the baseline
- Models fail to respect physical constraints
- no meaningful generality demonstrated  

