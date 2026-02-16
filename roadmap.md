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

#### **Phase 4: Project Retrospective**
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