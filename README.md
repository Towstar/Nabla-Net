# NeuralNetworkCPP

NeuralNetworkCPP is a dependency-free C++ scientific-ML learning library built
from first principles. Its current core is a configurable multilayer
perceptron with manual forward propagation, backpropagation, objective
callbacks, regularization, optimizer factories, deterministic batching, and a
report-returning training facade.

## Current state

Implemented and covered by executable self-checks:

- Dense MLP construction, validation, deterministic initialization, and
  configurable activation behavior.
- Stable sigmoid, forward caches, manual backward propagation, batch gradients,
  and finite-difference gradient checks.
- Binary cross-entropy, softmax cross-entropy, MSE, objective composition, L2,
  and L1-subgradient regularization infrastructure.
- `OptimizerSpec`, SGD, and Momentum SGD with optimizer state and reset
  behavior.
- Full-batch, mini-batch, stochastic training, deterministic shuffling,
  stopping conditions, and `TrainingReport` streaming.
- End-to-end XOR convergence through the public `train()` API.
- Wolfe and line-search groundwork for a future L-BFGS implementation.

The project remains a learning lab: Ethan writes implementation code in
production files, while Codex writes executable checks in `self_checks.hpp` and
`self_checks.cpp`, explains each requested function, and verifies the result.

## Current priority order

1. Implement AdaGrad, RMSProp, Adam, and AdamW.
2. Implement Elastic Net, Proximal L1, and Smooth L1.
3. Implement Exponential and Hinge objectives.
4. Implement Jacobian/JVP support.
5. Implement L-BFGS.
6. Implement PINN functionality.
7. Return to the earlier deferred features, including dropout, evaluation
   metrics, cross-validation, model averaging, and automatic differentiation.

The Adam paper should be read before Adam. The AdamW paper should be read
before AdamW. The remaining papers in `NNCPPP-Practice/local-notes/Research Papers`
are reserved for their corresponding later features.

## Repository guidance

The canonical current handoff is
`NNCPPP-Practice/local-notes/TRANSFER_CURRENT_AGENDA.md`. Older transfer notes
are retained only when they contain useful historical design context; the
current agenda and repository state take precedence.
