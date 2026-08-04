# NNCPP Practice Progress

Last updated: 2026-08-04

## Completed

1. Core manual-backprop MLP and self-check suite.
2. Wolfe line-search implementation moved to wolfe_analysis.cpp.
3. Wolfe declarations moved to wolfe_analysis.hpp.
4. Objective implementations moved to objective_functions.cpp.
5. Objective declarations moved to objective_functions.hpp.
6. Optimizer contracts/specifications moved to optimizers.hpp.
7. Optimizer metadata and validation moved to optimizers.cpp.
8. Optimizer class and factory skeletons added for SGD, AdaGrad, RMSProp, Adam, AdamW, and LBFGS.
9. Project files updated for the new source and header files.
10. All current translation units compile and link with C++23.

## Current status

The project is still incomplete.

Still stubbed:

1. MSE objective.
2. Exponential objective.
3. L2 regularization.
4. L1 regularization.
5. Elastic Net regularization.
6. Built-in optimizer factories.
7. All optimizer method bodies.
8. Trainer implementation.
9. Configurable activations.
10. Dropout.
11. Generalization reporting.
12. Model averaging.
13. Matrix-valued reverse-mode autodiff.

## Next implementation order

1. Finish MSE and exponential objective functions.
2. Replace their self-check stub expectations with exact value and gradient tests.
3. Implement L2 regularization.
4. Add L2 loss, gradient, coefficient, bias-inclusion, finite-value, and shape tests.
5. Implement L1 and Elastic Net.
6. Replace all remaining regularization stub tests.
7. Decide optimizer hyperparameter structs and how factories capture them.
8. Decide whether the trainer should return TrainingReport instead of void.
9. Resolve whether TrainingStepResult should contain optional Wolfe data.
10. Implement and test SGD.
11. Implement and test AdaGrad.
12. Implement and test RMSProp.
13. Implement and test Adam.
14. Implement and test AdamW.
15. Implement LBFGS last among the optimizers.
16. Implement the trainer.
17. Add configurable activations.
18. Add dropout with explicit train/evaluation mode.
19. Add automated evaluation and generalization statistics.
20. Add frequentist and Bayesian model averaging.
21. Build the matrix-valued reverse-mode autodiff engine.
22. Integrate autodiff as a second training backend.
23. Compare autodiff gradients against manual backprop before declaring completion.

## Long Road Lookahead

1. Optimizer/trainer abstraction.
2. SGD, AdaGrad, RMSProp, Adam, AdamW, and LBFGS.
3. Configurable hidden activations.
4. Dropout and train/evaluation modes.
5. Regularization terms.
6. Matrix-valued autodiff.
7. Autodiff integration into the MLP trainer.
8. Automated generalization-performance evaluation.
9. Bayesian and frequentist model averaging as the final milestone.
