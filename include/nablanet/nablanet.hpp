#pragma once

// Stable high-level umbrella include for consumers of the installed library.
// It exposes model construction, norms, objectives, optimizers, and training. CSV
// helpers and line-search implementation mechanics intentionally remain
// source-private implementation details.
#include "mlp.hpp"
#include "norms.hpp"
#include "objective_functions.hpp"
#include "optimizers.hpp"
#include "train.hpp"
