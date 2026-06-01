#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace algo {

constexpr double kMaxTimeLimitSeconds = 1800000.0;
constexpr int kMaxHamiltonianExactCells = 100;

struct SolverStats {
  bool found{false};
  double elapsed_ms{0.0};
  std::int64_t operations{0};
  std::vector<std::string> path_moves{};
  std::vector<std::pair<int, int>> path_cells{};
  std::string note{};
  std::size_t peak_memory_bytes{0};
};

struct PathCountStats {
  std::int64_t count{0};
  double elapsed_ms{0.0};
  std::int64_t operations{0};
  std::string note{};
  std::size_t peak_memory_bytes{0};
};

struct HamiltonianOptions {
  bool use_warnsdorff{false};
  bool use_connectivity_pruning{false};
  bool use_backjumping{false};
  double time_limit_s{kMaxTimeLimitSeconds};
};

using PuzzleState = std::array<int, 16>;

SolverStats solve_hamiltonian_path(int rows,
                                   int cols,
                                   std::pair<int, int> start,
                                   std::pair<int, int> finish,
                                   const HamiltonianOptions& options);

PathCountStats count_hamiltonian_paths(int rows,
                                       int cols,
                                       std::pair<int, int> start,
                                       std::pair<int, int> finish,
                                       const HamiltonianOptions& options);

int manhattan_distance(const PuzzleState& state);
int linear_conflict(const PuzzleState& state);
int heuristic_distance(const PuzzleState& state);
bool is_solvable(const PuzzleState& state);
PuzzleState random_solvable_state(std::optional<unsigned int> seed = std::nullopt,
                                  int shuffle_steps = 100);
SolverStats solve_puzzle_astar(const PuzzleState& start,
                               std::size_t max_nodes = 1200000,
                               double time_limit_s = kMaxTimeLimitSeconds);

PuzzleState goal_state();

}  // namespace algo

