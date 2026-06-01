#include "algorithms.hpp"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

namespace {

void print_usage() {
  std::cout
      << "Usage:\n"
      << "  ivan_algorithms hamiltonian <rows> <cols> <sr> <sc> <fr> <fc> [warnsdorff:0|1] "
         "[connectivity:0|1] [backjumping:0|1] [time_limit_s]\n"
      << "  ivan_algorithms hamiltonian-count <rows> <cols> <sr> <sc> <fr> <fc> [warnsdorff:0|1] "
         "[connectivity:0|1] [time_limit_s]\n"
      << "  ivan_algorithms puzzle-astar [seed] [shuffle_steps] [max_nodes] [time_limit_s]\n";
}

int to_int(const char* value) {
  return std::stoi(value);
}

double to_double(const char* value) {
  return std::stod(value);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    return 1;
  }

  const std::string mode = argv[1];

  if (mode == "hamiltonian") {
    if (argc < 8) {
      print_usage();
      return 1;
    }

    algo::HamiltonianOptions options;
    if (argc > 8) options.use_warnsdorff = to_int(argv[8]) != 0;
    if (argc > 9) options.use_connectivity_pruning = to_int(argv[9]) != 0;
    if (argc > 10) options.use_backjumping = to_int(argv[10]) != 0;
    if (argc > 11) options.time_limit_s = to_double(argv[11]);

    auto stats = algo::solve_hamiltonian_path(to_int(argv[2]),
                                              to_int(argv[3]),
                                              {to_int(argv[4]), to_int(argv[5])},
                                              {to_int(argv[6]), to_int(argv[7])},
                                              options);

    std::cout << "found=" << (stats.found ? "true" : "false") << "\n";
    std::cout << "elapsed_ms=" << stats.elapsed_ms << "\n";
    std::cout << "operations=" << stats.operations << "\n";
    std::cout << "path_length=" << stats.path_cells.size() << "\n";
    if (!stats.note.empty()) {
      std::cout << "note=" << stats.note << "\n";
    }
    return 0;
  }

  if (mode == "hamiltonian-count") {
    if (argc < 8) {
      print_usage();
      return 1;
    }

    algo::HamiltonianOptions options;
    if (argc > 8) options.use_warnsdorff = to_int(argv[8]) != 0;
    if (argc > 9) options.use_connectivity_pruning = to_int(argv[9]) != 0;
    if (argc > 10) options.time_limit_s = to_double(argv[10]);

    auto stats = algo::count_hamiltonian_paths(to_int(argv[2]),
                                               to_int(argv[3]),
                                               {to_int(argv[4]), to_int(argv[5])},
                                               {to_int(argv[6]), to_int(argv[7])},
                                               options);

    std::cout << "count=" << stats.count << "\n";
    std::cout << "elapsed_ms=" << stats.elapsed_ms << "\n";
    std::cout << "operations=" << stats.operations << "\n";
    if (!stats.note.empty()) {
      std::cout << "note=" << stats.note << "\n";
    }
    return 0;
  }

  if (mode == "puzzle-astar") {
    std::optional<unsigned int> seed;
    int shuffle_steps = 100;
    std::size_t max_nodes = 1200000;
    double time_limit_s = algo::kMaxTimeLimitSeconds;

    if (argc > 2) {
      seed = static_cast<unsigned int>(to_int(argv[2]));
    }
    if (argc > 3) {
      shuffle_steps = to_int(argv[3]);
    }
    if (argc > 4) {
      max_nodes = static_cast<std::size_t>(std::stoull(argv[4]));
    }
    if (argc > 5) {
      time_limit_s = to_double(argv[5]);
    }

    const auto start = algo::random_solvable_state(seed, shuffle_steps);
    const auto stats = algo::solve_puzzle_astar(start, max_nodes, time_limit_s);

    std::cout << "found=" << (stats.found ? "true" : "false") << "\n";
    std::cout << "elapsed_ms=" << stats.elapsed_ms << "\n";
    std::cout << "operations=" << stats.operations << "\n";
    std::cout << "moves=" << stats.path_moves.size() << "\n";
    if (!stats.note.empty()) {
      std::cout << "note=" << stats.note << "\n";
    }
    return 0;
  }

  print_usage();
  return 1;
}

