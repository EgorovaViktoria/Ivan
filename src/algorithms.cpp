#include "algorithms.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace algo {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::array<std::pair<int, int>, 4> kDirections4{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
constexpr std::array<std::tuple<char, int, int>, 4> kPuzzleMoves{{
    {'U', -1, 0},
    {'D', 1, 0},
    {'L', 0, -1},
    {'R', 0, 1},
}};

struct PuzzleStateHash {
  std::size_t operator()(const PuzzleState& s) const noexcept {
    std::size_t h = 0;
    for (int v : s) {
      h = h * 131u + static_cast<std::size_t>(v + 1);
    }
    return h;
  }
};

bool in_bounds(int r, int c, int rows, int cols) {
  return r >= 0 && r < rows && c >= 0 && c < cols;
}

int to_idx(int r, int c, int cols) {
  return r * cols + c;
}

std::pair<int, int> to_cell(int idx, int cols) {
  return {idx / cols, idx % cols};
}

std::vector<int> neighbors(int idx, int rows, int cols) {
  const auto [r, c] = to_cell(idx, cols);
  std::vector<int> out;
  out.reserve(4);
  for (const auto [dr, dc] : kDirections4) {
    const int nr = r + dr;
    const int nc = c + dc;
    if (in_bounds(nr, nc, rows, cols)) {
      out.push_back(to_idx(nr, nc, cols));
    }
  }
  return out;
}

bool unvisited_connectivity_ok(int rows,
                               int cols,
                               const std::vector<bool>& visited,
                               int finish) {
  const int target_len = rows * cols;
  std::vector<int> free;
  free.reserve(target_len);
  for (int i = 0; i < target_len; ++i) {
    if (!visited[i]) {
      free.push_back(i);
    }
  }
  if (free.empty()) {
    return true;
  }

  const int start = free.front();
  std::queue<int> q;
  q.push(start);
  std::vector<bool> seen(target_len, false);
  seen[start] = true;
  int reached = 1;

  while (!q.empty()) {
    int node = q.front();
    q.pop();
    for (int nxt : neighbors(node, rows, cols)) {
      if (visited[nxt] || seen[nxt]) {
        continue;
      }
      seen[nxt] = true;
      ++reached;
      q.push(nxt);
    }
  }

  if (reached != static_cast<int>(free.size())) {
    return false;
  }
  return visited[finish] || seen[finish];
}

std::vector<int> order_candidates(int current,
                                  int rows,
                                  int cols,
                                  const std::vector<bool>& visited,
                                  bool use_warnsdorff) {
  std::vector<int> candidates;
  for (int p : neighbors(current, rows, cols)) {
    if (!visited[p]) {
      candidates.push_back(p);
    }
  }
  if (!use_warnsdorff) {
    return candidates;
  }

  auto onward_degree = [&](int cell) {
    int degree = 0;
    for (int nxt : neighbors(cell, rows, cols)) {
      if (!visited[nxt]) {
        ++degree;
      }
    }
    return degree;
  };

  std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
    return onward_degree(a) < onward_degree(b);
  });
  return candidates;
}

std::array<std::pair<int, int>, 16> build_goal_pos() {
  std::array<std::pair<int, int>, 16> pos{};
  PuzzleState goal{};
  for (int i = 0; i < 15; ++i) {
    goal[i] = i + 1;
  }
  goal[15] = 0;
  for (int i = 0; i < 16; ++i) {
    pos[goal[i]] = {i / 4, i % 4};
  }
  return pos;
}

const std::array<std::pair<int, int>, 16>& goal_pos() {
  static const auto pos = build_goal_pos();
  return pos;
}

const std::array<std::vector<std::pair<char, int>>, 16>& puzzle_transitions() {
  static const auto transitions = [] {
    std::array<std::vector<std::pair<char, int>>, 16> t{};
    for (int zero = 0; zero < 16; ++zero) {
      const int zr = zero / 4;
      const int zc = zero % 4;
      for (const auto [code, dr, dc] : kPuzzleMoves) {
        const int nr = zr + dr;
        const int nc = zc + dc;
        if (!in_bounds(nr, nc, 4, 4)) {
          continue;
        }
        t[zero].push_back({code, nr * 4 + nc});
      }
    }
    return t;
  }();
  return transitions;
}

char reverse_move(char m) {
  switch (m) {
    case 'U':
      return 'D';
    case 'D':
      return 'U';
    case 'L':
      return 'R';
    case 'R':
      return 'L';
    default:
      return '\0';
  }
}

std::vector<std::tuple<char, PuzzleState, int>> next_states_with_zero(const PuzzleState& state,
                                                                       int zero_pos,
                                                                       char last_move) {
  std::vector<std::tuple<char, PuzzleState, int>> out;
  for (const auto [code, pos] : puzzle_transitions()[zero_pos]) {
    if (last_move != '\0' && reverse_move(last_move) == code) {
      continue;
    }
    PuzzleState next = state;
    std::swap(next[zero_pos], next[pos]);
    out.push_back({code, next, pos});
  }
  return out;
}

double elapsed_ms(const Clock::time_point& begin) {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

}  // namespace

PuzzleState goal_state() {
  PuzzleState g{};
  for (int i = 0; i < 15; ++i) {
    g[i] = i + 1;
  }
  g[15] = 0;
  return g;
}

SolverStats solve_hamiltonian_path(int rows,
                                   int cols,
                                   std::pair<int, int> start,
                                   std::pair<int, int> finish,
                                   const HamiltonianOptions& options) {
  const auto begin = Clock::now();
  const int target_len = rows * cols;

  if (target_len > kMaxHamiltonianExactCells) {
    return SolverStats{false,
                       0.0,
                       0,
                       {},
                       {},
                       "Размер слишком большой для точного перебора",
                       0};
  }

  const int start_idx = to_idx(start.first, start.second, cols);
  const int finish_idx = to_idx(finish.first, finish.second, cols);

  if (start_idx == finish_idx && target_len > 1) {
    return SolverStats{false, 0.0, 0, {}, {}, "Старт и финиш не могут совпадать", 0};
  }

  const int start_color = (start.first + start.second) % 2;
  const int finish_color = (finish.first + finish.second) % 2;

  if (target_len % 2 == 0 && start_color == finish_color) {
    return SolverStats{false,
                       0.0,
                       0,
                       {},
                       {},
                       "Для чётного числа клеток старт и финиш должны быть разного цвета",
                       0};
  }
  if (target_len % 2 == 1) {
    const int color0 = (target_len + 1) / 2;
    const int color1 = target_len / 2;
    const int majority_color = color0 > color1 ? 0 : 1;
    if (start_color != finish_color || start_color != majority_color) {
      return SolverStats{false,
                         0.0,
                         0,
                         {},
                         {},
                         "Для нечётного числа клеток старт и финиш должны быть на цвете большинства",
                         0};
    }
  }

  std::vector<bool> visited(target_len, false);
  visited[start_idx] = true;
  std::vector<int> path{start_idx};
  std::int64_t operations = 0;
  const auto deadline = begin + std::chrono::duration<double>(std::max(0.1, options.time_limit_s));

  std::function<std::pair<bool, int>(int, int)> dfs = [&](int cell, int depth) -> std::pair<bool, int> {
    if (Clock::now() > deadline) {
      return {false, std::max(0, depth - 1)};
    }

    ++operations;
    if (static_cast<int>(path.size()) == target_len) {
      return {cell == finish_idx, depth};
    }

    auto candidates = order_candidates(cell, rows, cols, visited, options.use_warnsdorff);
    if (options.use_backjumping && candidates.empty()) {
      return {false, std::max(0, depth - 2)};
    }

    int local_jump = depth - 1;
    for (int nxt : candidates) {
      if (nxt == finish_idx && static_cast<int>(path.size()) + 1 != target_len) {
        continue;
      }

      visited[nxt] = true;
      path.push_back(nxt);

      if (options.use_connectivity_pruning && !unvisited_connectivity_ok(rows, cols, visited, finish_idx)) {
        ++operations;
        path.pop_back();
        visited[nxt] = false;
        continue;
      }

      auto [found, jump_target] = dfs(nxt, depth + 1);
      if (found) {
        return {true, depth};
      }

      path.pop_back();
      visited[nxt] = false;
      local_jump = std::min(local_jump, jump_target);
      if (options.use_backjumping && jump_target < depth - 1) {
        return {false, jump_target};
      }
    }

    return {false, local_jump};
  };

  const auto [found, _] = dfs(start_idx, 0);
  std::vector<std::pair<int, int>> cells;
  if (found) {
    cells.reserve(path.size());
    for (int idx : path) {
      cells.push_back(to_cell(idx, cols));
    }
  }

  std::string note;
  if (!found && Clock::now() > deadline) {
    note = "Прервано по лимиту времени";
  }

  return SolverStats{found, elapsed_ms(begin), operations, {}, cells, note, 0};
}

PathCountStats count_hamiltonian_paths(int rows,
                                       int cols,
                                       std::pair<int, int> start,
                                       std::pair<int, int> finish,
                                       const HamiltonianOptions& options) {
  const auto begin = Clock::now();
  const int target_len = rows * cols;

  if (target_len > kMaxHamiltonianExactCells) {
    return PathCountStats{0, 0.0, 0, "Размер слишком большой для подсчёта", 0};
  }

  const int start_idx = to_idx(start.first, start.second, cols);
  const int finish_idx = to_idx(finish.first, finish.second, cols);
  if (start_idx == finish_idx && target_len > 1) {
    return PathCountStats{0, 0.0, 0, "Старт и финиш не могут совпадать", 0};
  }

  if (target_len == 1) {
    return PathCountStats{1, elapsed_ms(begin), 1, "", 0};
  }

  std::vector<bool> visited(target_len, false);
  visited[start_idx] = true;
  std::vector<int> path{start_idx};
  std::int64_t operations = 0;
  std::int64_t count = 0;
  const auto deadline = begin + std::chrono::duration<double>(std::max(0.1, options.time_limit_s));

  std::function<bool(int)> dfs = [&](int cell) {
    if (Clock::now() > deadline) {
      return false;
    }
    ++operations;

    if (static_cast<int>(path.size()) == target_len) {
      if (cell == finish_idx) {
        ++count;
      }
      return true;
    }

    auto candidates = order_candidates(cell, rows, cols, visited, options.use_warnsdorff);
    for (int nxt : candidates) {
      if (nxt == finish_idx && static_cast<int>(path.size()) + 1 != target_len) {
        continue;
      }
      visited[nxt] = true;
      path.push_back(nxt);

      if (options.use_connectivity_pruning && !unvisited_connectivity_ok(rows, cols, visited, finish_idx)) {
        ++operations;
        path.pop_back();
        visited[nxt] = false;
        continue;
      }

      if (!dfs(nxt)) {
        return false;
      }
      path.pop_back();
      visited[nxt] = false;
    }
    return true;
  };

  const bool completed = dfs(start_idx);
  std::string note;
  if (!completed) {
    note = "Подсчёт прерван по лимиту времени";
  }
  return PathCountStats{count, elapsed_ms(begin), operations, note, 0};
}

int manhattan_distance(const PuzzleState& state) {
  int dist = 0;
  const auto& pos = goal_pos();
  for (int idx = 0; idx < 16; ++idx) {
    const int value = state[idx];
    if (value == 0) {
      continue;
    }
    const int r = idx / 4;
    const int c = idx % 4;
    const auto [gr, gc] = pos[value];
    dist += std::abs(r - gr) + std::abs(c - gc);
  }
  return dist;
}

int linear_conflict(const PuzzleState& state) {
  auto max_disjoint_inversions = [](const std::vector<int>& goal_order) -> int {
    const int n = static_cast<int>(goal_order.size());
    if (n <= 1) {
      return 0;
    }
    std::unordered_map<int, int> memo;
    std::function<int(int)> dfs = [&](int mask) {
      if (mask == 0) {
        return 0;
      }
      auto it = memo.find(mask);
      if (it != memo.end()) {
        return it->second;
      }
      const int first = __builtin_ctz(mask);
      int best = dfs(mask & ~(1 << first));
      for (int j = first + 1; j < n; ++j) {
        if ((mask & (1 << j)) == 0) {
          continue;
        }
        if (goal_order[first] > goal_order[j]) {
          best = std::max(best, 1 + dfs(mask & ~(1 << first) & ~(1 << j)));
        }
      }
      memo[mask] = best;
      return best;
    };
    return dfs((1 << n) - 1);
  };

  int conflict = 0;
  const auto& pos = goal_pos();

  for (int row = 0; row < 4; ++row) {
    std::vector<int> row_goal_cols;
    for (int col = 0; col < 4; ++col) {
      const int value = state[row * 4 + col];
      if (value == 0) {
        continue;
      }
      const auto [goal_row, goal_col] = pos[value];
      if (goal_row == row) {
        row_goal_cols.push_back(goal_col);
      }
    }
    conflict += 2 * max_disjoint_inversions(row_goal_cols);
  }

  for (int col = 0; col < 4; ++col) {
    std::vector<int> col_goal_rows;
    for (int row = 0; row < 4; ++row) {
      const int value = state[row * 4 + col];
      if (value == 0) {
        continue;
      }
      const auto [goal_row, goal_col] = pos[value];
      if (goal_col == col) {
        col_goal_rows.push_back(goal_row);
      }
    }
    conflict += 2 * max_disjoint_inversions(col_goal_rows);
  }

  return conflict;
}

int heuristic_distance(const PuzzleState& state) {
  return manhattan_distance(state) + linear_conflict(state);
}

bool is_solvable(const PuzzleState& state) {
  std::vector<int> arr;
  arr.reserve(15);
  for (int v : state) {
    if (v != 0) {
      arr.push_back(v);
    }
  }

  int inv = 0;
  for (std::size_t i = 0; i < arr.size(); ++i) {
    for (std::size_t j = i + 1; j < arr.size(); ++j) {
      if (arr[i] > arr[j]) {
        ++inv;
      }
    }
  }

  int zero_index = 0;
  for (; zero_index < 16; ++zero_index) {
    if (state[zero_index] == 0) {
      break;
    }
  }
  const int zero_row_from_bottom = 4 - (zero_index / 4);
  return (inv + zero_row_from_bottom) % 2 == 1;
}

PuzzleState random_solvable_state(std::optional<unsigned int> seed, int shuffle_steps) {
  std::mt19937 rng(seed.has_value() ? *seed : std::random_device{}());
  PuzzleState state = goal_state();
  int zero = 15;
  int prev = -1;

  for (int i = 0; i < std::max(1, shuffle_steps); ++i) {
    std::vector<int> candidates;
    for (const auto [_, pos] : puzzle_transitions()[zero]) {
      if (pos == prev) {
        continue;
      }
      candidates.push_back(pos);
    }
    std::uniform_int_distribution<std::size_t> dist(0, candidates.size() - 1);
    const int nxt = candidates[dist(rng)];
    std::swap(state[zero], state[nxt]);
    prev = zero;
    zero = nxt;
  }

  return state;
}

SolverStats solve_puzzle_astar(const PuzzleState& start,
                               std::size_t max_nodes,
                               double time_limit_s) {
  const auto begin = Clock::now();
  const auto deadline = begin + std::chrono::duration<double>(std::max(0.1, time_limit_s));
  const PuzzleState goal = goal_state();

  if (start == goal) {
    return SolverStats{true, 0.0, 1, {}, {}, "", 0};
  }

  struct Node {
    int f;
    int g;
    PuzzleState state;
    int zero_pos;
    char last_move;
  };
  struct NodeCmp {
    bool operator()(const Node& a, const Node& b) const {
      return a.f > b.f;
    }
  };

  std::priority_queue<Node, std::vector<Node>, NodeCmp> open_heap;
  const int start_zero = static_cast<int>(std::find(start.begin(), start.end(), 0) - start.begin());
  open_heap.push(Node{heuristic_distance(start), 0, start, start_zero, '\0'});

  std::unordered_map<PuzzleState, int, PuzzleStateHash> g_score;
  g_score[start] = 0;

  std::unordered_map<PuzzleState, std::pair<PuzzleState, char>, PuzzleStateHash> parent;
  parent[start] = {start, '\0'};

  std::int64_t operations = 0;

  while (!open_heap.empty()) {
    if (Clock::now() > deadline) {
      return SolverStats{false, elapsed_ms(begin), operations, {}, {}, "A* прерван по лимиту времени", 0};
    }
    if (g_score.size() > max_nodes) {
      return SolverStats{false, elapsed_ms(begin), operations, {}, {}, "A* достиг лимита состояний", 0};
    }

    Node cur = open_heap.top();
    open_heap.pop();

    auto g_it = g_score.find(cur.state);
    if (g_it == g_score.end() || cur.g != g_it->second) {
      continue;
    }

    ++operations;
    if (cur.state == goal) {
      std::vector<std::string> path;
      PuzzleState node = cur.state;
      while (node != start) {
        const auto [prev, move] = parent[node];
        path.emplace_back(1, move);
        node = prev;
      }
      std::reverse(path.begin(), path.end());
      return SolverStats{true, elapsed_ms(begin), operations, path, {}, "", 0};
    }

    for (const auto& [move, nxt, nxt_zero] : next_states_with_zero(cur.state, cur.zero_pos, cur.last_move)) {
      const int next_g = cur.g + 1;
      const auto it = g_score.find(nxt);
      if (it != g_score.end() && next_g >= it->second) {
        continue;
      }

      g_score[nxt] = next_g;
      parent[nxt] = {cur.state, move};
      open_heap.push(Node{next_g + heuristic_distance(nxt), next_g, nxt, nxt_zero, move});
    }
  }

  return SolverStats{false, elapsed_ms(begin), operations, {}, {}, "A* не нашел решение", 0};
}

}  // namespace algo

