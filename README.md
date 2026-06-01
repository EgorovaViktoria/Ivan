## C++ версия с CMake

В репозиторий добавлена C++ реализация основных алгоритмов из `algorithms.py`:
- поиск гамильтонова пути на сетке;
- подсчёт числа гамильтоновых путей;
- решение 15-пазла алгоритмом A* с эвристикой Manhattan + linear conflict.

### Сборка

```bash
cmake -S . -B build
cmake --build build
```

### Запуск

```bash
# Поиск гамильтонова пути
./build/ivan_algorithms hamiltonian 5 5 0 0 4 4 1 1 1 30

# Подсчёт количества путей
./build/ivan_algorithms hamiltonian-count 4 4 0 0 3 3 1 1 30

# Решение случайного 15-пазла A*
./build/ivan_algorithms puzzle-astar 42 80 1200000 60
```

### Примечание

Оригинальные Python-файлы сохранены без удаления.
