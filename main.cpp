#include <bits/stdc++.h>
#include <ncurses.h>

using namespace std;

#define printd(expr) cout << #expr " = " << (expr) << endl
#define locale static
#define null   nullptr

class red_black_tree {
  enum colors {none, red, black};
  struct node {
    int key{};
    node *prev{null};
    node *left{null};
    node *right{null};
    colors color{none};
  };
  node *leaf{new node{0, null, null, null, black}};
  node *root{leaf};
public:
  red_black_tree() = default;
  void insert(int key) {
    auto p = &root;
    auto prev = leaf;
    while (*p != leaf and (*p)->key != key) {
      prev = *p;
      if ((*p)->key > key)
        p = &(*p)->left;
      else
        p = &(*p)->right;
    }
    if (*p == leaf) {
      *p = new node{key, prev, leaf, leaf, red};
      insert_fix(*p);
    }
  }
  void print() const {
    print(root, 0);
  }
  int depth() const {
    return depth(root);
  }
private:
  void left_rotate(node *p) {
    node *n = p->right;
    p->right = n->left;
    n->left = p;
    n->prev = p->prev;
    p->right->prev = p;
    if (p->prev == leaf)
      root = n;
    else if (p == p->prev->left)
      p->prev->left = n;
    else
      p->prev->right = n;
    p->prev = n;
  }
  void right_rotate(node *p) {
    node *n = p->left;
    p->left = n->right;
    n->right = p;
    n->prev = p->prev;
    p->left->prev = p;
    if (p->prev == leaf)
      root = n;
    else if (p == p->prev->left)
      p->prev->left = n;
    else
      p->prev->right = n;
    p->prev = n;
  }
  void insert_fix(node *p) {
    while (p != root and p->prev->color == red) {
      auto g = p->prev->prev;
      if (p->prev == g->left) {
        auto u = g->right;
        if (u->color == red) {
          p->prev->color = black;
          u->color = black;
          g->color = red;
          p = g;
        } else {
          if (p == p->prev->right) {
            p = p->prev;
            left_rotate(p);
          }
          p->prev->color = black;
          g->color = red;
          right_rotate(g);
        }
      } else {
        auto u = g->left;
        if (u->color == red) {
          p->prev->color = black;
          u->color = black;
          g->color = red;
          p = g;
        } else {
          if (p == p->prev->left) {
            p = p->prev;
            right_rotate(p);
          }
          p->prev->color = black;
          g->color = red;
          left_rotate(g);
        }
      }
    }
    root->color = black;
  }
  void print(const node *p, int n) const {
    if (p != leaf) {
      print(p->right, n + 1);
      for (int i = 0; i < n; i++)
        cout << "    ";
      if (p->color == red)
        cout << "\033[1;31m";
      else
        cout << "\033[1;30m";
      cout << p->key << "\033[0m" << endl;
      print(p->left, n + 1);
    }
  }
  int depth(const node *p) const {
    return p == leaf ? 0 : max(depth(p->left), depth(p->right)) + 1;
  }
private:
  red_black_tree(const red_black_tree &) = delete;
  void operator=(const red_black_tree &) = delete;
};

class avl_tree {
  struct node {
    int key{};
    int h{};
    node *prev{};
    node *left{};
    node *right{};
  };
  node *root{null};
public:
  avl_tree() = default;
  ~avl_tree() {

  }
  void insert(int key) {
    node **p = &root;
    node *prev = null;
    while (*p != null and (*p)->key != key) {
      prev = *p;
      if ((*p)->key > key)
        p = &(*p)->left;
      else
        p = &(*p)->right;
    }
    if (*p == null) {
      *p = new node{key, 0, prev, null, null};
      insert_fix(prev, key);
    }
  }
  void print() const {
    print(root, 0);
  }
  int depth() const {
    return depth(root);
  }
private:
  void left_rotate(node *p) {
    node *n = p->right;
    p->right = n->left;
    n->left = p;
    n->prev = p->prev;
    if (p->right)
      p->right->prev = p;
    if (p->prev == null)
      root = n;
    else if (p == p->prev->left)
      p->prev->left = n;
    else
      p->prev->right = n;
    p->prev = n;
    recalc_h(p);
    recalc_h(n);
  }
  void right_rotate(node *p) {
    node *n = p->left;
    p->left = n->right;
    n->right = p;
    n->prev = p->prev;
    if (p->left)
      p->left->prev = p;
    if (p->prev == null)
      root = n;
    else if (p == p->prev->left)
      p->prev->left = n;
    else
      p->prev->right = n;
    p->prev = n;
    recalc_h(p);
    recalc_h(n);
  }
  int height(node *p) {
    return p == null ? -1 : p->h;
  }
  void recalc_h(node *p) {
    p->h = max(height(p->left), height(p->right)) + 1;
  }
  void insert_fix(node *p, int key) {
    while (p != null) {
      int fac = height(p->left) - height(p->right);
      if (fac > 1) {
        if (p->left->key > key)
          right_rotate(p);
        else {
          left_rotate(p->left);
          right_rotate(p);
        }
        break;
      } else if (fac < -1) {
        if (p->right->key < key)
          left_rotate(p);
        else {
          right_rotate(p->right);
          left_rotate(p);
        }
        break;
      } else {
        recalc_h(p);
        p = p->prev;
      }
    }
  }
  void print(const node *p, int n) const {
    if (p != null) {
      print(p->right, n + 1);
      for (int i = 0; i < n; i++)
        cout << "    ";
      cout << p->key << endl;
      print(p->left, n + 1);
    }
  }
  int depth(const node *p) const {
    return p == null ? 0 : max(depth(p->left), depth(p->right)) + 1;
  }
private:
  avl_tree(const avl_tree &) = delete;
  void operator=(const avl_tree &) = delete;
};

using int_type = size_t;

struct graph_weighed {
  vector<vector<pair<int_type, int_type>>> g{};
  int_type n{};
  int_type m{};

  void load(const string &path) {
    ifstream in(path);
    if (!in.is_open())
      throw invalid_argument("The file [" + path + "] can't open...");
    in >> n;
    in >> m;
    g.resize(n);
    int_type u, v, w;
    for (int_type i = 0; i < m; i++) {
      in >> u >> v >> w;
      g[u].push_back({v, w});
    }
    if (!in.good())
      throw invalid_argument("In process read [" + path + "] occurred errors...");
    in.close();
  }
  void print() const {
    for (int_type i = 0; i < n; i++) {
      cout << i;
      for (auto &[v, w] : g[i]) {
        cout << " -> [" << v << ", " << w << "]";
      }
      cout << endl;
    }
  }
  auto begin() const {
    return g.begin();
  }
  auto end() const {
    return g.end();
  }
  auto operator[](int_type i) const {
    return g[i];
  }
};

struct p_queue {
  const vector<int_type> &d;
  vector<int_type> indices;
  vector<int_type> h;
  int_type size;
public:
  p_queue(const vector<int_type> &__d) : d(__d), indices(d.size(), -1), h(d.size()), size(0) {}
  ~p_queue() {}
  void push(int_type u) {
    h[size] = u;
    indices[u] = size;
    sift_up(size);
    size++;
  }
  int_type pop() {
    size--;
    int_type u = h[0];
    h[0] = h[size];
    indices[h[size]] = 0;
    indices[u] = -1;
    sift_down(0);
    return u;
  }
  void dec_key(int_type u) {
    sift_up(indices[u]);
  }
private:
  void sift_up(int_type i) {
    int_type p = (i - 1) >> 1;
    while (i > 0 and d[h[i]] < d[h[p]]) {
      swap(indices[h[i]], indices[h[p]]);
      swap(h[i], h[p]);
      i = p;
      p = (i - 1) >> 1;
    }
  }
  void sift_down(int_type i) {
    int_type ls = (i << 1) + 1;
    int_type rs = (i << 1) + 2;
    while (ls < size) {
      int_type k = ls;
      if (rs < size and d[h[rs]] < d[h[ls]])
        k = rs;
      if (d[h[i]] < d[h[k]])
        return;
      swap(indices[h[i]], indices[h[k]]);
      swap(h[i], h[k]);
      i = k;
      ls = (i << 1) + 1;
      rs = (i << 1) + 2;
    }
  }
};

locale const int_type inf = INT_MAX;

auto dijkstra(const graph_weighed& g, int_type s, int_type t) {
  vector<int_type> d(g.n, inf);
  vector<int_type> p(g.n, -1);
  p_queue q(d);

  d[s] = 0;

  for (int_type i = 0; i < g.n; i++)
    q.push(i);

  while (q.size > 0) {
    int_type u = q.pop();
    if (d[u] == inf)
      break;
    for (auto &[v, w] : g[u]) {
      if (d[u] + w < d[v]) {
        d[v] = d[u] + w;
        p[v] = u;
        q.dec_key(v);
      }
    }
  }

  return d;
}

struct graph {
  vector<vector<int>> g{};
  int n{};
  int m{};

  void load(const string &path) {
    ifstream in(path);
    if (!in.is_open())
      throw invalid_argument("The file [" + path + "] can't open...");
    in >> n;
    in >> m;
    g.resize(n);
    int u, v;
    for (int i = 0; i < m; i++) {
      in >> u >> v;
      g[u].push_back(v);
    }
    if (!in.good())
      throw invalid_argument("In process read [" + path + "] occurred errors...");
    in.close();
  }
  void print() const {
    for (int i = 0; i < n; i++) {
      cout << i;
      for (auto &v : g[i]) {
        cout << " -> " << v;
      }
      cout << endl;
    }
  }
  auto begin() const {
    return g.begin();
  }
  auto end() const {
    return g.end();
  }
  auto operator[](int i) const {
    return g[i];
  }
};

struct graph_inc {
  struct edge {
    int u{};
    int v{};
    int w{};
  };
  vector<edge> g{};
  int n{};
  int m{};

  void load(const string &path) {
    ifstream in(path);
    if (!in.is_open())
      throw invalid_argument("The file [" + path + "] can't open...");
    in >> n;
    in >> m;
    int u, v, w;
    for (int i = 0; i < m; i++) {
      in >> u >> v >> w;
      g.push_back({u, v, w});
    }
    if (!in.good())
      throw invalid_argument("In process read [" + path + "] occurred errors...");
    in.close();
  }
  void print() const {
    for (auto &[u, v, w] : g)
      cout << "(" << u << ", " << v << ", " << w << ")" << endl;
  }
  auto begin() const {
    return g.begin();
  }
  auto end() const {
    return g.end();
  }
  auto operator[](int i) const {
    return g[i];
  }
};

bool bellman_ford(const graph_inc &g, int s, int t) {
  vector<int> d(g.n, inf);
  vector<int> p(g.n, -1);
  d[s] = 0;

  for (int i = 0; i < g.n - 1; i++) {
    for (auto &[u, v, w] : g) {
      if (d[u] < inf and d[u] + w < d[v]) {
        d[v] = d[u] + w;
        p[v] = u;
      }
    }
  }

  for (auto &[u, v, w] : g) {
    if (d[v] > d[u] + w) {
      cerr << "cycle with nagative weight...!" << endl;
      return false;
    }
  }

  for (auto i : d)
    cout << i << " ";
  cout << endl;

  if (d[t] != inf) {
    cout << "length of path from " << s << " to " << t << " = " << d[t] << endl;
    stack<int> st;
    while (t >= 0) {
      st.push(t);
      t = p[t];
    }
    while (!st.empty()) {
      cout << st.top() << " ";
      st.pop();
    }
    cout << endl;
  }
  return true;
}

void sep(int n, char ch) {
  cout << "\033[1;31m";
  for (int i = 0; i < n; i++)
    cout << ch;
  cout << "\033[0m" << endl;
}

template <typename T>
struct matrix {
  int n{};
  int m{};
  vector<vector<T>> mx{};

  matrix(int __n, int __m, const T& __v = T{}) : n(__n), m(__m), mx(__n, vector<T>(__m, __v)) {

  }

  auto& operator[](int i) {
    return mx[i];
  }

  auto& operator[](int i) const {
    return mx[i];
  }
};

auto floyd_warshall(const graph_inc& g) {
  int n = g.n;
  matrix<int> d(n, n, inf);
  matrix<int> prev(n, n, -1);

  for (auto &[u, v, w] : g) {
    d[u][v] = w;
    prev[u][v] = u;
  }

  for (int i = 0; i < n; i++) {
    d[i][i] = 0;
    prev[i][i] = -1;
  }

  for (int k = 0; k < n; k++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (d[i][k] + d[k][j] < d[i][j]) {
          d[i][j] = d[i][k] + d[k][j];
          prev[i][j] = k;
        }
      }
    }
  }

  return d;
}

namespace typoi {
  using row = const std::string&;

  auto bmm_bc(row p) {
    const int sigma = 256;
    int n = p.size();
    vector<int> d(sigma, n);
    for (int i = 0; i < n - 1; i++)
      d[p[i]] = n - i - 1;
    return d;
  }

  auto bm_match(row t, row p) {
    auto d = bmm_bc(p);
    int n = t.size();
    int m = p.size();
    vector<int> indices;
    for (int i = m - 1; i < n; i += d[t[i]]) {
      int k = m - 1;
      int s = i;
      while (k >= 0 and p[k] == t[s]) {
        k--;
        s--;
      }
      if (k < 0)
        indices.push_back(s + 1);
    }
    return indices;
  }

  auto kmp_pr(row s) {
    int n = s.size();
    vector<int> pi(n, 0);
    for (int i = 1, k = 0; i < n; i++) {
      while (k > 0 and s[i] != s[k])
        k = pi[k - 1];
      if (s[i] == s[k])
        k++;
      pi[i] = k;
    }
    return pi;
  }

  auto kmp_match(row t, row p) {
    auto pi = kmp_pr(p);
    int n = t.size();
    int m = p.size();
    vector<int> indices;
    for (int i = 0, k = 0; i < n; i++) {
      while (k > 0 and t[i] != p[k])
        k = pi[k - 1];
      if (t[i] == p[k])
        k++;
      if (k == m) {
        indices.push_back(i - m + 1);
        k = pi[k - 1];
      }
    }
    return indices;
  }

  auto naive_match(row t, row p) {
    vector<int> indices;
    int n = t.length();
    int m = p.length();
    for (int i = 0; i <= n - m; i++) {
      int k = 0;
      while (k < m and t[i + k] == p[k])
        k++;
      if (k == m)
        indices.push_back(i);
    }
    return indices;
  }

  auto rabin_karp_match(row t, row p) {
    const size_t d = 256;
    const size_t q = 72057594037927931;

    size_t n = t.length();
    size_t m = p.length();

    size_t ht = 0, hp = 0;
    for (size_t i = 0; i < m; i++) {
      ht = (d * ht + t[i]) % q;
      hp = (d * hp + p[i]) % q;
    }

    size_t h = 1;
    for (size_t i = 0; i < m - 1; i++)
      h = (h * d) % q;

    vector<int> res;
    for (size_t i = 0; i <= n - m; i++) {
      if (ht == hp and t.compare(i, m, p) == 0)
        res.push_back(i);
      ht = (d * (ht - t[i] * h) + t[i + m]) % q;
    }

    return res;
  }

  auto z_function_stupid(row s) {
    int n = s.length();
    vector<int> z(n, 0);
    for (int i = 1; i < n; i++) {
      for (int j = 0, k = i; k < n and s[j] == s[k]; j++, k++) {
        z[i]++;
      }
    }
    return z;
  }

  auto z_function(row s) {
    int n = s.size();
    vector<int> z(n, 0);
    for (int i = 1, l = 0, r = 0; i < n; i++) {
      if (i <= r)
        z[i] = min(r - i + 1, z[i - l]);
      while (i + z[i] < n and s[z[i]] == s[i + z[i]])
        z[i]++;
      if (z[i] > 0 and i + z[i] - 1 > r) {
        l = i;
        r = i + z[i] - 1;
      }
    }
    return z;
  }

  auto z_match(row t, row p) {
    auto z = z_function(p + "\a" + t);
    vector<int> res;
    int n = t.size();
    int m = p.size();
    for (int i = 0; i <= n - m; i++) {
      if (z[m + 1 + i] == m)
        res.push_back(i);
    }
    return res;
  }
}

int partition(int a[], int l, int r) {
  int x = a[r];
  int k = l;
  for (int i = l; i < r; i++) {
    if (a[i] <= x) {
      swap(a[i], a[k]);
      k++;
    }
  }
  swap(a[k], a[r]);
  return k;
}

void qsort(int a[], int l, int r) {
  while (l < r) {
    int k = partition(a, l, r);
    qsort(a, l, k - 1);
    l = k + 1;
  }
}

void qsort(int a[], int n) {
  qsort(a, 0, n - 1);
}

struct leftist_heap {
  struct node {
    int key{};
    int depth{};
    node *left{};
    node *right{};
  };
  node *root{};
  size_t size{};
public:
  leftist_heap() = default;
  ~leftist_heap() {
  }
  void insert(int key) {
    root = merge(root, new node{key});
    size++;
  }
  int extract() {
    assert(root);
    node *p = root;
    root = merge(root->left, root->right);
    int value = p->key;
    delete p;
    size--;
    return value;
  }
  int depth() {
    return depth(root);
  }
  int max_depth() {
    return max_depth(root);
  }
private:
  int max_depth(node *p) {
    return p == null ? 0 : max(max_depth(p->left), max_depth(p->right)) + 1;
  }
  int depth(node *p) {
    return p == null ? 0 : p->depth;
  }
  void update(node *p) {
    p->depth = min(depth(p->left), depth(p->right)) + 1;
  }
  node* merge(node *p1, node *p2) {
    if (p1 == null) return p2;
    if (p2 == null) return p1;
    if (p1->key > p2->key)
      swap(p1, p2);
    p1->right = merge(p1->right, p2);
    if (depth(p1->right) > depth(p1->left))
      swap(p1->left, p1->right);
    update(p1);
    return p1;
  }
private:
  leftist_heap(const leftist_heap&) = delete;
  void operator=(const leftist_heap&) = delete;
};

namespace badidea {

  struct line_info {
    int low{};
    int high{};
    int len{};

    line_info() = default;
    line_info(int l, int h) {
      low = l;
      high = h;
      len = h - l + 1;
    }
  };

  struct line_set {
    std::vector<line_info> table{line_info{0, -1}};
    int sz{1};
  public:
    line_set() = default;
    void resize(int i, int d) {
      assert(i >= 0 and i < sz);
      table[i].high += d;
      table[i].len += d;
      for (i += 1; i < sz; i++) {
        table[i].low += d;
        table[i].high += d;
      }
    }
    void split(int i, int offset) {
      assert(i >= 0 and i < sz);
      assert(offset >= table[i].low and offset <= table[i].high);
      line_info info = table[i];
      table[i] = line_info(info.low, offset);
      table.insert(table.begin() + i + 1, line_info(offset + 1, info.high));
      sz++;
    }
    void erase(int i) {
      assert(i >= 0 and i < sz);
      this->resize(i, -table[i].len);
      if (sz > 1) {
        table.erase(table.begin() + i);
        sz--;
      }
    }
    void merge(int i) {
      assert(i >= 0 and i + 1 < sz);
      table[i] = line_info(table[i].low, table[i + 1].high);
      table.erase(table.begin() + i + 1);
      sz--;
    }
    const line_info& operator[](int i) {
      assert(i >= 0 and i < sz);
      return table[i];
    }
  };

  struct stack {
    vector<char> st{};
    int sz{};
    stack() = default;
    void push(char ch) {
      st.push_back(ch);
      sz++;
    }
    char pop() {
      assert(sz > 0);
      char ch = st.back();
      st.pop_back();
      sz--;
      return ch;
    }
    char& operator[](int i) {
      assert(i >= 0 and i < sz);
      return st[i];
    }
  };

  struct gap_buf {
    stack a{};
    stack b{};
    int sz{};
    gap_buf() = default;
    void insert(char ch, int offset) {
      assert(offset >= 0 and offset <= sz);
      if (offset != a.sz)
        move(offset);
      a.push(ch);
      sz++;
    }
    char erase(int offset) {
      assert(offset > 0 and offset <= sz);
      if (offset != a.sz)
        move(offset);
      char tmp = a.pop();
      sz--;
      return tmp;
    }
    char& operator[](int i) {
      assert(i >= 0 and i < sz);
      if (i < a.sz)
        return a[i];
      return b[b.sz - 1 - (i - a.sz)];
    }
  private:
    void move(int offset) {
      int cn = abs(a.sz - offset);
      if (offset < a.sz) {
        for (int i = 0; i < cn; i++)
          b.push(a.pop());
      }
      else {
        for (int i = 0; i < cn; i++)
          a.push(b.pop());
      }
    }
  };

  struct editor_driver {
    gap_buf text{};
    line_set lines{};

    editor_driver() = default;
    void insert(char ch, int line, int offset) {
      text.insert(ch, offset);
      lines.resize(line, 1);
      if (ch == '\n')
        lines.split(line, offset);
    }
    char erase(int line, int offset) {
      char ech = text.erase(offset);
      if (ech == '\n') {
        assert(line > 0);
        line--;
        lines.merge(line);
      }
      lines.resize(line, -1);
      return ech;
    }
    int line_len(int i) {
      assert(i >= 0 and i < lines.sz);
      int len = lines[i].len;
      if (lines[i].high >= lines[i].low and text[lines[i].high] == '\n')
        len--;
      return len;
    }
    void cut_line(int i) {
      assert(i >= 0 and i < lines.sz);
      int offset = lines[i].high + 1;
      while (offset > lines[i].low) {
        text.erase(offset);
        offset--;
      }
      lines.erase(i);
    }
  private:
    editor_driver(const editor_driver&) = delete;
    void operator=(const editor_driver&) = delete;
  };

  const int min_rows = 12;
  const int min_cols = 42;

#define ctrl_key(key) ((key) & 0x1F)

  class editor {
    editor_driver driver;
    int rows;
    int cols;
    int xpos;
    int ypos;
    int beg_line;
    int beg_char;
    bool is_run;
  public:
    editor() {
      ncurses_init();
      getmaxyx(stdscr, rows, cols);
      assert(rows >= min_rows);
      assert(cols >= min_cols);
      xpos = 0;
      ypos = 0;
      beg_line = 0;
      beg_char = 0;
      is_run = true;
    }
    ~editor() {
      endwin();
    }
    int run() {
      while (is_run) {
        press_process();
        print_buf();
      }
      return 0;
    }
  private:
    void press_process() {
      int ch = getch();
      switch (ch) {
      case ctrl_key('q'): is_run = false; break; // ctrl-q
      case ctrl_key('k'): press_cut_line(); break; // ctrl-k
      case '\t': press_tab(); break;
      case KEY_LEFT: press_move_left(); break;
      case KEY_RIGHT: press_move_right(); break;
      case KEY_UP: press_move_up(); break;
      case KEY_DOWN: press_move_down(); break;
      case KEY_BACKSPACE: press_backspace(); break;
      case KEY_HOME: press_home(); break;
      case KEY_END: press_end(); break;
      default:
        if (isprint(ch) || isspace(ch))
          insert(ch);
      }
    }
    void press_cut_line() {
      driver.cut_line(ypos + beg_line);
      xpos = 0;
      beg_char = 0;
      if (ypos + beg_line >= driver.lines.sz) {
        if (ypos > 0)
          ypos--;
        else
          beg_line--;
      }
    }
    void press_tab() {
      this->insert(' ');
      this->insert(' ');
    }
    void press_move_left() {
      if (xpos > 0)
        xpos--;
      else if (beg_char > 0)
        beg_char--;
      else if (ypos + beg_line > 0) {
        if (ypos > 0)
          ypos--;
        else
          beg_line--;
        cursor_to_end(line_len());
      }
    }
    void press_move_right() {
      int len = line_len();
      if (xpos < std::min(len, cols - 1))
        xpos++;
      else if (len - beg_char >= cols)
        beg_char++;
      else if (ypos + beg_line < driver.lines.sz - 1) {
        if (ypos < rows - 1)
          ypos++;
        else
          beg_line++;
        cursor_to_begin();
      }
    }
    void press_move_up() {
      if (ypos > 0)
        ypos--;
      else if (beg_line > 0)
        beg_line--;
      int len = line_len();
      if (xpos + beg_char > len)
        cursor_to_end(len);
    }
    void press_move_down() {
      if (ypos < std::min(driver.lines.sz - 1, rows - 1))
        ypos++;
      else if (driver.lines.sz - beg_line > rows)
        beg_line++;
      int len = line_len();
      if (xpos + beg_char > len)
        cursor_to_end(len);
    }
    void press_backspace() {
      int offset = driver.lines[ypos + beg_line].low + xpos + beg_char;
      if (offset > 0) {
        int len{};
        if (ypos + beg_line > 0)
          len = driver.lines[ypos + beg_line - 1].len - 1;
        int ch = driver.erase(ypos + beg_line, offset);
        if (ch == '\n') {
          if (beg_line > 0)
            beg_line--;
          else if (ypos > 0)
            ypos--;
          cursor_to_end(len);
        } else {
          if (beg_char > 0)
            beg_char--;
          else if (xpos > 0)
            xpos--;
        }
      }
    }
    void press_home() {
      cursor_to_begin();
    }
    void press_end() {
      cursor_to_end(line_len());
    }
    void insert(char ch) {
      int offset = driver.lines[ypos + beg_line].low + xpos + beg_char;
      driver.insert(ch, ypos + beg_line, offset);
      if (ch == '\n') {
        if (ypos < rows - 1)
          ypos++;
        else
          beg_line++;
        xpos = 0;
        beg_char = 0;
      }
      else if (xpos < cols - 1)
        xpos++;
      else
        beg_char++;
    }
    void ncurses_init() {
      initscr();
      raw();
      noecho();
      keypad(stdscr, true);
      cbreak();
    }
    void cursor_to_end(int len) {
      if (len < cols) {
        xpos = len;
        beg_char = 0;
      } else {
        xpos = cols - 1;
        beg_char = len - cols + 1;
      }
    }
    void cursor_to_begin() {
      xpos = 0;
      beg_char = 0;
    }
    int line_len() {
      return driver.line_len(ypos + beg_line);
    }
    void print_buf() {
      for (int i = 0; i < rows; i++) {
        move(i, 0);
        clrtoeol();
        if (i + beg_line < driver.lines.sz) {
          int len = driver.lines[i + beg_line].len;
          int offset = driver.lines[i + beg_line].low;
          for (int j = 0; j < cols and j + beg_char < len; j++) {
            printw("%c", driver.text[offset + j + beg_char]);
          }
        }
      }
      refresh();
      move(ypos, xpos);
    }
  };
}

int main() {
  badidea::editor ed;
  return ed.run();
  exit(0);
}
