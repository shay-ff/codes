snippet snips "all snippets"
/*
snips
dsu
mint
for
pbds
powr
pqs
sieve
tc
vec
readvec
template
gcd
deb
sieve
*/
endsnippet

snippet sieve "sieve"
const int max = $1;
int is_prime[max + 10];
void findPrimes(int max){
	is_prime[0] = is_prime[1] = false;
	for (int i = 2; i * i <= max; i++) {
	    if (is_prime[i]) {
	        for (int j = i * i; j <= max; j += i)
	            is_prime[j] = false;
	    }
	}
}
endsnippet

snippet template "template"
#include<bits/stdc++.h>
using namespace std;

#ifdef ONLINE_JUDGE
#define debug(x...)
#define testcase(tc)
#else
#include "debug.h"
#endif

#define int int64_t
#define all(v) (v).begin(),(v).end()
#define allr(v) (v).rbegin(),(v).rend()
#define show(x) cout << x << endl
#define yesno(x) cout << ((x) ? "YES\n" : "NO\n")
const int N = 200200, INF = 8e18, MOD = 1000000007; //   1000000009 , 1000000023 ,1000000007 , 998244353
// lb(x) : first greater or equal than x, ub(x) : first greater than x
template <class T> bool setmin(T &a, const T &b) {return b < a ? a = b, 1 : 0;}
template <class T> bool setmax(T &a, const T &b) {return b > a ? a = b, 1 : 0;}

signed main(){
	ios_base::sync_with_stdio(false); cin.tie(NULL);
	$1

	return 0;
}
endsnippet

snippet for "for"
for (int ${1:i} = 0; $1 < ${2:n}; $1++) {
	$0
}
endsnippet


snippet vect "vector"
vector<${1:int}> ${2:arr};$0
endsnippet


snippet readvec "read vector"
vector<${1:int}> ${2:arr}(${3:n});
for ($1 &val : $2) {
	cin >> val;
}
$0
endsnippet

snippet sort "read vector"
sort(${1:arr}.begin(), $1.end());$0
endsnippet


snippet gcd "gcd"
template<typename T>
T gcd(T a, T b) {
	while (a) {
		b %= a;
		swap(a, b);
	}
	return b;
}
endsnippet

snippet powr "binpow"
template<typename T>
T binpow(T a, T b) {
	T ans = 1;
	while (b) {
		if (b & 1) {
			ans = 1LL * ans * a % MOD;
		}
		a = 1LL * a * a % MOD;
		b >>= 1;
	}
	return ans;
}
endsnippet

snippet mint "modular_int"
const int  MOD = (int)$1;
struct mint{
    int val;
    mint(long long v = 0) { if (v < 0) v = v % MOD + MOD; if (v >= MOD) v %= MOD; val = v; }
    static int mod_inv(int a, int m = MOD) { int g = m, r = a, x = 0, y = 1; while (r != 0) { int q = g / r; g %= r; swap(g, r); x -= q * y; swap(x, y); } return x < 0 ? x + m : x; }
    explicit operator int() const { return val; }
    mint &operator+=(const mint &other) { val += other.val; if (val >= MOD) val -= MOD; return *this; }
    mint &operator-=(const mint &other) { val -= other.val; if (val < 0) val += MOD; return *this; }
    typedef unsigned long long ull;
    ull fast_mod(ull a, ull b, ull M = MOD) { long long ret = a * b - M * ull(1.L / M * a * b); return ret + M * (ret < 0) - M * (ret >= (long long)M); }
    mint &operator*=(const mint &other) { val = fast_mod((ull)val, other.val); return *this; }
    mint &operator/=(const mint &other) { return *this *= other.inv(); }
    friend mint operator+(const mint &a, const mint &b) { return mint(a) += b; }
    friend mint operator-(const mint &a, const mint &b) { return mint(a) -= b; }
    friend mint operator*(const mint &a, const mint &b) { return mint(a) *= b; }
    friend mint operator/(const mint &a, const mint &b) { return mint(a) /= b; }
    mint &operator++() { val = val == MOD - 1 ? 0 : val + 1; return *this; }
    mint &operator--() { val = val == 0 ? MOD - 1 : val - 1; return *this; }
    mint operator++(int32_t) { mint before = *this; ++*this; return before; }
    mint operator--(int32_t) { mint before = *this; --*this; return before; }
    mint operator-() const { return val == 0 ? 0 : MOD - val; }
    bool operator==(const mint &other) const { return val == other.val; }
    bool operator!=(const mint &other) const { return val != other.val; }
    mint inv() const { return mod_inv(val); }
    mint pow(long long p) const { assert(p >= 0); mint a = *this, result = 1; while (p > 0) { if (p & 1) result *= a; a *= a; p >>= 1; } return result; }
    friend ostream &operator<<(ostream &stream, const mint &m) { return stream << m.val; }
    friend istream &operator>>(istream &stream, mint &m) { return stream >> m.val; }
    friend void __print(const mint &x) { cerr << x.val; }
};

const int maxn = $2;
mint fact[maxn], invf[maxn];
mint C(int n, int r){
    assert(n >= r && r >= 0);
    if(r == 0)
        return mint(1);
    return fact[n] * invf[r] * (invf[n - r]);
}
void bincof(int a, int b){
    fact[0] = 1;
    for(int i = 1; i <= maxn; ++i){
        fact[i] = fact[i - 1] * i;
        invf[i] = mint(fact[i]).inv();
    }
}
endsnippet

snippet fenwick "Fenwick tree"
struct Fenwick {
	vector<ll> tree;
	int n;

	Fenwick(int n) : n(n) {
		tree.assign(n, 0);
	}

	void point_add(int pos, ll val) {
		for (; pos < n; pos |= (pos + 1)) {
			tree[pos] += val;
		}
	}

	ll find_sum(int r) { // [0, r]
		ll ans = 0;
		for (; r >= 0; r = (r & (r + 1)) - 1) {
			ans += tree[r];
		}
		return ans;
	}

	ll find_sum(int l, int r) { // [l, r)
		return find_sum(r - 1) - find_sum(l - 1);
	}
};
endsnippet

// rarely use them but its handy


snippet sufarr "suffix array"
const char C = 'a' - 1; // before first letter // change
const char maxchar = 'z'; // change

vector<int> suffarray(string s) { // without $ at the end
	vector<int> p, c, pn, cn, cnt;
	int n = (int)s.size();
	c.assign(n, 0);
	for (int i = 0; i < n; i++) {
		c[i] = s[i] - C;
	}
	for (int j = 0; j <= (maxchar - C); j++) {
		for (int i = 0; i < n; i++) {
			if (c[i] == j) {
				p.push_back(i);
			}
		}
	}
	int maxc = c[p.back()];
	pn.resize(n);
	for (int k = 0; (1 << k) <= 2 * n; k++) {
		for (int i = 0; i < n; i++) {
			pn[i] = ((p[i] -  (1 << k)) % n + n) % n;
		}
		cnt.assign(maxc + 3, 0);
		for (int i = 0; i < n; i++) {
			cnt[c[i] + 1]++;
		}
		for (int i = 1; i <= maxc + 2; i++) {
			cnt[i] += cnt[i - 1];
		}
		for (int i = 0; i < n; i++) {
			p[cnt[c[pn[i]]]++] = pn[i];
		}
		cn.assign(n, 0);
		cn[p[0]] = 1;
		for (int i = 1; i < n; i++) {
			if (c[p[i]] == c[p[i - 1]] && c[(p[i] + (1 << k)) % n] == c[(p[i - 1] + (1 << k)) % n]) {
				cn[p[i]] = cn[p[i - 1]];
			} else {
				cn[p[i]] = cn[p[i - 1]] + 1;
			}
		}
		maxc = cn[p.back()];
		c = cn;
	}
	return p;
}

vector<int> findlcp(string s, vector<int> p) {
	vector<int> lcp, mem;
	int n = (int)s.size();
	mem.resize(n);
	for (int i = 0; i < n; i++) {
		mem[p[i]] = i;
	}
	lcp.assign(n, 0);
	for (int i = 0; i < n; i++) {
		if (i > 0) {
			lcp[mem[i]] = max(lcp[mem[i - 1]] - 1, 0);
		}
		if (mem[i] == n - 1) {
			continue;
		}
		while (max(i, p[mem[i] + 1]) + lcp[mem[i]] < n && s[i + lcp[mem[i]]] == s[p[mem[i] + 1] + lcp[mem[i]]]) {
			lcp[mem[i]]++;
		}
	}
	return lcp;
}
endsnippet


snippet aho "aho-corasik"
struct aho {
	vector<vector<int> > g, gr;
	vector<string> str;
	int root;
	int sz;
	vector<ll> ending;
	vector<int> link;
	char firstlet;
	int numlet = 0;

	aho():
		g(),
		gr(),
		str(),
		root(0),
		sz(0),
		ending(),
		link() {}

	aho(vector<string> q, char firlet = 'a') { // change
		firstlet = firlet;
		sz = q.size();
		str = q;
		g.clear();
		gr.clear();
		ending.clear();
		link.clear();
		root = 0;
		ending.assign(1, 0);
		numlet = 0;
		for (int i = 0; i < q.size(); i++) {
			for (int j = 0; j < q[i].size(); j++) {
				numlet = q[i][j] - firstlet;
			}
		}
		numlet++;
		g.push_back(vector<int>(numlet, -1));
		for (int i = 0; i < q.size(); i++) {
			int v = root;
			for (int j = 0; j < q[i].size(); j++) {
				if (g[v][q[i][j] - firstlet] == -1) {
					g[v][q[i][j] - firstlet] = g.size();
					g.push_back(vector<int>(numlet, -1));
					ending.push_back(0);
				}
				v = g[v][q[i][j] - firstlet];
			}
			ending[v]++;
		}
		link.assign(g.size(), -1);
		link[root] = root;
		queue<int> que;
		que.push(root);
		while (que.size()) {
			int v = que.front();
			que.pop();
			for (int i = 0; i < numlet; i++) {
				if (g[v][i] == -1) {
					if (v == root) {
						g[v][i] = v;
					} else {
						g[v][i] = g[link[v]][i];
					}
				}
				else {
					que.push(g[v][i]);
					if (v == root) {
						link[g[v][i]] = v;
					} else {
						link[g[v][i]] = g[link[v]][i];
					}
				}
		}
		gr.resize(g.size());
		for (int i = 0; i < g.size(); i++) {
			if (i != root) {
				gr[link[i]].push_back(i);
			}
		}
		dfslink(root);
	}

	void dfslink(int v) {
		for (int u : gr[v]) {
			ending[u] += ending[v];
			dfslink(u);
		}
	}

	ll find(string s) { // change
		ll ans = 0;
		int v = root;
		for (int i = 0; i < s.size(); i++) {
			v = g[v][s[i] - firstlet];
			ans += ending[v];
		}
		return ans;
	}
};
endsnippet
	

snippet cht "convex hull trick"
typedef long long integer;

struct Line {
	integer k, b;
	Line():
		k(0),
		b(0) {}
	Line(integer k, integer b):
		k(k),
		b(b) {}

	ld operator()(ld x) {
		return x * (ld)k + (ld)b;
	}
};

const integer INF = 2e18; // change

struct CHT {
	vector<Line> lines;
	bool mini; // cht on minimum

	ld f(Line l1, Line l2) {
		return (ld)(l1.b - l2.b) / (ld)(l2.k - l1.k);
	}

	void addLine(integer k, integer b) {
		if (!mini) {
			k = -k;
			b = -b;
		}
		Line l(k, b);
		while (lines.size() > 1) {
			if (lines.back().k == k) {
				if (lines.back().b > b) {
					lines.pop_back();
				} else {
					break;
				}
				continue;
			}
			ld x1 = f(lines.back(), l);
			ld x2 = f(lines.back(), lines[lines.size() - 2]);
			if (x1 > x2) {
				break;
			}
			lines.pop_back();
		}
		if (!lines.size() || lines.back().k != k) {
			lines.push_back(l);
		}
	}

	CHT(vector<pair<integer, integer> > v, bool ok = 1) { // change
		mini = ok;
		lines.clear();
		for (int i = 0; i < v.size(); i++) {
			addLine(v[i].first, v[i].second);
		}
	}

	integer getmin(integer x) { //find of integer!
		if (!lines.size()) {
			return (mini ? INF : -INF);
		}
		int l = 0, r = lines.size();
		while (r - l > 1) {
			int mid = (r + l) / 2;
			if (f(lines[mid], lines[mid - 1]) <= (ld)x) {
				l = mid;
			} else {
				r = mid;
			}
		}
		integer ans = lines[l].k * x + lines[l].b;
		return (mini ? ans : -ans);
	}
};
endsnippet

snippet segtree "segment tree"
struct SegmentTree {
	// TO CHANGE

	struct Node { // set default values
		...

		template<typename T>
		void apply(int l, int r, T val) { // update value and save push
			...
		}
	};

	Node merge(const Node& left, const Node& right) {
		...
	}

	void push(int v, int l, int r) {
		if (tree[v].??? != ...) {
			int mid = (r + l) >> 1;
			int vl = v + 1, vr = v + ((mid - l) << 1);
			tree[vl].apply(l, mid, tree[v].???);
			tree[vr].apply(mid, r, tree[v].???);
			tree[v].??? = ...;
		}
	}

	// DEFAULT PART

	vector<Node> tree;
	int n;

	template<typename T>
	void build(int v, int l, int r, const vector<T>& arr) {
		if (l + 1 == r) {
			tree[v].apply(l, r, arr[l]);
			return;
		}
		int mid = (r + l) >> 1;
		int vl = v + 1, vr = v + ((mid - l) << 1);
		build(vl, l, mid, arr);
		build(vr, mid, r, arr);
		tree[v] = merge(tree[vl], tree[vr]);
	}

	void build(int v, int l, int r) {
		if (l + 1 == r) {
			return;
		}
		int mid = (r + l) >> 1;
		int vl = v + 1, vr = v + ((mid - l) << 1);
		build(vl, l, mid);
		build(vr, mid, r);
		tree[v] = merge(tree[vl], tree[vr]);
	}

	Node find(int v, int l, int r, int ql, int qr) {
		if (ql <= l && r <= qr) {
			return tree[v];
		}
		push(v, l, r);
		int mid = (r + l) >> 1;
		int vl = v + 1, vr = v + ((mid - l) << 1);
		if (qr <= mid) {
			return find(vl, l, mid, ql, qr);
		} else if (ql >= mid) {
			return find(vr, mid, r, ql, qr);
		} else {
			return merge(find(vl, l, mid, ql, qr), find(vr, mid, r, ql, qr));
		}
	}

	template<typename T>
	void update(int v, int l, int r, int ql, int qr, const T& newval) {
		if (ql <= l && r <= qr) {
			tree[v].apply(l, r, newval);
			return;
		}
		push(v, l, r);
		int mid = (r + l) >> 1;
		int vl = v + 1, vr = v + ((mid - l) << 1);
		if (ql < mid) {
			update(vl, l, mid, ql, qr, newval);
		}
		if (qr > mid) {
			update(vr, mid, r, ql, qr, newval);
		}
		tree[v] = merge(tree[vl], tree[vr]);
	}

	int find_first(int v, int l, int r, int ql, int qr, const function<bool(const Node&)>& predicate) {
		if (!predicate(tree[v])) {
			return -1;
		}
		if (l + 1 == r) {
			return l;
		}
		push(v, l, r);
		int mid = (r + l) >> 1;
		int vl = v + 1, vr = v + ((mid - l) << 1);
		if (ql < mid) {
			int lans = find_first(vl, l, mid, ql, qr, predicate);
			if (lans != -1) {
				return lans;
			}
		}
		if (qr > mid) {
			int rans = find_first(vr, mid, r, ql, qr, predicate);
			if (rans != -1) {
				return rans;
			}
		}
		return -1;
	}

	// INTERFACE

	SegmentTree(int n) : n(n) { // build from size with default values
		tree.resize(2 * n - 1);
		build(0, 0, n);
	}

	template<typename T>
	SegmentTree(const vector<T>& arr) { // build from vector
		n = arr.size();
		tree.resize(2 * n - 1);
		build(0, 0, n, arr);
	}

	Node find(int ql, int qr) { // find value on [ql, qr)
		return find(0, 0, n, ql, qr);
	}

	Node find(int qi) { // find value of position qi
		return find(0, 0, n, qi);
	}

	template<typename T>
	void update(int ql, int qr, const T& newval) { // update [ql, qr) with newval
		update(0, 0, n, ql, qr, newval);
	}

	template<typename T>
	void update(int qi, const T& newval) { // update position qi with newval
		update(0, 0, n, qi, qi + 1, newval);
	}

	int find_first(int ql, int qr, const function<bool(const Node&)>& predicate) { // find first index on [ql, qr) that satisfies predicate or -1 if none
		return find_first(0, 0, n, ql, qr, predicate);
	}

	int find_first(int ql, const function<bool(const Node&)>& predicate) { // find first index >= ql that satisfies predicate or -1 if none
		return find_first(0, 0, n, ql, n, predicate);
	}

	int find_first(const function<bool(const Node&)>& predicate) { // find first index that satisfies predicate or -1 if none
		return find_first(0, 0, n, 0, n, predicate);
	}
};
endsnippet
	
snippet centroid "centroid decomposition"
const int MAXN = ;

vector<int> g[MAXN], used, p, d;

int cnt;

int dfs(int v, int pr) {
	cnt++;
	d[v] = 1;
	for (int u : g[v]) {
		if (!used[u] && u != pr) {
			d[v] += dfs(u, v);
		}
	}
	return d[v];
}

int centroid(int v) {
	cnt = 0;
	dfs(v, -1);
	int pr = -1;
	while (true) {
		int z = -1;
		for (int u : g[v]) {
			if (!used[u] && u != pr && d[u] * 2 >= cnt) {
				z = u;
			}
		}
		if (z == -1) {
			break;
		}
		pr = v;
		v = z;
	}
	return v;
}

void go(int v, int pr) {
	v = centroid(v);
	p[v] = pr;
	used[v] = 1;

	for (int u : g[v]) {
		if (!used[u]) {
			go(u, v);
		}
	}
}
endsnippet
	

snippet sparse "sparse table"
template<typename T>
struct SparseTable {
	vector<vector<T>> sparse;
	function<T(const T&, const T&)> accum_func;

	SparseTable(const vector<T>& arr, const function<T(const T&, const T&)>& func) : accum_func(func) {
		int n = arr.size();
		int logn = 32 - __builtin_clz(n);
		sparse.resize(logn, vector<T>(n));
		sparse[0] = arr;
		for (int lg = 1; lg < logn; lg++) {
			for (int i = 0; i + (1 << lg) <= n; i++) {
				sparse[lg][i] = accum_func(sparse[lg - 1][i], sparse[lg - 1][i + (1 << (lg - 1))]);
			}
		}
	}

	T find(int l, int r) { // [l, r)
		int cur_log = 31 - __builtin_clz(r - l);
		return accum_func(sparse[cur_log][l], sparse[cur_log][r - (1 << cur_log)]);
	}
};
endsnippet
	


snippet decart "treap"
struct Node {
	int x;
	ll y;
	int sz;
	Node *left;
	Node *right;
	Node(int x = 0):
		x(x),
		y((ll)rnd()),
		sz(1),
		left(NULL),
		right(NULL) {}
};

int sz(Node *v) {
	return (v == NULL ? 0 : v->sz);
}

Node* upd(Node *v) {
	if (v != NULL) {
		v->sz = 1 + sz(v->left) + sz(v->right);
	}
	return v;
}

Node* merge(Node *l, Node *r) {
	if (l == NULL) {
		return r;
	}
	if (r == NULL) {
		return l;
	}
	if (l->y < r->y) {
		l = merge(l, r->left);
		r->left = l;
		r = upd(r);
		return r;
	}
	r = merge(l->right, r);
	l->right = r;
	l = upd(l);
	return l;
}

pair<Node*, Node*> keySplit(Node *v, int key) { // l's keys <= key, r's keys > key
	if (v == NULL) {
		return {v, v};
	}
	if (v->x <= key) {
		auto a = keySplit(v->right, key);
		v->right = a.first;
		v = upd(v);
		return {v, a.second};
	}
	auto a = keySplit(v->left, key);
	v->left = a.second;
	v = upd(v);
	return {a.first, v};
}

pair<Node*, Node*> sizeSplit(Node *v, int siz) { // l's size is siz
	if (!v) {
		return {v, v};
	}
	if (sz(v->left) >= siz) {
		auto a = sizeSplit(v->left, siz);
		v->left = a.second;
		v = upd(v);
		return {a.first, v};
	}
	auto a = sizeSplit(v->right, siz - sz(v->left) - 1);
	v->right = a.first;
	v = upd(v);
	return {v, a.second};
}

void gogo(Node *v) {
	if (v == NULL) {
		return;
	}
	gogo(v->left);
	cerr << v->x << endl;
	gogo(v->right);
}
endsnippet


snippet Fenwick2D "2D Fenwick tree"
struct Fenwick2D {
	vector<vector<ll>> tree;
	int n, m;

	Fenwick2D(int n, int m) : n(n), m(m) {
		tree.assign(n, vector<ll>(m, 0));
	}

	void point_add(int posx, int posy, ll val) {
		for (int x = posx; x < n; x |= (x + 1)) {
			for (int y = posy; y < m; y |= (y + 1)) {
				tree[x][y] += val;
			}
		}
	}

	ll find_sum(int rx, int ry) { // [0, rx] x [0, ry]
		ll ans = 0;
		for (int x = rx; x >= 0; x = (x & (x + 1)) - 1) {
			for (int y = ry; y >= 0; y = (y & (y + 1)) - 1) {
				ans += tree[x][y];
			}
		}
		return ans;
	}

	ll find_sum(int lx, int rx, int ly, int ry) { // [lx, rx) x [ly, ry)
		return find_sum(rx - 1, ry - 1) - find_sum(rx - 1, ly - 1) - find_sum(lx - 1, ry - 1) + find_sum(lx - 1, ly - 1);
	}
};
endsnippet

snippet modular "modular arithmetics"
template<int MODULO>
struct ModularInt {
	int value;

	ModularInt(ll llvalue) : value(llvalue % MODULO) {
		if (value < 0) {
			value += MODULO;
		}
	}

	ModularInt(const ModularInt<MODULO>& other) : value(other.value) {}

	inline void operator+=(ModularInt<MODULO> other) {
		value += other.value;
		if (value >= MODULO) {
			value -= MODULO;
		}
	}

	inline ModularInt<MODULO> operator+(ModularInt<MODULO> other) const {
		return ModularInt<MODULO>(value + other.value >= MODULO ? value + other.value - MODULO : value + other.value);
	}

	inline void operator-=(ModularInt<MODULO> other) {
		value -= other.value;
		if (value < 0) {
			value += MODULO;
		}
	}

	inline ModularInt<MODULO> operator-(ModularInt<MODULO> other) const {
		return ModularInt<MODULO>(value - other.value < 0 ? value - other.value + MODULO : value - other.value);
	}

	inline ModularInt<MODULO> operator-() const {
		return ModularInt<MODULO>(value == 0 ? value : MODULO - value);
	}

	inline ModularInt<MODULO>& operator++() {
		++value;
		if (value == MODULO) {
			value = 0;
		}
		return *this;
	}

	inline ModularInt<MODULO> operator++(int) {
		ModularInt<MODULO> old(*this);
		++value;
		if (value == MODULO) {
			value = 0;
		}
		return old;
	}

	inline ModularInt<MODULO>& operator--() {
		--value;
		if (value == -1) {
			value = MODULO - 1;
		}
		return *this;
	}

	inline ModularInt<MODULO> operator--(int) {
		ModularInt<MODULO> old(*this);
		--value;
		if (value == -1) {
			value = MODULO - 1;
		}
		return old;
	}

	inline ModularInt<MODULO> operator*(ModularInt<MODULO> other) const {
		return ModularInt<MODULO>(1LL * value * other.value);
	}

	inline void operator*=(ModularInt<MODULO> other) {
		value = 1LL * value * other.value % MODULO;
	}

	friend ModularInt<MODULO> binpow(ModularInt<MODULO> a, ll bll) {
		if (a.value == 0) {
			return ModularInt<MODULO>(bll == 0 ? 1 : 0);
		}
		int b = bll % (MODULO - 1);
		int ans = 1;
		while (b) {
			if (b & 1) {
				ans = 1LL * ans * a % MODULO;
			}
			a = 1LL * a * a % MODULO;
			b >>= 1;
		}
		return ModularInt<MODULO>(ans);
	}

	inline ModularInt<MODULO> inv() const {
		return binpow(*this, MODULO - 2);
	}

	inline ModularInt<MODULO> operator/(ModularInt<MODULO> other) const {
		return (*this) * other.inv();
	}

	inline void operator/=(ModularInt<MODULO> other) {
		value = 1LL * value * other.inv().value % MODULO;
	}

	inline bool operator==(ModularInt<MODULO> other) const {
		return value == other.value;
	}

	inline bool operator!=(ModularInt<MODULO> other) const {
		return value != other.value;
	}

	explicit operator int() const {
		return value;
	}

	explicit operator bool() const {
		return value;
	}

	explicit operator long long() const {
		return value;
	}

	friend istream& operator>>(istream& inp, const ModularInt<MODULO>& mint) {
		inp >> mint.value;
		return inp;
	}

	friend ostream& operator<<(ostream& out, const ModularInt<MODULO>& mint) {
		out << mint.value;
		return out;
	}
};

const int MOD = ;

typedef ModularInt<MOD> MInt;
endsnippet

snippet table "table graph"
int dx[] = {0, 1, 0, -1};
int dy[] = {1, 0, -1, 0};
int n, m; // DON'T MAKE THEM IN MAIN

bool check(int x, int y) {
	return x >= 0 && x < n && y >= 0 && y < m;
}
endsnippet

snippet { "block"
{
	$0
}
endsnippet

snippet dsu "Disjoint Set Union"
struct DSU {
	vector<int> pr;
	int n;

	DSU(int n) : n(n) {
		pr.resize(n);
		iota(pr.begin(), pr.end(), 0);
	}

	inline int findpr(int v) {
		return (v == pr[v] ? v : (pr[v] = findpr(pr[v])));
	}

	inline bool unite(int v, int u) {
		v = findpr(v);
		u = findpr(u);
		if (u == v) {
			return false;
		} else {
			pr[v] = u;
			return true;
		}
	}
};
endsnippet

snippet deb "debug output"
#ifdef ONPC
	void debug_print(string s) {
		cerr << "\"" << s << "\"";
	}

	void debug_print(const char* s) {
		debug_print((string)s);
	}

	void debug_print(bool val) {
		cerr << (val ? "true" : "false");
	}

	void debug_print(int val) {
		cerr << val;
	}

	void debug_print(ll val) {
		cerr << val;
	}

	template<typename F, typename S>
	void debug_print(pair<F, S> val) {
		cerr << "(";
		debug_print(val.first);
		cerr << ", ";
		debug_print(val.second);
		cerr << ")";
	}

	void debug_print(vector<bool> val) {
		cerr << "{";
		bool first = true;
		for (bool x : val) {
			if (!first) {
				cerr << ", ";
			} else {
				first = false;
			}
			debug_print(x);
		}
		cerr << "}";
	}

	template<typename T>
	void debug_print(T val) {
		cerr << "{";
		bool first = true;
		for (const auto &x : val) {
			if (!first) {
				cerr << ", ";
			} else {
				first = false;
			}
			debug_print(x);
		}
		cerr << "}";
	}

	void debug_print_collection() {
		cerr << endl;
	}

	template<typename First, typename... Args>
	void debug_print_collection(First val, Args... args) {
		cerr << " ";
		debug_print(val);
		debug_print_collection(args...);
	}

#define debug(...) cerr << "@@@ " << #__VA_ARGS__ << " ="; debug_print_collection(__VA_ARGS__);

#else
	#define debug(...) 
#endif
endsnippet