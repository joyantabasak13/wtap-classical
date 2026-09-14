#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace std;
using Clock = chrono::high_resolution_clock;

// ============================================================
// General Binary WTAP benchmark
//
// Model used here:
//   - n individual weapons
//   - m targets
//   - each weapon is assigned to exactly one target
//   - a target may receive zero, one, or multiple weapons
//
// P[i][j] = kill probability of weapon i against target j
// V[j]    = target value
//
// Survival objective:
//   C(D) = sum_j V[j] * product_i (1 - P[i][j] * D[i][j])
//
// Lower is better.
//
// Methods:
//   1) Exact exhaustive search (ground truth)
//   2) Greedy marginal-gain heuristic (matching the slide idea)
//   3) Simulated annealing
//   4) Repeated Hungarian heuristic
//
// IMPORTANT:
// Standard Hungarian is one-to-one. General WTAP allows multiple
// weapons on the same target. Therefore the "Hungarian" method below
// is a repeated-round heuristic: in each round it computes the best
// one-to-one assignment for the currently unused weapons and current
// marginal target values, updates survival, and repeats.
// ============================================================

struct Instance {
    int nWeapons;
    int nTargets;
    vector<vector<double>> P;
    vector<double> V;
};

struct Solution {
    vector<int> targetOfWeapon;  // targetOfWeapon[i] = target assigned to weapon i
    double cost = numeric_limits<double>::infinity();
    double milliseconds = 0.0;
    bool valid = true;
};

// ------------------------------------------------------------
// Objective
// ------------------------------------------------------------
double survivalCost(const Instance& ins, const vector<int>& assign) {
    vector<double> survival(ins.nTargets, 1.0);

    for (int i = 0; i < ins.nWeapons; ++i) {
        int j = assign[i];
        if (j >= 0 && j < ins.nTargets)
            survival[j] *= (1.0 - ins.P[i][j]);
    }

    double cost = 0.0;
    for (int j = 0; j < ins.nTargets; ++j)
        cost += ins.V[j] * survival[j];

    return cost;
}

// ------------------------------------------------------------
// Random instance generator
// ------------------------------------------------------------
Instance generateInstance(int nWeapons, int nTargets, uint64_t seed) {
    mt19937_64 rng(seed);

    // Avoid trivial 0/1 probabilities.
    uniform_real_distribution<double> pDist(0.10, 0.95);
    uniform_real_distribution<double> vDist(0.50, 1.50);

    Instance ins;
    ins.nWeapons = nWeapons;
    ins.nTargets = nTargets;
    ins.P.assign(nWeapons, vector<double>(nTargets));
    ins.V.resize(nTargets);

    for (int j = 0; j < nTargets; ++j)
        ins.V[j] = vDist(rng);

    for (int i = 0; i < nWeapons; ++i)
        for (int j = 0; j < nTargets; ++j)
            ins.P[i][j] = pDist(rng);

    return ins;
}

// ------------------------------------------------------------
// Exact exhaustive search
// Each weapon chooses one of m targets => m^n possibilities.
// This is the ground truth, but only practical for small cases.
// ------------------------------------------------------------
void exactDFS(const Instance& ins,
              int weapon,
              vector<int>& current,
              vector<double>& survival,
              double& bestCost,
              vector<int>& bestAssign) {
    if (weapon == ins.nWeapons) {
        double c = 0.0;
        for (int j = 0; j < ins.nTargets; ++j)
            c += ins.V[j] * survival[j];

        if (c < bestCost) {
            bestCost = c;
            bestAssign = current;
        }
        return;
    }

    for (int j = 0; j < ins.nTargets; ++j) {
        current[weapon] = j;

        double old = survival[j];
        survival[j] *= (1.0 - ins.P[weapon][j]);

        exactDFS(ins, weapon + 1, current, survival, bestCost, bestAssign);

        survival[j] = old;
    }
}

Solution solveExact(const Instance& ins, long double maxStates = 5.0e7L) {
    Solution sol;
    long double states = pow((long double)ins.nTargets,
                             (long double)ins.nWeapons);

    if (states > maxStates) {
        sol.valid = false;
        sol.cost = numeric_limits<double>::quiet_NaN();
        return sol;
    }

    auto start = Clock::now();

    vector<int> cur(ins.nWeapons, -1), best(ins.nWeapons, -1);
    vector<double> survival(ins.nTargets, 1.0);
    double bestCost = numeric_limits<double>::infinity();

    exactDFS(ins, 0, cur, survival, bestCost, best);

    auto stop = Clock::now();

    sol.targetOfWeapon = best;
    sol.cost = bestCost;
    sol.milliseconds =
        chrono::duration<double, milli>(stop - start).count();
    return sol;
}

// ------------------------------------------------------------
// Greedy marginal-gain heuristic
//
// Current survival of target j:
//   S[j] = product of (1-P) for already assigned weapons.
//
// Marginal expected reduction from assigning weapon i to target j:
//   gain(i,j) = V[j] * S[j] * P[i][j]
//
// Repeatedly choose the largest feasible gain.
// ------------------------------------------------------------
Solution solveGreedy(const Instance& ins) {
    auto start = Clock::now();

    vector<int> assign(ins.nWeapons, -1);
    vector<char> used(ins.nWeapons, false);
    vector<double> survival(ins.nTargets, 1.0);

    for (int step = 0; step < ins.nWeapons; ++step) {
        double bestGain = -1.0;
        int bestI = -1, bestJ = -1;

        for (int i = 0; i < ins.nWeapons; ++i) {
            if (used[i]) continue;

            for (int j = 0; j < ins.nTargets; ++j) {
                double gain = ins.V[j] * survival[j] * ins.P[i][j];

                if (gain > bestGain) {
                    bestGain = gain;
                    bestI = i;
                    bestJ = j;
                }
            }
        }

        if (bestI == -1) break;

        assign[bestI] = bestJ;
        used[bestI] = true;
        survival[bestJ] *= (1.0 - ins.P[bestI][bestJ]);
    }

    double cost = 0.0;
    for (int j = 0; j < ins.nTargets; ++j)
        cost += ins.V[j] * survival[j];

    auto stop = Clock::now();

    Solution sol;
    sol.targetOfWeapon = assign;
    sol.cost = cost;
    sol.milliseconds =
        chrono::duration<double, milli>(stop - start).count();
    return sol;
}

// ------------------------------------------------------------
// Simulated Annealing
//
// State: one target index for each weapon.
// Move: reassign one randomly chosen weapon to another target.
// ------------------------------------------------------------
Solution solveSA(const Instance& ins,
                 uint64_t seed,
                 int restarts = 10,
                 int iterationsPerRestart = 200000,
                 double startTemp = 0.20,
                 double endTemp = 1e-6) {
    auto totalStart = Clock::now();

    mt19937_64 rng(seed);
    uniform_int_distribution<int> targetDist(0, ins.nTargets - 1);
    uniform_int_distribution<int> weaponDist(0, ins.nWeapons - 1);
    uniform_real_distribution<double> u01(0.0, 1.0);

    double globalBest = numeric_limits<double>::infinity();
    vector<int> globalBestAssign;

    // Greedy gives SA a useful initial candidate.
    Solution greedy = solveGreedy(ins);

    for (int restart = 0; restart < restarts; ++restart) {
        vector<int> cur(ins.nWeapons);

        if (restart == 0) {
            cur = greedy.targetOfWeapon;
        } else {
            for (int i = 0; i < ins.nWeapons; ++i)
                cur[i] = targetDist(rng);
        }

        double curCost = survivalCost(ins, cur);
        double localBest = curCost;
        vector<int> localBestAssign = cur;

        for (int it = 0; it < iterationsPerRestart; ++it) {
            double frac = (iterationsPerRestart <= 1)
                              ? 1.0
                              : (double)it / (iterationsPerRestart - 1);

            // Geometric cooling.
            double T = startTemp * pow(endTemp / startTemp, frac);

            int i = weaponDist(rng);
            int oldTarget = cur[i];

            int newTarget = targetDist(rng);
            if (ins.nTargets > 1) {
                while (newTarget == oldTarget)
                    newTarget = targetDist(rng);
            }

            cur[i] = newTarget;
            double newCost = survivalCost(ins, cur);
            double delta = newCost - curCost;

            bool accept = false;
            if (delta <= 0.0) {
                accept = true;
            } else if (T > 0.0 && u01(rng) < exp(-delta / T)) {
                accept = true;
            }

            if (accept) {
                curCost = newCost;

                if (curCost < localBest) {
                    localBest = curCost;
                    localBestAssign = cur;
                }
            } else {
                cur[i] = oldTarget;
            }
        }

        if (localBest < globalBest) {
            globalBest = localBest;
            globalBestAssign = localBestAssign;
        }
    }

    auto totalStop = Clock::now();

    Solution sol;
    sol.targetOfWeapon = globalBestAssign;
    sol.cost = globalBest;
    sol.milliseconds =
        chrono::duration<double, milli>(totalStop - totalStart).count();
    return sol;
}

// ------------------------------------------------------------
// Hungarian algorithm for rectangular MIN-cost assignment.
//
// Input a[n][m]. It requires n <= m.
// Returns assignment[row] = selected column.
//
// Standard O(n^2 m) implementation using potentials.
// ------------------------------------------------------------
vector<int> hungarianMin(const vector<vector<double>>& a) {
    int n = (int)a.size();
    int m = n ? (int)a[0].size() : 0;

    if (n == 0) return {};
    if (n > m)
        throw runtime_error("hungarianMin requires rows <= columns.");

    const double INF = numeric_limits<double>::infinity();

    vector<double> u(n + 1), v(m + 1);
    vector<int> p(m + 1), way(m + 1);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        vector<double> minv(m + 1, INF);
        vector<char> used(m + 1, false);

        do {
            used[j0] = true;
            int i0 = p[j0];
            double delta = INF;
            int j1 = 0;

            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;

                double cur = a[i0 - 1][j - 1] - u[i0] - v[j];

                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }

                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }

            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else if (j > 0) {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
        } while (p[j0] != 0);

        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    vector<int> ans(n, -1);
    for (int j = 1; j <= m; ++j)
        if (p[j] != 0)
            ans[p[j] - 1] = j - 1;

    return ans;
}

// ------------------------------------------------------------
// One Hungarian round.
//
// We need one-to-one matching between a subset of unused weapons
// and target slots. If #weapons > #targets, select at most m
// weapons by transposing the problem.
//
// Cost is negative current marginal gain because Hungarian minimizes.
// ------------------------------------------------------------
vector<pair<int,int>> hungarianRound(
    const Instance& ins,
    const vector<int>& unusedWeapons,
    const vector<double>& survival) {

    int r = (int)unusedWeapons.size();
    int m = ins.nTargets;

    vector<pair<int,int>> chosen;

    if (r <= m) {
        // Rows = weapons, columns = targets.
        vector<vector<double>> cost(r, vector<double>(m));

        for (int rr = 0; rr < r; ++rr) {
            int i = unusedWeapons[rr];
            for (int j = 0; j < m; ++j) {
                double gain = ins.V[j] * survival[j] * ins.P[i][j];
                cost[rr][j] = -gain;
            }
        }

        vector<int> a = hungarianMin(cost);
        for (int rr = 0; rr < r; ++rr)
            if (a[rr] >= 0)
                chosen.push_back({unusedWeapons[rr], a[rr]});

    } else {
        // More unused weapons than targets.
        // Rows = targets, columns = unused weapons.
        vector<vector<double>> cost(m, vector<double>(r));

        for (int j = 0; j < m; ++j) {
            for (int cc = 0; cc < r; ++cc) {
                int i = unusedWeapons[cc];
                double gain = ins.V[j] * survival[j] * ins.P[i][j];
                cost[j][cc] = -gain;
            }
        }

        vector<int> a = hungarianMin(cost);
        for (int j = 0; j < m; ++j)
            if (a[j] >= 0)
                chosen.push_back({unusedWeapons[a[j]], j});
    }

    return chosen;
}

// ------------------------------------------------------------
// Repeated Hungarian heuristic
//
// Standard Hungarian cannot solve general multi-weapon-per-target
// WTAP directly. We use it in rounds:
//
//   1) solve best one-to-one matching under current marginal gains
//   2) assign those weapons
//   3) update target survival
//   4) repeat until all weapons are used
//
// This gives a meaningful Hungarian-based classical baseline.
// ------------------------------------------------------------
Solution solveRepeatedHungarian(const Instance& ins) {
    auto start = Clock::now();

    vector<int> assign(ins.nWeapons, -1);
    vector<char> used(ins.nWeapons, false);
    vector<double> survival(ins.nTargets, 1.0);

    int left = ins.nWeapons;

    while (left > 0) {
        vector<int> unused;
        for (int i = 0; i < ins.nWeapons; ++i)
            if (!used[i])
                unused.push_back(i);

        vector<pair<int,int>> round =
            hungarianRound(ins, unused, survival);

        if (round.empty())
            break;

        for (auto [i, j] : round) {
            if (used[i]) continue;

            used[i] = true;
            assign[i] = j;
            --left;

            survival[j] *= (1.0 - ins.P[i][j]);
        }
    }

    double cost = survivalCost(ins, assign);

    auto stop = Clock::now();

    Solution sol;
    sol.targetOfWeapon = assign;
    sol.cost = cost;
    sol.milliseconds =
        chrono::duration<double, milli>(stop - start).count();
    return sol;
}

// ------------------------------------------------------------
// Printing helpers
// ------------------------------------------------------------
void printInstance(const Instance& ins) {
    cout << "\nTarget values V:\n";
    for (double x : ins.V)
        cout << fixed << setprecision(4) << x << " ";
    cout << "\n\nKill-probability matrix P (weapons x targets):\n";

    for (int i = 0; i < ins.nWeapons; ++i) {
        for (int j = 0; j < ins.nTargets; ++j)
            cout << fixed << setprecision(4) << ins.P[i][j] << " ";
        cout << '\n';
    }
}

void printAssignment(const Solution& s) {
    for (int i = 0; i < (int)s.targetOfWeapon.size(); ++i)
        cout << "W" << i << "->T" << s.targetOfWeapon[i]
             << (i + 1 == (int)s.targetOfWeapon.size() ? "" : ", ");
    cout << '\n';
}

void printResult(const string& name,
                 const Solution& s,
                 const Solution* exact = nullptr) {
    cout << left << setw(24) << name;

    if (!s.valid) {
        cout << setw(18) << "SKIPPED"
             << setw(14) << "-"
             << setw(14) << "-"
             << '\n';
        return;
    }

    cout << right << fixed << setprecision(10)
         << setw(18) << s.cost
         << fixed << setprecision(3)
         << setw(14) << s.milliseconds;

    if (exact && exact->valid) {
        double gap = s.cost - exact->cost;
        double rel = (fabs(exact->cost) > 1e-15)
                         ? 100.0 * gap / fabs(exact->cost)
                         : 0.0;

        cout << fixed << setprecision(6)
             << setw(14) << rel;
    } else {
        cout << setw(14) << "-";
    }

    cout << '\n';
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int nWeapons, nTargets;
    int numCases = 1;
    uint64_t baseSeed = 123456789ULL;

    if (argc >= 3) {
        nWeapons = stoi(argv[1]);
        nTargets = stoi(argv[2]);

        if (argc >= 4)
            numCases = stoi(argv[3]);
        if (argc >= 5)
            baseSeed = stoull(argv[4]);
    } else {
        cout << "Number of weapons: ";
        cin >> nWeapons;

        cout << "Number of targets: ";
        cin >> nTargets;

        cout << "Number of random cases: ";
        cin >> numCases;

        cout << "Random seed: ";
        cin >> baseSeed;
    }

    if (nWeapons <= 0 || nTargets <= 0 || numCases <= 0) {
        cerr << "Invalid input.\n";
        return 1;
    }

    const long double MAX_EXACT_STATES = 5.0e7L;
    const int SA_RESTARTS = 10;
    const int SA_ITERS_PER_RESTART = 100000;

    // Append all runs to one CSV file.
    const string logFile = "wtap_results.csv";

    bool fileExists = false;
    {
        ifstream test(logFile);
        fileExists = test.good() && test.peek() != ifstream::traits_type::eof();
    }

    ofstream log(logFile, ios::app);
    if (!log) {
        cerr << "Could not open " << logFile << '\n';
        return 1;
    }

    if (!fileExists) {
        log << "weapons,targets,case,seed,method,survival,time_ms,gap_percent\n";
    }

    cout << fixed << setprecision(10);
    cout << "Case,Method,Survival,Time(ms),Gap(%)\n";

    for (int c = 0; c < numCases; ++c) {
        uint64_t seed =
            baseSeed + 0x9E3779B97F4A7C15ULL * (uint64_t)c;

        Instance ins = generateInstance(nWeapons, nTargets, seed);

        Solution exact = solveExact(ins, MAX_EXACT_STATES);
        Solution greedy = solveGreedy(ins);
        Solution hungarian = solveRepeatedHungarian(ins);
        // Solution sa = solveSA(
        //     ins,
        //     seed ^ 0xD1B54A32D192ED03ULL,
        //     SA_RESTARTS,
        //     SA_ITERS_PER_RESTART
        // );

        auto output = [&](const string& method, const Solution& s) {
            if (!s.valid) {
                cout << (c + 1) << "," << method << ",SKIPPED,-,-\n";

                log << nWeapons << ","
                    << nTargets << ","
                    << (c + 1) << ","
                    << seed << ","
                    << method << ","
                    << "SKIPPED,,\n";
                return;
            }

            double gap = numeric_limits<double>::quiet_NaN();

            if (exact.valid) {
                gap = (fabs(exact.cost) > 1e-15)
                    ? 100.0 * (s.cost - exact.cost) / fabs(exact.cost)
                    : 0.0;
            }

            cout << (c + 1) << ","
                 << method << ","
                 << s.cost << ","
                 << setprecision(3) << s.milliseconds << ","
                 << setprecision(6);

            if (exact.valid)
                cout << gap;
            else
                cout << "-";

            cout << '\n';
            cout << setprecision(10);

            log << nWeapons << ","
                << nTargets << ","
                << (c + 1) << ","
                << seed << ","
                << method << ","
                << setprecision(15) << s.cost << ","
                << setprecision(6) << s.milliseconds << ",";

            if (exact.valid)
                log << setprecision(10) << gap;

            log << '\n';
        };

        output("Exact", exact);
        output("Greedy", greedy);
        output("Hungarian", hungarian);
        // output("SA", sa);

        log.flush();
    }

    cout << "Results saved to " << logFile << '\n';
    return 0;
}