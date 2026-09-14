#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <sstream>
#include <set>
#include <string>
#include <tuple>
#include <vector>

using namespace std;
using Clock = chrono::high_resolution_clock;

// Randomized Regret-CWTAA defaults.
constexpr int    RR_TOP_K = 5;
constexpr double RR_ALPHA = 2.0;
constexpr int    RR_NUM_RESTARTS = 10;
constexpr double RR_EPSILON = 1e-12;

// Random-score perturbation (Idea 5):
// Each restart perturbs every base score multiplicatively:
//
//     S_tilde[i][j] = S[i][j] * (1 + epsilon_ij)
//
// where epsilon_ij ~ Uniform[-PRR_NOISE, +PRR_NOISE].
//
// The perturbed scores are used ONLY to construct the assignment.
// The returned solution is always evaluated with the original
// unperturbed WTAP objective.
constexpr double PRR_NOISE = 0.05;   // legacy full-score perturbation, +/- 5%

// Improved perturbation experiments:
// 1) Baseline-preserving hybrid:
//      include ordinary RR-CWTAA candidates plus perturbed candidates,
//      and keep the best under the true objective.
// 2) Smaller perturbation sweep for Gaussian-sensitive near-ties.
// 3) Priority-only perturbation:
//      perturb target regret priority, but ALWAYS assign the target's
//      true best currently feasible weapon according to the unperturbed score.
constexpr double RPP_NOISE = 0.005;   // +/- 0.5% regret-priority perturbation
constexpr int    HYBRID_RR_RESTARTS = 10;
constexpr int    HYBRID_PERTURBED_RESTARTS = 10;

// ============================================================
// WTAP benchmark: CWTAA vs Regret-CWTAA vs Randomized-Regret-CWTAA vs Perturbed-Randomized-Regret-CWTAA vs Repeated Hungarian
//
// Paper model:
//
//   n weapon types, m targets
//   P[i][j] = kill probability of weapon type i against target j
//   V[j]    = target value
//   W[i]    = available count of weapon type i
//   D[i][j] in {0,1}
//
// Objective:
//
//   C(D) = sum_j V[j] * product_i (1 - P[i][j]) ^ D[i][j]
//
// Constraints used by the paper's Algorithm 1:
//
//   sum_j D[i][j] <= W[i]      for each weapon type i
//   sum_i D[i][j] <= 1         for each target j
//
// Lower C(D) is better.
//
// IMPORTANT PAPER DETAIL:
// The paper uses n = m, W[i] >= 1, and sum_i W[i] = m.
// Therefore, for the exact paper benchmark, W[i] = 1 for every i.
// The generator below still implements the stated general rule for n <= m:
// initialize every W[i] = 1, then distribute the remaining m-n units randomly.
// ============================================================

enum class PDistribution {
    Uniform,
    Beta,
    Gaussian
};

string distName(PDistribution d) {
    if (d == PDistribution::Uniform) return "Uniform";
    if (d == PDistribution::Beta) return "Beta(2,5)";
    return "Gaussian(0.7,0.15)_clipped";
}

PDistribution parseDistribution(const string& sRaw) {
    string s = sRaw;
    transform(s.begin(), s.end(), s.begin(),
              [](unsigned char c) { return static_cast<char>(tolower(c)); });

    if (s == "uniform" || s == "u")
        return PDistribution::Uniform;
    if (s == "beta" || s == "b" || s == "beta(2,5)")
        return PDistribution::Beta;
    if (s == "gaussian" || s == "normal" || s == "g")
        return PDistribution::Gaussian;

    throw invalid_argument(
        "Unknown distribution: " + sRaw +
        ". Use uniform, beta, or gaussian."
    );
}


struct Instance {
    int nTypes = 0;
    int nTargets = 0;
    vector<vector<double>> P;
    vector<double> V;
    vector<int> W;
};

struct Solution {
    // typeOfTarget[j] = weapon type assigned to target j, or -1 if unassigned.
    vector<int> typeOfTarget;

    // Paper objective.
    double totalSurvivalValue = numeric_limits<double>::infinity();

    // C(D) / sum_j V[j]. This is a target-value-weighted survival fraction.
    double normalizedSurvival = numeric_limits<double>::infinity();

    // Unweighted mean of each target's survival probability.
    double meanSurvivalProbability = numeric_limits<double>::infinity();

    double milliseconds = 0.0;
    bool valid = true;
};

// ------------------------------------------------------------
// Paper-faithful data generation
//
// P:
//   Uniform:  U[0,1]
//   Beta:     Beta(2,5)
//   Gaussian: N(0.7,0.15), clipped to [0,1]
//
// V:
//   U[0.1,10.0]
//
// W:
//   W[i] = 1 initially.
//   Distribute remaining m-n weapons uniformly at random.
// ------------------------------------------------------------
Instance generateInstance(
    int nTypes,
    int nTargets,
    PDistribution distribution,
    uint64_t seed
) {
    if (nTypes <= 0 || nTargets <= 0)
        throw invalid_argument("nTypes and nTargets must be positive.");

    if (nTypes > nTargets)
        throw invalid_argument(
            "Paper-style W generation requires nTypes <= nTargets "
            "(because W[i] >= 1 and sum W[i] = nTargets)."
        );

    mt19937_64 rng(seed);

    uniform_real_distribution<double> uniformP(0.0, 1.0);
    uniform_real_distribution<double> valueDist(0.1, 10.0);
    normal_distribution<double> gaussianP(0.7, 0.15);

    // Beta(a,b) via Gamma(a,1)/(Gamma(a,1)+Gamma(b,1)).
    gamma_distribution<double> gammaA(2.0, 1.0);
    gamma_distribution<double> gammaB(5.0, 1.0);

    Instance ins;
    ins.nTypes = nTypes;
    ins.nTargets = nTargets;
    ins.P.assign(nTypes, vector<double>(nTargets, 0.0));
    ins.V.resize(nTargets);
    ins.W.assign(nTypes, 1);

    for (int j = 0; j < nTargets; ++j)
        ins.V[j] = valueDist(rng);

    for (int i = 0; i < nTypes; ++i) {
        for (int j = 0; j < nTargets; ++j) {
            double x = 0.0;

            if (distribution == PDistribution::Uniform) {
                x = uniformP(rng);
            }
            else if (distribution == PDistribution::Beta) {
                double a = gammaA(rng);
                double b = gammaB(rng);
                x = a / (a + b);
            }
            else {
                x = gaussianP(rng);
                x = max(0.0, min(1.0, x));
            }

            ins.P[i][j] = x;
        }
    }

    // Paper: initialize W_i = 1 and distribute remaining m-n weapons
    // uniformly at random. For n=m, remaining=0, hence all W_i=1.
    int remaining = nTargets - nTypes;
    if (remaining > 0) {
        uniform_int_distribution<int> typeDist(0, nTypes - 1);
        for (int k = 0; k < remaining; ++k)
            ++ins.W[typeDist(rng)];
    }

    return ins;
}

// ------------------------------------------------------------
// Evaluate the paper objective and two normalized metrics.
// Because Algorithm 1 permits at most one weapon type per target,
// target j has survival 1 if unassigned, otherwise 1-P[i][j].
// ------------------------------------------------------------
void evaluateSolution(const Instance& ins, Solution& sol) {
    double total = 0.0;
    double sumV = 0.0;
    double sumSurvivalProb = 0.0;

    for (int j = 0; j < ins.nTargets; ++j) {
        double survival = 1.0;
        int i = sol.typeOfTarget[j];

        if (i >= 0) {
            if (i >= ins.nTypes)
                throw runtime_error("Invalid weapon type in solution.");
            survival = 1.0 - ins.P[i][j];
        }

        total += ins.V[j] * survival;
        sumV += ins.V[j];
        sumSurvivalProb += survival;
    }

    sol.totalSurvivalValue = total;
    sol.normalizedSurvival = (sumV > 0.0) ? total / sumV : 0.0;
    sol.meanSurvivalProbability =
        (ins.nTargets > 0) ? sumSurvivalProb / ins.nTargets : 0.0;
}

// ------------------------------------------------------------
// CWTAA -- implemented as Algorithm 1 in the paper.
//
// Step 1:
//   S[i][j] = P[i][j] * V[j]
//
// Step 2:
//   Repeatedly choose global argmax S[i][j].
//   If W[i] > 0:
//       D[i][j] = 1
//       W[i]--
//       zero target column j
//   else:
//       zero weapon row i
//
// The implementation deliberately follows the paper's matrix-zeroing
// semantics rather than replacing it with a different greedy heuristic.
// ------------------------------------------------------------
Solution solveCWTAA(const Instance& ins) {
    auto start = Clock::now();

    const int n = ins.nTypes;
    const int m = ins.nTargets;

    vector<vector<double>> S(n, vector<double>(m, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            S[i][j] = ins.P[i][j] * ins.V[j];

    vector<int> w = ins.W;
    vector<int> typeOfTarget(m, -1);

    int weaponsLeft = accumulate(w.begin(), w.end(), 0);

    while (weaponsLeft > 0) {
        double bestScore = 0.0;
        int bestI = -1;
        int bestJ = -1;

        // Algorithm 1: argmax over the scoring matrix.
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (S[i][j] > bestScore) {
                    bestScore = S[i][j];
                    bestI = i;
                    bestJ = j;
                }
            }
        }

        // Equivalent to paper condition max(S) > 0.
        if (bestI < 0 || bestScore <= 0.0)
            break;

        if (w[bestI] > 0) {
            // Assign one unit of weapon type bestI to target bestJ.
            typeOfTarget[bestJ] = bestI;
            --w[bestI];
            --weaponsLeft;

            // Paper: zero the whole target column after assignment.
            for (int i = 0; i < n; ++i)
                S[i][bestJ] = 0.0;
        }
        else {
            // Paper: if weapon type is exhausted, zero its row.
            for (int j = 0; j < m; ++j)
                S[bestI][j] = 0.0;
        }
    }

    auto stop = Clock::now();

    Solution sol;
    sol.typeOfTarget = move(typeOfTarget);
    sol.milliseconds =
        chrono::duration<double, milli>(stop - start).count();

    evaluateSolution(ins, sol);
    return sol;
}


// ------------------------------------------------------------
// Regret-First CWTAA
//
// Uses the same paper score:
//   S[i][j] = P[i][j] * V[j]
//
// At every step, for each still-unassigned target j:
//
//   best(j)   = largest feasible S[i][j] among weapon types with W[i] > 0
//   second(j) = second-largest feasible S[i][j]
//
// Define target regret:
//
//   regret(j) = best(j) - second(j)
//
// The target with the largest regret is assigned first to its
// best currently available weapon type.
//
// Intuition:
// A large regret means that target j has one especially valuable
// weapon choice and a much worse fallback. Assigning such a target
// early protects scarce high-quality pairings from being consumed
// by other targets.
//
// Tie-breaking (deterministic):
//   1) larger regret
//   2) larger best score
//   3) smaller target index
//
// This is NOT the paper's original CWTAA. It is a new heuristic
// benchmarked against paper-faithful CWTAA and Repeated Hungarian.
// ------------------------------------------------------------
Solution solveRegretCWTAA(const Instance& ins) {
    auto start = Clock::now();

    const int n = ins.nTypes;
    const int m = ins.nTargets;

    vector<vector<double>> S(n, vector<double>(m, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            S[i][j] = ins.P[i][j] * ins.V[j];

    vector<int> w = ins.W;
    vector<char> targetUsed(m, false);
    vector<int> typeOfTarget(m, -1);

    int unitsLeft = accumulate(w.begin(), w.end(), 0);
    int targetsLeft = m;

    const double EPS = 1e-15;

    while (unitsLeft > 0 && targetsLeft > 0) {
        int chosenTarget = -1;
        int chosenWeapon = -1;
        double chosenRegret = -numeric_limits<double>::infinity();
        double chosenBestScore = -numeric_limits<double>::infinity();

        // Compute regret for every still-unassigned target.
        for (int j = 0; j < m; ++j) {
            if (targetUsed[j]) continue;

            double bestScore = -numeric_limits<double>::infinity();
            double secondBestScore = -numeric_limits<double>::infinity();
            int bestWeapon = -1;

            for (int i = 0; i < n; ++i) {
                if (w[i] <= 0) continue;

                double score = S[i][j];

                if (score > bestScore) {
                    secondBestScore = bestScore;
                    bestScore = score;
                    bestWeapon = i;
                }
                else if (score > secondBestScore) {
                    secondBestScore = score;
                }
            }

            if (bestWeapon < 0)
                continue;

            // If only one weapon type remains feasible, losing it would
            // leave no fallback. Treat fallback score as 0.
            if (!isfinite(secondBestScore))
                secondBestScore = 0.0;

            double regret = bestScore - secondBestScore;

            bool better = false;

            if (regret > chosenRegret + EPS) {
                better = true;
            }
            else if (fabs(regret - chosenRegret) <= EPS) {
                if (bestScore > chosenBestScore + EPS) {
                    better = true;
                }
                else if (fabs(bestScore - chosenBestScore) <= EPS) {
                    if (chosenTarget < 0 || j < chosenTarget)
                        better = true;
                }
            }

            if (better) {
                chosenRegret = regret;
                chosenBestScore = bestScore;
                chosenTarget = j;
                chosenWeapon = bestWeapon;
            }
        }

        if (chosenTarget < 0 || chosenWeapon < 0)
            break;

        typeOfTarget[chosenTarget] = chosenWeapon;
        targetUsed[chosenTarget] = true;

        --w[chosenWeapon];
        --unitsLeft;
        --targetsLeft;
    }

    auto stop = Clock::now();

    Solution sol;
    sol.typeOfTarget = move(typeOfTarget);
    sol.milliseconds =
        chrono::duration<double, milli>(stop - start).count();

    evaluateSolution(ins, sol);
    return sol;
}


// ------------------------------------------------------------
// Randomized Regret-CWTAA (RR-CWTAA)
//
// At each step:
//   regret(j) = best feasible score for target j
//             - second-best feasible score for target j
//
// Build a restricted candidate list (RCL) containing the top-K
// targets by regret. Sample one target from that RCL with
// probability proportional to (regret + epsilon)^alpha, then
// assign that target to its best currently available weapon.
// Repeat the full randomized construction several times and
// return the solution with the lowest paper objective C(D).
// ------------------------------------------------------------
Solution solveRandomizedRegretCWTAA(
    const Instance& ins,
    uint64_t seed,
    int topK = RR_TOP_K,
    double alpha = RR_ALPHA,
    int numRestarts = RR_NUM_RESTARTS
) {
    auto overallStart = Clock::now();

    if (topK <= 0) throw invalid_argument("RR-CWTAA topK must be positive.");
    if (alpha < 0.0) throw invalid_argument("RR-CWTAA alpha must be nonnegative.");
    if (numRestarts <= 0) throw invalid_argument("RR-CWTAA numRestarts must be positive.");

    const int n = ins.nTypes;
    const int m = ins.nTargets;

    vector<vector<double>> S(n, vector<double>(m, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            S[i][j] = ins.P[i][j] * ins.V[j];

    struct Candidate {
        int target = -1;
        int weapon = -1;
        double regret = 0.0;
        double bestScore = 0.0;
    };

    Solution bestOverall;
    bestOverall.totalSurvivalValue = numeric_limits<double>::infinity();

    mt19937_64 rng(seed ^ 0xA0761D6478BD642FULL);

    for (int restart = 0; restart < numRestarts; ++restart) {
        vector<int> w = ins.W;
        vector<char> targetUsed(m, false);
        vector<int> typeOfTarget(m, -1);

        int unitsLeft = accumulate(w.begin(), w.end(), 0);
        int targetsLeft = m;

        while (unitsLeft > 0 && targetsLeft > 0) {
            vector<Candidate> candidates;
            candidates.reserve(targetsLeft);

            for (int j = 0; j < m; ++j) {
                if (targetUsed[j]) continue;

                double bestScore = -numeric_limits<double>::infinity();
                double secondBestScore = -numeric_limits<double>::infinity();
                int bestWeapon = -1;

                for (int i = 0; i < n; ++i) {
                    if (w[i] <= 0) continue;
                    double score = S[i][j];

                    if (score > bestScore) {
                        secondBestScore = bestScore;
                        bestScore = score;
                        bestWeapon = i;
                    } else if (score > secondBestScore) {
                        secondBestScore = score;
                    }
                }

                if (bestWeapon < 0) continue;
                if (!isfinite(secondBestScore)) secondBestScore = 0.0;

                Candidate c;
                c.target = j;
                c.weapon = bestWeapon;
                c.bestScore = bestScore;
                c.regret = max(0.0, bestScore - secondBestScore);
                candidates.push_back(c);
            }

            if (candidates.empty()) break;

            sort(candidates.begin(), candidates.end(),
                 [](const Candidate& a, const Candidate& b) {
                     if (a.regret != b.regret) return a.regret > b.regret;
                     if (a.bestScore != b.bestScore) return a.bestScore > b.bestScore;
                     return a.target < b.target;
                 });

            int k = min(topK, static_cast<int>(candidates.size()));
            vector<double> weights(k, 0.0);
            double weightSum = 0.0;

            for (int q = 0; q < k; ++q) {
                double base = candidates[q].regret + RR_EPSILON;
                double weight = (alpha == 0.0) ? 1.0 : pow(base, alpha);
                if (!isfinite(weight) || weight < 0.0) weight = 0.0;
                weights[q] = weight;
                weightSum += weight;
            }

            int selected = 0;
            if (weightSum <= 0.0 || !isfinite(weightSum)) {
                uniform_int_distribution<int> pick(0, k - 1);
                selected = pick(rng);
            } else {
                uniform_real_distribution<double> draw(0.0, weightSum);
                double x = draw(rng);
                double cumulative = 0.0;
                selected = k - 1;
                for (int q = 0; q < k; ++q) {
                    cumulative += weights[q];
                    if (x <= cumulative) {
                        selected = q;
                        break;
                    }
                }
            }

            const Candidate chosen = candidates[selected];
            typeOfTarget[chosen.target] = chosen.weapon;
            targetUsed[chosen.target] = true;
            --w[chosen.weapon];
            --unitsLeft;
            --targetsLeft;
        }

        Solution candidateSol;
        candidateSol.typeOfTarget = move(typeOfTarget);
        evaluateSolution(ins, candidateSol);

        if (candidateSol.totalSurvivalValue < bestOverall.totalSurvivalValue)
            bestOverall = move(candidateSol);
    }

    auto overallStop = Clock::now();
    bestOverall.milliseconds =
        chrono::duration<double, milli>(overallStop - overallStart).count();
    return bestOverall;
}


// ------------------------------------------------------------
// Perturbed Randomized Regret-CWTAA (PRR-CWTAA)
//
// Idea 5 is incorporated into Randomized Regret-CWTAA by perturbing
// the score matrix independently on every restart:
//
//   baseScore(i,j) = P[i][j] * V[j]
//
//   perturbedScore(i,j)
//       = baseScore(i,j) * (1 + epsilon_ij)
//
//   epsilon_ij ~ Uniform[-noiseLevel, +noiseLevel]
//
// Regret and the randomized top-K selection are computed from the
// perturbed scores. This intentionally changes some near-tie greedy
// decisions and explores neighboring assignment trajectories.
//
// IMPORTANT:
// The final assignment is evaluated using the ORIGINAL P and V,
// never the perturbed scores. Therefore the reported C(D) remains
// exactly the paper objective.
//
// As with RR-CWTAA, the whole construction is repeated and the
// lowest true survival-cost solution is returned.
// ------------------------------------------------------------
Solution solvePerturbedRandomizedRegretCWTAA(
    const Instance& ins,
    uint64_t seed,
    int topK = RR_TOP_K,
    double alpha = RR_ALPHA,
    int numRestarts = RR_NUM_RESTARTS,
    double noiseLevel = PRR_NOISE
) {
    auto overallStart = Clock::now();

    if (topK <= 0)
        throw invalid_argument("PRR-CWTAA topK must be positive.");
    if (alpha < 0.0)
        throw invalid_argument("PRR-CWTAA alpha must be nonnegative.");
    if (numRestarts <= 0)
        throw invalid_argument("PRR-CWTAA numRestarts must be positive.");
    if (noiseLevel < 0.0 || noiseLevel >= 1.0)
        throw invalid_argument("PRR-CWTAA noiseLevel must be in [0,1).");

    const int n = ins.nTypes;
    const int m = ins.nTargets;

    // Original, unperturbed score matrix.
    vector<vector<double>> baseS(n, vector<double>(m, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            baseS[i][j] = ins.P[i][j] * ins.V[j];

    Solution bestOverall;
    bestOverall.totalSurvivalValue = numeric_limits<double>::infinity();

    mt19937_64 rng(seed ^ 0xE7037ED1A0B428DBULL);
    uniform_real_distribution<double> noiseDist(-noiseLevel, noiseLevel);

    struct Candidate {
        int target = -1;
        int weapon = -1;
        double regret = 0.0;
        double bestScore = 0.0;
    };

    for (int restart = 0; restart < numRestarts; ++restart) {
        // Fresh perturbation for each restart.
        vector<vector<double>> S(n, vector<double>(m, 0.0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                const double eps = noiseDist(rng);
                S[i][j] = baseS[i][j] * (1.0 + eps);
            }
        }

        vector<int> w = ins.W;
        vector<char> targetUsed(m, false);
        vector<int> typeOfTarget(m, -1);

        int unitsLeft = accumulate(w.begin(), w.end(), 0);
        int targetsLeft = m;

        while (unitsLeft > 0 && targetsLeft > 0) {
            vector<Candidate> candidates;
            candidates.reserve(targetsLeft);

            for (int j = 0; j < m; ++j) {
                if (targetUsed[j]) continue;

                double bestScore = -numeric_limits<double>::infinity();
                double secondBestScore = -numeric_limits<double>::infinity();
                int bestWeapon = -1;

                for (int i = 0; i < n; ++i) {
                    if (w[i] <= 0) continue;

                    const double score = S[i][j];

                    if (score > bestScore) {
                        secondBestScore = bestScore;
                        bestScore = score;
                        bestWeapon = i;
                    }
                    else if (score > secondBestScore) {
                        secondBestScore = score;
                    }
                }

                if (bestWeapon < 0)
                    continue;

                if (!isfinite(secondBestScore))
                    secondBestScore = 0.0;

                Candidate c;
                c.target = j;
                c.weapon = bestWeapon;
                c.bestScore = bestScore;
                c.regret = max(0.0, bestScore - secondBestScore);
                candidates.push_back(c);
            }

            if (candidates.empty())
                break;

            sort(
                candidates.begin(),
                candidates.end(),
                [](const Candidate& a, const Candidate& b) {
                    if (a.regret != b.regret)
                        return a.regret > b.regret;
                    if (a.bestScore != b.bestScore)
                        return a.bestScore > b.bestScore;
                    return a.target < b.target;
                }
            );

            const int k = min(topK, static_cast<int>(candidates.size()));

            vector<double> weights(k, 0.0);
            double weightSum = 0.0;

            for (int q = 0; q < k; ++q) {
                const double base = candidates[q].regret + RR_EPSILON;
                double weight = (alpha == 0.0) ? 1.0 : pow(base, alpha);

                if (!isfinite(weight) || weight < 0.0)
                    weight = 0.0;

                weights[q] = weight;
                weightSum += weight;
            }

            int selected = 0;

            if (weightSum <= 0.0 || !isfinite(weightSum)) {
                uniform_int_distribution<int> pick(0, k - 1);
                selected = pick(rng);
            }
            else {
                uniform_real_distribution<double> draw(0.0, weightSum);
                const double x = draw(rng);

                double cumulative = 0.0;
                selected = k - 1;

                for (int q = 0; q < k; ++q) {
                    cumulative += weights[q];
                    if (x <= cumulative) {
                        selected = q;
                        break;
                    }
                }
            }

            const Candidate chosen = candidates[selected];

            typeOfTarget[chosen.target] = chosen.weapon;
            targetUsed[chosen.target] = true;
            --w[chosen.weapon];
            --unitsLeft;
            --targetsLeft;
        }

        Solution candidateSol;
        candidateSol.typeOfTarget = move(typeOfTarget);

        // Evaluate using ORIGINAL P and V only.
        evaluateSolution(ins, candidateSol);

        if (candidateSol.totalSurvivalValue < bestOverall.totalSurvivalValue)
            bestOverall = move(candidateSol);
    }

    auto overallStop = Clock::now();
    bestOverall.milliseconds =
        chrono::duration<double, milli>(overallStop - overallStart).count();

    return bestOverall;
}


// ------------------------------------------------------------
// Helper: one randomized-regret construction.
//
// If priorityNoise > 0, ONLY the regret used for target priority is
// perturbed:
//
//   perturbedRegret(j) = trueRegret(j) * (1 + eps_j)
//
// while the selected target is always assigned to its TRUE best
// currently feasible weapon under the unperturbed score matrix.
//
// This preserves weapon preference and injects randomness only into
// the order in which vulnerable targets are handled.
// ------------------------------------------------------------
Solution constructRandomizedRegretPriorityPerturbed(
    const Instance& ins,
    mt19937_64& rng,
    int topK,
    double alpha,
    double priorityNoise
) {
    const int n = ins.nTypes;
    const int m = ins.nTargets;

    vector<vector<double>> S(n, vector<double>(m, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            S[i][j] = ins.P[i][j] * ins.V[j];

    vector<int> w = ins.W;
    vector<char> targetUsed(m, false);
    vector<int> typeOfTarget(m, -1);

    int unitsLeft = accumulate(w.begin(), w.end(), 0);
    int targetsLeft = m;

    uniform_real_distribution<double> priorityNoiseDist(
        -priorityNoise, priorityNoise
    );

    struct Candidate {
        int target = -1;
        int trueBestWeapon = -1;
        double trueRegret = 0.0;
        double perturbedRegret = 0.0;
        double trueBestScore = 0.0;
    };

    while (unitsLeft > 0 && targetsLeft > 0) {
        vector<Candidate> candidates;
        candidates.reserve(targetsLeft);

        for (int j = 0; j < m; ++j) {
            if (targetUsed[j]) continue;

            double bestScore = -numeric_limits<double>::infinity();
            double secondBestScore = -numeric_limits<double>::infinity();
            int bestWeapon = -1;

            for (int i = 0; i < n; ++i) {
                if (w[i] <= 0) continue;

                const double score = S[i][j];

                if (score > bestScore) {
                    secondBestScore = bestScore;
                    bestScore = score;
                    bestWeapon = i;
                }
                else if (score > secondBestScore) {
                    secondBestScore = score;
                }
            }

            if (bestWeapon < 0)
                continue;

            if (!isfinite(secondBestScore))
                secondBestScore = 0.0;

            const double trueRegret =
                max(0.0, bestScore - secondBestScore);

            double perturbation = 0.0;
            if (priorityNoise > 0.0)
                perturbation = priorityNoiseDist(rng);

            Candidate c;
            c.target = j;
            c.trueBestWeapon = bestWeapon;
            c.trueBestScore = bestScore;
            c.trueRegret = trueRegret;
            c.perturbedRegret =
                max(0.0, trueRegret * (1.0 + perturbation));

            candidates.push_back(c);
        }

        if (candidates.empty())
            break;

        // RCL is formed by the perturbed target-priority regret.
        sort(
            candidates.begin(),
            candidates.end(),
            [](const Candidate& a, const Candidate& b) {
                if (a.perturbedRegret != b.perturbedRegret)
                    return a.perturbedRegret > b.perturbedRegret;
                if (a.trueBestScore != b.trueBestScore)
                    return a.trueBestScore > b.trueBestScore;
                return a.target < b.target;
            }
        );

        const int k = min(topK, static_cast<int>(candidates.size()));

        vector<double> weights(k, 0.0);
        double weightSum = 0.0;

        for (int q = 0; q < k; ++q) {
            const double base =
                candidates[q].perturbedRegret + RR_EPSILON;

            double weight =
                (alpha == 0.0) ? 1.0 : pow(base, alpha);

            if (!isfinite(weight) || weight < 0.0)
                weight = 0.0;

            weights[q] = weight;
            weightSum += weight;
        }

        int selected = 0;

        if (weightSum <= 0.0 || !isfinite(weightSum)) {
            uniform_int_distribution<int> pick(0, k - 1);
            selected = pick(rng);
        }
        else {
            uniform_real_distribution<double> draw(0.0, weightSum);
            const double x = draw(rng);

            double cumulative = 0.0;
            selected = k - 1;

            for (int q = 0; q < k; ++q) {
                cumulative += weights[q];
                if (x <= cumulative) {
                    selected = q;
                    break;
                }
            }
        }

        const Candidate chosen = candidates[selected];

        // Crucial design choice:
        // use the TRUE best weapon, not a perturbed weapon ranking.
        typeOfTarget[chosen.target] = chosen.trueBestWeapon;
        targetUsed[chosen.target] = true;

        --w[chosen.trueBestWeapon];
        --unitsLeft;
        --targetsLeft;
    }

    Solution sol;
    sol.typeOfTarget = move(typeOfTarget);
    evaluateSolution(ins, sol);
    return sol;
}


// ------------------------------------------------------------
// Regret-Priority Perturbation RR-CWTAA (RPP-RR-CWTAA)
//
// Repeats randomized-regret constructions where only target priority
// is slightly perturbed. The best solution under the TRUE objective
// is returned.
//
// This is intended to preserve the successful weapon-ranking behavior
// of RR-CWTAA while exploring nearby target-ordering trajectories.
// ------------------------------------------------------------
Solution solveRegretPriorityPerturbedRR(
    const Instance& ins,
    uint64_t seed,
    int topK = RR_TOP_K,
    double alpha = RR_ALPHA,
    int numRestarts = RR_NUM_RESTARTS,
    double priorityNoise = RPP_NOISE
) {
    auto start = Clock::now();

    if (topK <= 0)
        throw invalid_argument("RPP-RR topK must be positive.");
    if (alpha < 0.0)
        throw invalid_argument("RPP-RR alpha must be nonnegative.");
    if (numRestarts <= 0)
        throw invalid_argument("RPP-RR numRestarts must be positive.");
    if (priorityNoise < 0.0 || priorityNoise >= 1.0)
        throw invalid_argument("RPP-RR priorityNoise must be in [0,1).");

    mt19937_64 rng(seed ^ 0x8EBC6AF09C88C6E3ULL);

    Solution bestOverall;
    bestOverall.totalSurvivalValue =
        numeric_limits<double>::infinity();

    for (int r = 0; r < numRestarts; ++r) {
        Solution cand =
            constructRandomizedRegretPriorityPerturbed(
                ins, rng, topK, alpha, priorityNoise
            );

        if (cand.totalSurvivalValue <
            bestOverall.totalSurvivalValue) {
            bestOverall = move(cand);
        }
    }

    auto stop = Clock::now();
    bestOverall.milliseconds =
        chrono::duration<double, milli>(stop - start).count();

    return bestOverall;
}


// ------------------------------------------------------------
// Hybrid RR + Priority-Perturbed RR
//
// Baseline-preserving hybrid requested:
//
//   - HYBRID_RR_RESTARTS ordinary RR constructions (noise=0)
//   - HYBRID_PERTURBED_RESTARTS priority-perturbed RR constructions
//   - keep the best solution under the TRUE paper objective
//
// Therefore, for the SAME included ordinary-RR candidates, the hybrid
// cannot be worse than its baseline pool because the baseline
// candidates are explicitly part of the candidate set.
// ------------------------------------------------------------
Solution solveHybridRandomizedRegret(
    const Instance& ins,
    uint64_t seed,
    int topK = RR_TOP_K,
    double alpha = RR_ALPHA,
    int rrRestarts = HYBRID_RR_RESTARTS,
    int perturbedRestarts = HYBRID_PERTURBED_RESTARTS,
    double priorityNoise = RPP_NOISE
) {
    auto start = Clock::now();

    if (rrRestarts <= 0 || perturbedRestarts <= 0)
        throw invalid_argument(
            "Hybrid restart counts must be positive."
        );

    Solution bestOverall;
    bestOverall.totalSurvivalValue =
        numeric_limits<double>::infinity();

    // Separate deterministic streams for reproducibility.
    mt19937_64 rrRng(seed ^ 0x589965CC75374CC3ULL);
    mt19937_64 pertRng(seed ^ 0x1D8E4E27C47D124FULL);

    // Ordinary RR candidate pool.
    for (int r = 0; r < rrRestarts; ++r) {
        Solution cand =
            constructRandomizedRegretPriorityPerturbed(
                ins, rrRng, topK, alpha, 0.0
            );

        if (cand.totalSurvivalValue <
            bestOverall.totalSurvivalValue) {
            bestOverall = move(cand);
        }
    }

    // Priority-perturbed candidate pool.
    for (int r = 0; r < perturbedRestarts; ++r) {
        Solution cand =
            constructRandomizedRegretPriorityPerturbed(
                ins, pertRng, topK, alpha, priorityNoise
            );

        if (cand.totalSurvivalValue <
            bestOverall.totalSurvivalValue) {
            bestOverall = move(cand);
        }
    }

    auto stop = Clock::now();
    bestOverall.milliseconds =
        chrono::duration<double, milli>(stop - start).count();

    return bestOverall;
}


// ============================================================
// BEST-OF-PERTURBATION RANDOMIZED REGRET
//
// Requested experiment:
//   - 1 unperturbed Randomized-Regret construction
//   - for each perturbation level:
//         0.01%, 0.05%, 0.1%, 0.5%, 1%, 5%
//     run 3 independently perturbed constructions
//   - evaluate EVERY candidate with the ORIGINAL WTAP objective
//   - return ONLY the best candidate
//   - record which perturbation level / repetition produced it
//
// Perturbation is multiplicative on the score matrix:
//
//   S[i][j]       = P[i][j] * V[j]
//   S_tilde[i][j] = S[i][j] * (1 + eps_ij)
//   eps_ij ~ Uniform[-noise, +noise]
//
// IMPORTANT:
// Perturbed scores are used only to construct a solution.
// Final C(D) is always evaluated using the original P and V.
// ============================================================

struct BestPerturbationResult {
    Solution solution;
    string source;          // "Original" or "Perturbed"
    double noisePercent;    // e.g. 0.5 means 0.5%
    int repetition;         // 0 for original, 1..3 for perturbed
};

// One randomized-regret construction using a supplied score matrix.
Solution constructRRFromScoreMatrix(
    const Instance& ins,
    const vector<vector<double>>& S,
    mt19937_64& rng,
    int topK = RR_TOP_K,
    double alpha = RR_ALPHA
) {
    const int n = ins.nTypes;
    const int m = ins.nTargets;

    vector<int> w = ins.W;
    vector<char> targetUsed(m, false);
    vector<int> typeOfTarget(m, -1);

    int unitsLeft = accumulate(w.begin(), w.end(), 0);
    int targetsLeft = m;

    struct Candidate {
        int target = -1;
        int weapon = -1;
        double regret = 0.0;
        double bestScore = 0.0;
    };

    while (unitsLeft > 0 && targetsLeft > 0) {
        vector<Candidate> candidates;
        candidates.reserve(targetsLeft);

        for (int j = 0; j < m; ++j) {
            if (targetUsed[j]) continue;

            double bestScore = -numeric_limits<double>::infinity();
            double secondBestScore = -numeric_limits<double>::infinity();
            int bestWeapon = -1;

            for (int i = 0; i < n; ++i) {
                if (w[i] <= 0) continue;

                const double score = S[i][j];

                if (score > bestScore) {
                    secondBestScore = bestScore;
                    bestScore = score;
                    bestWeapon = i;
                }
                else if (score > secondBestScore) {
                    secondBestScore = score;
                }
            }

            if (bestWeapon < 0)
                continue;

            if (!isfinite(secondBestScore))
                secondBestScore = 0.0;

            Candidate c;
            c.target = j;
            c.weapon = bestWeapon;
            c.bestScore = bestScore;
            c.regret = max(0.0, bestScore - secondBestScore);
            candidates.push_back(c);
        }

        if (candidates.empty())
            break;

        sort(
            candidates.begin(),
            candidates.end(),
            [](const Candidate& a, const Candidate& b) {
                if (a.regret != b.regret)
                    return a.regret > b.regret;
                if (a.bestScore != b.bestScore)
                    return a.bestScore > b.bestScore;
                return a.target < b.target;
            }
        );

        const int k = min(topK, static_cast<int>(candidates.size()));

        vector<double> weights(k, 0.0);
        double weightSum = 0.0;

        for (int q = 0; q < k; ++q) {
            const double base = candidates[q].regret + RR_EPSILON;
            double weight = (alpha == 0.0) ? 1.0 : pow(base, alpha);

            if (!isfinite(weight) || weight < 0.0)
                weight = 0.0;

            weights[q] = weight;
            weightSum += weight;
        }

        int selected = 0;

        if (weightSum <= 0.0 || !isfinite(weightSum)) {
            uniform_int_distribution<int> pick(0, k - 1);
            selected = pick(rng);
        }
        else {
            uniform_real_distribution<double> draw(0.0, weightSum);
            const double x = draw(rng);

            double cumulative = 0.0;
            selected = k - 1;

            for (int q = 0; q < k; ++q) {
                cumulative += weights[q];
                if (x <= cumulative) {
                    selected = q;
                    break;
                }
            }
        }

        const Candidate chosen = candidates[selected];

        typeOfTarget[chosen.target] = chosen.weapon;
        targetUsed[chosen.target] = true;
        --w[chosen.weapon];
        --unitsLeft;
        --targetsLeft;
    }

    Solution sol;
    sol.typeOfTarget = move(typeOfTarget);

    // Always evaluate on the ORIGINAL WTAP instance.
    evaluateSolution(ins, sol);
    return sol;
}


BestPerturbationResult solveBestPerturbationRR(
    const Instance& ins,
    uint64_t seed,
    int topK = RR_TOP_K,
    double alpha = RR_ALPHA
) {
    auto start = Clock::now();

    const int n = ins.nTypes;
    const int m = ins.nTargets;

    vector<vector<double>> baseS(n, vector<double>(m, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            baseS[i][j] = ins.P[i][j] * ins.V[j];

    // Percent values requested by the user.
    // Convert percent -> fraction before applying perturbation.
    const vector<double> noisePercent = {
        0.01, 0.05, 0.10, 0.50, 1.00, 5.00
    };

    BestPerturbationResult best;
    best.solution.totalSurvivalValue =
        numeric_limits<double>::infinity();

    // --------------------------------------------------------
    // Candidate 0: ONE ordinary, unperturbed RR construction.
    // --------------------------------------------------------
    {
        mt19937_64 rng(seed ^ 0x243F6A8885A308D3ULL);
        Solution sol =
            constructRRFromScoreMatrix(ins, baseS, rng, topK, alpha);

        best.solution = move(sol);
        best.source = "Original";
        best.noisePercent = 0.0;
        best.repetition = 0;
    }

    // --------------------------------------------------------
    // Perturbed candidates:
    // 6 perturbation levels x 3 independent repetitions = 18.
    // --------------------------------------------------------
    for (size_t level = 0; level < noisePercent.size(); ++level) {
        const double pct = noisePercent[level];
        const double noise = pct / 100.0;  // e.g. 0.01% -> 0.0001

        for (int rep = 1; rep <= 3; ++rep) {
            // Independent deterministic RNG stream per level/repetition.
            const uint64_t streamSeed =
                seed
                ^ (0x9E3779B97F4A7C15ULL * (level + 1))
                ^ (0xD1B54A32D192ED03ULL * static_cast<uint64_t>(rep));

            mt19937_64 rng(streamSeed);
            uniform_real_distribution<double> noiseDist(-noise, noise);

            vector<vector<double>> perturbedS(
                n, vector<double>(m, 0.0)
            );

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    const double eps = noiseDist(rng);
                    perturbedS[i][j] =
                        baseS[i][j] * (1.0 + eps);
                }
            }

            Solution sol =
                constructRRFromScoreMatrix(
                    ins, perturbedS, rng, topK, alpha
                );

            if (sol.totalSurvivalValue <
                best.solution.totalSurvivalValue) {

                best.solution = move(sol);
                best.source = "Perturbed";
                best.noisePercent = pct;
                best.repetition = rep;
            }
        }
    }

    auto stop = Clock::now();

    // Report TOTAL search time for all 19 candidate constructions.
    best.solution.milliseconds =
        chrono::duration<double, milli>(stop - start).count();

    return best;
}

// ------------------------------------------------------------
// Hungarian algorithm for rectangular MIN-cost assignment.
// Requires rows <= columns.
// Returns assignment[row] = selected column.
// Complexity O(rows^2 * columns).
// ------------------------------------------------------------
vector<int> hungarianMin(const vector<vector<double>>& a) {
    const int n = static_cast<int>(a.size());
    const int m = n ? static_cast<int>(a[0].size()) : 0;

    if (n == 0) return {};
    if (n > m)
        throw runtime_error("hungarianMin requires rows <= columns.");

    const double INF = numeric_limits<double>::infinity();

    vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
    vector<int> p(m + 1, 0), way(m + 1, 0);

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
                }
                else if (j > 0) {
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

    vector<int> assignment(n, -1);
    for (int j = 1; j <= m; ++j) {
        if (p[j] != 0)
            assignment[p[j] - 1] = j - 1;
    }

    return assignment;
}

// ------------------------------------------------------------
// Repeated Hungarian baseline.
//
// One "capacity round" exposes one currently available unit from
// every weapon type with W[i] > 0 and matches those active types
// one-to-one to distinct currently unassigned targets.
//
// Score is the SAME P[i][j] * V[j] signal used by CWTAA.
// Hungarian minimizes, so cost = -P[i][j] * V[j].
//
// After the round:
//   - each matched type loses one available unit;
//   - each matched target is permanently removed;
//   - repeat until all units/targets are exhausted.
//
// Under the paper's exact benchmark n=m and sum W_i=m with W_i>=1,
// every W_i is necessarily 1. Therefore Repeated Hungarian performs
// exactly ONE Hungarian round in those paper-faithful experiments.
// ------------------------------------------------------------
Solution solveRepeatedHungarian(const Instance& ins) {
    auto start = Clock::now();

    const int n = ins.nTypes;
    const int m = ins.nTargets;

    vector<int> w = ins.W;
    vector<char> targetUsed(m, false);
    vector<int> typeOfTarget(m, -1);

    int unitsLeft = accumulate(w.begin(), w.end(), 0);
    int targetsLeft = m;

    while (unitsLeft > 0 && targetsLeft > 0) {
        vector<int> activeTypes;
        vector<int> remainingTargets;

        for (int i = 0; i < n; ++i)
            if (w[i] > 0)
                activeTypes.push_back(i);

        for (int j = 0; j < m; ++j)
            if (!targetUsed[j])
                remainingTargets.push_back(j);

        if (activeTypes.empty() || remainingTargets.empty())
            break;

        // With the paper's W construction and total units=m,
        // the number of active types cannot exceed remaining targets
        // in the intended repeated-round process.
        //
        // For robustness, if it does, transpose the matching.
        if (activeTypes.size() <= remainingTargets.size()) {
            const int r = static_cast<int>(activeTypes.size());
            const int c = static_cast<int>(remainingTargets.size());

            vector<vector<double>> cost(r, vector<double>(c, 0.0));

            for (int rr = 0; rr < r; ++rr) {
                int i = activeTypes[rr];
                for (int cc = 0; cc < c; ++cc) {
                    int j = remainingTargets[cc];
                    cost[rr][cc] = -(ins.P[i][j] * ins.V[j]);
                }
            }

            vector<int> match = hungarianMin(cost);

            for (int rr = 0; rr < r; ++rr) {
                int cc = match[rr];
                if (cc < 0) continue;

                int i = activeTypes[rr];
                int j = remainingTargets[cc];

                if (w[i] <= 0 || targetUsed[j]) continue;

                typeOfTarget[j] = i;
                targetUsed[j] = true;
                --w[i];
                --unitsLeft;
                --targetsLeft;
            }
        }
        else {
            // Rows = remaining targets, columns = active types.
            const int r = static_cast<int>(remainingTargets.size());
            const int c = static_cast<int>(activeTypes.size());

            vector<vector<double>> cost(r, vector<double>(c, 0.0));

            for (int rr = 0; rr < r; ++rr) {
                int j = remainingTargets[rr];
                for (int cc = 0; cc < c; ++cc) {
                    int i = activeTypes[cc];
                    cost[rr][cc] = -(ins.P[i][j] * ins.V[j]);
                }
            }

            vector<int> match = hungarianMin(cost);

            for (int rr = 0; rr < r; ++rr) {
                int cc = match[rr];
                if (cc < 0) continue;

                int j = remainingTargets[rr];
                int i = activeTypes[cc];

                if (w[i] <= 0 || targetUsed[j]) continue;

                typeOfTarget[j] = i;
                targetUsed[j] = true;
                --w[i];
                --unitsLeft;
                --targetsLeft;
            }
        }
    }

    auto stop = Clock::now();

    Solution sol;
    sol.typeOfTarget = move(typeOfTarget);
    sol.milliseconds =
        chrono::duration<double, milli>(stop - start).count();

    evaluateSolution(ins, sol);
    return sol;
}


// ============================================================================
// FINAL COMPREHENSIVE EXPERIMENT DRIVER
//
// Experiment grid requested:
//
// Problem sizes:         50, 75, 100, 125
// Cases per setting:     20
// Distributions:         Uniform, Beta(2,5), Gaussian(0.7,0.15) clipped
//
// Randomized-Regret:
//   top-K:               2, 5, 10
//   alpha:               1, 2, 3, 5
//   budgets:             10, 20
//
// Perturbation:
//   levels:              0.01%, 0.05%, 0.10%
//   applied to P itself:
//      P_tilde[i][j] = clip(P[i][j] * (1 + eps_ij), 0, 1)
//      eps_ij ~ U(-delta,+delta)
//
// Techniques:
//   1. CWTAA baseline
//   2. Deterministic Regret-CWTAA baseline
//   3. Repeated Hungarian baseline
//   4. RR
//   5. RR + perturbation
//   6. RR + target permutation
//   7. RR + perturbation + target permutation
//   8. CWTAA + perturbation
//   9. CWTAA + target permutation
//  10. CWTAA + perturbation + target permutation
//
// IMPORTANT FAIR-BUDGET DESIGN:
// We execute at most 20 stochastic runs for each hyperparameter setting.
// The "budget 10" result uses runs 1..10.
// The "budget 20" result uses runs 1..20.
// Thus best-of-10 is a true subset of best-of-20 and we do not waste time
// rerunning the same experiment under separate budget labels.
//
// IMPORTANT EVALUATION RULE:
// Perturbed/permuted instances are used ONLY to construct an assignment.
// Every final assignment is mapped back (when necessary) and evaluated using
// the ORIGINAL, unperturbed WTAP instance.
//
// OUTPUT FILES:
//   wtap_final_all_runs.csv
//   wtap_final_case_summary.csv
//   wtap_final_overall_summary.csv
//
// "Quality / kill" metrics recorded:
//   - total_survival_value                    (lower is better)
//   - normalized_survival                     (lower is better)
//   - mean_survival_probability               (lower is better)
//   - total_destroyed_value = sum(V)-survival (higher is better)
//   - normalized_kill = 1-normalized_survival (higher is better)
//   - mean_kill_probability                   (higher is better)
//   - runtime_ms
//
// Each stochastic row also contains enough seed information to reproduce it.
// ============================================================================

static const vector<int> EXP_SIZES = {50, 75, 100, 125};
static const vector<int> EXP_TOP_K = {2, 5, 10};
static const vector<double> EXP_ALPHA = {1.0, 2.0, 3.0, 5.0};
static const vector<int> EXP_BUDGETS = {10, 20};
static const vector<double> EXP_PERTURB_PCT = {0.01, 0.05, 0.10};

static constexpr int EXP_CASES = 20;
static constexpr int EXP_MAX_RUNS = 20;
static constexpr uint64_t EXP_DEFAULT_SEED = 123456789ULL;

// ----------------------------------------------------------------------------
// Deterministic 64-bit seed mixer.
// ----------------------------------------------------------------------------
uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

uint64_t deriveSeed(
    uint64_t instanceSeed,
    uint64_t techniqueTag,
    int runIndex,
    int topK = 0,
    int alphaCode = 0,
    int perturbCode = 0
) {
    uint64_t x = instanceSeed;
    x ^= mix64(techniqueTag);
    x ^= mix64(static_cast<uint64_t>(runIndex) + 0x100ULL);
    x ^= mix64(static_cast<uint64_t>(topK) + 0x200ULL);
    x ^= mix64(static_cast<uint64_t>(alphaCode) + 0x300ULL);
    x ^= mix64(static_cast<uint64_t>(perturbCode) + 0x400ULL);
    return mix64(x);
}

int alphaCode(double alpha) {
    return static_cast<int>(llround(alpha * 1000.0));
}

int perturbCode(double pct) {
    return static_cast<int>(llround(pct * 1000000.0));
}

// ----------------------------------------------------------------------------
// Metrics.
// ----------------------------------------------------------------------------
struct QualityMetrics {
    double survival = numeric_limits<double>::infinity();
    double normalizedSurvival = numeric_limits<double>::infinity();
    double meanSurvivalProbability = numeric_limits<double>::infinity();

    double destroyedValue = -numeric_limits<double>::infinity();
    double normalizedKill = -numeric_limits<double>::infinity();
    double meanKillProbability = -numeric_limits<double>::infinity();

    double runtimeMs = 0.0;
    bool valid = false;
};

QualityMetrics makeMetrics(const Instance& original, const Solution& sol) {
    QualityMetrics q;
    q.survival = sol.totalSurvivalValue;
    q.normalizedSurvival = sol.normalizedSurvival;
    q.meanSurvivalProbability = sol.meanSurvivalProbability;
    q.runtimeMs = sol.milliseconds;
    q.valid = sol.valid;

    const double totalV =
        accumulate(original.V.begin(), original.V.end(), 0.0);

    q.destroyedValue = totalV - q.survival;
    q.normalizedKill = 1.0 - q.normalizedSurvival;
    q.meanKillProbability = 1.0 - q.meanSurvivalProbability;

    return q;
}

// ----------------------------------------------------------------------------
// Transform helpers.
// ----------------------------------------------------------------------------
Instance makePerturbedInstance(
    const Instance& original,
    double perturbationPercent,
    uint64_t perturbSeed
) {
    Instance out = original;

    const double delta = perturbationPercent / 100.0;
    mt19937_64 rng(perturbSeed);
    uniform_real_distribution<double> noise(-delta, delta);

    for (int i = 0; i < original.nTypes; ++i) {
        for (int j = 0; j < original.nTargets; ++j) {
            double p = original.P[i][j] * (1.0 + noise(rng));
            out.P[i][j] = max(0.0, min(1.0, p));
        }
    }
    return out;
}

struct PermutedInstance {
    Instance instance;

    // perm[k] = original target index stored at permuted position k.
    vector<int> perm;
};

PermutedInstance makeTargetPermutedInstance(
    const Instance& input,
    uint64_t permutationSeed
) {
    const int n = input.nTypes;
    const int m = input.nTargets;

    mt19937_64 rng(permutationSeed);

    vector<int> perm(m);
    iota(perm.begin(), perm.end(), 0);
    shuffle(perm.begin(), perm.end(), rng);

    Instance out;
    out.nTypes = n;
    out.nTargets = m;
    out.W = input.W;
    out.V.resize(m);
    out.P.assign(n, vector<double>(m, 0.0));

    for (int k = 0; k < m; ++k) {
        const int jOriginal = perm[k];
        out.V[k] = input.V[jOriginal];

        for (int i = 0; i < n; ++i)
            out.P[i][k] = input.P[i][jOriginal];
    }

    return {move(out), move(perm)};
}

Solution mapPermutationBackAndEvaluate(
    const Instance& original,
    const Solution& permutedSolution,
    const vector<int>& perm,
    double runtimeMs
) {
    Solution mapped;
    mapped.typeOfTarget.assign(original.nTargets, -1);

    for (int k = 0; k < original.nTargets; ++k) {
        const int originalTarget = perm[k];
        mapped.typeOfTarget[originalTarget] =
            permutedSolution.typeOfTarget[k];
    }

    evaluateSolution(original, mapped);
    mapped.milliseconds = runtimeMs;
    return mapped;
}

// ----------------------------------------------------------------------------
// One RR construction on whichever instance is supplied.
// The assignment is returned in that instance's current target indexing.
// ----------------------------------------------------------------------------
Solution runRRRaw(
    const Instance& searchInstance,
    int topK,
    double alpha,
    uint64_t rrSeed
) {
    const int n = searchInstance.nTypes;
    const int m = searchInstance.nTargets;

    vector<vector<double>> S(n, vector<double>(m, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            S[i][j] = searchInstance.P[i][j] * searchInstance.V[j];

    mt19937_64 rng(rrSeed);
    return constructRRFromScoreMatrix(
        searchInstance, S, rng, topK, alpha
    );
}

// ----------------------------------------------------------------------------
// Comprehensive single-run RR wrapper.
//
// usePerturbation=false/usePermutation=false -> ordinary RR
// true/false -> perturbed RR
// false/true -> target-permuted RR
// true/true -> perturbed + target-permuted RR
//
// Separate RNG streams are used for perturbation, permutation, and RR draws.
// This prevents a shuffle from merely advancing the RR random-number stream.
// ----------------------------------------------------------------------------
Solution runRRTechnique(
    const Instance& original,
    int topK,
    double alpha,
    bool usePerturbation,
    double perturbationPercent,
    bool usePermutation,
    uint64_t masterRunSeed
) {
    auto start = Clock::now();

    const uint64_t perturbSeed =
        mix64(masterRunSeed ^ 0xABC98388FB8FAC03ULL);
    const uint64_t permutationSeed =
        mix64(masterRunSeed ^ 0x8CB92BA72F3D8DD7ULL);
    const uint64_t rrSeed =
        mix64(masterRunSeed ^ 0xDB4F0B9175AE2165ULL);

    Instance search = original;

    if (usePerturbation)
        search = makePerturbedInstance(
            original, perturbationPercent, perturbSeed
        );

    if (usePermutation) {
        PermutedInstance pi =
            makeTargetPermutedInstance(search, permutationSeed);

        Solution temp =
            runRRRaw(pi.instance, topK, alpha, rrSeed);

        auto stop = Clock::now();
        const double ms =
            chrono::duration<double, milli>(stop - start).count();

        return mapPermutationBackAndEvaluate(
            original, temp, pi.perm, ms
        );
    }

    Solution temp = runRRRaw(search, topK, alpha, rrSeed);

    // If perturbation was used, runRRRaw evaluated on the perturbed
    // search instance. Re-evaluate the exact same assignment on original P.
    Solution finalSol;
    finalSol.typeOfTarget = temp.typeOfTarget;
    evaluateSolution(original, finalSol);

    auto stop = Clock::now();
    finalSol.milliseconds =
        chrono::duration<double, milli>(stop - start).count();

    return finalSol;
}

// ----------------------------------------------------------------------------
// CWTAA transform wrapper.
//
// This lets us test:
//   CWTAA + perturbation
//   CWTAA + target permutation
//   CWTAA + perturbation + target permutation
//
// Again, construction may use transformed data, but evaluation uses original.
// ----------------------------------------------------------------------------
Solution runCWTAAWithTransforms(
    const Instance& original,
    bool usePerturbation,
    double perturbationPercent,
    bool usePermutation,
    uint64_t masterRunSeed
) {
    auto start = Clock::now();

    const uint64_t perturbSeed =
        mix64(masterRunSeed ^ 0xA24BAED4963EE407ULL);
    const uint64_t permutationSeed =
        mix64(masterRunSeed ^ 0x9FB21C651E98DF25ULL);

    Instance search = original;

    if (usePerturbation)
        search = makePerturbedInstance(
            original, perturbationPercent, perturbSeed
        );

    if (usePermutation) {
        PermutedInstance pi =
            makeTargetPermutedInstance(search, permutationSeed);

        Solution temp = solveCWTAA(pi.instance);

        auto stop = Clock::now();
        const double ms =
            chrono::duration<double, milli>(stop - start).count();

        return mapPermutationBackAndEvaluate(
            original, temp, pi.perm, ms
        );
    }

    Solution temp = solveCWTAA(search);

    Solution finalSol;
    finalSol.typeOfTarget = temp.typeOfTarget;
    evaluateSolution(original, finalSol);

    auto stop = Clock::now();
    finalSol.milliseconds =
        chrono::duration<double, milli>(stop - start).count();

    return finalSol;
}

// ----------------------------------------------------------------------------
// CSV records.
// ----------------------------------------------------------------------------
struct RunRecord {
    int weaponTypes = 0;
    int targets = 0;
    string distribution;
    int caseNumber = 0;
    uint64_t instanceSeed = 0;

    string family;
    string technique;

    int topK = 0;
    double alpha = 0.0;

    bool perturbation = false;
    double perturbationPercent = 0.0;

    bool permutation = false;

    int runIndex = 1;
    int maxBudget = 1;
    uint64_t runSeed = 0;

    QualityMetrics quality;
};

void writeAllRun(
    ofstream& out,
    const RunRecord& r
) {
    out
        << r.weaponTypes << ","
        << r.targets << ","
        << "\"" << r.distribution << "\"" << ","
        << r.caseNumber << ","
        << r.instanceSeed << ","
        << r.family << ","
        << r.technique << ","
        << r.topK << ","
        << fixed << setprecision(4) << r.alpha << ","
        << (r.perturbation ? 1 : 0) << ","
        << setprecision(4) << r.perturbationPercent << ","
        << (r.permutation ? 1 : 0) << ","
        << r.runIndex << ","
        << r.maxBudget << ","
        << r.runSeed << ","
        << setprecision(15)
        << r.quality.survival << ","
        << r.quality.normalizedSurvival << ","
        << r.quality.meanSurvivalProbability << ","
        << r.quality.destroyedValue << ","
        << r.quality.normalizedKill << ","
        << r.quality.meanKillProbability << ","
        << setprecision(9)
        << r.quality.runtimeMs << ","
        << (r.quality.valid ? 1 : 0)
        << "\n";
}

// ----------------------------------------------------------------------------
// Summary statistics.
// ----------------------------------------------------------------------------
double vecMean(const vector<double>& v) {
    if (v.empty()) return 0.0;
    long double s = 0.0L;
    for (double x : v) s += x;
    return static_cast<double>(s / v.size());
}

double vecStd(const vector<double>& v) {
    if (v.size() <= 1) return 0.0;
    const double mu = vecMean(v);
    long double ss = 0.0L;
    for (double x : v) {
        const long double d = x - mu;
        ss += d * d;
    }
    return sqrt(static_cast<double>(ss / (v.size() - 1)));
}

double vecMedian(vector<double> v) {
    if (v.empty()) return 0.0;
    sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n & 1) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct SummaryKey {
    int weaponTypes = 0;
    int targets = 0;
    string distribution;

    string family;
    string technique;

    int topK = 0;
    double alpha = 0.0;

    bool perturbation = false;
    double perturbationPercent = 0.0;
    bool permutation = false;

    int budget = 1;

    bool operator<(const SummaryKey& o) const {
        return tie(
            weaponTypes, targets, distribution,
            family, technique,
            topK, alpha,
            perturbation, perturbationPercent,
            permutation, budget
        ) <
        tie(
            o.weaponTypes, o.targets, o.distribution,
            o.family, o.technique,
            o.topK, o.alpha,
            o.perturbation, o.perturbationPercent,
            o.permutation, o.budget
        );
    }
};

struct OverallAccumulator {
    vector<double> bestSurvivalPerCase;
    vector<double> meanSurvivalPerCase;

    vector<double> bestNormalizedSurvivalPerCase;
    vector<double> bestMeanSurvivalProbabilityPerCase;

    vector<double> bestDestroyedValuePerCase;
    vector<double> bestNormalizedKillPerCase;
    vector<double> bestMeanKillProbabilityPerCase;

    vector<double> meanSingleRunRuntimePerCase;
    vector<double> totalRuntimePerCase;

    vector<double> allRunSurvival;
    vector<double> allRunRuntime;
};

// ----------------------------------------------------------------------------
// Summarize the first `budget` runs of a stochastic configuration.
// For deterministic baseline methods, budget=1.
// ----------------------------------------------------------------------------
void summarizeCaseGroup(
    const Instance& original,
    int caseNumber,
    uint64_t instanceSeed,
    const SummaryKey& key,
    const vector<RunRecord>& runs,
    ofstream& caseOut,
    map<SummaryKey, OverallAccumulator>& overall
) {
    if (runs.empty())
        return;

    const int take =
        min(key.budget, static_cast<int>(runs.size()));

    vector<double> survival;
    vector<double> normSurvival;
    vector<double> meanSurvivalProb;
    vector<double> destroyed;
    vector<double> normKill;
    vector<double> meanKill;
    vector<double> runtimes;

    survival.reserve(take);
    normSurvival.reserve(take);
    meanSurvivalProb.reserve(take);
    destroyed.reserve(take);
    normKill.reserve(take);
    meanKill.reserve(take);
    runtimes.reserve(take);

    int bestIdx = 0;

    for (int r = 0; r < take; ++r) {
        const auto& q = runs[r].quality;

        survival.push_back(q.survival);
        normSurvival.push_back(q.normalizedSurvival);
        meanSurvivalProb.push_back(q.meanSurvivalProbability);

        destroyed.push_back(q.destroyedValue);
        normKill.push_back(q.normalizedKill);
        meanKill.push_back(q.meanKillProbability);

        runtimes.push_back(q.runtimeMs);

        if (q.survival <
            runs[bestIdx].quality.survival)
            bestIdx = r;
    }

    const RunRecord& best = runs[bestIdx];

    const double totalRuntime =
        accumulate(runtimes.begin(), runtimes.end(), 0.0);

    caseOut
        << key.weaponTypes << ","
        << key.targets << ","
        << "\"" << key.distribution << "\"" << ","
        << caseNumber << ","
        << instanceSeed << ","
        << key.family << ","
        << key.technique << ","
        << key.topK << ","
        << fixed << setprecision(4) << key.alpha << ","
        << (key.perturbation ? 1 : 0) << ","
        << setprecision(4) << key.perturbationPercent << ","
        << (key.permutation ? 1 : 0) << ","
        << key.budget << ","
        << take << ","
        << best.runIndex << ","
        << best.runSeed << ","

        << setprecision(15)
        << best.quality.survival << ","
        << vecMean(survival) << ","
        << vecStd(survival) << ","
        << vecMedian(survival) << ","
        << *min_element(survival.begin(), survival.end()) << ","
        << *max_element(survival.begin(), survival.end()) << ","

        << best.quality.normalizedSurvival << ","
        << vecMean(normSurvival) << ","
        << best.quality.meanSurvivalProbability << ","
        << vecMean(meanSurvivalProb) << ","

        << best.quality.destroyedValue << ","
        << vecMean(destroyed) << ","
        << best.quality.normalizedKill << ","
        << vecMean(normKill) << ","
        << best.quality.meanKillProbability << ","
        << vecMean(meanKill) << ","

        << setprecision(9)
        << vecMean(runtimes) << ","
        << vecMedian(runtimes) << ","
        << totalRuntime
        << "\n";

    OverallAccumulator& a = overall[key];

    a.bestSurvivalPerCase.push_back(
        best.quality.survival);
    a.meanSurvivalPerCase.push_back(
        vecMean(survival));

    a.bestNormalizedSurvivalPerCase.push_back(
        best.quality.normalizedSurvival);
    a.bestMeanSurvivalProbabilityPerCase.push_back(
        best.quality.meanSurvivalProbability);

    a.bestDestroyedValuePerCase.push_back(
        best.quality.destroyedValue);
    a.bestNormalizedKillPerCase.push_back(
        best.quality.normalizedKill);
    a.bestMeanKillProbabilityPerCase.push_back(
        best.quality.meanKillProbability);

    a.meanSingleRunRuntimePerCase.push_back(
        vecMean(runtimes));
    a.totalRuntimePerCase.push_back(
        totalRuntime);

    for (int r = 0; r < take; ++r) {
        a.allRunSurvival.push_back(
            runs[r].quality.survival);
        a.allRunRuntime.push_back(
            runs[r].quality.runtimeMs);
    }
}

// ----------------------------------------------------------------------------
// Generate one reproducible stochastic group of max 20 runs.
// ----------------------------------------------------------------------------
template <typename Runner>
vector<RunRecord> generateRuns(
    const Instance& original,
    int caseNumber,
    uint64_t instanceSeed,
    const string& family,
    const string& technique,
    int topK,
    double alpha,
    bool perturbation,
    double perturbPct,
    bool permutation,
    uint64_t techniqueTag,
    Runner runner,
    ofstream& allRunsOut
) {
    vector<RunRecord> rows;
    rows.reserve(EXP_MAX_RUNS);

    for (int r = 1; r <= EXP_MAX_RUNS; ++r) {
        const uint64_t runSeed =
            deriveSeed(
                instanceSeed,
                techniqueTag,
                r,
                topK,
                alphaCode(alpha),
                perturbCode(perturbPct)
            );

        Solution sol = runner(runSeed);
        QualityMetrics q = makeMetrics(original, sol);

        RunRecord rec;
        rec.weaponTypes = original.nTypes;
        rec.targets = original.nTargets;
        rec.distribution = ""; // caller replaces below
        rec.caseNumber = caseNumber;
        rec.instanceSeed = instanceSeed;
        rec.family = family;
        rec.technique = technique;
        rec.topK = topK;
        rec.alpha = alpha;
        rec.perturbation = perturbation;
        rec.perturbationPercent = perturbPct;
        rec.permutation = permutation;
        rec.runIndex = r;
        rec.maxBudget = EXP_MAX_RUNS;
        rec.runSeed = runSeed;
        rec.quality = q;

        rows.push_back(rec);
    }

    return rows;
}

// ----------------------------------------------------------------------------
// Write an already-populated run vector and its 10/20 case summaries.
// ----------------------------------------------------------------------------
void outputStochasticGroup(
    vector<RunRecord>& rows,
    const string& distribution,
    const Instance& original,
    int caseNumber,
    uint64_t instanceSeed,
    ofstream& allRunsOut,
    ofstream& caseSummaryOut,
    map<SummaryKey, OverallAccumulator>& overall
) {
    if (rows.empty()) return;

    for (RunRecord& r : rows) {
        r.distribution = distribution;
        writeAllRun(allRunsOut, r);
    }

    for (int budget : EXP_BUDGETS) {
        SummaryKey key;
        key.weaponTypes = original.nTypes;
        key.targets = original.nTargets;
        key.distribution = distribution;
        key.family = rows[0].family;
        key.technique = rows[0].technique;
        key.topK = rows[0].topK;
        key.alpha = rows[0].alpha;
        key.perturbation = rows[0].perturbation;
        key.perturbationPercent =
            rows[0].perturbationPercent;
        key.permutation = rows[0].permutation;
        key.budget = budget;

        summarizeCaseGroup(
            original,
            caseNumber,
            instanceSeed,
            key,
            rows,
            caseSummaryOut,
            overall
        );
    }
}

// ----------------------------------------------------------------------------
// Deterministic baseline output.
// ----------------------------------------------------------------------------
void outputBaseline(
    const Instance& original,
    const string& distribution,
    int caseNumber,
    uint64_t instanceSeed,
    const string& family,
    const string& technique,
    uint64_t runSeed,
    const Solution& sol,
    ofstream& allRunsOut,
    ofstream& caseSummaryOut,
    map<SummaryKey, OverallAccumulator>& overall
) {
    RunRecord r;
    r.weaponTypes = original.nTypes;
    r.targets = original.nTargets;
    r.distribution = distribution;
    r.caseNumber = caseNumber;
    r.instanceSeed = instanceSeed;
    r.family = family;
    r.technique = technique;
    r.runIndex = 1;
    r.maxBudget = 1;
    r.runSeed = runSeed;
    r.quality = makeMetrics(original, sol);

    writeAllRun(allRunsOut, r);

    vector<RunRecord> rows = {r};

    SummaryKey key;
    key.weaponTypes = original.nTypes;
    key.targets = original.nTargets;
    key.distribution = distribution;
    key.family = family;
    key.technique = technique;
    key.budget = 1;

    summarizeCaseGroup(
        original, caseNumber, instanceSeed,
        key, rows, caseSummaryOut, overall
    );
}

// ----------------------------------------------------------------------------
// Run one complete benchmark block: size x distribution x cases.
// ----------------------------------------------------------------------------
void runExperimentBlock(
    int n,
    PDistribution dist,
    int numCases,
    uint64_t baseSeed,
    ofstream& allRunsOut,
    ofstream& caseSummaryOut,
    map<SummaryKey, OverallAccumulator>& overall
) {
    const string dname = distName(dist);

    cout
        << "\n============================================================\n"
        << "Size " << n << " x " << n
        << " | " << dname
        << " | cases=" << numCases << "\n"
        << "============================================================\n";

    for (int c = 0; c < numCases; ++c) {
        const int caseNumber = c + 1;

        const uint64_t instanceSeed =
            mix64(
                baseSeed
                ^ (static_cast<uint64_t>(n) << 32)
                ^ (static_cast<uint64_t>(
                       static_cast<int>(dist) + 1) << 24)
                ^ static_cast<uint64_t>(caseNumber)
            );

        Instance ins =
            generateInstance(n, n, dist, instanceSeed);

        // ====================================================
        // BASELINES
        // ====================================================
        {
            Solution s = solveCWTAA(ins);
            outputBaseline(
                ins, dname, caseNumber, instanceSeed,
                "BASELINE", "CWTAA",
                deriveSeed(instanceSeed, 1, 1),
                s,
                allRunsOut, caseSummaryOut, overall
            );
        }

        {
            Solution s = solveRegretCWTAA(ins);
            outputBaseline(
                ins, dname, caseNumber, instanceSeed,
                "BASELINE", "RegretCWTAA",
                deriveSeed(instanceSeed, 2, 1),
                s,
                allRunsOut, caseSummaryOut, overall
            );
        }

        {
            Solution s = solveRepeatedHungarian(ins);
            outputBaseline(
                ins, dname, caseNumber, instanceSeed,
                "BASELINE", "RepeatedHungarian",
                deriveSeed(instanceSeed, 3, 1),
                s,
                allRunsOut, caseSummaryOut, overall
            );
        }

        // ====================================================
        // RR FAMILY
        // K x alpha:
        //   RR
        //   RR+Perturb
        //   RR+Permutation
        //   RR+Perturb+Permutation
        // ====================================================
        for (int k : EXP_TOP_K) {
            for (double alpha : EXP_ALPHA) {

                // ---------------- ordinary RR ----------------
                {
                    auto rows = generateRuns(
                        ins, caseNumber, instanceSeed,
                        "RR", "RR",
                        k, alpha,
                        false, 0.0, false,
                        1001,
                        [&](uint64_t seed) {
                            return runRRTechnique(
                                ins, k, alpha,
                                false, 0.0,
                                false, seed
                            );
                        },
                        allRunsOut
                    );

                    outputStochasticGroup(
                        rows, dname, ins,
                        caseNumber, instanceSeed,
                        allRunsOut, caseSummaryOut, overall
                    );
                }

                // ------------ RR + target permutation --------
                {
                    auto rows = generateRuns(
                        ins, caseNumber, instanceSeed,
                        "RR", "RR_Permutation",
                        k, alpha,
                        false, 0.0, true,
                        1002,
                        [&](uint64_t seed) {
                            return runRRTechnique(
                                ins, k, alpha,
                                false, 0.0,
                                true, seed
                            );
                        },
                        allRunsOut
                    );

                    outputStochasticGroup(
                        rows, dname, ins,
                        caseNumber, instanceSeed,
                        allRunsOut, caseSummaryOut, overall
                    );
                }

                for (double pct : EXP_PERTURB_PCT) {
                    // -------- RR + perturbation --------------
                    {
                        auto rows = generateRuns(
                            ins, caseNumber, instanceSeed,
                            "RR", "RR_Perturbation",
                            k, alpha,
                            true, pct, false,
                            1003,
                            [&](uint64_t seed) {
                                return runRRTechnique(
                                    ins, k, alpha,
                                    true, pct,
                                    false, seed
                                );
                            },
                            allRunsOut
                        );

                        outputStochasticGroup(
                            rows, dname, ins,
                            caseNumber, instanceSeed,
                            allRunsOut, caseSummaryOut, overall
                        );
                    }

                    // ----- RR + perturbation + permutation ----
                    {
                        auto rows = generateRuns(
                            ins, caseNumber, instanceSeed,
                            "RR",
                            "RR_Perturbation_Permutation",
                            k, alpha,
                            true, pct, true,
                            1004,
                            [&](uint64_t seed) {
                                return runRRTechnique(
                                    ins, k, alpha,
                                    true, pct,
                                    true, seed
                                );
                            },
                            allRunsOut
                        );

                        outputStochasticGroup(
                            rows, dname, ins,
                            caseNumber, instanceSeed,
                            allRunsOut, caseSummaryOut, overall
                        );
                    }
                }
            }
        }

        // ====================================================
        // CWTAA DIVERSIFICATION FAMILY
        // No K / alpha because CWTAA does not use them.
        // ====================================================

        // ---------------- CWTAA + permutation ----------------
        {
            auto rows = generateRuns(
                ins, caseNumber, instanceSeed,
                "CWTAA_DIVERSIFIED",
                "CWTAA_Permutation",
                0, 0.0,
                false, 0.0, true,
                2001,
                [&](uint64_t seed) {
                    return runCWTAAWithTransforms(
                        ins,
                        false, 0.0,
                        true, seed
                    );
                },
                allRunsOut
            );

            outputStochasticGroup(
                rows, dname, ins,
                caseNumber, instanceSeed,
                allRunsOut, caseSummaryOut, overall
            );
        }

        for (double pct : EXP_PERTURB_PCT) {
            // --------------- CWTAA + perturbation ------------
            {
                auto rows = generateRuns(
                    ins, caseNumber, instanceSeed,
                    "CWTAA_DIVERSIFIED",
                    "CWTAA_Perturbation",
                    0, 0.0,
                    true, pct, false,
                    2002,
                    [&](uint64_t seed) {
                        return runCWTAAWithTransforms(
                            ins,
                            true, pct,
                            false, seed
                        );
                    },
                    allRunsOut
                );

                outputStochasticGroup(
                    rows, dname, ins,
                    caseNumber, instanceSeed,
                    allRunsOut, caseSummaryOut, overall
                );
            }

            // ------- CWTAA + perturbation + permutation ------
            {
                auto rows = generateRuns(
                    ins, caseNumber, instanceSeed,
                    "CWTAA_DIVERSIFIED",
                    "CWTAA_Perturbation_Permutation",
                    0, 0.0,
                    true, pct, true,
                    2003,
                    [&](uint64_t seed) {
                        return runCWTAAWithTransforms(
                            ins,
                            true, pct,
                            true, seed
                        );
                    },
                    allRunsOut
                );

                outputStochasticGroup(
                    rows, dname, ins,
                    caseNumber, instanceSeed,
                    allRunsOut, caseSummaryOut, overall
                );
            }
        }

        cout
            << "Finished case "
            << caseNumber << "/" << numCases
            << " for " << n << "x" << n
            << " " << dname << "\n";
    }
}

// ----------------------------------------------------------------------------
// Overall summary.
// ----------------------------------------------------------------------------
void writeOverallSummary(
    ofstream& out,
    const map<SummaryKey, OverallAccumulator>& overall
) {
    for (const auto& kv : overall) {
        const SummaryKey& k = kv.first;
        const OverallAccumulator& a = kv.second;

        out
            << k.weaponTypes << ","
            << k.targets << ","
            << "\"" << k.distribution << "\"" << ","
            << k.family << ","
            << k.technique << ","
            << k.topK << ","
            << fixed << setprecision(4) << k.alpha << ","
            << (k.perturbation ? 1 : 0) << ","
            << setprecision(4) << k.perturbationPercent << ","
            << (k.permutation ? 1 : 0) << ","
            << k.budget << ","
            << a.bestSurvivalPerCase.size() << ","

            << setprecision(15)
            << vecMean(a.bestSurvivalPerCase) << ","
            << vecStd(a.bestSurvivalPerCase) << ","
            << vecMedian(a.bestSurvivalPerCase) << ","

            << vecMean(a.meanSurvivalPerCase) << ","
            << vecStd(a.meanSurvivalPerCase) << ","

            << vecMean(a.bestNormalizedSurvivalPerCase) << ","
            << vecMean(a.bestMeanSurvivalProbabilityPerCase) << ","

            << vecMean(a.bestDestroyedValuePerCase) << ","
            << vecMean(a.bestNormalizedKillPerCase) << ","
            << vecMean(a.bestMeanKillProbabilityPerCase) << ","

            << setprecision(9)
            << vecMean(a.meanSingleRunRuntimePerCase) << ","
            << vecMedian(a.meanSingleRunRuntimePerCase) << ","
            << vecMean(a.totalRuntimePerCase) << ","

            << setprecision(15)
            << vecMean(a.allRunSurvival) << ","
            << vecStd(a.allRunSurvival) << ","
            << setprecision(9)
            << vecMean(a.allRunRuntime)
            << "\n";
    }
}

// ----------------------------------------------------------------------------
// CLI:
//
// No arguments:
//   Full requested experiment:
//     4 sizes x 3 distributions x 20 cases.
//
// Quick/custom square run:
//   ./program N DIST CASES [SEED]
//
// Examples:
//   ./program 50 uniform 2
//   ./program 125 gaussian 20 123456789
//
// This custom mode still runs the COMPLETE hyperparameter/technique grid.
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    ofstream allRunsOut("wtap_final_all_runs.csv");
    ofstream caseSummaryOut("wtap_final_case_summary.csv");
    ofstream overallSummaryOut("wtap_final_overall_summary.csv");

    if (!allRunsOut ||
        !caseSummaryOut ||
        !overallSummaryOut) {
        cerr << "ERROR: Could not create output CSV files.\n";
        return 1;
    }

    allRunsOut
        << "weapon_types,targets,distribution,case,instance_seed,"
           "family,technique,top_k,alpha,"
           "uses_perturbation,perturbation_percent,"
           "uses_target_permutation,"
           "run_index,max_budget,run_seed,"
           "total_survival_value,normalized_survival,"
           "mean_survival_probability,"
           "total_destroyed_value,normalized_kill,"
           "mean_kill_probability,"
           "runtime_ms,valid\n";

    caseSummaryOut
        << "weapon_types,targets,distribution,case,instance_seed,"
           "family,technique,top_k,alpha,"
           "uses_perturbation,perturbation_percent,"
           "uses_target_permutation,budget,runs_used,"
           "best_run_index,best_run_seed,"
           "best_survival,mean_survival,std_survival,"
           "median_survival,min_survival,max_survival,"
           "best_normalized_survival,mean_normalized_survival,"
           "best_mean_survival_probability,"
           "mean_mean_survival_probability,"
           "best_destroyed_value,mean_destroyed_value,"
           "best_normalized_kill,mean_normalized_kill,"
           "best_mean_kill_probability,"
           "mean_mean_kill_probability,"
           "mean_single_run_runtime_ms,"
           "median_single_run_runtime_ms,"
           "total_search_runtime_ms\n";

    overallSummaryOut
        << "weapon_types,targets,distribution,"
           "family,technique,top_k,alpha,"
           "uses_perturbation,perturbation_percent,"
           "uses_target_permutation,budget,num_cases,"
           "mean_best_survival,std_best_survival,"
           "median_best_survival,"
           "mean_case_average_survival,"
           "std_case_average_survival,"
           "mean_best_normalized_survival,"
           "mean_best_mean_survival_probability,"
           "mean_best_destroyed_value,"
           "mean_best_normalized_kill,"
           "mean_best_mean_kill_probability,"
           "mean_single_run_runtime_ms,"
           "median_case_mean_run_runtime_ms,"
           "mean_total_search_runtime_ms,"
           "mean_all_individual_run_survival,"
           "std_all_individual_run_survival,"
           "mean_all_individual_run_runtime_ms\n";

    map<SummaryKey, OverallAccumulator> overall;

    try {
        if (argc == 1) {
            const vector<PDistribution> dists = {
                PDistribution::Uniform,
                PDistribution::Beta,
                PDistribution::Gaussian
            };

            for (PDistribution d : dists) {
                for (int n : EXP_SIZES) {
                    runExperimentBlock(
                        n, d, EXP_CASES,
                        EXP_DEFAULT_SEED,
                        allRunsOut,
                        caseSummaryOut,
                        overall
                    );
                }
            }
        }
        else if (argc == 4 || argc == 5) {
            const int n = stoi(argv[1]);
            const PDistribution d =
                parseDistribution(argv[2]);
            const int cases = stoi(argv[3]);
            const uint64_t seed =
                (argc == 5)
                ? stoull(argv[4])
                : EXP_DEFAULT_SEED;

            runExperimentBlock(
                n, d, cases, seed,
                allRunsOut,
                caseSummaryOut,
                overall
            );
        }
        else {
            cerr
                << "Usage:\n"
                << "  " << argv[0]
                << "                         # full experiment\n"
                << "  " << argv[0]
                << " N DIST CASES [SEED]     # custom square experiment\n";
            return 1;
        }
    }
    catch (const exception& e) {
        cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    writeOverallSummary(
        overallSummaryOut, overall
    );

    cout
        << "\n============================================================\n"
        << "EXPERIMENT COMPLETE\n"
        << "============================================================\n"
        << "Every individual run:\n"
        << "  wtap_final_all_runs.csv\n\n"
        << "Per-case best/mean/std/median/runtime at budgets 10 and 20:\n"
        << "  wtap_final_case_summary.csv\n\n"
        << "Across-case aggregate results:\n"
        << "  wtap_final_overall_summary.csv\n";

    return 0;
}
