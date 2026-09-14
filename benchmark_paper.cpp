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

// ------------------------------------------------------------
// Statistics helpers
// ------------------------------------------------------------
struct Aggregate {
    int count = 0;
    long double sumCost = 0.0L;
    long double sumNorm = 0.0L;
    long double sumMeanProb = 0.0L;
    long double sumMs = 0.0L;

    void add(const Solution& s) {
        ++count;
        sumCost += s.totalSurvivalValue;
        sumNorm += s.normalizedSurvival;
        sumMeanProb += s.meanSurvivalProbability;
        sumMs += s.milliseconds;
    }

    double avgCost() const {
        return count ? static_cast<double>(sumCost / count) : 0.0;
    }
    double avgNorm() const {
        return count ? static_cast<double>(sumNorm / count) : 0.0;
    }
    double avgMeanProb() const {
        return count ? static_cast<double>(sumMeanProb / count) : 0.0;
    }
    double avgMs() const {
        return count ? static_cast<double>(sumMs / count) : 0.0;
    }
};

PDistribution parseDistribution(const string& s) {
    string x = s;
    transform(x.begin(), x.end(), x.begin(),
              [](unsigned char c) { return static_cast<char>(tolower(c)); });

    if (x == "uniform" || x == "u")
        return PDistribution::Uniform;
    if (x == "beta" || x == "b")
        return PDistribution::Beta;
    if (x == "gaussian" || x == "normal" || x == "g")
        return PDistribution::Gaussian;

    throw invalid_argument(
        "Distribution must be uniform, beta, or gaussian."
    );
}



struct RunRecord {
    int nTypes = 0;
    int nTargets = 0;
    string distribution;
    int caseNumber = 0;
    uint64_t instanceSeed = 0;

    string method;
    string variant;
    double perturbationPercent = 0.0;
    int runIndex = 0;
    uint64_t runSeed = 0;

    Solution solution;

    bool bestWithinVariant = false;
    bool bestWithinMethod = false;
    bool bestOverallCase = false;
};

double meanOf(const vector<double>& x) {
    if (x.empty()) return 0.0;
    long double s = 0.0L;
    for (double v : x) s += v;
    return static_cast<double>(s / x.size());
}

double stddevOf(const vector<double>& x) {
    if (x.size() <= 1) return 0.0;
    const double mu = meanOf(x);
    long double ss = 0.0L;
    for (double v : x) {
        long double d = static_cast<long double>(v) - mu;
        ss += d * d;
    }
    return sqrt(static_cast<double>(ss / (x.size() - 1)));
}

string pctLabel(double pct) {
    ostringstream oss;
    if (fabs(pct - 0.01) < 1e-12) oss << "0.01%";
    else if (fabs(pct - 0.05) < 1e-12) oss << "0.05%";
    else if (fabs(pct - 0.10) < 1e-12) oss << "0.10%";
    else if (fabs(pct - 0.50) < 1e-12) oss << "0.50%";
    else if (fabs(pct - 1.00) < 1e-12) oss << "1.00%";
    else if (fabs(pct - 5.00) < 1e-12) oss << "5.00%";
    else oss << fixed << setprecision(4) << pct << "%";
    return oss.str();
}

// ------------------------------------------------------------
// One UNPERTURBED randomized-regret construction.
// A fresh RNG seed gives an independent randomized trajectory.
// ------------------------------------------------------------
Solution runOneUnperturbedRR(
    const Instance& ins,
    uint64_t runSeed,
    int topK = RR_TOP_K,
    double alpha = RR_ALPHA
) {
    auto start = Clock::now();

    const int n = ins.nTypes;
    const int m = ins.nTargets;

    vector<vector<double>> S(n, vector<double>(m, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            S[i][j] = ins.P[i][j] * ins.V[j];

    mt19937_64 rng(runSeed);
    Solution sol = constructRRFromScoreMatrix(ins, S, rng, topK, alpha);

    auto end = Clock::now();
    sol.milliseconds =
        chrono::duration<double, milli>(end - start).count();

    return sol;
}

// ------------------------------------------------------------
// Idea 9: TARGET-PERMUTED, NON-PERTURBED Randomized Regret.
//
// For each run:
//   1. Randomly permute the TARGET order.
//   2. Reorder P's columns and V consistently.
//   3. Run ordinary Randomized-Regret on that permuted,
//      NON-PERTURBED instance.
//   4. Map the assignment back to the original target indices.
//   5. Evaluate the mapped assignment on the ORIGINAL instance.
//
// No numerical perturbation is applied to P or S.
// ------------------------------------------------------------
Solution runOneTargetPermutedRR(
    const Instance& original,
    uint64_t runSeed,
    int topK = RR_TOP_K,
    double alpha = RR_ALPHA
) {
    auto start = Clock::now();

    const int n = original.nTypes;
    const int m = original.nTargets;

    mt19937_64 rng(runSeed);

    // perm[k] = original target index placed at permuted position k.
    vector<int> perm(m);
    iota(perm.begin(), perm.end(), 0);
    shuffle(perm.begin(), perm.end(), rng);

    Instance permuted;
    permuted.nTypes = n;
    permuted.nTargets = m;
    permuted.W = original.W;
    permuted.V.resize(m);
    permuted.P.assign(n, vector<double>(m, 0.0));

    for (int k = 0; k < m; ++k) {
        const int originalTarget = perm[k];
        permuted.V[k] = original.V[originalTarget];

        for (int i = 0; i < n; ++i) {
            permuted.P[i][k] = original.P[i][originalTarget];
        }
    }

    vector<vector<double>> S(n, vector<double>(m, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < m; ++k) {
            S[i][k] = permuted.P[i][k] * permuted.V[k];
        }
    }

    // Use the same RNG after the shuffle, so the whole run is
    // exactly reproducible from runSeed.
    Solution permutedSol =
        constructRRFromScoreMatrix(permuted, S, rng, topK, alpha);

    // Map target assignments back to ORIGINAL target indices.
    Solution mapped;
    mapped.typeOfTarget.assign(m, -1);

    for (int k = 0; k < m; ++k) {
        const int originalTarget = perm[k];
        mapped.typeOfTarget[originalTarget] =
            permutedSol.typeOfTarget[k];
    }

    // All methods are compared using the original problem.
    evaluateSolution(original, mapped);

    auto end = Clock::now();
    mapped.milliseconds =
        chrono::duration<double, milli>(end - start).count();

    return mapped;
}

// ------------------------------------------------------------
// One INPUT-PERTURBED randomized-regret construction.
//
// IMPORTANT: this perturbs P itself:
//
//   P_tilde[i][j] = clip( P[i][j] * (1 + eps_ij), 0, 1 )
//
// where eps_ij ~ U[-delta, +delta].
//
// The RR search sees P_tilde through
//
//   S_tilde[i][j] = P_tilde[i][j] * V[j]
//
// BUT the returned assignment is evaluated using the ORIGINAL
// unperturbed instance P. Therefore all objective values remain
// directly comparable.
// ------------------------------------------------------------
Solution runOneInputPerturbedRR(
    const Instance& original,
    uint64_t runSeed,
    double perturbationPercent,
    int topK = RR_TOP_K,
    double alpha = RR_ALPHA
) {
    auto start = Clock::now();

    const int n = original.nTypes;
    const int m = original.nTargets;

    const double delta = perturbationPercent / 100.0;

    mt19937_64 rng(runSeed);
    uniform_real_distribution<double> noiseDist(-delta, delta);

    vector<vector<double>> perturbedS(n, vector<double>(m, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            const double eps = noiseDist(rng);

            double perturbedP =
                original.P[i][j] * (1.0 + eps);

            perturbedP = max(0.0, min(1.0, perturbedP));

            perturbedS[i][j] =
                perturbedP * original.V[j];
        }
    }

    // constructRRFromScoreMatrix uses perturbedS for the search,
    // but evaluates the final assignment using `original`.
    Solution sol = constructRRFromScoreMatrix(
        original, perturbedS, rng, topK, alpha
    );

    auto end = Clock::now();
    sol.milliseconds =
        chrono::duration<double, milli>(end - start).count();

    return sol;
}

struct OverallBucket {
    vector<double> allCosts;
    vector<double> allNorms;
    vector<double> allMeanProb;
    vector<double> allTimes;

    vector<double> bestCostPerCase;
    vector<double> bestNormPerCase;
    vector<double> bestMeanProbPerCase;
    vector<double> totalTimePerCase;
};

// Add one row to the all-runs CSV.
void writeRunRecord(ofstream& allRuns, const RunRecord& r) {
    allRuns
        << r.nTypes << ","
        << r.nTargets << ","
        << "\"" << r.distribution << "\"" << ","
        << r.caseNumber << ","
        << r.instanceSeed << ","
        << r.method << ","
        << r.variant << ","
        << fixed << setprecision(4)
        << r.perturbationPercent << ","
        << r.runIndex << ","
        << r.runSeed << ","
        << setprecision(15)
        << r.solution.totalSurvivalValue << ","
        << r.solution.normalizedSurvival << ","
        << r.solution.meanSurvivalProbability << ","
        << setprecision(9)
        << r.solution.milliseconds << ","
        << (r.bestWithinVariant ? 1 : 0) << ","
        << (r.bestWithinMethod ? 1 : 0) << ","
        << (r.bestOverallCase ? 1 : 0)
        << "\n";
}

void runBenchmark(
    int nTypes,
    int nTargets,
    PDistribution dist,
    int numCases,
    uint64_t baseSeed,
    ofstream& allRuns,
    ofstream& caseSummary,
    map<string, OverallBucket>& overall
) {
    // Fair-search comparison:
    // 19 independent unperturbed RR runs versus
    // 18 perturbed RR runs = 3 runs at each of 6 perturbation levels.
    //
    // We use 19 unperturbed runs because the earlier perturbation search
    // had 19 total candidates (1 original + 18 perturbed). This gives us
    // a clean equal-budget randomized baseline for later analysis.
    const int UNPERTURBED_RR_RUNS = 19;
    const int PERTURBED_RUNS_PER_LEVEL = 3;
    const int TARGET_PERMUTATION_RR_RUNS = 10;

    const vector<double> perturbationLevelsPercent = {
        0.01, 0.05, 0.10, 0.50, 1.00, 5.00
    };

    cout << "\n============================================================\n";
    cout << "nTypes=" << nTypes
         << ", nTargets=" << nTargets
         << ", distribution=" << distName(dist)
         << ", cases=" << numCases << "\n";
    cout << "Per instance:\n";
    cout << "  CWTAA:                     1 run\n";
    cout << "  Deterministic Regret:      1 run\n";
    cout << "  Randomized Regret:        19 unperturbed runs\n";
    cout << "  Input-Perturbed RR:       18 runs "
            "(3 each at 0.01%,0.05%,0.1%,0.5%,1%,5%)\n";
    cout << "  Target-Permuted RR:       10 NON-PERTURBED runs\n";
    cout << "  Repeated Hungarian:        1 run\n";
    cout << "All individual runs are written to CSV.\n";
    cout << "Perturbation is applied to P[i][j], not directly to S[i][j].\n";
    cout << "All solutions are evaluated on the ORIGINAL P matrix.\n";
    cout << "============================================================\n";

    for (int c = 0; c < numCases; ++c) {
        const uint64_t instanceSeed =
            baseSeed
            + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(c)
            + 0xD1B54A32D192ED03ULL * static_cast<uint64_t>(nTypes)
            + 0x94D049BB133111EBULL *
              static_cast<uint64_t>(static_cast<int>(dist) + 1);

        Instance ins =
            generateInstance(nTypes, nTargets, dist, instanceSeed);

        vector<RunRecord> rows;

        auto addRow = [&](const string& method,
                          const string& variant,
                          double pct,
                          int runIndex,
                          uint64_t runSeed,
                          const Solution& sol) {
            RunRecord r;
            r.nTypes = nTypes;
            r.nTargets = nTargets;
            r.distribution = distName(dist);
            r.caseNumber = c + 1;
            r.instanceSeed = instanceSeed;
            r.method = method;
            r.variant = variant;
            r.perturbationPercent = pct;
            r.runIndex = runIndex;
            r.runSeed = runSeed;
            r.solution = sol;
            rows.push_back(r);
        };

        // ----------------------------------------------------
        // 1) CWTAA
        // ----------------------------------------------------
        {
            Solution sol = solveCWTAA(ins);
            addRow(
                "CWTAA", "CWTAA",
                0.0, 1, instanceSeed, sol
            );
        }

        // ----------------------------------------------------
        // 2) Deterministic Regret-CWTAA
        // ----------------------------------------------------
        {
            Solution sol = solveRegretCWTAA(ins);
            addRow(
                "RegretCWTAA", "DeterministicRegret",
                0.0, 1, instanceSeed, sol
            );
        }

        // ----------------------------------------------------
        // 3) Unperturbed Randomized-Regret: 19 runs
        // ----------------------------------------------------
        for (int r = 1; r <= UNPERTURBED_RR_RUNS; ++r) {
            const uint64_t runSeed =
                instanceSeed
                ^ (0xA24BAED4963EE407ULL *
                   static_cast<uint64_t>(r));

            Solution sol =
                runOneUnperturbedRR(ins, runSeed);

            addRow(
                "RandomizedRegret",
                "UnperturbedRR",
                0.0,
                r,
                runSeed,
                sol
            );
        }

        // ----------------------------------------------------
        // 4) INPUT-PERTURBED RR:
        //    6 levels x 3 independent runs = 18 runs
        // ----------------------------------------------------
        for (size_t l = 0;
             l < perturbationLevelsPercent.size();
             ++l) {

            const double pct = perturbationLevelsPercent[l];

            for (int r = 1;
                 r <= PERTURBED_RUNS_PER_LEVEL;
                 ++r) {

                const uint64_t runSeed =
                    instanceSeed
                    ^ (0x9E3779B97F4A7C15ULL *
                       static_cast<uint64_t>(l + 1))
                    ^ (0xD1B54A32D192ED03ULL *
                       static_cast<uint64_t>(r));

                Solution sol =
                    runOneInputPerturbedRR(
                        ins, runSeed, pct
                    );

                addRow(
                    "PerturbedRandomizedRegret",
                    "InputPerturb_" + pctLabel(pct),
                    pct,
                    r,
                    runSeed,
                    sol
                );
            }
        }

        // ----------------------------------------------------
        // 5) IDEA 9: Target-permuted NON-PERTURBED RR
        //    10 independent target permutations per case.
        // ----------------------------------------------------
        for (int r = 1; r <= TARGET_PERMUTATION_RR_RUNS; ++r) {
            const uint64_t runSeed =
                instanceSeed
                ^ (0xBF58476D1CE4E5B9ULL *
                   static_cast<uint64_t>(r))
                ^ 0x94D049BB133111EBULL;

            Solution sol =
                runOneTargetPermutedRR(ins, runSeed);

            addRow(
                "PermutedRandomizedRegret",
                "TargetPermutationRR",
                0.0,
                r,
                runSeed,
                sol
            );
        }

        // ----------------------------------------------------
        // 6) Repeated Hungarian
        // ----------------------------------------------------
        {
            Solution sol = solveRepeatedHungarian(ins);
            addRow(
                "RepeatedHungarian",
                "RepeatedHungarian",
                0.0, 1, instanceSeed, sol
            );
        }

        // ----------------------------------------------------
        // Mark best rows.
        // ----------------------------------------------------
        const double TOL = 1e-12;

        double overallBest = numeric_limits<double>::infinity();
        map<string, double> bestByVariant;
        map<string, double> bestByMethod;

        for (const RunRecord& r : rows) {
            overallBest =
                min(overallBest, r.solution.totalSurvivalValue);

            auto vit = bestByVariant.find(r.variant);
            if (vit == bestByVariant.end())
                bestByVariant[r.variant] =
                    r.solution.totalSurvivalValue;
            else
                vit->second =
                    min(vit->second, r.solution.totalSurvivalValue);

            auto mit = bestByMethod.find(r.method);
            if (mit == bestByMethod.end())
                bestByMethod[r.method] =
                    r.solution.totalSurvivalValue;
            else
                mit->second =
                    min(mit->second, r.solution.totalSurvivalValue);
        }

        for (RunRecord& r : rows) {
            r.bestWithinVariant =
                fabs(r.solution.totalSurvivalValue -
                     bestByVariant[r.variant]) <= TOL;

            r.bestWithinMethod =
                fabs(r.solution.totalSurvivalValue -
                     bestByMethod[r.method]) <= TOL;

            r.bestOverallCase =
                fabs(r.solution.totalSurvivalValue -
                     overallBest) <= TOL;

            writeRunRecord(allRuns, r);
        }

        // ----------------------------------------------------
        // Create per-case summaries for every variant.
        // ----------------------------------------------------
        map<string, vector<const RunRecord*>> byVariant;
        for (const RunRecord& r : rows)
            byVariant[r.variant].push_back(&r);

        auto writeCaseVariant =
            [&](const string& family,
                const string& variant,
                const vector<const RunRecord*>& v) {

            vector<double> costs, norms, probs, times;
            costs.reserve(v.size());
            norms.reserve(v.size());
            probs.reserve(v.size());
            times.reserve(v.size());

            double bestCost =
                numeric_limits<double>::infinity();
            double bestNorm = 0.0;
            double bestProb = 0.0;
            int bestRun = -1;
            uint64_t bestRunSeed = 0;
            double pct = 0.0;

            for (const RunRecord* p : v) {
                costs.push_back(
                    p->solution.totalSurvivalValue);
                norms.push_back(
                    p->solution.normalizedSurvival);
                probs.push_back(
                    p->solution.meanSurvivalProbability);
                times.push_back(
                    p->solution.milliseconds);

                pct = p->perturbationPercent;

                if (p->solution.totalSurvivalValue <
                    bestCost) {
                    bestCost =
                        p->solution.totalSurvivalValue;
                    bestNorm =
                        p->solution.normalizedSurvival;
                    bestProb =
                        p->solution.meanSurvivalProbability;
                    bestRun = p->runIndex;
                    bestRunSeed = p->runSeed;
                }
            }

            const double totalTime =
                accumulate(times.begin(), times.end(), 0.0);

            caseSummary
                << nTypes << ","
                << nTargets << ","
                << "\"" << distName(dist) << "\"" << ","
                << (c + 1) << ","
                << instanceSeed << ","
                << family << ","
                << variant << ","
                << fixed << setprecision(4)
                << pct << ","
                << v.size() << ","
                << bestRun << ","
                << bestRunSeed << ","
                << setprecision(15)
                << bestCost << ","
                << meanOf(costs) << ","
                << stddevOf(costs) << ","
                << *min_element(costs.begin(), costs.end()) << ","
                << *max_element(costs.begin(), costs.end()) << ","
                << bestNorm << ","
                << meanOf(norms) << ","
                << bestProb << ","
                << meanOf(probs) << ","
                << setprecision(9)
                << meanOf(times) << ","
                << totalTime << ","
                << (fabs(bestCost - overallBest) <= TOL ? 1 : 0)
                << "\n";

            string overallKey =
                to_string(nTypes) + "|" +
                to_string(nTargets) + "|" +
                distName(dist) + "|" +
                family + "|" + variant;

            OverallBucket& ob = overall[overallKey];

            for (size_t z = 0; z < costs.size(); ++z) {
                ob.allCosts.push_back(costs[z]);
                ob.allNorms.push_back(norms[z]);
                ob.allMeanProb.push_back(probs[z]);
                ob.allTimes.push_back(times[z]);
            }

            ob.bestCostPerCase.push_back(bestCost);
            ob.bestNormPerCase.push_back(bestNorm);
            ob.bestMeanProbPerCase.push_back(bestProb);
            ob.totalTimePerCase.push_back(totalTime);
        };

        // Individual deterministic methods and each RR variant.
        for (const auto& kv : byVariant) {
            const string& variant = kv.first;
            const vector<const RunRecord*>& v = kv.second;
            const string family = v.front()->method;
            writeCaseVariant(family, variant, v);
        }

        // ----------------------------------------------------
        // Extra combined summaries useful for analysis:
        //   - best of all 19 unperturbed RR runs
        //   - best of all 18 perturbed RR runs
        //   - best of ALL randomized RR searches (37 total)
        // ----------------------------------------------------
        vector<const RunRecord*> unperturbedRR;
        vector<const RunRecord*> allPerturbedRR;
        vector<const RunRecord*> targetPermutedRR;
        vector<const RunRecord*> allRandomizedSearch;

        for (const RunRecord& r : rows) {
            if (r.method == "RandomizedRegret") {
                unperturbedRR.push_back(&r);
                allRandomizedSearch.push_back(&r);
            }
            else if (r.method ==
                     "PerturbedRandomizedRegret") {
                allPerturbedRR.push_back(&r);
                allRandomizedSearch.push_back(&r);
            }
            else if (r.method ==
                     "PermutedRandomizedRegret") {
                targetPermutedRR.push_back(&r);
                allRandomizedSearch.push_back(&r);
            }
        }

        writeCaseVariant(
            "RandomizedRegret",
            "UnperturbedRR_BestOf19",
            unperturbedRR
        );

        writeCaseVariant(
            "PerturbedRandomizedRegret",
            "PerturbedRR_BestOf18",
            allPerturbedRR
        );

        writeCaseVariant(
            "PermutedRandomizedRegret",
            "TargetPermutationRR_BestOf10",
            targetPermutedRR
        );

        writeCaseVariant(
            "CombinedRandomizedSearch",
            "BestOf47_RR_Perturbed_Permuted",
            allRandomizedSearch
        );

        // Terminal: compact one-line comparison for this case.
        auto bestCostOf = [&](const string& variant) {
            double b = numeric_limits<double>::infinity();
            for (const RunRecord& r : rows)
                if (r.variant == variant)
                    b = min(b, r.solution.totalSurvivalValue);
            return b;
        };

        double bestUnpert =
            numeric_limits<double>::infinity();
        double bestPert =
            numeric_limits<double>::infinity();
        double bestPerm =
            numeric_limits<double>::infinity();
        double cwtaaCost =
            numeric_limits<double>::infinity();
        double regretCost =
            numeric_limits<double>::infinity();
        double hungCost =
            numeric_limits<double>::infinity();

        for (const RunRecord& r : rows) {
            if (r.method == "RandomizedRegret")
                bestUnpert = min(
                    bestUnpert,
                    r.solution.totalSurvivalValue
                );
            else if (r.method ==
                     "PerturbedRandomizedRegret")
                bestPert = min(
                    bestPert,
                    r.solution.totalSurvivalValue
                );
            else if (r.method ==
                     "PermutedRandomizedRegret")
                bestPerm = min(
                    bestPerm,
                    r.solution.totalSurvivalValue
                );
            else if (r.method == "CWTAA")
                cwtaaCost = r.solution.totalSurvivalValue;
            else if (r.method == "RegretCWTAA")
                regretCost = r.solution.totalSurvivalValue;
            else if (r.method == "RepeatedHungarian")
                hungCost = r.solution.totalSurvivalValue;
        }

        cout << "Case " << setw(2) << (c + 1)
             << " | CWTAA=" << fixed << setprecision(6)
             << cwtaaCost
             << " | Regret=" << regretCost
             << " | RR19=" << bestUnpert
             << " | Pert18=" << bestPert
             << " | Perm10=" << bestPerm
             << " | Hungarian=" << hungCost
             << "\n";
    }
}

// ------------------------------------------------------------
// Write experiment-wide aggregate summaries after every benchmark
// block has finished.
// ------------------------------------------------------------
void writeOverallSummary(
    ofstream& out,
    const map<string, OverallBucket>& overall
) {
    for (const auto& kv : overall) {
        const string& key = kv.first;
        const OverallBucket& b = kv.second;

        vector<string> parts;
        string current;
        for (char ch : key) {
            if (ch == '|') {
                parts.push_back(current);
                current.clear();
            }
            else current.push_back(ch);
        }
        parts.push_back(current);

        if (parts.size() != 5)
            continue;

        const int nTypes = stoi(parts[0]);
        const int nTargets = stoi(parts[1]);
        const string& dist = parts[2];
        const string& family = parts[3];
        const string& variant = parts[4];

        out
            << nTypes << ","
            << nTargets << ","
            << "\"" << dist << "\"" << ","
            << family << ","
            << variant << ","
            << b.bestCostPerCase.size() << ","
            << b.allCosts.size() << ","
            << setprecision(15)
            << meanOf(b.allCosts) << ","
            << stddevOf(b.allCosts) << ","
            << meanOf(b.bestCostPerCase) << ","
            << stddevOf(b.bestCostPerCase) << ","
            << meanOf(b.bestNormPerCase) << ","
            << meanOf(b.bestMeanProbPerCase) << ","
            << setprecision(9)
            << meanOf(b.allTimes) << ","
            << meanOf(b.totalTimePerCase)
            << "\n";
    }
}

// ------------------------------------------------------------
// Usage:
//
// 1) Full paper-style benchmark:
//      ./wtap_full_analysis
//
// 2) Square custom benchmark:
//      ./wtap_full_analysis N DIST CASES [SEED]
//
//    Example:
//      ./wtap_full_analysis 125 gaussian 20 123456789
//
// 3) General nTypes <= nTargets:
//      ./wtap_full_analysis NTYPES NTARGETS DIST CASES SEED
//
//    Note: the 5-argument general form includes the seed explicitly
//    to avoid the argc ambiguity present in the older benchmark.
//
// Output:
//   wtap_all_runs.csv
//   wtap_case_summary.csv
//   wtap_overall_summary.csv
//
// Idea 9 adds 10 target-permuted, NON-PERTURBED RR runs per case.
// ------------------------------------------------------------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    const uint64_t DEFAULT_SEED = 123456789ULL;

    ofstream allRuns("wtap_all_runs.csv");
    ofstream caseSummary("wtap_case_summary.csv");
    ofstream overallSummary("wtap_overall_summary.csv");

    if (!allRuns || !caseSummary || !overallSummary) {
        cerr << "Could not create output CSV files.\n";
        return 1;
    }

    allRuns
        << "weapon_types,targets,distribution,case,instance_seed,"
           "method,variant,perturbation_percent,run_index,run_seed,"
           "total_survival_value,normalized_survival,"
           "mean_survival_probability,time_ms,"
           "best_within_variant,best_within_method,"
           "best_overall_case\n";

    caseSummary
        << "weapon_types,targets,distribution,case,instance_seed,"
           "method,variant,perturbation_percent,num_runs,"
           "best_run_index,best_run_seed,"
           "best_total_survival,mean_total_survival,"
           "std_total_survival,min_total_survival,max_total_survival,"
           "best_normalized_survival,mean_normalized_survival,"
           "best_mean_survival_probability,"
           "mean_mean_survival_probability,"
           "mean_time_ms,total_time_ms,best_overall_case\n";

    overallSummary
        << "weapon_types,targets,distribution,method,variant,"
           "num_cases,total_runs,"
           "mean_all_run_survival,std_all_run_survival,"
           "mean_best_per_case_survival,std_best_per_case_survival,"
           "mean_best_per_case_normalized_survival,"
           "mean_best_per_case_survival_probability,"
           "mean_single_run_time_ms,"
           "mean_total_search_time_per_case_ms\n";

    map<string, OverallBucket> overall;

    try {
        if (argc == 1) {
            const vector<int> scales = {
                50, 75, 100, 125
            };

            const vector<PDistribution> dists = {
                PDistribution::Uniform,
                PDistribution::Beta,
                PDistribution::Gaussian
            };

            for (PDistribution d : dists) {
                for (int n : scales) {
                    runBenchmark(
                        n, n, d, 20, DEFAULT_SEED,
                        allRuns, caseSummary, overall
                    );
                }
            }
        }
        else if (argc == 4 || argc == 5) {
            // N DIST CASES [SEED]
            int n = stoi(argv[1]);
            PDistribution d =
                parseDistribution(argv[2]);
            int cases = stoi(argv[3]);
            uint64_t seed =
                (argc == 5)
                    ? stoull(argv[4])
                    : DEFAULT_SEED;

            runBenchmark(
                n, n, d, cases, seed,
                allRuns, caseSummary, overall
            );
        }
        else if (argc == 6) {
            // NTYPES NTARGETS DIST CASES SEED
            int nTypes = stoi(argv[1]);
            int nTargets = stoi(argv[2]);
            PDistribution d =
                parseDistribution(argv[3]);
            int cases = stoi(argv[4]);
            uint64_t seed = stoull(argv[5]);

            runBenchmark(
                nTypes, nTargets, d, cases, seed,
                allRuns, caseSummary, overall
            );
        }
        else {
            cerr
                << "Usage:\n"
                << "  " << argv[0] << "\n"
                << "  " << argv[0]
                << " N DIST CASES [SEED]\n"
                << "  " << argv[0]
                << " NTYPES NTARGETS DIST CASES SEED\n";
            return 1;
        }
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    writeOverallSummary(overallSummary, overall);

    cout << "\nSaved EVERY individual run to:\n"
         << "  wtap_all_runs.csv\n";
    cout << "Saved per-case method/variant summaries to:\n"
         << "  wtap_case_summary.csv\n";
    cout << "Saved aggregate summaries to:\n"
         << "  wtap_overall_summary.csv\n";

    return 0;
}
