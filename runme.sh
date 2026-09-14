# g++ -O3 -std=c++17 wtap_classical.cpp -o wtap

# ./wtap 4096 4096 1 44
# ./wtap 8192 4096 1 44
# ./wtap 16384 4096 1 44

# ./wtap 8192 8192 1 44
# ./wtap 16384 8192 1 44
# ./wtap 32768 8192 1 44

# ./wtap 16384 16384 1 44
# ./wtap 32768 16384 1 44
# ./wtap 65536 16384 1 44

g++ -O3 -std=c++17 regret_k.cpp -o wtap_regret
./wtap_regret

