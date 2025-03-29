/**
 * @file polarfly_dual_optimizer.c
 * @brief Finds optimal PolarFly topology parameters ('q') based on switch radix
 * and target *host* count, optimizing for both minimum switches and maximum
 * network degree (minimum oversubscription).
 * @author Christian Martin
 * @author christian.j.martin@gmail.com
 *
 * @details
 * This program implements a search algorithm to identify suitable 'q' values
 * for constructing PolarFly topologies based on Erdős-Rényi polarity graphs (ER_q).
 * PolarFly is a diameter-2 network topology.
 *
 * The program takes the total available switch radix ('r') and the target minimum
 * number of connected hosts ('H_target') as input.
 *
 * It operates under the model where the ER_q graph defines the connections
 * between the switches/routers. Each switch has a total radix 'r'.
 * The PolarFly network itself consumes 'k = q + 1' ports per switch.
 * The remaining ports, 'p = r - k', are used to connect hosts directly to each switch.
 * The total number of switches is 'N = q^2 + q + 1'.
 * The total number of hosts supported is 'H = N * p'.
 *
 * The search considers candidate 'q' values that are prime powers and satisfy
 * the constraint `q <= r / 2`. This allows for a Polarfly topology to support
 * at least half as many hosts as internal network connections.
 *
 * It identifies two potentially different optimal configurations:
 * 1. Minimum Switches: Finds the *smallest* valid 'q' that meets H >= H_target.
 * This minimizes the number of switches (N) required.
 * 2. Maximum Network Degree: Finds the *largest* valid 'q' that meets H >= H_target.
 * This maximizes the network degree 'k' used for the PolarFly fabric,
 * minimizing the host ports 'p' per switch, thus reducing network
 * oversubscription, potentially at the cost of more switches (N).
 *
 * Based on concepts from "PolarFly: A Cost-Effective and Flexible Low-Diameter Topology"
 * by Lakhotia et al.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <limits.h> // For LLONG_MAX

// Define constants if needed
#define MAX_STR_LEN 256

// --- Data Structures ---

/**
 * @brief Represents a specific PolarFly network configuration including hosts.
 * Stores the parameters defining the topology and its capacity.
 */
typedef struct {
    int q;                      /**< The prime power defining the ER_q graph. */
    int network_degree;         /**< The degree required for PolarFly fabric (k = q + 1). */
    int hosts_per_switch;       /**< Ports remaining for hosts per switch (p = r - k). */
    long long num_switches;     /**< The total number of switches/nodes (N = q^2 + q + 1). */
    long long total_hosts;      /**< The total number of hosts supported (H = N * p). */
    bool valid;                 /**< Flag indicating if this struct holds a valid calculation result. */
} PolarflyHostConfig;

// --- Function Prototypes ---

// Utility Functions
bool is_prime(int n);
bool is_prime_power(int n);
long long integer_ceil_divide(long long numerator, long long denominator);
void format_with_commas_ll(long long num, char *str);

// Core Logic
void find_optimal_polarfly_hosts_dual(int r_radix, long long H_target,
                                     PolarflyHostConfig *min_switches_config,
                                     PolarflyHostConfig *max_net_degree_config);

// Output & Input
void print_polarfly_host_config(const PolarflyHostConfig *config, const char *title, int r_radix, long long H_target);
int get_int_input(const char *prompt, int min_value, int max_value);
long long get_long_long_input(const char *prompt, long long min_value, long long max_value);


// --- Utility Function Implementations ---

/**
 * @brief Checks if a positive integer 'n' is a prime number.
 * @details Uses trial division up to the square root of 'n'. Efficient enough
 * for the expected range of 'q' values. Handles base cases 1, 2, 3
 * and optimizes by checking factors 6k ± 1.
 * @param n The integer to check.
 * @return true if 'n' is prime, false otherwise.
 */
bool is_prime(int n) {
    // Primes must be greater than 1
    if (n <= 1) return false;
    // 2 and 3 are prime
    if (n <= 3) return true;
    // Eliminate multiples of 2 and 3 early
    if (n % 2 == 0 || n % 3 == 0) return false;

    // Check factors of the form 6k ± 1 up to sqrt(n)
    // All primes > 3 are of this form.
    for (int i = 5; i * i <= n; i = i + 6) {
        if (n % i == 0 || n % (i + 2) == 0)
            return false; // Found a factor
    }
    // No factors found, it's prime
    return true;
}

/**
 * @brief Checks if a positive integer 'n' is a prime power (p^m where p is prime, m >= 1).
 * @details First checks if 'n' itself is prime (p^1 case). If not, iterates through
 * prime bases 'p' up to sqrt(n) and checks if 'n' is a power (p^m, m>=2)
 * of that prime base.
 * @param n The integer to check.
 * @return true if 'n' is a prime power, false otherwise.
 */
bool is_prime_power(int n) {
    // Prime powers must be > 1
    if (n <= 1) return false;

    // Case 1: n is prime (p^1)
    if (is_prime(n)) return true;

    // Case 2: n = p^m where p is prime and m >= 2
    // Iterate through potential prime bases 'p' up to sqrt(n)
    int limit = sqrt(n);
    for (int p = 2; p <= limit; ++p) {
        if (is_prime(p)) {
            // Check if n is a power of this prime p
            long long current_power = (long long)p * p; // Start with m=2
            while (current_power <= n && current_power > 0) { // Loop while power <= n and no overflow
                if (current_power == n) {
                    return true; // Found n = p^m
                }
                // Check for potential overflow before calculating next power p^(m+1)
                if (n == 0 || p == 0) break; // Prevent division by zero below
                if (LLONG_MAX / p < current_power) {
                    break; // Next multiplication would overflow
                }
                current_power *= p; // Calculate next power
            }
        }
    }
    // If no prime base p was found such that n = p^m (m>=2)
    return false;
}


/**
 * @brief Calculates ceil(numerator / denominator) using integer arithmetic.
 * @param numerator The numerator (must be non-negative).
 * @param denominator The denominator (must be positive).
 * @return The ceiling of the division, or -1 on division by zero error.
 */
long long integer_ceil_divide(long long numerator, long long denominator) {
    if (denominator <= 0) { // Denominator must be positive
        fprintf(stderr, "Error: Non-positive denominator in integer_ceil_divide.\n");
        return -1; // Indicate error
    }
    if (numerator < 0) { // Numerator should be non-negative for this context
         fprintf(stderr, "Error: Negative numerator in integer_ceil_divide.\n");
         return -1; // Indicate error
    }
    if (numerator == 0) {
        return 0;
    }
    // Standard integer ceiling formula: (numerator + denominator - 1) / denominator
    // Check for potential overflow in (numerator + denominator - 1)
    if (LLONG_MAX - (denominator - 1) < numerator) {
         fprintf(stderr, "Error: Overflow detected in integer_ceil_divide for numerator=%lld, denominator=%lld\n", numerator, denominator);
         return -1; // Indicate overflow error
    }
    return (numerator + denominator - 1) / denominator;
}

/**
 * @brief Formats a long long integer with commas as thousands separators.
 * @param num The number to format.
 * @param str Output buffer to store the formatted string. Must be large enough (e.g., 50 chars).
 */
void format_with_commas_ll(long long num, char *str) {
    char temp[50];                  // Temporary buffer
    sprintf(temp, "%lld", num);     // Convert number to string
    int len = strlen(temp);
    int j = 0;                      // Output buffer index
    int lead = len % 3;             // Digits in first group (1, 2, or 3)
    if (lead == 0 && len > 0) lead = 3; // Adjust if length is multiple of 3

    // Copy digits, inserting commas
    for (int i = 0; i < len; i++) {
        // Insert comma before starting a new group (after the first group)
        if (i == lead && i != len) {
            str[j++] = ',';
            lead += 3; // Set position for next comma
        }
        str[j++] = temp[i]; // Copy digit
    }
    str[j] = '\0'; // Null-terminate the output string
}

// --- Core Logic Implementation (Dual Search) ---

/**
 * @brief Searches for optimal PolarFly 'q' values optimizing for both
 * minimum switches and maximum network degree (min oversubscription).
 * @details Iterates 'q' upwards through valid prime powers within the constraints.
 * Stores the first valid configuration found (minimizes switches N) and
 * the last valid configuration found (maximizes network degree k).
 * @param r_radix The total switch radix available.
 * @param H_target The minimum number of hosts required.
 * @param min_switches_config Pointer to struct to store the result with min switches (min q).
 * Its 'valid' field will be true if a solution is found.
 * @param max_net_degree_config Pointer to struct to store the result with max network degree (max q).
 * Its 'valid' field will be true if a solution is found.
 */
void find_optimal_polarfly_hosts_dual(int r_radix, long long H_target,
                                     PolarflyHostConfig *min_switches_config,
                                     PolarflyHostConfig *max_net_degree_config)
{
    // Initialize output structs as invalid
    min_switches_config->valid = false;
    max_net_degree_config->valid = false;

    // --- Input Validation ---
    // Need at least r=4 to potentially have q=2 and p=r-(q+1)=4-3=1 host port.
    if (r_radix < 4) {
         fprintf(stderr, "Error: Switch radix r_radix must be at least 4 to potentially support hosts in this model.\n");
         return;
    }
     if (H_target < 1) {
         fprintf(stderr, "Error: Target host count H_target must be at least 1.\n");
         return;
    }

    // --- Determine Search Range for q ---
    // constraint: q <= r / 2
    int q_max_r = r_radix / 2;
    // Smallest possible prime power is q=2. Check if constraint allows it.
    if (q_max_r < 2) {
         fprintf(stderr, "Error: Constraint q <= r_radix/2 (%d) is too restrictive. No prime power q >= 2 possible.\n", q_max_r);
         return;
    }

    // --- Log Search Parameters ---
    printf("Searching for optimal PolarFly configurations (Dual Objective)...\n");
    printf("  Switch Radix (r) = %d\n", r_radix);
    printf("  Target Hosts (H_target) >= %lld\n", H_target);
    printf("  Constraint: 2 <= q <= r/2 (%d), q is prime power\n", q_max_r);

    // --- Iterate q upwards ---
    // Start from the smallest possible prime power (q=2) up to the max allowed (q_max_r).
    for (int q = 2; q <= q_max_r; ++q) {

        // Constraint: q must be a prime power
        if (!is_prime_power(q)) {
            continue; // Skip non-prime-powers
        }

        // --- Calculate Configuration Parameters for current q ---
        int k_network_degree = q + 1;
        int p_hosts_per_switch = r_radix - k_network_degree;

        // Constraint: Must have non-negative ports for hosts.
        // Also skip if p=0 hosts/switch but H_target > 0.
        if (p_hosts_per_switch <= 0 && H_target > 0) {
             continue;
        }

        // Calculate number of switches N = q^2 + q + 1
        long long q_ll = q; // Use long long for intermediate calculations
        long long N_switches;
        // Check intermediate overflow during N calculation
        if (q_ll > 0 && LLONG_MAX / q_ll < q_ll) { N_switches = LLONG_MAX; } else { N_switches = q_ll * q_ll; }
        if (LLONG_MAX - q_ll < N_switches) { N_switches = LLONG_MAX; } else { N_switches += q_ll; }
        if (LLONG_MAX - 1 < N_switches) { N_switches = LLONG_MAX; } else { N_switches += 1; }

        if (N_switches == LLONG_MAX) {
             fprintf(stderr, "Warning: Overflow calculating N_switches for q=%d. Skipping q.\n", q);
             continue; // Cannot calculate total hosts if N overflows
        }

        // Calculate total hosts H = N * p
        long long H_total_hosts;
         if (p_hosts_per_switch == 0) {
            H_total_hosts = 0;
         }
         // Check overflow during H calculation
         else if (N_switches > 0 && LLONG_MAX / N_switches < p_hosts_per_switch) {
             H_total_hosts = LLONG_MAX; // Saturate if overflow
             fprintf(stderr, "Warning: Overflow calculating H_total_hosts for q=%d. Skipping q.\n", q);
             continue; // Cannot check H >= H_target if H overflows
        } else {
            H_total_hosts = N_switches * p_hosts_per_switch;
        }

        // --- Check Host Constraint ---
        // Constraint: H >= H_target
        if (H_total_hosts >= H_target) {
            // --- Valid Configuration Found for this q ---

            // Populate a temporary config struct for this valid q
            PolarflyHostConfig current_valid_config;
            current_valid_config.q = q;
            current_valid_config.network_degree = k_network_degree;
            current_valid_config.hosts_per_switch = p_hosts_per_switch;
            current_valid_config.num_switches = N_switches;
            current_valid_config.total_hosts = H_total_hosts;
            current_valid_config.valid = true;

            // Store the *first* valid one found for the "Minimum Switches" objective.
            // This works because we iterate q upwards.
            if (!min_switches_config->valid) {
                *min_switches_config = current_valid_config;
                 printf("  Found candidate for Min Switches (Min q): q=%d (N=%lld, k=%d, p=%d, H=%lld)\n",
                        q, N_switches, k_network_degree, p_hosts_per_switch, H_total_hosts);
            }

            // *Always* update the "Maximum Network Degree" result with the current valid one.
            // As we iterate q upwards, this struct will hold the valid config with the largest q at the end.
            *max_net_degree_config = current_valid_config;
            // Optional: Log update for max q candidate
            // printf("  Updating candidate for Max Network Degree (Max q): q=%d\n", q);
        }
        // Continue loop to check higher q values up to q_max_r
    } // End for loop over q

    // --- Final Output ---
    if (!min_switches_config->valid) { // If no valid config was ever found
         printf("No valid PolarFly configuration found for the given parameters and constraints.\n");
         // Add check to see if H_target was too high even for q_max_r (similar to previous version)
          if (q_max_r >= 2) {
             int q_test = q_max_r;
             while (q_test >= 2 && !is_prime_power(q_test)) { q_test--; } // Find largest valid q <= q_max_r
             if (q_test >= 2) {
                  int p_at_max = r_radix - (q_test + 1);
                  if (p_at_max >= 0) {
                       long long q_ll = q_test;
                       long long N_at_max = (LLONG_MAX - 1 - q_ll < q_ll * q_ll) ? LLONG_MAX : (q_ll*q_ll + q_ll + 1);
                       if (N_at_max != LLONG_MAX) {
                           long long H_at_max = (p_at_max == 0) ? 0 : ((LLONG_MAX / N_at_max < p_at_max) ? LLONG_MAX : N_at_max * p_at_max);
                           if (H_at_max != LLONG_MAX && H_at_max < H_target) {
                                 printf("  Note: Target H %lld is larger than max possible H (%lld) for largest valid q (%d) <= %d.\n", H_target, H_at_max, q_test, q_max_r);
                           }
                       }
                  }
             }
         }
    } else {
        printf("Optimal configuration search complete.\n");
        // Report the final q values found for each objective
        printf("  Min Switches optimal q = %d\n", min_switches_config->q);
        printf("  Max Network Degree optimal q = %d\n", max_net_degree_config->q);
    }
}

// --- Output Formatting ---

/**
 * @brief Prints the details of an optimal PolarFly host configuration.
 * @param config Pointer to the PolarflyHostConfig struct (must be valid).
 * @param title Title string indicating the optimization objective (e.g., "Minimum Switches").
 * @param r_radix The original total switch radix constraint input by the user.
 * @param H_target The original target host count input by the user.
 */
void print_polarfly_host_config(const PolarflyHostConfig *config, const char *title, int r_radix, long long H_target) {
    // Check if the passed config pointer is valid and the config itself is marked valid
    if (!config || !config->valid) {
        printf("\n--- No valid configuration found for '%s' objective ---\n", title);
        return;
    }

    // Format large numbers for readability
    char target_h_str[MAX_STR_LEN], actual_h_str[MAX_STR_LEN], actual_n_str[MAX_STR_LEN];
    format_with_commas_ll(H_target, target_h_str);
    format_with_commas_ll(config->total_hosts, actual_h_str);
    format_with_commas_ll(config->num_switches, actual_n_str);

    // Print configuration details with the provided title
    printf("\n=========== Optimal PolarFly (%s) ===========\n", title);
    printf("Target Hosts:            >= %s\n", target_h_str);
    printf("Switch Radix (r):        %d\n", r_radix);
    printf("Constraint:              q <= r/2 (%d), q is prime power\n", r_radix / 2); // Show constraint
    printf("---------------------------------------------------------\n");
    printf("Optimal Prime Power (q): %d\n", config->q);
    printf("Network Degree (k=q+1):  %d\n", config->network_degree);
    printf("Hosts per Switch (p=r-k):%d\n", config->hosts_per_switch);
    printf("Total Switches (N=q²+q+1):%s\n", actual_n_str);
    printf("Total Hosts (H=N*p):     %s\n", actual_h_str);
    printf("=========================================================\n");
}

// --- User Input Handling ---

/**
 * @brief Prompts the user for integer input and validates it within a range.
 * @param prompt The message to display to the user.
 * @param min_value The minimum acceptable value.
 * @param max_value The maximum acceptable value (or <=0 for no upper limit).
 * @return The validated integer input from the user. Handles EOF.
 */
int get_int_input(const char *prompt, int min_value, int max_value) {
    int value;
    char buffer[MAX_STR_LEN]; // Use defined constant
    while (1) {
        printf("%s", prompt);
        // Read a line of input
        if (fgets(buffer, sizeof(buffer), stdin) == NULL) {
             // Handle EOF (Ctrl+D) or read error
             if (feof(stdin)) {
                 printf("\nEOF detected, exiting.\n");
                 exit(1); // Exit program cleanly
             }
            perror("fgets error"); // Print system error
            // Decide how to handle other errors, maybe exit or retry
            fprintf(stderr, "Input error, please try again or Ctrl+C to exit.\n");
            continue;
        }
        // Remove trailing newline character if present
        buffer[strcspn(buffer, "\n")] = 0;

        // Attempt to parse integer using strtol for better error checking
        char *endptr;
        long val_l = strtol(buffer, &endptr, 10); // Base 10 conversion

        // Check for various parsing errors
        if (endptr == buffer || *endptr != '\0' || val_l < INT_MIN || val_l > INT_MAX) {
            printf("  Invalid input. Please enter an integer.\n");
            continue;
        }
        value = (int)val_l; // Convert valid long to int

        // Check if value is within the specified logical range
        if (value < min_value || (max_value > 0 && value > max_value)) {
            if (max_value > 0) printf("  Value must be between %d and %d.\n", min_value, max_value);
            else printf("  Value must be at least %d.\n", min_value);
            continue;
        }
        return value; // Input is valid
    }
}

/**
 * @brief Prompts the user for long long integer input and validates it within a range.
 * @param prompt The message to display to the user.
 * @param min_value The minimum acceptable value.
 * @param max_value The maximum acceptable value (or <=0 for no upper limit).
 * @return The validated long long input from the user. Handles EOF.
 */
long long get_long_long_input(const char *prompt, long long min_value, long long max_value) {
    long long value;
    char buffer[MAX_STR_LEN];
    while (1) {
        printf("%s", prompt);
        if (fgets(buffer, sizeof(buffer), stdin) == NULL) {
             if (feof(stdin)) { printf("\nEOF detected, exiting.\n"); exit(1); }
             perror("fgets error");
             fprintf(stderr, "Input error, please try again or Ctrl+C to exit.\n");
             continue;
        }
        buffer[strcspn(buffer, "\n")] = 0;
        char *endptr;
        // Use strtoll for long long
        value = strtoll(buffer, &endptr, 10);

        // Check for parsing errors
        if (endptr == buffer || *endptr != '\0') {
            // Note: Range errors (LLONG_MIN, LLONG_MAX) are implicitly handled by strtoll setting errno,
            // which could be checked explicitly if needed.
            printf("  Invalid input. Please enter a valid integer.\n");
            continue;
        }
        // Check logical range
        if (value < min_value || (max_value > 0 && value > max_value)) {
            if (max_value > 0) printf("  Value must be between %lld and %lld.\n", min_value, max_value);
            else printf("  Value must be at least %lld.\n", min_value);
            continue;
        }
        return value; // Input is valid
    }
}

// --- Main Program Entry Point ---
/**
 * @brief Main function: Gets user input, calls the dual-objective search,
 * and prints both optimal results found (if they exist).
 * @param argc Argument count (unused).
 * @param argv Argument vector (unused).
 * @return 0 on success, 1 on error.
 */
int main(int argc, char *argv[]) {
    // Suppress unused parameter warnings if not using argc/argv
    (void)argc;
    (void)argv;

    // Print program header
    printf("\n===== PolarFly Network Optimizer (Dual Objective) =====\n");
    printf("Finds optimal 'q' for PolarFly, optimizing for both:\n");
    printf("  1) Minimum Switches (smallest valid q, q <= r/2)\n");
    printf("  2) Maximum Network Degree / Minimum Oversubscription (largest valid q, q <= r/2)\n");
    printf("Constraints: Host target met, q is prime power.\n");

    // Get user inputs with validation
    // Minimum radix r=4 allows q=2, k=3, p=1 (smallest possible useful config)
    int r_radix = get_int_input("Enter Total Switch Radix (r >= 4): ", 4, 8192); // Example upper limit
    long long H_target = get_long_long_input("Enter Target Minimum Host Count (H_target >= 1): ", 1, -1); // -1 indicates no upper limit check

    // Declare structs to hold the two results
    PolarflyHostConfig min_switch_config;
    PolarflyHostConfig max_net_degree_config;

    // Find the optimal configurations by calling the dual search function
    find_optimal_polarfly_hosts_dual(r_radix, H_target, &min_switch_config, &max_net_degree_config);

    // Print the results, clearly labeling each objective
    print_polarfly_host_config(&min_switch_config, "Minimum Switches (Min q)", r_radix, H_target);

    // Only print the second result if it's valid and *different* from the first one
    if (max_net_degree_config.valid && (!min_switch_config.valid || max_net_degree_config.q != min_switch_config.q)) {
         print_polarfly_host_config(&max_net_degree_config, "Max Network Degree (Max q)", r_radix, H_target);
    } else if (min_switch_config.valid && max_net_degree_config.valid && max_net_degree_config.q == min_switch_config.q) {
        // If both are valid and have the same q, mention that they converged
        printf("\n(Note: The 'Minimum Switches' and 'Maximum Network Degree' objectives yield the same optimal q = %d)\n\n", min_switch_config.q);
    }

    return 0; // Indicate successful execution
}
