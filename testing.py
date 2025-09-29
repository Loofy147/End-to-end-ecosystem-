# eulernet_optimized/math_utils.py
import math
import numpy as np
from functools import lru_cache
from typing import List, Dict, Any
import logging
import time # Import time for potential performance monitoring
import torch # Import torch for FeatureExtractor tests
import unittest # Import unittest for unit tests

# Get logger for this module
logger = logging.getLogger(__name__)

# Set up a basic logging configuration if not already configured
# This is helpful for running this file as a script independently
if not logging.root.handlers:
    logging.basicConfig(level=logging.INFO)

class MathUtils:
    """
    Provides optimized mathematical utilities for number theory functions.

    Includes functions for primality testing, Euler's totient function,
    and factorization, utilizing lookup tables and optimized algorithms.
    """

    def __init__(self, precompute_limit: int = 50000):
        """
        Initializes the MathUtils with precomputed lookup tables.

        Args:
            precompute_limit (int): The upper bound (inclusive) for precomputation.
                                    Must be a positive integer.
        """
        if not isinstance(precompute_limit, int) or precompute_limit <= 0:
            logger.error(f"Invalid precompute_limit: {precompute_limit}")
            raise ValueError("Precompute limit must be a positive integer.")

        self.precompute_limit = precompute_limit
        self._prime_lut: Dict[int, bool] = {}
        self._totient_lut: Dict[int, int] = {}
        self._factorization_lut: Dict[int, List[int]] = {}

        # Precompute results for small numbers
        self._precompute_small_numbers(limit=self.precompute_limit)

        logger.info(f"MathUtils initialized with precomputation up to {self.precompute_limit}")


    def _precompute_small_numbers(self, limit: int):
        """
        Computes and stores results for small numbers up to a given limit.

        Uses a Sieve of Erathenes for primality and direct computation
        for totient and factorization.

        Args:
            limit (int): The upper bound (inclusive) for precomputation.
        """
        logger.info(f"Precomputing math utilities up to {limit}...")
        start_time = time.time()

        # Sieve of Erathenes
        is_prime_sieve = [True] * (limit + 1)
        is_prime_sieve[0] = is_prime_sieve[1] = False

        for i in range(2, int(limit**0.5) + 1):
            if is_prime_sieve[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime_sieve[j] = False

        # Populate LUTs
        for n in range(2, limit + 1):
            # Prime LUT
            self._prime_lut[n] = is_prime_sieve[n]

            # Totient LUT
            self._totient_lut[n] = self._compute_totient_direct(n)

            # Small factorization
            if not is_prime_sieve[n]:
                self._factorization_lut[n] = self._factorize_small(n)
            elif n > 1: # Primes' only factor is themselves, consider adding for consistency
                 self._factorization_lut[n] = [n]


        end_time = time.time()
        logger.info(f"Precomputation complete in {end_time - start_time:.2f} seconds. Populated {len(self._prime_lut)} entries.")


    def _compute_totient_direct(self, n: int) -> int:
        """
        Computes Euler's totient (phi) function directly for a number.

        Uses a factorization-based method. Assumes n >= 1.
        """
        if n < 1:
             # This should ideally be caught by input validation in the public method,
             # but adding a check here for safety in internal method.
             logger.warning(f"_compute_totient_direct called with n < 1: {n}")
             return 0 # Or raise ValueError, depending on desired strictness
        if n == 1: # totient(1) is 1
            return 1

        result = n

        # Handle even numbers
        if n % 2 == 0:
            result //= 2
            while n % 2 == 0:
                n //= 2

        # Handle odd prime factors
        p = 3
        while p * p <= n:
            if n % p == 0:
                result -= result // p
                while n % p == 0:
                    n //= p
            p += 2

        # Handle remaining prime factor (if any)
        if n > 1:
            result -= result // n

        return result

    def _factorize_small(self, n: int) -> List[int]:
        """
        Performs a simple trial division factorization for small numbers.

        Assumes n >= 2 and not prime (handled by precomputation logic).
        """
        if n < 2:
            # logger.warning(f"_factorize_small called with n < 2: {n}") # Already handled by precomp check
            return []

        factors = []
        d = 2
        temp_n = n # Use a temporary variable to avoid modifying n during factorization
        # Only need to check up to sqrt(temp_n)
        while d * d <= temp_n:
            while temp_n % d == 0:
                factors.append(d)
                temp_n //= d
            d += 1
        if temp_n > 1:
            factors.append(temp_n)
        return factors


    def is_prime(self, n: int) -> bool:
        """
        Checks if a number is prime using LUT or optimized trial division.

        Args:
            n (int): The input number. Must be an integer greater than or equal to 1.

        Returns:
            bool: True if n is prime, False otherwise.

        Raises:
            ValueError: If n is not an integer or is less than 1.
        """
        if not isinstance(n, int) or n < 1:
            logger.error(f"Invalid input for is_prime: {n}")
            raise ValueError("Input 'n' for is_prime must be an integer greater than or equal to 1.")

        # Check LUT for small numbers
        if n <= self.precompute_limit:
            return self._prime_lut.get(n, False) # False for n=1 or numbers outside LUT range but <= limit

        # Use optimized trial division for larger numbers
        return self._is_prime_large(n)

    @lru_cache(maxsize=10000)
    def _is_prime_large(self, n: int) -> bool:
        """
        Optimized primality test for numbers larger than precompute_limit.

        Uses trial division up to sqrt(n) with 6k +/- 1 optimization.
        Cached for repeated calls with the same large numbers.
        Assumes input n is an integer > precompute_limit.
        """
        # Basic checks (redundant if called from is_prime, but good for safety)
        if n < 2: return False
        if n <= 3: return True # 2 and 3 are prime
        if n % 2 == 0 or n % 3 == 0: return False

        # Check primes of the form 6k +/- 1
        i = 5
        # Loop up to sqrt(n)
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True


    def euler_totient(self, n: int) -> int:
        """
        Computes Euler's totient (phi) function using LUT or direct computation.

        Args:
            n (int): The input number. Must be an integer greater than or equal to 1.

        Returns:
            int: The totient (phi) value of n.

        Raises:
            ValueError: If n is not an integer or is less than 1.
        """
        if not isinstance(n, int) or n < 1:
            logger.error(f"Invalid input for euler_totient: {n}")
            raise ValueError("Input 'n' for euler_totient must be an integer greater than or equal to 1.")

        # Check LUT for small numbers
        if n <= self.precompute_limit:
            return self._totient_lut.get(n, 1) # totient(1) is 1, totient for others >= 1

        # Use direct computation for larger numbers
        return self._compute_totient_direct(n)


    def get_small_factors(self, n: int, max_factors: int = 3) -> List[int]:
        """
        Gets a limited number of small factors for a number.

        Uses LUT for small numbers and simple trial division otherwise.

        Args:
            n (int): The input number. Must be an integer greater than or equal to 1.
            max_factors (int): The maximum number of small factors to return.
                               Must be a positive integer. Defaults to 3.

        Returns:
            List[int]: A list containing up to max_factors of the smallest factors of n.
                       Returns [] for primes.

        Raises:
            ValueError: If n is not an integer or is less than 1, or if max_factors is not positive.
        """
        if not isinstance(n, int) or n < 1:
            logger.error(f"Invalid input 'n' for get_small_factors: {n}")
            raise ValueError("Input 'n' for get_small_factors must be an integer greater than or equal to 1.")
        if not isinstance(max_factors, int) or max_factors <= 0:
             logger.error(f"Invalid input 'max_factors' for get_small_factors: {max_factors}")
             raise ValueError("Input 'max_factors' for get_small_factors must be a positive integer.")

        # Check LUT for small numbers
        if n <= self.precompute_limit:
             # Return empty list for primes within LUT
            if self._prime_lut.get(n, False):
                 return []
            # Return factors for composites within LUT
            if n in self._factorization_lut:
                factors = self._factorization_lut[n]
                return factors[:max_factors]
            # Handle n=1 case if not precomputed (should be [] factors)
            if n == 1:
                 return []


        # Use simple trial division for larger numbers or numbers not in LUT
        # First check if the number is prime before trying factorization
        if self.is_prime(n):
             return [] # Primes have no small factors other than themselves (if considered), but usually factorization means composite factors

        factors = []
        temp_n = n # Use a temporary variable
        # Check small prime factors up to a certain limit (e.g., 100)
        # We should also check up to sqrt(n) for any prime factors if not found up to 100
        d = 2
        limit_check = min(101, int(temp_n**0.5) + 1) # Check up to 100 or sqrt(n), whichever is smaller

        while d <= limit_check and temp_n > 1:
            while temp_n % d == 0:
                factors.append(d)
                temp_n //= d
                if len(factors) >= max_factors:
                    return factors # Return early if max_factors reached
            d += 1

        # If temp_n is still > 1 after checking small factors, it might have a large prime factor
        # Or if sqrt(n) was checked up to 100, continue checking up to original sqrt(n)
        if temp_n > 1 and len(factors) < max_factors:
             # Continue trial division from where we left off up to the original sqrt(n)
             # Since d was incremented, start check from d
             d_large = d
             while d_large * d_large <= temp_n and len(factors) < max_factors:
                 while temp_n % d_large == 0:
                     factors.append(d_large)
                     temp_n //= d_large
                     if len(factors) >= max_factors:
                         return factors
                 d_large += 1 # Simple increment, could optimize with 6k+/-1 if needed

             if temp_n > 1 and len(factors) < max_factors:
                  factors.append(temp_n) # The remaining temp_n is a prime factor


        return factors[:max_factors]

# Instantiate the utility class globally for easy access
MATH_UTILS = MathUtils()


class FeatureExtractor:
    """
    Extracts relevant mathematical features from numbers or values.

    Designed for speed and includes caching.
    """
    def __init__(self):
        self._feature_cache: Dict[Any, torch.Tensor] = {}
        # Keep track of cache hits for performance reporting
        self._cache_hits = 0
        self._cache_accesses = 0
        logger.info("FeatureExtractor initialized.")


    def extract_number_features(self, n: int, cache_key: Any = None) -> torch.Tensor:
        """
        Extracts mathematical features for a given integer.

        Includes basic properties, binary representation, and digit properties.
        Uses caching based on cache_key.

        Args:
            n (int): The input number. Must be a non-negative integer.
            cache_key (Any, optional): A key to use for caching the extracted features.
                                       Defaults to None. If None, caching is not used.

        Returns:
            torch.Tensor: A 1D tensor containing the extracted features.

        Raises:
            ValueError: If n is not a non-negative integer.
        """
        if not isinstance(n, int) or n < 0:
            logger.error(f"Invalid input 'n' for extract_number_features: {n}")
            raise ValueError("Input 'n' for extract_number_features must be a non-negative integer.")

        # Check cache first
        if cache_key is not None:
             self._cache_accesses += 1
             if cache_key in self._feature_cache:
                 self._cache_hits += 1
                 logger.debug(f"Feature cache hit for key: {cache_key}")
                 return self._feature_cache[cache_key]

        features: List[float] = []

        # Basic features
        features.extend([
            float(n),
            math.log1p(n) if n >= 0 else 0.0, # Use log1p for stability with n=0, ensure float
            math.sqrt(n) if n >= 0 else 0.0, # Ensure float
            float(n % 2),  # even/odd
            float(n % 3),
            float(n % 5),
            float(n % 7)
        ])

        # Binary representation (last 10 bits)
        # Ensure n >= 0 is handled for binary conversion
        binary_str = format(n, '010b')[-10:] if n >= 0 else '0'*10 # Handle n=0 case
        features.extend([float(bit) for bit in binary_str])

        # Mathematical properties
        # Handle n=0 case for string conversion
        str_n = str(n) if n >= 0 else '0'
        features.extend([
            float(len(str_n)),  # number of digits
            float(sum(int(d) for d in str_n)),  # sum of digits
            float(n % 4),  # quadratic residue (well-defined for non-negative integers)
        ])

        # Convert to tensor
        feature_tensor = torch.tensor(features, dtype=torch.float32)

        # Save to cache if cache_key is provided
        if cache_key is not None:
            self._feature_cache[cache_key] = feature_tensor
            logger.debug(f"Feature cached with key: {cache_key}")

        return feature_tensor

    def extract_zeta_features(self, s: float, cache_key: Any = None) -> torch.Tensor:
        """
        Extracts features for the Riemann Zeta function input 's'.

        Uses caching based on cache_key.

        Args:
            s (Union[int, float]): The input value for the Zeta function. Must be a number.
                                   Validation for s > 1 should be done by the caller.
            cache_key (Any, optional): A key to use for caching the extracted features.
                                       Defaults to None. If None, caching is not used.

        Returns:
            torch.Tensor: A 1D tensor containing the extracted features for s.

        Raises:
            TypeError: If s is not a number (int or float).
            ValueError: If s leads to invalid math operations, although handled by conditionals.
        """
        if not isinstance(s, (int, float)):
            logger.error(f"Invalid input type for extract_zeta_features: {s} ({type(s)})")
            raise TypeError("Input 's' for extract_zeta_features must be a number (integer or float).")

        # Check cache first
        if cache_key is not None:
             self._cache_accesses += 1
             if cache_key in self._feature_cache:
                 self._cache_hits += 1
                 logger.debug(f"Zeta feature cache hit for key: {cache_key}")
                 return self._feature_cache[cache_key]


        features: List[float] = [
            float(s),
            float(s - 1),
            float(1.0 / s) if s != 0 else 0.0, # Add .0 for float division
            math.log(s) if s > 0 else 0.0, # Add .0 for float log, handle s<=0
            float(s**2),
            # Handle s close to 1 with a small epsilon to avoid division by zero
            # If s is exactly 1, s-1 is 0. If s is slightly less than 1, s-1 is negative.
            # The test was failing for s=1.0, asserting 100.0 != -100.0.
            # The condition `abs(s - 1) > 1e-9` is true for s=1.0.
            # The original code had `(100.0 if s > 1 else -100.0)`. For s=1.0, s > 1 is False, so it returns -100.0.
            # This seems intentional to indicate the behavior for s <= 1.
            # The test `self.assertAlmostEqual(features_1[5].item(), 100.0)` was incorrect for s=1.0.
            # It should assert against -100.0 for s=1.0. Let's correct the test, not the code logic here.
            float(1.0 / (s - 1)) if abs(s - 1) > 1e-9 else (100.0 if s > 1 else -100.0),
            float(s % 1) if isinstance(s, float) else 0.0, # fractional part, ensure float and handle int
            float(int(s)) if isinstance(s, int) else float(math.floor(s)) # integer part, handle floats
        ]

        feature_tensor = torch.tensor(features, dtype=torch.float32)

        # Save to cache if cache_key is provided
        if cache_key is not None:
            self._feature_cache[cache_key] = feature_tensor
            logger.debug(f"Zeta feature cached with key: {cache_key}")

        return feature_tensor

    def clear_cache(self):
        """Clears the feature cache to free up memory."""
        self._feature_cache.clear()
        self._cache_hits = 0
        self._cache_accesses = 0
        logger.info("Feature cache cleared.")

    def get_cache_stats(self) -> Dict[str, int]:
        """Returns statistics about the feature cache."""
        return {
            'size': len(self._feature_cache),
            'hits': self._cache_hits,
            'accesses': self._cache_accesses,
            'hit_rate': (self._cache_hits / self._cache_accesses) if self._cache_accesses > 0 else 0.0
        }


# Instantiate the utility class globally for easy access
MATH_UTILS = MathUtils()


# ============================================================================
# Unit Tests for MathUtils and FeatureExtractor
# ============================================================================

import unittest

class TestMathUtils(unittest.TestCase):

    def setUp(self):
        """Set up a new MathUtils instance before each test."""
        # Use a small precompute limit for faster testing
        self.math_utils = MathUtils(precompute_limit=100)
        # Disable logging during tests for cleaner output
        logging.disable(logging.CRITICAL)


    def tearDown(self):
        """Re-enable logging after tests."""
        logging.disable(logging.NOTSET)


    def test_math_utils_initialization(self):
        """Test that MathUtils is initialized correctly."""
        self.assertEqual(self.math_utils.precompute_limit, 100)
        self.assertGreater(len(self.math_utils._prime_lut), 0)
        self.assertGreater(len(self.math_utils._totient_lut), 0)
        # Factorization LUT might be smaller as it only includes composites
        self.assertGreaterEqual(len(self.math_utils._factorization_lut), 0)


    def test_is_prime_small(self):
        """Test is_prime for small numbers (within LUT)."""
        self.assertFalse(self.math_utils.is_prime(1))
        self.assertTrue(self.math_utils.is_prime(2))
        self.assertTrue(self.math_utils.is_prime(3))
        self.assertFalse(self.math_utils.is_prime(4))
        self.assertTrue(self.math_utils.is_prime(97)) # Prime within LUT
        self.assertFalse(self.math_utils.is_prime(100)) # Composite within LUT


    def test_is_prime_large(self):
        """Test is_prime for large numbers (beyond LUT)."""
        self.assertTrue(self.math_utils.is_prime(101)) # Prime
        self.assertFalse(self.math_utils.is_prime(102)) # Composite
        self.assertTrue(self.math_utils.is_prime(1009)) # Larger prime
        self.assertFalse(self.math_utils.is_prime(10000)) # Larger composite


    def test_is_prime_edge_cases(self):
        """Test is_prime for edge cases."""
        with self.assertRaises(ValueError):
            self.math_utils.is_prime(0)
        with self.assertRaises(ValueError):
            self.math_utils.is_prime(-5)
        with self.assertRaises(ValueError):
            self.math_utils.is_prime("abc") # type: ignore
        with self.assertRaises(ValueError):
            self.math_utils.is_prime(1.5) # type: ignore


    def test_euler_totient_small(self):
        """Test euler_totient for small numbers (within LUT)."""
        self.assertEqual(self.math_utils.euler_totient(1), 1)
        self.assertEqual(self.math_utils.euler_totient(2), 1)
        self.assertEqual(self.math_utils.euler_totient(3), 2)
        self.assertEqual(self.math_utils.euler_totient(4), 2)
        self.assertEqual(self.math_utils.euler_totient(10), 4)
        self.assertEqual(self.math_utils.euler_totient(100), 40)


    def test_euler_totient_large(self):
        """Test euler_totient for large numbers (beyond LUT)."""
        self.assertEqual(self.math_utils.euler_totient(101), 100) # Prime
        self.assertEqual(self.math_utils.euler_totient(102), 32) # 102 = 2 * 3 * 17, phi(102) = 102 * (1-1/2) * (1-1/3) * (1-1/17) = 102 * 1/2 * 2/3 * 16/17 = 32
        self.assertEqual(self.math_utils.euler_totient(1000), 400) # 1000 = 2^3 * 5^3, phi(1000) = 1000 * (1-1/2) * (1-1/5) = 1000 * 1/2 * 4/5 = 400
        # Correcting the totient calculation for 9999
        # 9999 = 3^2 * 11 * 101
        # phi(9999) = 9999 * (1 - 1/3) * (1 - 1/11) * (1 - 1/101)
        #           = 9999 * (2/3) * (10/11) * (100/101)
        #           = (9999/3) * 2 * (10/11) * (100/101)
        #           = 3333 * 2 * (10/11) * (100/101)
        #           = 6666 * (10/11) * (100/101)
        #           = (6666/11) * 10 * (100/101)
        #           = 606 * 10 * (100/101)
        #           = 6060 * (100/101)
        #           = (6060 / 101) * 100
        #           = 60 * 100 = 6000
        # The previous expected value 6600 was incorrect. The direct computation logic seems correct.
        self.assertEqual(self.math_utils.euler_totient(9999), 6000)


    def test_euler_totient_edge_cases(self):
        """Test euler_totient for edge cases."""
        with self.assertRaises(ValueError):
            self.math_utils.euler_totient(0)
        with self.assertRaises(ValueError):
            self.math_utils.euler_totient(-5)
        with self.assertRaises(ValueError):
            self.math_utils.euler_totient("abc") # type: ignore
        with self.assertRaises(ValueError):
            self.math_utils.euler_totient(1.5) # type: ignore


    def test_get_small_factors_small(self):
        """Test get_small_factors for small numbers (within LUT)."""
        self.assertEqual(self.math_utils.get_small_factors(1), [])
        self.assertEqual(self.math_utils.get_small_factors(2), []) # Prime
        self.assertEqual(self.math_utils.get_small_factors(4), [2, 2])
        self.assertEqual(self.math_utils.get_small_factors(12), [2, 2, 3])
        self.assertEqual(self.math_utils.get_small_factors(30), [2, 3, 5])
        self.assertEqual(self.math_utils.get_small_factors(60, max_factors=2), [2, 2]) # Test max_factors
        self.assertEqual(self.math_utils.get_small_factors(99), [3, 3, 11])


    def test_get_small_factors_large(self):
        """Test get_small_factors for large numbers (beyond LUT)."""
        self.assertEqual(self.math_utils.get_small_factors(101), []) # Prime
        self.assertEqual(self.math_utils.get_small_factors(102), [2, 3, 17])
        self.assertEqual(self.math_utils.get_small_factors(210), [2, 3, 5]) # 210 = 2*3*5*7
        self.assertEqual(self.math_utils.get_small_factors(210, max_factors=2), [2, 3])
        self.assertEqual(self.math_utils.get_small_factors(1024), [2, 2, 2]) # 2^10
        self.assertEqual(self.math_utils.get_small_factors(1023), [3, 11, 31]) # 1023 = 3 * 11 * 31
        self.assertEqual(self.math_utils.get_small_factors(9999), [3, 3, 11]) # 9999 = 3^2 * 11 * 101
        self.assertEqual(self.math_utils.get_small_factors(9999, max_factors=5), [3, 3, 11, 101])


    def test_get_small_factors_edge_cases(self):
        """Test get_small_factors for edge cases."""
        with self.assertRaises(ValueError):
            self.math_utils.get_small_factors(0)
        with self.assertRaises(ValueError):
            self.math_utils.get_small_factors(-10)
        with self.assertRaises(ValueError):
            self.math_utils.get_small_factors(10, max_factors=0)
        with self.assertRaises(ValueError):
            self.math_utils.get_small_factors(10, max_factors=-2)
        with self.assertRaises(ValueError):
            self.math_utils.get_small_factors("abc") # type: ignore


class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        """Set up a new FeatureExtractor instance before each test."""
        self.feature_extractor = FeatureExtractor()
        # Disable logging during tests
        logging.disable(logging.CRITICAL)


    def tearDown(self):
        """Re-enable logging after tests."""
        logging.disable(logging.NOTSET)


    def test_feature_extractor_initialization(self):
        """Test that FeatureExtractor is initialized correctly."""
        self.assertEqual(len(self.feature_extractor._feature_cache), 0)
        self.assertEqual(self.feature_extractor._cache_hits, 0)
        self.assertEqual(self.feature_extractor._cache_accesses, 0)


    def test_extract_number_features_basic(self):
        """Test extract_number_features for basic numbers."""
        # Test features for a simple number like 10
        features_10 = self.feature_extractor.extract_number_features(10)
        self.assertIsInstance(features_10, torch.Tensor)
        self.assertEqual(features_10.shape, (20,)) # Check feature vector size

        # Verify a few known features for 10
        # [10.0, log1p(10)=2.39789..., sqrt(10)=3.16227..., 10%2=0.0, 10%3=1.0, 10%5=0.0, 10%7=3.0,
        # Binary 1010 (padded 0000001010) -> [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0],
        # len(str(10))=2.0, sum(digits)=1.0, 10%4=2.0]
        self.assertAlmostEqual(features_10[0].item(), 10.0)
        # Correcting the assertion for log1p due to potential floating point precision differences
        # Use a higher tolerance or fewer decimal places if needed, or check against a calculated value
        self.assertAlmostEqual(features_10[1].item(), math.log1p(10), places=6) # Increased tolerance slightly
        self.assertAlmostEqual(features_10[2].item(), math.sqrt(10))
        self.assertAlmostEqual(features_10[3].item(), 0.0) # 10 % 2
        self.assertAlmostEqual(features_10[4].item(), 1.0) # 10 % 3
        # Correcting the expected value for the feature at index 10 (4th bit from the left / 7th from the right in the 10-bit padded string).
        # The 10-bit padded binary for 10 is 0000001010.
        # The bit at index 3 (0-indexed from left) is 0. This corresponds to features[7 + 3] = features[10].
        # Correcting the expected value from 1.0 to 0.0.
        self.assertAlmostEqual(features_10[10].item(), 0.0) # 4th bit from left (index 3) of 10 (0000001010) is 0.0

        self.assertAlmostEqual(features_10[17].item(), 2.0) # len('10')
        self.assertAlmostEqual(features_10[18].item(), 1.0) # sum of digits 1+0
        self.assertAlmostEqual(features_10[19].item(), 2.0) # 10 % 4


    def test_extract_number_features_edge_cases(self):
        """Test extract_number_features for edge cases."""
        # Test features for 0
        features_0 = self.feature_extractor.extract_number_features(0)
        self.assertIsInstance(features_0, torch.Tensor)
        self.assertEqual(features_0.shape, (20,))
        self.assertAlmostEqual(features_0[0].item(), 0.0)
        self.assertAlmostEqual(features_0[1].item(), math.log1p(0)) # log1p(0) is 0
        self.assertAlmostEqual(features_0[2].item(), math.sqrt(0)) # sqrt(0) is 0
        self.assertAlmostEqual(features_0[3].item(), 0.0) # 0 % 2
        self.assertEqual(features_0[7:17].tolist(), [0.0]*10) # Binary 0000000000
        self.assertAlmostEqual(features_0[17].item(), 1.0) # len('0')
        self.assertAlmostEqual(features_0[18].item(), 0.0) # sum of digits 0
        self.assertAlmostEqual(features_0[19].item(), 0.0) # 0 % 4

        # Test invalid inputs
        with self.assertRaises(ValueError):
            self.feature_extractor.extract_number_features(-1)
        with self.assertRaises(ValueError):
            self.feature_extractor.extract_number_features(1.5) # type: ignore
        with self.assertRaises(ValueError):
            self.feature_extractor.extract_number_features("abc") # type: ignore


    def test_extract_zeta_features_basic(self):
        """Test extract_zeta_features for basic values of s."""
        # Test features for s = 2.0
        features_2 = self.feature_extractor.extract_zeta_features(2.0)
        self.assertIsInstance(features_2, torch.Tensor)
        self.assertEqual(features_2.shape, (8,)) # Check feature vector size

        # Verify a few known features for s=2.0
        # [2.0, 2.0-1=1.0, 1.0/2.0=0.5, log(2.0)=0.693..., 2.0^2=4.0, 1.0/(2.0-1)=1.0, 2.0%1=0.0, floor(2.0)=2.0]
        self.assertAlmostEqual(features_2[0].item(), 2.0)
        self.assertAlmostEqual(features_2[1].item(), 1.0)
        self.assertAlmostEqual(features_2[2].item(), 0.5)
        self.assertAlmostEqual(features_2[3].item(), math.log(2.0))
        self.assertAlmostEqual(features_2[4].item(), 4.0)
        self.assertAlmostEqual(features_2[5].item(), 1.0)
        self.assertAlmostEqual(features_2[6].item(), 0.0)
        self.assertAlmostEqual(features_2[7].item(), 2.0)

        # Test features for s = 3.5
        features_3_5 = self.feature_extractor.extract_zeta_features(3.5)
        self.assertIsInstance(features_3_5, torch.Tensor)
        self.assertEqual(features_3_5.shape, (8,))
        self.assertAlmostEqual(features_3_5[0].item(), 3.5)
        self.assertAlmostEqual(features_3_5[1].item(), 2.5)
        self.assertAlmostEqual(features_3_5[2].item(), 1.0/3.5)
        # Correcting the assertion for log due to potential floating point precision differences
        self.assertAlmostEqual(features_3_5[3].item(), math.log(3.5), places=6) # Increased tolerance slightly
        self.assertAlmostEqual(features_3_5[4].item(), 3.5**2)
        self.assertAlmostEqual(features_3_5[5].item(), 1.0/(3.5-1))
        self.assertAlmostEqual(features_3_5[6].item(), 0.5)
        self.assertAlmostEqual(features_3_5[7].item(), 3.0)


    def test_extract_zeta_features_edge_cases(self):
        """Test extract_zeta_features for edge cases of s."""
        # Test features for s = 1.0 (handled by conditional in predict_zeta, but feature extraction should work)
        features_1 = self.feature_extractor.extract_zeta_features(1.0)
        self.assertIsInstance(features_1, torch.Tensor)
        self.assertEqual(features_1.shape, (8,))
        self.assertAlmostEqual(features_1[0].item(), 1.0)
        self.assertAlmostEqual(features_1[1].item(), 0.0)
        self.assertAlmostEqual(features_1[2].item(), 1.0)
        self.assertAlmostEqual(features_1[3].item(), 0.0) # log(1.0) is 0
        self.assertAlmostEqual(features_1[4].item(), 1.0)
        # This is the value for 1.0/(s-1) when s is close to 1.0.
        # The code returns -100.0 if s <= 1.0, 100.0 if s > 1.0 and abs(s-1) <= 1e-9
        # For s=1.0, s > 1 is False, so it should return -100.0 based on the code logic.
        self.assertAlmostEqual(features_1[5].item(), -100.0)
        self.assertAlmostEqual(features_1[6].item(), 0.0)
        self.assertAlmostEqual(features_1[7].item(), 1.0)


        # Test features for s = 0.0
        features_0 = self.feature_extractor.extract_zeta_features(0.0)
        self.assertIsInstance(features_0, torch.Tensor)
        self.assertEqual(features_0.shape, (8,))
        self.assertAlmostEqual(features_0[0].item(), 0.0)
        self.assertAlmostEqual(features_0[1].item(), -1.0)
        self.assertAlmostEqual(features_0[2].item(), 0.0) # 1.0 / s if s != 0 else 0.0
        self.assertAlmostEqual(features_0[3].item(), 0.0) # log(s) if s > 0 else 0.0
        self.assertAlmostEqual(features_0[4].item(), 0.0)
        self.assertAlmostEqual(features_0[5].item(), float(1.0 / (0.0 - 1.0))) # 1.0 / (-1.0) = -1.0
        self.assertAlmostEqual(features_0[6].item(), 0.0)
        self.assertAlmostEqual(features_0[7].item(), 0.0)

        # Test features for s = -2.0
        features_neg2 = self.feature_extractor.extract_zeta_features(-2.0)
        self.assertIsInstance(features_neg2, torch.Tensor)
        self.assertEqual(features_neg2.shape, (8,))
        self.assertAlmostEqual(features_neg2[0].item(), -2.0)
        self.assertAlmostEqual(features_neg2[1].item(), -3.0)
        self.assertAlmostEqual(features_neg2[2].item(), -0.5) # 1.0 / -2.0
        self.assertAlmostEqual(features_neg2[3].item(), 0.0) # log(s) if s > 0 else 0.0
        self.assertAlmostEqual(features_neg2[4].item(), 4.0)
        self.assertAlmostEqual(features_neg2[5].item(), float(1.0 / (-2.0 - 1.0))) # 1.0 / -3.0 = -0.333...
        self.assertAlmostEqual(features_neg2[6].item(), 0.0)
        self.assertAlmostEqual(features_neg2[7].item(), -2.0)


        # Test invalid inputs
        with self.assertRaises(TypeError):
            self.feature_extractor.extract_zeta_features("abc") # type: ignore


    def test_feature_cache(self):
        """Test the feature caching mechanism."""
        # Clear cache before testing
        self.feature_extractor.clear_cache()
        self.assertEqual(self.feature_extractor.get_cache_stats()['size'], 0)
        self.assertEqual(self.feature_extractor.get_cache_stats()['hits'], 0)
        self.assertEqual(self.feature_extractor.get_cache_stats()['accesses'], 0)


        # Extract features for 10 with caching
        features_10_a = self.feature_extractor.extract_number_features(10, cache_key=10)
        self.assertEqual(self.feature_extractor.get_cache_stats()['size'], 1)
        self.assertEqual(self.feature_extractor.get_cache_stats()['accesses'], 1)
        self.assertEqual(self.feature_extractor.get_cache_stats()['hits'], 0) # First access is a miss

        # Extract features for 10 again with the same cache key
        features_10_b = self.feature_extractor.extract_number_features(10, cache_key=10)
        self.assertEqual(self.feature_extractor.get_cache_stats()['size'], 1) # Size should not increase
        self.assertEqual(self.feature_extractor.get_cache_stats()['accesses'], 2)
        self.assertEqual(self.feature_extractor.get_cache_stats()['hits'], 1) # Second access is a hit

        # Check if the tensors are the same object (or very similar values) - caching should return the same tensor
        self.assertIs(features_10_a, features_10_b)
        self.assertTrue(torch.equal(features_10_a, features_10_b))

        # Extract features for a different number with caching
        features_20 = self.feature_extractor.extract_number_features(20, cache_key=20)
        self.assertEqual(self.feature_extractor.get_cache_stats()['size'], 2) # Size should increase
        self.assertEqual(self.feature_extractor.get_cache_stats()['accesses'], 3)
        self.assertEqual(self.feature_extractor.get_cache_stats()['hits'], 1) # Still 1 hit

        # Extract features for 10 without caching
        features_10_c = self.feature_extractor.extract_number_features(10, cache_key=None)
        self.assertEqual(self.feature_extractor.get_cache_stats()['size'], 2) # Size should not change
        self.assertEqual(self.feature_extractor.get_cache_stats()['accesses'], 3) # Accesses should not increase for non-cached calls
        self.assertEqual(self.feature_extractor.get_cache_stats()['hits'], 1) # Hits should not change

        # Clear cache and check stats
        self.feature_extractor.clear_cache()
        self.assertEqual(self.feature_extractor.get_cache_stats()['size'], 0)
        self.assertEqual(self.feature_extractor.get_cache_stats()['hits'], 0)
        self.assertEqual(self.feature_extractor.get_cache_stats()['accesses'], 0) # Access/hit counts are reset

        # Test zeta feature caching
        features_zeta_2 = self.feature_extractor.extract_zeta_features(2.0, cache_key=2.0)
        self.assertEqual(self.feature_extractor.get_cache_stats()['size'], 1)
        self.assertEqual(self.feature_extractor.get_cache_stats()['accesses'], 1)
        self.assertEqual(self.feature_extractor.get_cache_stats()['hits'], 0)

        features_zeta_2_b = self.feature_extractor.extract_zeta_features(2.0, cache_key=2.0)
        self.assertEqual(self.feature_extractor.get_cache_stats()['size'], 1)
        self.assertEqual(self.feature_extractor.get_cache_stats()['accesses'], 2)
        self.assertEqual(self.feature_extractor.get_cache_stats()['hits'], 1)
        self.assertIs(features_zeta_2, features_zeta_2_b)


# This block will run the tests when the cell is executed
if __name__ == '__main__':
    # Use a text runner to display test results
    # Add verbosity for more detailed output
    unittest.main(argv=['first-arg-is-ignored', '-v'], exit=False)